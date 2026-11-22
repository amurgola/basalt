// src/core/suspendable.cpp - see suspendable.hpp.
#include "basalt/core/suspendable.hpp"

#include "basalt/core/on_device.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/memory.hpp"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

namespace basalt::core::suspendable {

namespace {

// A smaller allocation stays with the runtime's allocator: a reserved range is a whole number of granules.
constexpr uint64_t kMinBytes = 32ull << 20;
constexpr uint64_t kHostAlign = 4096;

// How a sleeping allocation's host copy is held, fastest to copy first.
enum class Host { None, Mapped, Locked, Plain };

struct Block {
    uint8_t* va = nullptr;
    uint64_t bytes = 0;              // the mapped size, a whole number of granules
    unsigned long long handle = 0;   // its physical memory; 0 while its bytes are in host memory
    int device = 0;
    uint8_t* host = nullptr;         // its bytes while asleep
    Host held = Host::None;
    uint64_t host_mapped = 0;        // what vmm::free_host takes back
};

std::mutex mu;
std::vector<Block> blocks;
bool keep_host = false;   // --keep-sleep-ram

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// Page-locked memory the device reads directly copies several times faster than ordinary memory, which is the
// fallback when the lock is refused.
bool host_alloc(Block& b) {
    b.host = vmm::alloc_host(b.bytes, b.host_mapped);
    if (b.host != nullptr) { b.held = Host::Mapped; return true; }
    void* p = nullptr;
    if (cudaHostAlloc(&p, (size_t) b.bytes, cudaHostAllocDefault) == cudaSuccess) {
        b.host = (uint8_t*) p;
        b.held = Host::Locked;
        return true;
    }
    (void) cudaGetLastError();
    b.host = (uint8_t*) platform::alloc_aligned((size_t) b.bytes, kHostAlign);
    b.held = b.host != nullptr ? Host::Plain : Host::None;
    return b.host != nullptr;
}

void host_free(Block& b) {
    if (b.held == Host::Mapped) vmm::free_host(b.host, b.host_mapped);
    else if (b.held == Host::Locked) (void) cudaFreeHost(b.host);
    else if (b.held == Host::Plain) platform::free_aligned(b.host);
    b.host = nullptr;
    b.held = Host::None;
    b.host_mapped = 0;
}

bool copy(void* dst, const void* src, uint64_t bytes) {
    if (cudaMemcpy(dst, src, (size_t) bytes, cudaMemcpyDefault) == cudaSuccess) return true;
    (void) cudaGetLastError();
    return false;
}

// The thread giving the last wake's host copies back. Never destroyed: a process may end while it runs.
std::thread* releasing = nullptr;

// Waits for those copies to be given back (under mu).
void settle() {
    if (releasing == nullptr) return;
    releasing->join();
    delete releasing;
    releasing = nullptr;
}

}  // namespace

cudaError_t alloc_bytes(void** out, size_t n) {
    *out = nullptr;
    const uint64_t granule = n >= kMinBytes && vmm::supported() ? vmm::granularity() : 0;
    if (granule == 0) return cudaMalloc(out, n);
    const uint64_t bytes = (n + granule - 1) / granule * granule;
    uint8_t* va = vmm::reserve(bytes);
    if (va == nullptr) return cudaMalloc(out, n);
    const unsigned long long handle = vmm::map(va, bytes);
    if (handle == 0) {
        vmm::release_range(va, bytes);
        return cudaErrorMemoryAllocation;
    }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) device = 0;
    std::lock_guard<std::mutex> lock(mu);
    Block b;
    b.va = va;
    b.bytes = bytes;
    b.handle = handle;
    b.device = device;
    blocks.push_back(b);
    *out = va;
    return cudaSuccess;
}

cudaError_t free(void* p) {
    if (p == nullptr) return cudaSuccess;
    Block b;
    {
        std::lock_guard<std::mutex> lock(mu);
        const auto it = std::find_if(blocks.begin(), blocks.end(), [&](const Block& k) { return k.va == p; });
        if (it == blocks.end()) return cudaFree(p);
        b = *it;
        blocks.erase(it);
        if (blocks.empty()) settle();   // the last allocation goes: nothing of a wake is left behind it
    }
    const OnDevice on(b.device);
    if (b.handle != 0) (void) vmm::unmap(b.va, b.bytes, b.handle);
    host_free(b);
    vmm::release_range(b.va, b.bytes);
    return cudaSuccess;
}

bool available() {
    std::lock_guard<std::mutex> lock(mu);
    return !blocks.empty();
}

uint64_t bytes() {
    std::lock_guard<std::mutex> lock(mu);
    uint64_t n = 0;
    for (const Block& b : blocks) n += b.bytes;
    return n;
}

void keep_host_memory(bool on) {
    std::lock_guard<std::mutex> lock(mu);
    keep_host = on;
}

uint64_t host_bytes() {
    std::lock_guard<std::mutex> lock(mu);
    uint64_t n = 0;
    for (const Block& b : blocks)
        if (b.host != nullptr) n += b.bytes;
    return n;
}

bool asleep() {
    std::lock_guard<std::mutex> lock(mu);
    return std::any_of(blocks.begin(), blocks.end(), [](const Block& b) { return b.handle == 0; });
}

bool sleep(std::string& err) {
    std::lock_guard<std::mutex> lock(mu);
    settle();
    const auto t0 = Clock::now();
    // every host copy first: no VRAM is released until all of them are held
    std::vector<Block*> moved;
    uint64_t n = 0, direct = 0;
    double taking = 0.0;
    for (Block& b : blocks) {
        if (b.handle == 0) continue;
        const OnDevice on(b.device);
        const auto t1 = Clock::now();
        const bool held = cudaDeviceSynchronize() == cudaSuccess && (b.host != nullptr || host_alloc(b));
        taking += seconds_since(t1);
        if (!held || !copy(b.host, b.va, b.bytes)) {
            err = held ? "copying the device memory to host memory failed"
                       : "no host memory for " + std::to_string(b.bytes >> 20) + " MiB of device memory";
            host_free(b);
            for (Block* m : moved) host_free(*m);
            return false;
        }
        moved.push_back(&b);
        n += b.bytes;
        if (b.held != Host::Plain) direct += b.bytes;
    }
    for (Block* m : moved) {
        const OnDevice on(m->device);
        (void) vmm::unmap(m->va, m->bytes, m->handle);
        m->handle = 0;
    }
    basalt::log::info("basalt sleep", "%.2f GiB of device memory in host memory in %.2f s (%.2f s taking the memory, "
                      "%.2f GiB of it page-locked); its VRAM is free",
                      (double) n / 1073741824.0, seconds_since(t0), taking, (double) direct / 1073741824.0);
    return true;
}

bool wake(std::string& err) {
    std::lock_guard<std::mutex> lock(mu);
    settle();
    const auto t0 = Clock::now();
    uint64_t n = 0;
    double backing = 0.0;
    std::vector<Block> spent;   // the host copies whose bytes are back on their card
    for (Block& b : blocks) {
        if (b.handle != 0) continue;
        const OnDevice on(b.device);
        const auto t1 = Clock::now();
        b.handle = vmm::map(b.va, b.bytes);
        backing += seconds_since(t1);
        if (b.handle == 0) {
            err = "no device memory for " + std::to_string(b.bytes >> 20) + " MiB of the sleeping state";
            break;
        }
        if (!copy(b.va, b.host, b.bytes)) {
            (void) vmm::unmap(b.va, b.bytes, b.handle);
            b.handle = 0;
            err = "copying the sleeping state back to the device failed";
            break;
        }
        n += b.bytes;
        if (keep_host) continue;   // the next sleep copies into the same memory
        spent.push_back(b);
        b.host = nullptr;
        b.held = Host::None;
        b.host_mapped = 0;
    }
    if (err.empty())
        basalt::log::info("basalt sleep", "%.2f GiB back on the device in %.2f s (%.2f s backing the addresses)",
                          (double) n / 1073741824.0, seconds_since(t0), backing);
    // giving the page-locked copies back takes as long as taking them did: the engine serves meanwhile
    if (!spent.empty())
        releasing = new std::thread([spent]() mutable {
            for (Block& b : spent) {
                const OnDevice on(b.device);
                host_free(b);
            }
        });
    return err.empty();
}

}  // namespace basalt::core::suspendable
