// src/core/suspendable.cpp - see suspendable.hpp.
#include "basalt/core/suspendable.hpp"

#include "basalt/core/on_device.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/memory.hpp"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <vector>

namespace basalt::core::suspendable {

namespace {

// A smaller allocation stays with the runtime's allocator: a reserved range is a whole number of granules.
constexpr uint64_t kMinBytes = 32ull << 20;
constexpr uint64_t kHostAlign = 4096;
// A host copy is held in pieces: the driver refuses one page-locked allocation much larger than this, and a piece
// can be copied while the next one is taken.
constexpr uint64_t kPiece = 2ull << 30;

struct Piece {
    uint8_t* at = nullptr;
    bool page_locked = false;   // how it was allocated: page-locked memory copies several times faster
};

struct Block {
    uint8_t* va = nullptr;
    uint64_t bytes = 0;              // the mapped size, a whole number of granules
    unsigned long long handle = 0;   // its physical memory; 0 while its bytes are in host memory
    int device = 0;
    std::vector<Piece> host;         // its bytes while asleep, and the memory a wake left in place
    bool spent = false;              // awake again: `host` is given back when the engine is idle
};

std::mutex mu;
std::vector<Block> blocks;
bool keep_host = false;   // --keep-sleep-ram

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

size_t pieces_of(const Block& b) { return (size_t) ((b.bytes + kPiece - 1) / kPiece); }
uint64_t piece_bytes(const Block& b, size_t i) { return std::min<uint64_t>(kPiece, b.bytes - (uint64_t) i * kPiece); }

// Page-locked when the lock is given, ordinary memory otherwise.
Piece take_piece(uint64_t bytes) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, (size_t) bytes, cudaHostAllocDefault) == cudaSuccess) return {(uint8_t*) p, true};
    (void) cudaGetLastError();
    return {(uint8_t*) platform::alloc_aligned((size_t) bytes, kHostAlign), false};
}

void host_free(Block& b) {
    for (const Piece& p : b.host) {
        if (p.page_locked) (void) cudaFreeHost(p.at);
        else platform::free_aligned(p.at);
    }
    b.host.clear();
}

// One stream for a block's copies: a piece's copy runs while the next piece is taken or queued.
struct CopyStream {
    cudaStream_t stream = nullptr;
    CopyStream() { if (cudaStreamCreate(&stream) != cudaSuccess) { (void) cudaGetLastError(); stream = nullptr; } }
    ~CopyStream() { if (stream != nullptr) (void) cudaStreamDestroy(stream); }
    bool copy(void* dst, const void* src, uint64_t bytes) {
        const cudaError_t e = stream != nullptr ? cudaMemcpyAsync(dst, src, (size_t) bytes, cudaMemcpyDefault, stream)
                                                : cudaMemcpy(dst, src, (size_t) bytes, cudaMemcpyDefault);
        if (e != cudaSuccess) (void) cudaGetLastError();
        return e == cudaSuccess;
    }
    bool finish() {
        if (stream == nullptr || cudaStreamSynchronize(stream) == cudaSuccess) return true;
        (void) cudaGetLastError();
        return false;
    }
};

// The block's bytes into host memory, taking piece by piece what a wake did not leave in place.
// `taking` grows by the time spent taking memory, `locked` by the page-locked bytes.
bool to_host(Block& b, double& taking, uint64_t& locked, std::string& err) {
    b.spent = false;
    CopyStream copies;
    for (size_t i = 0; i < pieces_of(b); ++i) {
        const uint64_t n = piece_bytes(b, i);
        if (i >= b.host.size()) {
            const auto t1 = Clock::now();
            b.host.push_back(take_piece(n));
            taking += seconds_since(t1);
            if (b.host.back().at == nullptr) {
                b.host.pop_back();
                err = "no host memory for " + std::to_string(b.bytes >> 20) + " MiB of device memory";
                return false;
            }
        }
        if (b.host[i].page_locked) locked += n;
        if (!copies.copy(b.host[i].at, b.va + (uint64_t) i * kPiece, n)) {
            err = "copying the device memory to host memory failed";
            return false;
        }
    }
    if (copies.finish()) return true;
    err = "copying the device memory to host memory failed";
    return false;
}

bool to_device(Block& b) {
    CopyStream copies;
    for (size_t i = 0; i < pieces_of(b); ++i)
        if (!copies.copy(b.va + (uint64_t) i * kPiece, b.host[i].at, piece_bytes(b, i))) return false;
    return copies.finish();
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
        for (size_t i = 0; i < b.host.size(); ++i) n += piece_bytes(b, i);
    return n;
}

bool asleep() {
    std::lock_guard<std::mutex> lock(mu);
    return std::any_of(blocks.begin(), blocks.end(), [](const Block& b) { return b.handle == 0; });
}

bool sleep(std::string& err) {
    std::lock_guard<std::mutex> lock(mu);
    const auto t0 = Clock::now();
    // every host copy first: no VRAM is released until all of them are held
    std::vector<Block*> moved;
    uint64_t n = 0, locked = 0;
    double taking = 0.0;
    for (Block& b : blocks) {
        if (b.handle == 0) continue;
        const OnDevice on(b.device);
        if (cudaDeviceSynchronize() != cudaSuccess) err = "the device did not finish its work";
        if (!err.empty() || !to_host(b, taking, locked, err)) {
            (void) cudaGetLastError();
            host_free(b);
            for (Block* m : moved) host_free(*m);
            return false;
        }
        moved.push_back(&b);
        n += b.bytes;
    }
    for (Block* m : moved) {
        const OnDevice on(m->device);
        (void) vmm::unmap(m->va, m->bytes, m->handle);
        m->handle = 0;
    }
    basalt::log::info("basalt sleep", "%.2f GiB of device memory in host memory in %.2f s (%.2f s taking the memory, "
                      "%.2f GiB of it page-locked); its VRAM is free",
                      (double) n / 1073741824.0, seconds_since(t0), taking, (double) locked / 1073741824.0);
    return true;
}

bool wake(std::string& err) {
    std::lock_guard<std::mutex> lock(mu);
    const auto t0 = Clock::now();
    uint64_t n = 0;
    double backing = 0.0;
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
        if (!to_device(b)) {
            (void) vmm::unmap(b.va, b.bytes, b.handle);
            b.handle = 0;
            err = "copying the sleeping state back to the device failed";
            break;
        }
        n += b.bytes;
        b.spent = !keep_host;   // kept, the next sleep copies into the same memory
    }
    if (err.empty())
        basalt::log::info("basalt sleep", "%.2f GiB back on the device in %.2f s (%.2f s backing the addresses)",
                          (double) n / 1073741824.0, seconds_since(t0), backing);
    return err.empty();
}

bool give_back_one() {
    std::lock_guard<std::mutex> lock(mu);
    for (Block& b : blocks) {
        if (!b.spent || b.host.empty()) continue;
        const OnDevice on(b.device);
        const Piece p = b.host.back();
        b.host.pop_back();
        if (p.page_locked) (void) cudaFreeHost(p.at);
        else platform::free_aligned(p.at);
        if (b.host.empty()) b.spent = false;
        return true;
    }
    return false;
}

}  // namespace basalt::core::suspendable
