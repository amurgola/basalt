// Device memory a sleeping engine gives back (suspendable.hpp).  A large allocation sits behind a reserved range:
// sleeping releases its VRAM and keeps its bytes, waking puts the same bytes back at the same address, and freeing
// works asleep or awake.  A small allocation is the runtime's own and is never moved.  Needs a CUDA device with
// virtual memory management (exits 77 without one).
#include "basalt/core/suspendable.hpp"
#include "basalt/core/vmm.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {
int fail(const char* what, const std::string& err = "") {
    std::fprintf(stderr, "FAIL %s %s\n", what, err.c_str());
    return 1;
}

size_t free_vram() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f;
}

std::vector<uint8_t> pattern(size_t bytes, uint32_t seed) {
    std::vector<uint8_t> v(bytes);
    uint32_t x = seed;
    for (uint8_t& b : v) {
        x = x * 1664525u + 1013904223u;
        b = (uint8_t) (x >> 24);
    }
    return v;
}

bool put(void* dev, const std::vector<uint8_t>& bytes) {
    return cudaMemcpy(dev, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) == cudaSuccess;
}

bool holds(const void* dev, const std::vector<uint8_t>& want) {
    std::vector<uint8_t> got(want.size());
    return cudaMemcpy(got.data(), dev, got.size(), cudaMemcpyDeviceToHost) == cudaSuccess && got == want;
}
}  // namespace

int main() {
    namespace suspendable = basalt::core::suspendable;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    if (cudaSetDevice(0) != cudaSuccess || !basalt::core::vmm::supported()) {
        std::puts("no virtual memory management on the device: skipped");
        return 77;
    }
    const size_t big = 96ull << 20, odd = (64ull << 20) + 12345, small = 1ull << 20;
    const size_t slack = 8ull << 20;
    uint8_t* a = nullptr;
    float* b = nullptr;
    void* s = nullptr;
    if (suspendable::alloc(&a, big) != cudaSuccess || suspendable::alloc(&b, odd) != cudaSuccess ||
        suspendable::alloc(&s, small) != cudaSuccess)
        return fail("alloc");
    if (!suspendable::available() || suspendable::bytes() < big + odd) return fail("the large allocations sit behind no range");
    if (suspendable::bytes() >= big + odd + small + slack) return fail("the small allocation was given a range");
    const std::vector<uint8_t> pa = pattern(big, 1), pb = pattern(odd, 2), ps = pattern(small, 3);
    if (!put(a, pa) || !put(b, pb) || !put(s, ps)) return fail("filling the allocations");

    std::string err;
    const size_t awake = free_vram();
    if (suspendable::asleep()) return fail("asleep before any sleep");
    if (!suspendable::sleep(err)) return fail("sleep", err);
    if (!suspendable::asleep()) return fail("not asleep after a sleep");
    if (free_vram() + slack < awake + big + odd) return fail("a sleep released too little VRAM");
    if (!holds(s, ps)) return fail("the small allocation changed over a sleep");
    if (!suspendable::sleep(err)) return fail("a second sleep", err);
    if (!suspendable::wake(err)) return fail("wake", err);
    if (suspendable::asleep()) return fail("asleep after a wake");
    if (!holds(a, pa) || !holds(b, pb)) return fail("the bytes changed over a sleep");
    if (!suspendable::wake(err)) return fail("a second wake", err);

    // freed asleep: its range and its host copy go, the others wake as they were
    if (!suspendable::sleep(err)) return fail("the sleep before a free", err);
    if (suspendable::free(a) != cudaSuccess) return fail("freeing a sleeping allocation");
    if (!suspendable::wake(err)) return fail("the wake after a free", err);
    if (!holds(b, pb) || !holds(s, ps)) return fail("the bytes changed when a sleeping allocation was freed");

    // kept: the host memory of a sleep is still held after the wake, and the next sleep and wake use it
    // a wake leaves its host memory to be given back piece by piece
    if (suspendable::host_bytes() < odd) return fail("a wake left no host memory to give back");
    int given = 0;
    while (suspendable::give_back_one()) ++given;
    if (given == 0 || suspendable::host_bytes() != 0) return fail("the host memory of a wake was not given back");
    if (!holds(b, pb)) return fail("the bytes changed when the host memory was given back");
    suspendable::keep_host_memory(true);
    if (!suspendable::sleep(err) || !suspendable::wake(err)) return fail("a kept sleep and wake", err);
    if (suspendable::host_bytes() < odd) return fail("the host memory was not kept over a wake");
    const std::vector<uint8_t> pc = pattern(odd, 4);
    if (!put(b, pc) || !suspendable::sleep(err) || !suspendable::wake(err)) return fail("a second kept sleep and wake", err);
    if (!holds(b, pc)) return fail("the bytes changed over a kept sleep");
    suspendable::keep_host_memory(false);

    if (suspendable::free(b) != cudaSuccess || suspendable::free(s) != cudaSuccess || suspendable::free(nullptr) != cudaSuccess)
        return fail("free");
    if (suspendable::available() || suspendable::bytes() != 0) return fail("an allocation is left");
    std::puts("suspendable_test: ok");
    return 0;
}
