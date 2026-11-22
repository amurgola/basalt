// include/basalt/core/suspendable.hpp - device memory a sleeping engine gives back.
//
// `alloc` stands where cudaMalloc would. Where the device has virtual memory management, a large allocation sits
// behind a reserved address range: `sleep` copies every such allocation to host memory and releases its VRAM, and
// `wake` backs the same addresses again and copies the bytes back. Captured graphs and kernel arguments hold
// addresses into these ranges, so nothing else has to know the engine slept.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace basalt::core::suspendable {

/// Device memory on the current device, as cudaMalloc gives it.
cudaError_t alloc_bytes(void** out, size_t bytes);
/// The same for a typed pointer.
template <class T> cudaError_t alloc(T** out, size_t bytes) {
    void* p = nullptr;
    const cudaError_t e = alloc_bytes(&p, bytes);
    *out = static_cast<T*>(p);
    return e;
}
/// Frees what `alloc` returned, asleep or awake.
cudaError_t free(void* p);
/// Whether `sleep` has anything to release: at least one allocation sits behind a reserved range.
bool available();
/// The device bytes `sleep` releases, which is also the host memory it takes.
uint64_t bytes();
/// Copies every allocation to host memory and releases its VRAM. The caller has made sure no work is running.
/// False with `err` when the host memory is not there or a copy failed: every allocation is on its card as before.
bool sleep(std::string& err);
/// Backs every range again and copies its bytes back. False with `err` when a card has no room: the allocations that
/// did not come back stay in host memory, and another `wake` goes on from there.
bool wake(std::string& err);
/// Whether an allocation is in host memory.
bool asleep();

}  // namespace basalt::core::suspendable
