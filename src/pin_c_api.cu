/*
 * pin_c_api.cu — register and unregister host buffers with the GPU driver.
 *
 * The device copies into and out of phantom's staging buffers.  Those are large
 * pageable allocations, and copies to unregistered pageable memory can be
 * pathological: on GH200 at 11.3M particles, 12% of the density downloads stalled
 * (up to 54 s each, 1843 s of a 1867 s run) while the median was 36 ms.
 * Registering the buffer once removes on-demand page population from every copy.
 *
 * The caller owns the lifetime: register right after allocating, unregister right
 * before freeing.  A registration that outlives its memory would leave the driver
 * holding a range the allocator may hand out again.
 *
 * Failure is not fatal -- the copies still work, just unpinned -- so both calls
 * clear the sticky error and return quietly.
 */

#include <cstddef>

#include "util/cuda_utils.hpp"

extern "C" void cosmo_pin_host(void* ptr, size_t nbytes)
{
    if (ptr == nullptr || nbytes == 0) return;
    if (hipHostRegister(ptr, nbytes, hipHostRegisterDefault) != hipSuccess)
        (void)hipGetLastError();
}

extern "C" void cosmo_unpin_host(void* ptr)
{
    if (ptr == nullptr) return;
    if (hipHostUnregister(ptr) != hipSuccess)
        (void)hipGetLastError();
}
