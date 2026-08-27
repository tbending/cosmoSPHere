/*
 * gpu_state.cu — the one GpuState shared by the density and force entry points.
 *
 * DELIBERATELY LEAKED.  A static thrust::device_vector runs its destructor during
 * static teardown, which happens AFTER the CUDA runtime has unloaded, so cudaFree
 * fails, thrust throws, and the process aborts with
 *     what(): CUDA free failed: cudaErrorCudartUnloading: driver shutting down
 * after the run has otherwise completed successfully.  Allocating with new and never
 * deleting avoids the exit-time CUDA call; the driver reclaims the device memory when
 * the process ends.  Any future static holding device memory needs the same.
 */

#include "gpu_state.hpp"

GpuState& gpuState()
{
    static GpuState* s = new GpuState();
    return *s;
}
