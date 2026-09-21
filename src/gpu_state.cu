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

void sizeParticleArrays(GpuState& s, int n)
{
    if (s.sizedFor == n) return;

    // Particle data and the solve's working set
    for (auto* v : {&s.x, &s.y, &s.z, &s.h, &s.rho, &s.gradh,
                    &s.vx, &s.vy, &s.vz, &s.ax, &s.ay, &s.az,
                    &s.sortTmp, &s.divv, &s.ddivvdt, &s.xi, &s.xferStage})
        v->resize(n);

    // Force pass inputs, per-particle factors and outputs
    for (auto* v : {&s.pro2, &s.spsound, &s.alphaAV, &s.u,
                    &s.hsqinv, &s.hinv, &s.rhoh, &s.rho1,
                    &s.grkfac, &s.pres, &s.auterm, &s.divfac,
                    &s.fx, &s.fy, &s.fz, &s.f4, &s.vsigmax, &s.divvF})
        v->resize(n);

    // Integer maps and scratch.  activeLeaves is NOT here: it is nLeaves long.
    for (auto* v : {&s.order, &s.particleLeaf, &s.converged,
                    &s.activeParticles, &s.activeTmp, &s.activeLeavesTmp})
        v->resize(n);

    s.keys.resize(n);

    s.sizedFor = n;
}
