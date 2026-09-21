/*
 * arrays_c_api.cu — the array interface: sizing the device side, and moving results off it.
 *
 * The host owns the data and decides what moves.  So the particle-length device arrays
 * are sized here, once per particle count, from the host's gpu_arrays_init, rather than
 * by whichever kernel first happened to touch one; and the results are fetched by the
 * host asking for a slot, rather than by the compute entry points copying on their way
 * out.
 *
 * The tree-shaped arrays are not covered by the sizing — their lengths follow nLeaves
 * and numNodes, which do not exist until the tree has been built.  See sizeParticleArrays.
 *
 * Uploads still happen inside the compute entry points; moving them is the other half of
 * this split and is harder, because an upload before the tree exists lands in phantom
 * order and one after it has to be gathered into Hilbert order.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <thrust/scatter.h>

#include "arrays.hpp"
#include "gpu_check.hpp"
#include "gpu_state.hpp"
#include "util/cuda_utils.hpp"

//! @brief Size the device's particle-length arrays for n particles.  Idempotent.
extern "C" void cosmo_arrays_init(int n)
{
    if (n <= 0) return;
    sizeParticleArrays(gpuState(), n);
}

//! @brief The device arrays a slot stands for, in the host's component order.
static std::vector<const thrust::device_vector<double>*> slotComponents(const GpuState& s, int slot)
{
    switch (slot)
    {
        case COSMO_HSML:      return {&s.h};
        case COSMO_DENS_OUT:  return {&s.rho, &s.gradh};
        case COSMO_GRAD_OUT:  return {&s.divv, &s.xi, &s.ddivvdt};
        case COSMO_FORCE_OUT: return {&s.fx, &s.fy, &s.fz, &s.f4, &s.vsigmax, &s.divvF};
        default:              return {};
    }
}

/*! @brief Copy a slot back to the host, into ncomp contiguous runs of n doubles.
 *
 * Results are in Hilbert order; s.order maps sorted index -> phantom index, so each
 * component is scattered into the staging array and copied out in phantom's order.  The
 * host never sees the sorted ordering.
 *
 * Time spent here is accumulated into the state and reported as download= by
 * COSMO_DENS_STATS, so the figure still covers the transfers now that they have moved
 * out of the compute calls.
 */
extern "C" void cosmo_download(int slot, double* host, int n)
{
    GpuState& s = gpuState();
    if (n <= 0) return;

    if (s.sizedFor != n)
    {
        std::fprintf(stderr,
            "FATAL: cosmo_download(slot=%d) with n=%d but the device arrays are sized "
            "for %d\n", slot, n, s.sizedFor);
        std::abort();
    }
    const auto comps = slotComponents(s, slot);
    if (comps.empty())
    {
        std::fprintf(stderr, "FATAL: cosmo_download called for slot %d, which is not an "
                             "output slot\n", slot);
        std::abort();
    }

    cudaEvent_t t0, t1;
    for (auto* e : {&t0, &t1}) checkGpuErrors(hipEventCreate(e));
    HIP_CHECK(hipEventRecord(t0));

    const size_t nbytes = static_cast<size_t>(n) * sizeof(double);
    for (size_t c = 0; c < comps.size(); ++c)
    {
        thrust::scatter(comps[c]->begin(), comps[c]->end(), s.order.begin(), s.dStage.begin());
        HIP_CHECK(hipMemcpy(host + c*static_cast<size_t>(n), rawPtr(s.dStage), nbytes,
                            hipMemcpyDeviceToHost));
    }

    HIP_CHECK(hipEventRecord(t1));
    checkGpuErrors(hipEventSynchronize(t1));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    s.downloadSeconds += ms * 1e-3;
    for (auto* e : {&t0, &t1}) HIP_CHECK(hipEventDestroy(*e));
}
