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

#include <thrust/gather.h>
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
static std::vector<thrust::device_vector<double>*> slotComponents(GpuState& s, int slot)
{
    switch (slot)
    {
        case COSMO_POS:       return {&s.x, &s.y, &s.z};
        case COSMO_HSML:      return {&s.h};
        case COSMO_VEL:       return {&s.vx, &s.vy, &s.vz};
        case COSMO_ACCEL:     return {&s.ax, &s.ay, &s.az};
        case COSMO_DENS_OUT:  return {&s.rho, &s.gradh};
        case COSMO_GRAD_OUT:  return {&s.divv, &s.xi, &s.ddivvdt};
        case COSMO_THERMO:    return {&s.pro2, &s.spsound, &s.alphaAV, &s.u};
        case COSMO_FORCE_OUT: return {&s.fx, &s.fy, &s.fz, &s.f4, &s.vsigmax, &s.divvF};
        default:              return {};
    }
}

//! @brief Checks every transfer makes.  Aborts rather than guess.
static std::vector<thrust::device_vector<double>*> checkedSlot(GpuState& s, int slot, int n,
                                                               const char* what)
{
    if (s.sizedFor != n)
    {
        std::fprintf(stderr, "FATAL: %s(slot=%d) with n=%d but the device arrays are "
                             "sized for %d\n", what, slot, n, s.sizedFor);
        std::abort();
    }
    auto comps = slotComponents(s, slot);
    if (comps.empty())
    {
        std::fprintf(stderr, "FATAL: %s called for slot %d, which has no arrays\n", what, slot);
        std::abort();
    }
    return comps;
}

//! @brief Requires an ordering, i.e. that a density solve has built the tree.
static void requireOrder(const GpuState& s, int slot, int n, const char* what)
{
    if ((int)s.order.size() != n)
    {
        std::fprintf(stderr, "FATAL: %s(slot=%d) with no ordering: a density solve must "
                             "have run first\n", what, slot);
        std::abort();
    }
}

/*! @brief Times a transfer and adds it to a counter, however the caller leaves.
 *
 * The events are the only honest measure here: the copies are asynchronous with respect
 * to the host, so timing them from the host would mostly measure the launch.
 */
struct XferTimer
{
    double& sink;
    cudaEvent_t t0, t1;

    explicit XferTimer(double& counter) : sink(counter)
    {
        checkGpuErrors(hipEventCreate(&t0));
        checkGpuErrors(hipEventCreate(&t1));
        HIP_CHECK(hipEventRecord(t0));
    }
    ~XferTimer()
    {
        HIP_CHECK(hipEventRecord(t1));
        checkGpuErrors(hipEventSynchronize(t1));
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        sink += ms * 1e-3;
        HIP_CHECK(hipEventDestroy(t0));
        HIP_CHECK(hipEventDestroy(t1));
    }
};

/*! @brief Send a slot to the device, optionally putting it in the device's own order.
 *
 * sorted = false is for before the tree for this particle set exists: the data lands in
 * phantom order and the tree build sorts it as part of producing the ordering.
 * sorted = true is for afterwards, when it has to be gathered through s.order to match
 * everything else on the device.  One call cannot choose between them, because during a
 * solve the PREVIOUS step's ordering is still there and looks perfectly valid.
 */
static void uploadSlot(int slot, const double* host, int n, bool sorted, const char* what)
{
    GpuState& s = gpuState();
    if (n <= 0) return;
    auto comps = checkedSlot(s, slot, n, what);
    if (sorted) requireOrder(s, slot, n, what);

    XferTimer timer(s.uploadSeconds);
    const size_t nbytes = static_cast<size_t>(n) * sizeof(double);
    for (size_t c = 0; c < comps.size(); ++c)
    {
        const double* hc = host + c*static_cast<size_t>(n);
        if (!sorted)
        {
            HIP_CHECK(hipMemcpy(rawPtr(*comps[c]), hc, nbytes, hipMemcpyHostToDevice));
        }
        else
        {
            // staged, because a gather cannot read and write the same array
            HIP_CHECK(hipMemcpy(rawPtr(s.xferStage), hc, nbytes, hipMemcpyHostToDevice));
            thrust::gather(s.order.begin(), s.order.end(), s.xferStage.begin(), comps[c]->begin());
        }
    }
}

/*! @brief How many components a slot has here.
 *
 * So the host can check its own table against this one at start-up.  The two lists --
 * ibun_* in phantom's gpu_arrays and CosmoSlot above -- are the same numbers by
 * convention, and a silent disagreement would transfer the wrong array rather than fail.
 * Returns 0 for a slot this side does not know.
 */
extern "C" int cosmo_slot_ncomp(int slot)
{
    return (int)slotComponents(gpuState(), slot).size();
}

//! @brief Send a slot in phantom's order; the tree build will sort it.
extern "C" void cosmo_upload(int slot, const double* host, int n)
{
    uploadSlot(slot, host, n, /*sorted=*/false, "cosmo_upload");
}

//! @brief Send a slot and put it in the device's order; needs a tree.
extern "C" void cosmo_upload_sorted(int slot, const double* host, int n)
{
    uploadSlot(slot, host, n, /*sorted=*/true, "cosmo_upload_sorted");
}

/*! @brief Copy a slot back, into ncomp contiguous runs of n doubles.
 *
 * Results are in Hilbert order; scattering through s.order puts them back in phantom's,
 * so the host never sees the sorted ordering.
 *
 * Time here, and in the uploads, is accumulated into the state and reported as upload=
 * and download= by COSMO_DENS_STATS: the transfers sit outside the compute calls now, so
 * the figures cover the whole step rather than one solve.
 */
extern "C" void cosmo_download(int slot, double* host, int n)
{
    GpuState& s = gpuState();
    if (n <= 0) return;
    auto comps = checkedSlot(s, slot, n, "cosmo_download");
    requireOrder(s, slot, n, "cosmo_download");

    XferTimer timer(s.downloadSeconds);
    const size_t nbytes = static_cast<size_t>(n) * sizeof(double);
    for (size_t c = 0; c < comps.size(); ++c)
    {
        thrust::scatter(comps[c]->begin(), comps[c]->end(), s.order.begin(), s.xferStage.begin());
        HIP_CHECK(hipMemcpy(host + c*static_cast<size_t>(n), rawPtr(s.xferStage), nbytes,
                            hipMemcpyDeviceToHost));
    }
}
