/*
 * force_c_api.cu — C-linkage entry point for the GPU force pass.
 *
 * Mirrors phantom's structure: densityiterate and force are two separate calls
 * (deriv.F90 :139 and :195), so this is its own entry point.  It rebuilds nothing —
 * the tree, the Hilbert-sorted particles, the converged h, rho, gradh, velocities and
 * the leaf bookkeeping were all left in gpuState() by densityiterate_gpu_c.
 *
 * PARTICLE ORDERING — read before adding an argument.
 * Phantom's arrays are in phantom's order; everything in the state is Hilbert-sorted.
 * s.order maps sorted index -> phantom index, so:
 *   - a NEW input must be gathered:   thrust::gather(order.begin(), order.end(),
 *                                                    uploaded.begin(), sorted.begin())
 *   - every output must be scattered: thrust::scatter(sorted.begin(), sorted.end(),
 *                                                     order.begin(), out.begin())
 * Getting this wrong does not crash, it silently permutes the particles.
 */

#include "force.hpp"
#include "gpu_state.hpp"

#include <cstdio>
#include <cstdlib>

#include "util/cuda_utils.hpp"

#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

// Output arrays are deliberately absent: this currently builds the symmetric j-leaf
// list and nothing else, so there is nothing to write back yet.  fx/fy/fz/dudt get
// added to the signature together with the force kernel.
extern "C" void force_gpu_c(
    int n,
    double pmass,
    const double* pro2,
    const double* spsound,
    const double* alphaAV,
    const double* u,
    double beta,
    double alphau,
    double* fx,
    double* fy,
    double* fz,
    double* f4,
    double* vsigmax)
{
    GpuState& s = gpuState();

    // Refuse rather than run on an absent or mismatched tree.  Repeated calls on the
    // same tree are legitimate (see GpuState::token).
    if (!s.readyForForce(n))
    {
        std::fprintf(stderr,
            "FATAL: force_gpu_c called with no GPU density solve for this particle set "
            "(n=%d state.ngas=%d token=%llu)\n",
            n, s.ngas, (unsigned long long)s.token);
        std::abort();
    }

    ForceTimings ft;
    buildForceJLeafList(s, ft); //this builds the neighbour lists for force calculation
	//nactive particles is not included here? Ignored as of 0829

    //preparation other arguments for force kernel here


	// Upload pro2 in PHANTOM order.
	thrust::device_vector<double> d_pro2_phantom(pro2, pro2 + n);
	thrust::device_vector<double> d_spsound_phantom(
    spsound, spsound + n);

	thrust::device_vector<double> d_alphaAV_phantom(
	    alphaAV, alphaAV + n);
	
	thrust::device_vector<double> d_u_phantom(
	    u, u + n);
	
	// Gather into the Hilbert order already stored in GpuState.
	thrust::device_vector<double> d_pro2(n);
	
	thrust::device_vector<double> d_spsound(n);
	thrust::device_vector<double> d_alphaAV(n);
	thrust::device_vector<double> d_u(n);

	thrust::gather(
	    s.order.begin(),
	    s.order.end(),
	    d_pro2_phantom.begin(),
	    d_pro2.begin());

	thrust::gather(
	    s.order.begin(),
	    s.order.end(),
	    d_spsound_phantom.begin(),
	    d_spsound.begin());
	
	thrust::gather(
	    s.order.begin(),
	    s.order.end(),
	    d_alphaAV_phantom.begin(),
	    d_alphaAV.begin());
	
	thrust::gather(
	    s.order.begin(),
	    s.order.end(),
	    d_u_phantom.begin(),
	    d_u.begin());


	thrust::device_vector<double> d_fx(n, 0.0);
	thrust::device_vector<double> d_fy(n, 0.0);
	thrust::device_vector<double> d_fz(n, 0.0);
	thrust::device_vector<double> d_f4(n, 0.0);
	thrust::device_vector<double> d_vsigmax(n, 0.0);
	
	constexpr int forceBlockSize = 256;
	


    //end preparation for other arguments

	//conventionally there are 256 threads per block so use 256 here

	// =======================================================================
	// >>> CALL THE FORCE KERNEL HERE <<<
	sphForceKernelJList<<<iceil(n, forceBlockSize), forceBlockSize>>>(
	    rawPtr(s.x), rawPtr(s.y), rawPtr(s.z),
	    rawPtr(s.vx), rawPtr(s.vy), rawPtr(s.vz),
	    rawPtr(s.h), rawPtr(s.rho), rawPtr(s.gradh),
	    rawPtr(d_pro2),
	    rawPtr(d_spsound),
	    rawPtr(d_alphaAV),
	    rawPtr(d_u),
	    rawPtr(d_fx), rawPtr(d_fy), rawPtr(d_fz), rawPtr(d_f4),
	    rawPtr(d_vsigmax),
	    n, pmass, beta, alphau,
	    rawPtr(s.particleLeaf),
	    rawPtr(s.jcount), rawPtr(s.jlist),
	    rawPtr(s.layout));
    // =======================================================================
	checkGpuErrors(cudaGetLastError());
	HIP_CHECK(hipDeviceSynchronize()); //check errors
    //scatter and download results:

	thrust::device_vector<double> d_out(n);
	
	thrust::scatter(
	    d_fx.begin(), d_fx.end(),
	    s.order.begin(), d_out.begin());
	HIP_CHECK(hipMemcpy(
	    fx, rawPtr(d_out),
	    static_cast<size_t>(n) * sizeof(double),
	    hipMemcpyDeviceToHost));
	
	thrust::scatter(
	    d_fy.begin(), d_fy.end(),
	    s.order.begin(), d_out.begin());
	HIP_CHECK(hipMemcpy(
	    fy, rawPtr(d_out),
	    static_cast<size_t>(n) * sizeof(double),
	    hipMemcpyDeviceToHost));
	
	thrust::scatter(
	    d_fz.begin(), d_fz.end(),
	    s.order.begin(), d_out.begin());
	HIP_CHECK(hipMemcpy(
	    fz, rawPtr(d_out),
	    static_cast<size_t>(n) * sizeof(double),
	    hipMemcpyDeviceToHost));
	
	thrust::scatter(
	    d_f4.begin(), d_f4.end(),
	    s.order.begin(), d_out.begin());
	HIP_CHECK(hipMemcpy(
	    f4, rawPtr(d_out),
	    static_cast<size_t>(n) * sizeof(double),
	    hipMemcpyDeviceToHost));

	thrust::scatter(
	    d_vsigmax.begin(),
	    d_vsigmax.end(),
	    s.order.begin(),
	    d_out.begin());
	
	HIP_CHECK(hipMemcpy(
	    vsigmax,
	    rawPtr(d_out),
	    static_cast<size_t>(n) * sizeof(double),
	    hipMemcpyDeviceToHost));

	//end scatter and download results

    //(void)pmass;   // until the kernel lands

    // Same env gate as the density solve, so one setting shows the whole picture.
    static const bool stats = (std::getenv("COSMO_DENS_STATS") != nullptr);
    if (stats)
        std::fprintf(stderr, "COSMO_FORCE n=%d leaves=%d | upsweep=%.2f jbuild=%.2f "
                             "total=%.2f\n",
                     s.ngas, s.nLeaves,
                     1e3*ft.hmaxUpsweep, 1e3*ft.jleafBuild,
                     1e3*(ft.hmaxUpsweep + ft.jleafBuild));
}
