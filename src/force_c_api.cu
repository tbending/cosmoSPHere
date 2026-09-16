/*
 * force_c_api.cu — C-linkage entry point for the GPU force pass.
 *
 * Mirrors phantom's structure: densityiterate and force are two separate calls
 * (deriv.F90 :139 and :195), so this is its own entry point.  It rebuilds no tree:
 * the Hilbert ordering, positions, h, gradh and the leaf bookkeeping were left in
 * gpuState() by densityiterate_gpu_c.  The work is done by computeForces (force.cu).
 *
 * PARTICLE ORDERING — read before adding an argument (applies in computeForces).
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

#include <chrono>
#include <cstdio>
#include <cstdlib>

extern "C" void force_gpu_c(
    int n,
    double pmass,
    const double* vx,
    const double* vy,
    const double* vz,
    const double* pro2,
    const double* spsound,
    const double* alphaAV,
    const double* u,
    double beta,
    double alphau,
    int disc_viscosity,   // nonzero: phantom's disc_viscosity form of the artificial viscosity
    double* fx,
    double* fy,
    double* fz,
    double* f4,
    double* vsigmax,
    double* divv)
{
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
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

    ForceFields f{vx, vy, vz, pro2, spsound, alphaAV, u,
                  fx, fy, fz, f4, vsigmax, divv};
    ForceTimings ft;
    computeForces(s, f, pmass, beta, alphau, disc_viscosity != 0, ft);

    // Same env gate as the density solve, so one setting shows the whole picture.
    // wall is measured on the host around the whole call; wall - gpusum is host-side
    // cost that none of the device phases see (allocation, vector construction).
    static const bool stats = (std::getenv("COSMO_DENS_STATS") != nullptr);
    if (stats)
    {
        const double wall = std::chrono::duration<double>(clk::now() - t0).count();
        const double gpu  = ft.hmaxUpsweep + ft.jleafBuild + ft.upload + ft.kernel + ft.download;
        std::fprintf(stderr, "COSMO_FORCE n=%d leaves=%d | upsweep=%.2f jbuild=%.2f "
                             "upload=%.2f kernel=%.2f download=%.2f | gpusum=%.2f "
                             "wall=%.2f unaccounted=%.2f\n",
                     s.ngas, s.nLeaves,
                     1e3*ft.hmaxUpsweep, 1e3*ft.jleafBuild,
                     1e3*ft.upload, 1e3*ft.kernel, 1e3*ft.download,
                     1e3*gpu, 1e3*wall, 1e3*(wall - gpu));
    }
}
