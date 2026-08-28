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

// Output arrays are deliberately absent: this currently builds the symmetric j-leaf
// list and nothing else, so there is nothing to write back yet.  fx/fy/fz/dudt get
// added to the signature together with the force kernel.
extern "C" void force_gpu_c(int n, double pmass)
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
    buildForceJLeafList(s, ft);

    // =======================================================================
    // >>> CALL THE FORCE KERNEL HERE <<<
    // =======================================================================

    (void)pmass;   // until the kernel lands

    // Same env gate as the density solve, so one setting shows the whole picture.
    static const bool stats = (std::getenv("COSMO_DENS_STATS") != nullptr);
    if (stats)
        std::fprintf(stderr, "COSMO_FORCE n=%d leaves=%d | upsweep=%.2f jbuild=%.2f "
                             "total=%.2f\n",
                     s.ngas, s.nLeaves,
                     1e3*ft.hmaxUpsweep, 1e3*ft.jleafBuild,
                     1e3*(ft.hmaxUpsweep + ft.jleafBuild));
}
