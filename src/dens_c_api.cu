/*
 * dens_c_api.cu — C-linkage entry point for the Cornerstone GPU density solver.
 *
 * Phantom's Fortran code cannot call solveDensH() directly (it takes
 * std::vector arguments).  This thin wrapper accepts flat C arrays,
 * copies them into the vectors solveDensH() expects, and writes back
 * the results.
 *
 * Outputs (all arrays length n unless noted):
 *   h        — converged smoothing lengths (in/out, updated in-place)
 *   rho      — SPH-summed density at the converged h (out)
 *   gradh    — d(rho)/d(h), fully normalised (out)
 *              Phantom wants 1/omega; the Fortran wrapper in
 *              gpu_dens_iface.F90 performs that conversion, and needs rho
 *              in order to do it.
 *   divv     — div v (out)
 *   dvdx     — velocity gradient tensor, 9*n, laid out to match Fortran's
 *              dvdx(1:9,i), i.e. dvdx[9*i + c] (out)
 *   ddivvdt  — d(div v)/dt for the Cullen & Dehnen switch (out)
 *
 * The last three replace the CPU densityiterate(icall=3) sweep that the
 * phantom GPU path used to run after the GPU solve; vx/vy/vz and ax/ay/az
 * (= fxyzu + fext) are the extra inputs they need.
 *
 * Naming convention: densityiterate_gpu_c mirrors the Fortran subroutine
 * densityiterate_gpu, with a _c suffix indicating the C binding.
 * The underscore appended by Fortran bind(C) is handled by the bind(C)
 * attribute on the Fortran side — the C symbol here has no trailing
 * underscore.
 */

#include "density.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void densityiterate_gpu_c(
    double*       h,         // in/out: smoothing lengths
    double*       rho,       // out:    density
    double*       gradh_out, // out:    d(rho)/d(h) normalised
    double*       divv,      // out:    div v
    double*       dvdx,      // out:    velocity gradient tensor, 9*n
    double*       ddivvdt,   // out:    d(div v)/dt
    const double* x,
    const double* y,
    const double* z,
    const double* vx,
    const double* vy,
    const double* vz,
    const double* ax,
    const double* ay,
    const double* az,
    int           n,
    double        pmass)
{
    // Set COSMO_DENS_STATS=1 for a one-line phase breakdown per solve on stderr.
    // Costs two clock reads when off.
    static const bool stats = (std::getenv("COSMO_DENS_STATS") != nullptr);
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    std::vector<double> h_vec(h, h + n);
    std::vector<double> rho_vec(n, 0.0);
    std::vector<double> gradh_vec(n, 0.0);
    const std::vector<double> x_vec(x, x + n);
    const std::vector<double> y_vec(y, y + n);
    const std::vector<double> z_vec(z, z + n);

    // Velocities, accelerations and the gradient outputs go straight through as
    // raw pointers — they are already flat arrays, so there is nothing to copy.
    GradFields grads{vx, vy, vz, ax, ay, az, divv, dvdx, ddivvdt};

    auto t1 = clk::now();
    DensTimings t = solveDensH(h_vec, rho_vec, gradh_vec, x_vec, y_vec, z_vec, pmass,
                               KernelMode::FLAT_PARTICLE, &grads);
    auto t2 = clk::now();

    for (int i = 0; i < n; ++i) {
        h[i]         = h_vec[i];
        rho[i]       = rho_vec[i];
        gradh_out[i] = gradh_vec[i];
    }
    auto t3 = clk::now();

    if (stats) {
        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const double solve = ms(t1, t2);
        const double gpu   = 1e3 * (t.upload + t.bboxAndSetup + t.keysAndSort + t.treeBuild
                                  + t.nodeCenters + t.jleafBuild + t.densKernel
                                  + t.gradJleafBuild + t.gradKernel + t.download);
        std::fprintf(stderr,
            "COSMO_STATS n=%d leaves=%d iters=%d | vecin=%.2f upload=%.2f bbox=%.2f "
            "keysort=%.2f tree=%.2f nodes=%.2f jbuild=%.2f nrkern=%.2f gjbuild=%.2f "
            "gradkern=%.2f download=%.2f | gpusum=%.2f "
            "solve=%.2f unaccounted=%.2f vecout=%.2f total=%.2f\n",
            t.nParticles, t.nLeavesOut, t.itersRun,
            ms(t0, t1), 1e3*t.upload, 1e3*t.bboxAndSetup, 1e3*t.keysAndSort,
            1e3*t.treeBuild, 1e3*t.nodeCenters, 1e3*t.jleafBuild, 1e3*t.densKernel,
            1e3*t.gradJleafBuild, 1e3*t.gradKernel, 1e3*t.download,
            gpu, solve, solve - gpu, ms(t2, t3), ms(t0, t3));
    }
}
