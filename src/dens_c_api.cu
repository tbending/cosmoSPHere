/*
 * dens_c_api.cu — C-linkage entry point for the Cornerstone GPU density solver.
 *
 * C linkage for solveDensH().  Every array goes straight through: inputs are
 * uploaded from the caller's arrays and results are written into them by the
 * device copies, so nothing is staged here.
 *
 * Outputs (all arrays length n unless noted):
 *   h        — converged smoothing lengths (in/out, updated in-place)
 *   rho      — SPH-summed density at the converged h (out)
 *   gradh    — d(rho)/d(h), fully normalised (out)
 *              Phantom wants 1/omega; the Fortran wrapper in
 *              gpu_dens_iface.F90 performs that conversion, and needs rho
 *              in order to do it.
 *   divv     — div v (out)
 *   xi       — Cullen & Dehnen xi limiter (out), formed on the device from
 *              the velocity gradient tensor, which is not returned
 *   ddivvdt  — d(div v)/dt for the Cullen & Dehnen switch (out)
 *
 * periodic/box select periodic boundaries, as phantom's PERIODIC: pairs and tree
 * nodes are taken at their nearest image in the box.  The caller must have wrapped
 * the particles into the box (phantom's cross_boundary).
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
#include "gpu_state.hpp"
#include "gpu_check.hpp"
#include "kernel.hpp"
#include "util/cuda_utils.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void densityiterate_gpu_c(
    double*       h,         // in/out: smoothing lengths
    double*       rho,       // out:    density
    double*       gradh_out, // out:    d(rho)/d(h) normalised
    double*       divv,      // out:    div v
    double*       xi,        // out:    xi limiter
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
    double        pmass,
    int           periodic,  // nonzero: periodic in x, y and z
    const double* box,       // {xmin, xmax, ymin, ymax, zmin, zmax}; read only if periodic
    double        tolh,      // Newton tolerance on |dh/h|, phantom's tolh
    double        hfact)     // rho = pmass (hfact/h)^3, phantom's hfact
{
    // Set COSMO_DENS_STATS=1 for a one-line phase breakdown per solve on stderr.
    // Costs two clock reads when off.
    // The host sizes the device arrays through cosmo_arrays_init.  Refuse rather than
    // resize behind it: a mismatch here means the two sides disagree about the particle
    // count, which would otherwise show up as silent out-of-bounds device writes.
    if (gpuState().sizedFor != n)
    {
        std::fprintf(stderr,
            "FATAL: %s called with n=%d but the device arrays are sized for %d "
            "(cosmo_arrays_init not called, or called with a different count)\n",
            "densityiterate_gpu_c", n, gpuState().sizedFor);
        std::abort();
    }

    static const bool stats = (std::getenv("COSMO_DENS_STATS") != nullptr);
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    GradFields grads{vx, vy, vz, ax, ay, az, divv, xi, ddivvdt};

    auto t1 = clk::now();
    DensTimings t = solveDensH(h, rho, gradh_out, x, y, z, n, pmass,
                               KernelMode::FLAT_PARTICLE, &grads,
                               periodic ? box : nullptr, tolh, hfact);
    auto t2 = clk::now();

    // COSMO_MEM: one line per run with device memory in use and the host high-water
    // mark, taken after the first solve, so the largest problem that fits on a given
    // card can be extrapolated from a small run.
    static bool memReported = false;
    if (!memReported && std::getenv("COSMO_MEM"))
    {
        memReported = true;
        size_t freeB = 0, totalB = 0;
        HIP_CHECK(hipMemGetInfo(&freeB, &totalB));
        long hwmKB = 0;
        if (FILE* fp = std::fopen("/proc/self/status", "r"))
        {
            char line[256];
            while (std::fgets(line, sizeof line, fp))
                if (std::strncmp(line, "VmHWM:", 6) == 0) { std::sscanf(line + 6, "%ld", &hwmKB); break; }
            std::fclose(fp);
        }
        std::fprintf(stderr,
            "COSMO_MEM n=%d device_used=%.2f GB of %.2f GB (%.0f bytes/particle) host_peak=%.2f GB\n",
            n, (totalB - freeB) / 1e9, totalB / 1e9, double(totalB - freeB) / n, hwmKB / 1e6);
    }

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
            "solve=%.2f unaccounted=%.2f total=%.2f\n",
            t.nParticles, t.nLeavesOut, t.itersRun,
            ms(t0, t1), 1e3*t.upload, 1e3*t.bboxAndSetup, 1e3*t.keysAndSort,
            1e3*t.treeBuild, 1e3*t.nodeCenters, 1e3*t.jleafBuild, 1e3*t.densKernel,
            1e3*t.gradJleafBuild, 1e3*t.gradKernel, 1e3*t.download,
            gpu, solve, solve - gpu, ms(t0, t2));
    }
}

// The support radius, in units of h, of the kernel this library was built with, so the
// caller can check it matches its own (KERNEL=cubic -> 2, KERNEL=quintic -> 3).
extern "C" double cosmo_kernel_radius(void)
{
    return sph::radkernel;
}
