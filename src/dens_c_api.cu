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
    std::vector<double> h_vec(h, h + n);
    std::vector<double> rho_vec(n, 0.0);
    std::vector<double> gradh_vec(n, 0.0);
    const std::vector<double> x_vec(x, x + n);
    const std::vector<double> y_vec(y, y + n);
    const std::vector<double> z_vec(z, z + n);

    // Velocities, accelerations and the gradient outputs go straight through as
    // raw pointers — they are already flat arrays, so there is nothing to copy.
    GradFields grads{vx, vy, vz, ax, ay, az, divv, dvdx, ddivvdt};

    solveDensH(h_vec, rho_vec, gradh_vec, x_vec, y_vec, z_vec, pmass,
               KernelMode::FLAT_PARTICLE, &grads);

    for (int i = 0; i < n; ++i) {
        h[i]         = h_vec[i];
        rho[i]       = rho_vec[i];
        gradh_out[i] = gradh_vec[i];
    }
}
