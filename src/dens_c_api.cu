/*
 * dens_c_api.cu — C-linkage entry point for the Cornerstone GPU density solver.
 *
 * Phantom's Fortran code cannot call solveDensH() directly (it takes
 * std::vector arguments).  This thin wrapper accepts flat C arrays,
 * copies them into the vectors solveDensH() expects, and writes back
 * the results.
 *
 * Outputs (all arrays length n):
 *   h        — converged smoothing lengths (in/out, updated in-place)
 *   rho      — density  rho_i = pmass * (hfact/h_i)^3  (out)
 *   gradh    — d(rho)/d(h), fully normalised (out)
 *              Phantom wants 1/omega; the Fortran wrapper in
 *              gpu_dens_iface.F90 performs that conversion.
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
    const double* x,
    const double* y,
    const double* z,
    int           n,
    double        pmass)
{
    std::vector<double> h_vec(h, h + n);
    std::vector<double> rho_vec(n, 0.0);
    std::vector<double> gradh_vec(n, 0.0);
    const std::vector<double> x_vec(x, x + n);
    const std::vector<double> y_vec(y, y + n);
    const std::vector<double> z_vec(z, z + n);

    solveDensH(h_vec, rho_vec, gradh_vec, x_vec, y_vec, z_vec, pmass);

    for (int i = 0; i < n; ++i) {
        h[i]         = h_vec[i];
        rho[i]       = rho_vec[i];
        gradh_out[i] = gradh_vec[i];
    }
}
