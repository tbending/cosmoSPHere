/*
 * kernel.hpp — M4 cubic spline SPH kernel.
 *
 * Direct port of the branch-free Fortran m4_kern from cosmoSPHere/src/kernel.f90.
 * Decorated HOST_DEVICE_FUN so it compiles for both CPU (host) and GPU (device).
 *
 * Physical constants match the Fortran code exactly:
 *   radkernel = 2.0  (kernel support radius in units of h)
 *   hfact     = 1.2  (h-rho relationship: rho = pmass * (hfact/h)^3)
 *   cnormk    = 1/pi (3D M4 normalisation)
 */

#pragma once

#include <cmath>
#include "util/annotation.hpp"

namespace sph
{

// Kernel constants — identical values to the Fortran module.
// Use a literal for pi so the value is available in device code without
// relying on M_PI being defined by <cmath> in every compiler context.
constexpr double pi        = 3.141592653589793;
constexpr double radkernel = 2.0;
constexpr double radk2     = 4.0;   // radkernel^2
constexpr double hfact     = 1.2;
constexpr double cnormk    = 1.0 / pi;  // 3D M4 normalisation

/*! @brief Branch-free M4 cubic spline kernel and its derivative.
 *
 * @param[in]  q     dimensionless separation q = r/h, must be >= 0
 * @param[out] wij   kernel value  W(q,h) / (cnormk * h^3 * pmass) [accumulated dimensionless]
 * @param[out] grwij dW/dq         [accumulated dimensionless]
 *
 * The caller multiplies by cnormk*pmass/h^3 (and cnormk*pmass/h^4 for gradient).
 */
HOST_DEVICE_FUN inline void m4_kern(double q, double& wij, double& grwij)
{
    double cA1 = (2.0 - q > 0.0) ? (2.0 - q) : 0.0;
    double cB1 = (1.0 - q > 0.0) ? (1.0 - q) : 0.0;

    double cA2 = cA1 * cA1;
    double cA3 = cA2 * cA1;
    double cB2 = cB1 * cB1;
    double cB3 = cB2 * cB1;

    wij   =  0.25  * (cA3 - 4.0 * cB3);
    grwij =  0.25  * (-3.0 * cA2 + 12.0 * cB2);
}

} // namespace sph
