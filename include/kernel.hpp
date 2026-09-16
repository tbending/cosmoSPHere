/*
 * kernel.hpp — the SPH smoothing kernel: M4 cubic spline (default) or M6 quintic.
 *
 * Chosen at compile time, as in phantom (KERNEL=cubic|quintic there, passed through
 * to this build, which defines COSMO_KERNEL_QUINTIC for the quintic).  Everything that
 * depends on the kernel reads it from here: sph::kern, the support radius radkernel
 * (the tree walks search out to radkernel*h) and the normalisation cnormk.  hfact is
 * not a kernel constant: phantom passes its runtime value with each density solve;
 * hfact_default is only phantom's default for the kernel.
 *
 * Decorated HOST_DEVICE_FUN so it compiles for both CPU (host) and GPU (device).
 * Constants and formulae match phantom's kernel_cubic.f90 and kernel_quintic.f90.
 */

#pragma once

#include <cmath>
#include "util/annotation.hpp"

namespace sph
{

// Use a literal for pi so the value is available in device code without
// relying on M_PI being defined by <cmath> in every compiler context.
constexpr double pi = 3.141592653589793;

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

/*! @brief M6 quintic spline and its derivative, as phantom's kernel_quintic.f90.
 *
 * Same conventions as m4_kern; q2 = q*q as the caller already has it.
 */
HOST_DEVICE_FUN inline void m6_kern(double q2, double q, double& wij, double& grwij)
{
    if (q < 1.0)
    {
        const double q4 = q2 * q2;
        wij   = -10.0 * q4 * q + 30.0 * q4 - 60.0 * q2 + 66.0;
        grwij = q * (-50.0 * q2 * q + 120.0 * q2 - 120.0);
    }
    else if (q < 2.0)
    {
        const double a = q - 3.0, b = q - 2.0;
        const double a2 = a * a, b2 = b * b;
        wij   = -(a2 * a2 * a) + 6.0 * (b2 * b2 * b);
        grwij = -5.0 * (a2 * a2) + 30.0 * (b2 * b2);
    }
    else if (q < 3.0)
    {
        const double a = q - 3.0, a2 = a * a;
        wij   = -(a2 * a2 * a);
        grwij = -5.0 * (a2 * a2);
    }
    else
    {
        wij = 0.0; grwij = 0.0;
    }
}

#ifdef COSMO_KERNEL_QUINTIC
constexpr char   kernelname[]  = "M_6 quintic";
constexpr double radkernel     = 3.0;
constexpr double radk2         = 9.0;                  // radkernel^2
constexpr double cnormk        = 1.0 / (120.0 * pi);   // 3D M6 normalisation
constexpr double hfact_default = 1.0;
HOST_DEVICE_FUN inline void kern(double q2, double q, double& wij, double& grwij)
{
    m6_kern(q2, q, wij, grwij);
}
#else
constexpr char   kernelname[]  = "M_4 cubic";
constexpr double radkernel     = 2.0;
constexpr double radk2         = 4.0;                  // radkernel^2
constexpr double cnormk        = 1.0 / pi;             // 3D M4 normalisation
constexpr double hfact_default = 1.2;
HOST_DEVICE_FUN inline void kern(double /*q2*/, double q, double& wij, double& grwij)
{
    m4_kern(q, wij, grwij);
}
#endif

} // namespace sph
