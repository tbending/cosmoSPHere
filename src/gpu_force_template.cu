// Each thread will correspond to one particle with sorted label i when executing this kernel
#include "force.hpp"
#include "kernel.hpp"
#include "tree.cuh"

#include <cmath>

__global__ void sphForceKernelJList(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ vz,
    const double* __restrict__ h,
    const double* __restrict__ gradh,
    const double* __restrict__ pro2,
    const double* __restrict__ spsound,
    const double* __restrict__ alphaAV,
    const double* __restrict__ u,
    double* __restrict__ fx,
    double* __restrict__ fy,
    double* __restrict__ fz,
    double* __restrict__ f4,
    double* __restrict__ vsigmax,
    double* __restrict__ divv,
    int n,
    double pmass,
    double beta,
    double alphau,
    const int* __restrict__ particleLeaf,
    const int* __restrict__ jcount,
    const int* __restrict__ jlist,
    const unsigned* __restrict__ layout)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    const double xi = x[i], yi = y[i], zi = z[i];
    const double hi        = h[i];
    const double hi_sq_inv = 1.0 / (hi * hi);
    const double hi_4_inv  = hi_sq_inv * hi_sq_inv;

    // rhoh(hi), as in phantom's force.F90, not the summed density
    const double hfoh_i = sph::hfact / hi;
    const double rhoi   = pmass * hfoh_i * hfoh_i * hfoh_i;

    const double grad_i      = gradh[i];   // d(rho)/dh at fixed q, from the density solve
    const double dhdrhoi     = -hi / (3.0 * rhoi);
    const double omegai      = 1.0 - dhdrhoi * grad_i;
    const double omega_inv_i = 1 / omegai;

    const double pro2i   = pro2[i];
    const double vwavei  = spsound[i];
    const double alphai  = alphaAV[i];
    const double eni     = u[i];
    const double rho1i   = 1.0 / rhoi;
    const double pri     = pro2i * rhoi * rhoi;
    const double autermi = 0.5 * pmass * rho1i * alphau;

    double itermx = 0.0;   // force from neighbours inside h_i
    double itermy = 0.0;
    double itermz = 0.0;
    double jtermx = 0.0;   // force from neighbours inside h_j
    double jtermy = 0.0;
    double jtermz = 0.0;

    double f4sum     = 0.0;   // du/dt
    double vsigmax_i = 0.0;
    double divv_s    = 0.0;

    const int iLeaf = particleLeaf[i];
    const int jBase = iLeaf * MAX_J_PER_LEAF;
    const int nj    = jcount[iLeaf];

    for (int jl = 0; jl < nj; ++jl)
    {
        const int jLeaf = jlist[jBase + jl];
        for (unsigned j = layout[jLeaf]; j < layout[jLeaf + 1]; ++j)
        {
            double dx  = xi - x[j];
            double dy  = yi - y[j];
            double dz  = zi - z[j];
            double dr2 = dx*dx + dy*dy + dz*dz;
            if (!(dr2 > 0.0)) continue;
            double qij2 = (dx*dx + dy*dy + dz*dz) * hi_sq_inv;

            const double hj        = h[j];
            const double hj_sq_inv = 1.0 / (hj * hj);
            const double hj_4_inv  = hj_sq_inv * hj_sq_inv;

            double qj_ij2 = (dx*dx + dy*dy + dz*dz) * hj_sq_inv;

            double dr    = sqrt(dr2);
            double runix = dx / dr;   // unit vector (r_i - r_j) / |r_i - r_j|
            double runiy = dy / dr;
            double runiz = dz / dr;
            const double dvxij = vx[i] - vx[j];
            const double dvyij = vy[i] - vy[j];
            const double dvzij = vz[i] - vz[j];
            const double projv = dvxij * runix + dvyij * runiy + dvzij * runiz;

            // rhoh(hj)
            const double hfoh_j = sph::hfact / hj;
            const double rhoj   = pmass * hfoh_j * hfoh_j * hfoh_j;

            const double rho1j   = 1.0 / rhoj;
            const double pro2j   = pro2[j];
            const double vwavej  = spsound[j];
            const double alphaj  = alphaAV[j];
            const double enj     = u[j];

            const double prj     = pro2j * rhoj * rhoj;
            const double autermj = 0.5 * pmass * rho1j * alphau;

            // signal speeds: vsig* with alpha = 1 for the timestep,
            // vsigav* with the particle's alpha for the viscosity
            const double vsigi   = fmax(vwavei - beta * projv, 0.0);
            const double vsigavi = fmax(alphai * vwavei - beta * projv, 0.0);
            const double vsigj   = fmax(vwavej - beta * projv, 0.0);
            const double vsigavj = fmax(alphaj * vwavej - beta * projv, 0.0);

            const double pair_vsigmax = fmax(vsigi, vsigj);

            double qrho2i = 0.0;
            double qrho2j = 0.0;

            const double denij  = eni - enj;
            const double rhoav1 = 2.0 / (rhoi + rhoj);
            const double vsigu  = sqrt(fabs(pri - prj) * rhoav1);

            if (projv < 0.0)
            {
                qrho2i = -0.5 * rho1i * vsigavi * projv;
                qrho2j = -0.5 * rho1j * vsigavj * projv;
            }

            if (qij2 < sph::radk2)
            {
                vsigmax_i = fmax(vsigmax_i, pair_vsigmax);

                double qij, wij, grwij;
                qij = sqrt(qij2);
                sph::m4_kern(qij, wij, grwij);

                // div v, as in sphGradientsKernel
                const double rij1_divv       = 1.0 / (dr + 2.220446049250313e-16);
                const double rij1grkern_divv = rij1_divv * grwij;

                const double runix_divv = dx * rij1grkern_divv * pmass;
                const double runiy_divv = dy * rij1grkern_divv * pmass;
                const double runiz_divv = dz * rij1grkern_divv * pmass;

                divv_s += dvxij * runix_divv
                        + dvyij * runiy_divv
                        + dvzij * runiz_divv;

                const double hfacgrkerni = hi_4_inv * sph::cnormk * omega_inv_i;   // hfacgrkern in force.F90
                const double gradkerni   = grwij * hfacgrkerni;                    // F_ij(h_i) / omega_i

                const double gradpi = pmass * (pro2i + qrho2i) * gradkerni;

                itermx += -gradpi * runix;
                itermy += -gradpi * runiy;
                itermz += -gradpi * runiz;

                // du/dt: p dV work, viscous heating, conductivity
                const double pdvtermi     = pmass * pro2i * projv * gradkerni;
                const double dudtdissi    = pmass * qrho2i * projv * gradkerni;
                const double dendisstermi = vsigu * denij * autermi * gradkerni;

                f4sum += pdvtermi;
                f4sum += dudtdissi;
                f4sum += dendisstermi;
            }

            if (qj_ij2 < sph::radk2)
            {
                vsigmax_i = fmax(vsigmax_i, pair_vsigmax);

                double qj_ij, wj_ij, grwj_ij;
                qj_ij = sqrt(qj_ij2);
                sph::m4_kern(qj_ij, wj_ij, grwj_ij);

                const double dhdrhoj     = -hj / (3.0 * rhoj);
                const double grad_j      = gradh[j];
                const double omegaj      = 1.0 - dhdrhoj * grad_j;
                const double omega_inv_j = 1 / omegaj;

                double hfacgrkernj = hj_4_inv * sph::cnormk * omega_inv_j;
                double gradkernj   = grwj_ij * hfacgrkernj;   // F_ij(h_j) / omega_j

                const double gradpj = pmass * (pro2j + qrho2j) * gradkernj;

                jtermx -= gradpj * runix;
                jtermy -= gradpj * runiy;
                jtermz -= gradpj * runiz;

                const double dendisstermj = vsigu * denij * autermj * gradkernj;

                f4sum += dendisstermj;
            }
        }
    }

    // Dead particles (h <= 0) have rhoi <= 0: return zeros for them.
    if (!(rhoi > 0.0))
    {
        fx[i]      = 0.0;
        fy[i]      = 0.0;
        fz[i]      = 0;
        f4[i]      = 0;
        vsigmax[i] = 0.0;
        divv[i]    = 0.0;
        return;
    }

    fx[i]      = itermx + jtermx;
    fy[i]      = itermy + jtermy;
    fz[i]      = itermz + jtermz;
    f4[i]      = f4sum;
    vsigmax[i] = vsigmax_i;

    const double omega_inv_divv = (omegai > 0.0) ? (1.0 / omegai) : 1.0;
    const double term_divv      = sph::cnormk * omega_inv_divv * hi_4_inv / rhoi;

    divv[i] = -divv_s * term_divv;
}
