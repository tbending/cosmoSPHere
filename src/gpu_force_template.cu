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
    //const double* __restrict__ rho,
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
	//int i = activeParticles[idx];   // actual particle index

    const double xi = x[i], yi = y[i], zi = z[i];
    const double hi        = h[i];
    const double hi_sq_inv = 1.0 / (hi * hi);
	const double hi_4_inv = hi_sq_inv * hi_sq_inv;

	const double hfoh_i = sph::hfact / hi;
	const double rhoi = pmass * hfoh_i * hfoh_i * hfoh_i;   // rhoh(hi), to mirror phantom

	const double grad_i = gradh[i];//partial rho partial h at constant qij, already evaluated in the density loop
    const double dhdrhoi = -hi / (3.0 * rhoi);
    //const double grad_i  = gradhi * sph::cnormk * pmass * hi41; //= Sum[(-qij*f'(qij) - 3 * f(qij)) * Cnorm * m * 1/hi^4, j] = partial rhoi partial hi (while keeping q fixed), this is no longer needed bc of above
    const double omegai  = 1.0 - dhdrhoi * grad_i; //this is the actual omega used for force as well
	const double omega_inv_i = 1 / omegai;
	
	const double pro2i = pro2[i];
	const double vwavei = spsound[i];
	const double alphai = alphaAV[i];
	const double eni = u[i];
	const double rho1i  = 1.0 / rhoi;
	const double pri    = pro2i * rhoi * rhoi;
	const double autermi = 0.5 * pmass * rho1i * alphau;
	//const double rho_i = rho[i]
    //const double dhdrhoi = -hi / (3.0 * rho_i);
	//const double grad_i = grad_h[i]
    //double rhoi  = 0.0;
    //double gradhi = 0.0;
	
	double itermx = 0.0; //this corresponds to the 1st term in the force summation that includes neighbours within hi
	double itermy = 0.0;
	double itermz = 0.0;
	double jtermx = 0.0; //this correspond to the 2nd term in the force summation that includes neighbours within hj
	double jtermy = 0.0;
	double jtermz = 0.0;

	double f4sum = 0.0; //internal energy derivative
	double vsigmax_i = 0.0;
    double divv_s = 0.0;

    const int iLeaf = particleLeaf[i];
    const int jBase = iLeaf * MAX_J_PER_LEAF;
    const int nj    = jcount[iLeaf];

    for (int jl = 0; jl < nj; ++jl)
    {
        const int jLeaf = jlist[jBase + jl];
        for (unsigned j = layout[jLeaf]; j < layout[jLeaf + 1]; ++j)
        {
            double dx   = xi - x[j];
            double dy   = yi - y[j];
            double dz   = zi - z[j];
            double dr2 = dx*dx + dy*dy + dz*dz;
			if (!(dr2 > 0.0)) continue;
            double qij2 = (dx*dx + dy*dy + dz*dz) * hi_sq_inv;

//to find j-neighbours and calc. forces


    		const double hj        = h[j];
    		const double hj_sq_inv = 1.0 / (hj * hj);
			const double hj_4_inv = hj_sq_inv * hj_sq_inv;

			double qj_ij2 = (dx*dx + dy*dy + dz*dz) * hj_sq_inv;

			double dr = sqrt(dr2);
			double runix = dx / dr; //this is the x-component of rij^hat := (ri - rj) / abs(rij)
			double runiy = dy / dr; //same as above, but y-component
			double runiz = dz / dr; //same as above, but z-component
			const double dvxij = vx[i] - vx[j]; 
			const double dvyij = vy[i] - vy[j];
			const double dvzij = vz[i] - vz[j];
			const double projv = dvxij * runix + dvyij * runiy + dvzij * runiz;

			const double hfoh_j = sph::hfact / hj;
			const double rhoj = pmass * hfoh_j * hfoh_j * hfoh_j;   // rhoh(hj), to mirror phantom, so that rho as an array does not need to be copied to the GPU

			const double rho1j   = 1.0 / rhoj;
			const double pro2j   = pro2[j];
			const double vwavej  = spsound[j];
			const double alphaj  = alphaAV[j];
			const double enj     = u[j];
			
			const double prj     = pro2j * rhoj * rhoj;
			const double autermj = 0.5 * pmass * rho1j * alphau;
			
 			//const double vsigi = fmax(vwavei - beta * projv, 0.0);//this is for time-stepping with alpha taken to be 1
	
			//const double vsigavi =
			    //fmax(alphai * vwavei - beta * projv, 0.0);//this is for the qro in force terms
		
			//const double vwavej = spsound[j];
			//const double alphaj = alphaAV[j];
//vsig parameters
			const double vsigi =
			    fmax(vwavei - beta * projv, 0.0);
			
			const double vsigavi =
			    fmax(alphai * vwavei - beta * projv, 0.0);
			
			const double vsigj =
			    fmax(vwavej - beta * projv, 0.0);
			
			const double vsigavj =
			    fmax(alphaj * vwavej - beta * projv, 0.0);
			
			const double pair_vsigmax =
			    fmax(vsigi, vsigj);

			//vsigmax_i = fmax(vsigmax_i, vsigi);


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



				//mirroring sphGradientsKernel
                const double rij1_divv = 1.0 / (dr + 2.220446049250313e-16);
                const double rij1grkern_divv = rij1_divv * grwij;

                const double runix_divv = dx * rij1grkern_divv * pmass;
                const double runiy_divv = dy * rij1grkern_divv * pmass;
                const double runiz_divv = dz * rij1grkern_divv * pmass;

                divv_s += dvxij * runix_divv
                        + dvyij * runiy_divv
                        + dvzij * runiz_divv;



				const double hfacgrkerni = hi_4_inv * sph::cnormk * omega_inv_i;//mirrors the Fortran definition of hfacgrkern
				const double gradkerni = grwij * hfacgrkerni; //this is now (1/omega) * Fij(hi) as in the Fortran definition as well
				
				const double gradpi =
				    pmass * (pro2i + qrho2i) * gradkerni;

   				itermx += -gradpi * runix; //updating components of the 1st term in the force summation: 
				itermy += -gradpi * runiy;
				itermz += -gradpi * runiz;
				//double hfacgrkernj = hj
                //rhoi  += wij;
                //gradhi += -qij * grwij - 3.0 * wij;

				//calculate internal energy derivative

				//this is now pdv term to mirror the original Fortran code

				const double pdvtermi =
				    pmass * pro2i * projv * gradkerni;
				
				const double dudtdissi =
				    pmass * qrho2i * projv * gradkerni; //qrho2i does not appear in the paper directly but 
				
				const double dendisstermi =
				    vsigu * denij * autermi * gradkerni;


				f4sum += pdvtermi;
				f4sum += dudtdissi;
				f4sum += dendisstermi;

            }

			if (qj_ij2 < sph::radk2)

			{
							
				vsigmax_i = fmax(vsigmax_i, pair_vsigmax);
				
				double qj_ij, wj_ij, grwj_ij;
            	qj_ij = sqrt(qj_ij2);//might need to be changed to qj_ij2
            	sph::m4_kern(qj_ij, wj_ij, grwj_ij);
							
				// double pro2j = pro2[j];
												//fij = sph::cnormk * hi_4_inv * grwij
				//const double rhoj = rho[j];
			    const double dhdrhoj = -hj / (3.0 * rhoj);
				const double grad_j = gradh[j];
		    	const double omegaj  = 1.0 - dhdrhoj * grad_j; //this is the actual omega used for force as well
				const double omega_inv_j = 1 / omegaj;
												
				double hfacgrkernj = hj_4_inv * sph::cnormk * omega_inv_j;//mirrors the Fortran definition of hfacgrkern
				double gradkernj = grwj_ij * hfacgrkernj; //this is now (1/omega) * Fij(hi) as in the Fortran definition as wel

				const double gradpj =
				    pmass * (pro2j + qrho2j) * gradkernj;

				jtermx -= gradpj * runix;
				jtermy -= gradpj * runiy;
				jtermz -= gradpj * runiz;
										
				const double dendisstermj =
				    vsigu * denij * autermj * gradkernj;
				
				f4sum += dendisstermj;					
			}//end non-SPH neighbour

        }//end j-loop
    }//end jl-loop
	
	
    // Newton–Raphson update (identical to sphDensityKernel).
    //const double hi_old  = hi;
    //const double hi1     = 1.0 / hi;
    //const double hi31    = hi1 * hi1 * hi1;
    //const double hi41    = hi31 * hi1;
    //const double rho_i   = rhoi  * sph::cnormk * pmass * hi31;
    // Guard: if rhoi==0 (no neighbours found — j-leaf list too small or particle
    // outside domain), skip the Newton update and mark unconverged.  Without this
    // guard, h_new becomes NaN/Inf which cascades into the next iteration's
    // hmax-leaf computation, causing the DFS to visit the entire tree.
    if (!(rhoi > 0.0))
    {
        fx[i]       = 0.0;
        fy[i]     = 0.0;
        fz[i] = 0;
		f4[i] = 0;
		vsigmax[i] = 0.0;
        divv[i] = 0.0;
        return;
    }
	
	//keep the above safeguard for now
	
	fx[i] = itermx + jtermx;
	fy[i] = itermy + jtermy;
	fz[i] = itermz + jtermz;	
	f4[i] = f4sum;
	vsigmax[i] = vsigmax_i;

    const double omega_inv_divv = (omegai > 0.0) ? (1.0 / omegai) : 1.0;
    const double term_divv = sph::cnormk * omega_inv_divv * hi_4_inv / rhoi;

    divv[i] = -divv_s * term_divv;
    // dhdrho uses rhoh(h)=pmass*(hfact/h)^3, matching the CPU (part.F90 dhdrho),
    // NOT the SPH sum rho_i.  See sphDensityKernel for the rationale.

    // omega uses NORMALISED grad_i, matching Fortran dens.f90.
    // Guard: avoid sign flip when omegai ≤ 0 (mirrors Fortran finish_cell).
    //const double safe_omega = (omegai > 0.0) ? omegai : fabs(omegai + 1e-300);
    //const double hi_new_raw = hi - funci * dhdrhoi / safe_omega;
    // Clamp step to ±20% per iteration (mirrors Fortran finish_cell in dens.F90).
    //double hi_new;
    //if      (hi_new_raw > 1.2 * hi) hi_new = 1.2 * hi;
    //else if (hi_new_raw < 0.8 * hi) hi_new = 0.8 * hi;
    //else                             hi_new = hi_new_raw;

    //rho[i]       = rho_i;
    //gradh[i]     = grad_i;
    //h[i]         = hi_new;
    //..converged[i] = (fabs((hi_new - hi) / hi_old) < HTOL) ? 1 : 0;
}

