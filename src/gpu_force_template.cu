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

    const double* __restrict__ h,           // in
    const double* __restrict__ rho,         // in, this is the rho found from the density calculation earlier
    const double* __restrict__ gradh,       // in, this is inputed after sphGradientsKernel computes it
	const double* __restrict__ pro2,         // in, this is P/rho^2 computed by get_stress in force.f90 in PHANTOM
    //int*          converged,   // in
	double* __restrict__ fx,
    double* __restrict__ fy,
    double* __restrict__ fz,   
	double* __restrict__ f4,   
	int           n,        
    //int           nActive,
    //const int*    __restrict__ activeParticles,
    double        pmass,
    const int*       __restrict__ particleLeaf,
    const int*       __restrict__ jcount,
    const int*       __restrict__ jlist,
    const unsigned*  __restrict__ layout)//force array
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;    
	//int i = activeParticles[idx];   // actual particle index

    const double xi = x[i], yi = y[i], zi = z[i];
    const double hi        = h[i];
    const double hi_sq_inv = 1.0 / (hi * hi);
	const double hi_4_inv = hi_sq_inv * hi_sq_inv;

	const double rhoi = rho[i];
	const double grad_i = gradh[i];//partial rho partial h at constant qij, already evaluated in the density loop
    const double dhdrhoi = -hi / (3.0 * rhoi);
    //const double grad_i  = gradhi * sph::cnormk * pmass * hi41; //= Sum[(-qij*f'(qij) - 3 * f(qij)) * Cnorm * m * 1/hi^4, j] = partial rhoi partial hi (while keeping q fixed), this is no longer needed bc of above
    const double omegai  = 1.0 - dhdrhoi * grad_i; //this is the actual omega used for force as well
	const double omega_inv_i = 1 / omegai;
	
	const double pro2i = pro2[i];
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

            if (qij2 < sph::radk2)
            {
                double qij, wij, grwij;
                qij = sqrt(qij2);
                sph::m4_kern(qij, wij, grwij);

				double hfacgrkerni = hi_4_inv * sph::cnormk * omega_inv_i;//mirrors the Fortran definition of hfacgrkern
				double gradkerni = grwij * hfacgrkerni; //this is now (1/omega) * Fij(hi) as in the Fortran definition as well
				itermx += -pmass * pro2i * gradkerni * runix; //updating components of the 1st term in the force summation: 
				itermy += -pmass * pro2i * gradkerni * runiy;
				itermz += -pmass * pro2i * gradkerni * runiz;
				//double hfacgrkernj = hj
                //rhoi  += wij;
                //gradhi += -qij * grwij - 3.0 * wij;

				//calculate internal energy derivative

				const double dvxij = vx[i] - vx[j]; 
				const double dvyij = vy[i] - vy[j];
				const double dvzij = vz[i] - vz[j];

				const double vproji = dvxij * runix + dvyij * runiy + dvzij * runiz;
 
				f4sum += pmass * pro2i * vproji * gradkerni;

            }

			if (qj_ij2 < sph::radk2)

			{
											
				double qj_ij, wj_ij, grwj_ij;
            	qj_ij = sqrt(qj_ij2);//might need to be changed to qj_ij2
            	sph::m4_kern(qj_ij, wj_ij, grwj_ij);
							
				const double pro2j = pro2[j];
												//fij = sph::cnormk * hi_4_inv * grwij
				const double rhoj = rho[j];
			    const double dhdrhoj = -hj / (3.0 * rhoj);
				const double grad_j = gradh[j];
		    	const double omegaj  = 1.0 - dhdrhoj * grad_j; //this is the actual omega used for force as well
				const double omega_inv_j = 1 / omegaj;
												
				double hfacgrkernj = hj_4_inv * sph::cnormk * omega_inv_j;//mirrors the Fortran definition of hfacgrkern
				double gradkernj = grwj_ij * hfacgrkernj; //this is now (1/omega) * Fij(hi) as in the Fortran definition as well
				jtermx += -pmass * pro2j * gradkernj * runix; //updating components of the 1st term in the force summation: 
				jtermy += -pmass * pro2j * gradkernj * runiy;
				jtermz += -pmass * pro2j * gradkernj * runiz;
											
											
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
        return;
    }
	
	//keep the above safeguard for now
	
	fx[i] = itermx + jtermx;
	fy[i] = itermy + jtermy;
	fz[i] = itermz + jtermz;	
	f4[i] = f4sum;
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

