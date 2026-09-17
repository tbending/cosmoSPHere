/*
 * force.hpp — GPU force pass.
 *
 * Phantom runs density and force as two passes (deriv.F90 :139 and :195), so this is
 * a separate entry point rather than a tail on the density solve.  It reuses the tree,
 * the Hilbert ordering and gradh left in GpuState by solveDensH; rebuilding them here
 * would cost ~10.8 ms of a 33.7 ms solve, every step, for nothing.  Positions, h and
 * velocities are passed in again, because phantom calls force a second time after the
 * particles have moved (the leapfrog corrector, icall=2).
 */

#pragma once

#include "gpu_check.hpp"
#include "gpu_state.hpp"

//! @brief Per-phase cost of one force pass, in seconds.
struct ForceTimings
{
    double hmaxUpsweep = 0.0;
    double jleafBuild  = 0.0;
    double upload      = 0.0;   // host -> device copies and gathers into Hilbert order
    double kernel      = 0.0;   // sphForceKernelJList
    double download    = 0.0;   // scatters back to phantom order and device -> host copies
};

// Host arrays for one force pass, in phantom order, all length n.  Raw pointers rather
// than std::vector, as for GradFields: they come straight from Fortran through the C API.
// Positions and h are not here: the force pass uses the solve's copies in GpuState.
struct ForceFields
{
    // inputs
    const double* vx; const double* vy; const double* vz;   // read on the corrector only
    const double* pro2;      // P / rho^2
    const double* spsound;   // sound speed
    const double* alphaAV;   // artificial viscosity alpha
    const double* u;         // specific thermal energy
    // outputs
    double* fx; double* fy; double* fz; double* f4;   // fxyzu(1:4)
    double* vsigmax;         // max signal speed over neighbours, for the Courant timestep
    double* divv;            // div v
};

/*! @brief Rebuild the j-leaf lists with the SYMMETRIC (gather + scatter) criterion.
 *
 * Density asks only "is j inside MY kernel?" — radius 2*hmax_i.  The force sum is
 * symmetric (force.F90:1287 takes a pair when q2i < radkern2 .OR. q2j < radkern2), so
 * it also needs the SCATTER neighbours: j with r_ij < 2*h_j but r_ij >= 2*h_i.  Those
 * contribute to the force on i but never to its density, and without them the pair
 * force is not antisymmetric, so momentum is not conserved.
 *
 * Requires s.hmax_leaf valid for EVERY leaf at the converged h — the full-tree pass
 * at the top of solveDensH's gradient sweep leaves it that way.  Propagates it to
 * every node (hmax cannot be derived from an SFC key the way node geometry can) and
 * overwrites s.jlist / s.jcount / s.jOffset.
 *
 * Measured on sedov (176900 particles, 41 dumps to t=0.1): 0% larger than the gather
 * list at uniform ICs, ~7% once the blast develops, peaking at 8.3%.  It stays small
 * because leaf size and h are both set by local density, so 2h is ~1 leaf width
 * everywhere and a big-h leaf is also a big leaf.  Longest list seen on sedov: 433
 * j-leaves; the torus reached 1494.
 */
void buildForceJLeafList(GpuState& s, ForceTimings& ft);


/*! @brief SPH forces for every particle.
 *
 * Rebuilds the symmetric j-leaf lists if the tree has changed, uploads the inputs and
 * gathers them into Hilbert order, runs the force kernel, and scatters the outputs back
 * to phantom order.  Positions and h are taken from the solve's copies in the state, and
 * velocities uploaded only when they may have changed since the solve.  Requires a
 * density solve for the same particle set (GpuState::readyForForce), with positions not
 * moved since it.
 */
void computeForces(GpuState& s, const ForceFields& f, double pmass, double beta,
                   double alphau, ForceTimings& ft);
