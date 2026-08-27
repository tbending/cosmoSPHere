/*
 * density.hpp — Public interface for the Cornerstone GPU density solver.
 *
 * Includable from NON-GPU translation units: no thrust, no cornerstone.  A consumer
 * that just wants densities — a resolution-changing tool, an analysis pass — includes
 * this and nothing else.
 *
 * The device buffers and the octree live in the process-wide GpuState (gpu_state.hpp)
 * and persist between calls, so a caller that solves repeatedly pays the allocation
 * cost once.  Callers never see it.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>


// Selects which GPU density kernel is used inside the Newton iteration.
//
//   FLAT_PARTICLE  — sphDensityKernelJList: flat launch over active particles.
//                    Block size 256, all threads active → 100% occupancy.
//                    j-leaf list read from global memory per thread.
//                    Wins in late iterations (few unconverged particles).
//
//   WARP_PER_LEAF  — sphDensityKernelLeafWarp: one block (BUCKET_SIZE=64
//                    threads) per active i-leaf.  j-leaf list loaded into
//                    shared memory cooperatively → 64× address reuse.
//                    Wins in iteration 1 (full device, perfect alignment);
//                    loses badly in late iterations (idle lanes for partial
//                    leaves and re-accumulated converged particles).
enum class KernelMode { FLAT_PARTICLE, WARP_PER_LEAF };

// Per-call timing breakdown returned by solveDensH.
struct DensTimings
{
    double     upload;          // host-to-device transfer
    double     bboxAndSetup;    // bounding box (GPU) + misc setup
    double     keysAndSort;     // Hilbert key compute + GPU sort
    double     treeBuild;       // cornerstone + linked tree
    double     nodeCenters;     // nodeFpCentersKernel
    double     jleafBuild;      // j-leaf list construction (all iterations)
    double     densKernel;      // density kernel time (all iterations)
    double     gradJleafBuild;  // j-leaf rebuild over ALL leaves before the grad sweep
    double     gradKernel;      // post-convergence gradient sweep (0 if not requested)
    double     download;        // device-to-host result transfer
    int        itersRun;        // Newton iterations performed
    int        nParticles;      // particles solved for
    int        nLeavesOut;      // leaves in the final tree
    KernelMode kernelMode;      // which kernel was used
};

// Optional second-pass fields: the SPH velocity and acceleration gradients that
// phantom's densityiterate computes alongside the density.  Supplying this makes
// solveDensH run ONE extra neighbour sweep after Newton convergence, evaluated at
// the converged h, and fill divv/dvdx/ddivvdt.  rho and gradh are re-evaluated in
// the same sweep so that every returned field belongs to the same h (the Newton
// loop necessarily leaves rho/gradh one step behind h).  Pass nullptr to skip the
// sweep and get the density-only behaviour.
//
// Raw pointers rather than std::vector: these come straight from Fortran arrays
// through the C API, and the inputs are read-only, so there is nothing to gain
// from copying them into vectors first.  All arrays are length n except dvdx.
struct GradFields
{
    // inputs
    const double* vx; const double* vy; const double* vz;   // velocity
    const double* ax; const double* ay; const double* az;   // acceleration (fxyzu+fext)
    // outputs
    double* divv;      // div v
    double* dvdx;      // velocity gradient tensor, 9*n, particle-major: dvdx[9*i + c]
    double* ddivvdt;   // d(div v)/dt, the Cullen & Dehnen switch source term
};

// Solve for smoothing lengths h and densities rho for all particles.
// h_host is in/out; rho_host and gradh_host are output-only.
// x/y/z and pmass are read-only inputs.
// mode selects the GPU density kernel (default: FLAT_PARTICLE).
// grads is optional — see GradFields above.
//
// Leaves the tree, the Hilbert-sorted particle data and the leaf bookkeeping in the
// shared state, so a repeated solve reuses the allocations — see gpu_state.hpp.
DensTimings solveDensH(std::vector<double>& h_host,
                       std::vector<double>& rho_host,
                       std::vector<double>& gradh_host,
                       const std::vector<double>& x_host,
                       const std::vector<double>& y_host,
                       const std::vector<double>& z_host,
                       double pmass,
                       KernelMode mode = KernelMode::FLAT_PARTICLE,
                       const GradFields* grads = nullptr);
