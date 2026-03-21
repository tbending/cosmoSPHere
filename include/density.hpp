/*
 * density.hpp — Public interface for the Cornerstone GPU density solver.
 *
 * Only include this header from non-GPU translation units (e.g. main.cu
 * could include it, but caller only needs DensTimings + solveDensH).
 * The full GPU implementation lives in density.cu.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>

// Error-checking macro for all HIP API calls.
// Prints file/line on failure and aborts.
#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t _e = (call);                                                \
        if (_e != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP error %s:%d  %s\n",                     \
                         __FILE__, __LINE__, hipGetErrorString(_e));           \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

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
    double     download;        // device-to-host result transfer
    int        itersRun;        // Newton iterations performed
    KernelMode kernelMode;      // which kernel was used
};

// Solve for smoothing lengths h and densities rho for all particles.
// h_host is in/out; rho_host and gradh_host are output-only.
// x/y/z and pmass are read-only inputs.
// mode selects the GPU density kernel (default: FLAT_PARTICLE).
DensTimings solveDensH(std::vector<double>& h_host,
                       std::vector<double>& rho_host,
                       std::vector<double>& gradh_host,
                       const std::vector<double>& x_host,
                       const std::vector<double>& y_host,
                       const std::vector<double>& z_host,
                       double pmass,
                       KernelMode mode = KernelMode::FLAT_PARTICLE);
