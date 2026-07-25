/*
 * density.cuh — GPU density + smoothing-length solver using a Cornerstone octree.
 *
 * Pipeline:
 *   1. Compute 64-bit Hilbert keys for all particles on the GPU.
 *   2. Sort particles into Hilbert order (GPU radix sort via Thrust).
 *   3. Build adaptive cornerstone leaf tree (GPU, updateOctreeGpu loop).
 *   4. Build fully-linked internal octree (GPU, buildLinkedTreeGpu).
 *   5. Compute floating-point node centres and half-sizes (GPU kernel).
 *   6. Run one (or more) Newton–Raphson steps: each GPU thread traverses the
 *      octree for its own particle, accumulates rho and grad-h, and updates h.
 *
 * The Newton step faithfully reproduces the Fortran solve_dens_h logic from
 * cosmoSPHere/src/dens.f90 — same kernel, same variable reuse order, same
 * meaning of gradhi (unnormalised accumulator) in the omega formula.
 *
 * Convergence tolerance and maximum iterations are compile-time tunable below.
 */

// DensTimings struct and solveDensH declaration live in density.hpp.
#include "density.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>

#include "util/annotation.hpp"
#include "util/cuda_utils.hpp"
#include "sfc/box.hpp"
#include "sfc/hilbert.hpp"
#include "tree/csarray.hpp"
#include "tree/csarray_gpu.cuh"
#include "tree/octree.hpp"
#include "tree/octree_gpu.cuh"

#include "kernel.hpp"

using namespace cstone;

// Maximum particles per octree leaf node.
// Smaller = finer tree (more nodes, shorter j-loops per node).
// 64 is a good balance for O(50) neighbours; tune if needed.
static constexpr unsigned BUCKET_SIZE = 64;

// Newton iteration parameters.
static constexpr int    MAX_ITER = 10;
static constexpr double HTOL     = 1.0e-4;

// ---------------------------------------------------------------------------
// GPU kernel: compute Hilbert keys for all particles
// ---------------------------------------------------------------------------
__global__ void computeHilbertKeysKernel(const double* __restrict__ x,
                                         const double* __restrict__ y,
                                         const double* __restrict__ z,
                                         uint64_t* __restrict__ keys,
                                         int n,
                                         Box<double> box)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        keys[i] = hilbert3D<uint64_t>(x[i], y[i], z[i], box);
}

// ---------------------------------------------------------------------------
// GPU kernel: compute floating-point node centres and half-sizes from prefixes
// ---------------------------------------------------------------------------
__global__ void nodeFpCentersKernel(const uint64_t* __restrict__ prefixes,
                                    int numNodes,
                                    Vec3<double>* __restrict__ centers,
                                    Vec3<double>* __restrict__ sizes,
                                    Box<double> box)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numNodes) return;

    uint64_t prefix   = prefixes[i];
    uint64_t startKey = decodePlaceholderBit(prefix);
    unsigned level    = decodePrefixLength(prefix) / 3;
    IBox     nodeBox  = hilbertIBox(startKey, level);

    util::tie(centers[i], sizes[i]) = centerAndSize<uint64_t>(nodeBox, box);
}

// ---------------------------------------------------------------------------
// GPU kernel: one Newton step per particle.
//
// Performs a depth-first octree traversal (iterative, fixed-size stack).
// Accumulates the M4 kernel sum and gradient sum for particle i,
// then applies one Newton–Raphson update to h[i].
//
// h[i] is both input (current estimate) and output (updated estimate).
// rho[i]    receives the converged density (cnormk-normalised, with pmass).
// gradh[i]  receives ∂ρ/∂h (fully normalised) — useful for grad-h terms.
// converged[i] is set to 1 if |h_new - h| / h < HTOL.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Device helper: accumulate M4 kernel sums over all particles in a leaf node.
// ---------------------------------------------------------------------------
__device__ inline void accumulateLeaf(
    const unsigned* __restrict__ layout,
    const TreeNodeIndex* __restrict__ internalToLeaf,
    TreeNodeIndex child,
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    double xi, double yi, double zi, double hi_sq_inv,
    double& rhoi, double& gradhi)
{
    TreeNodeIndex leafIdx = internalToLeaf[child];
    for (unsigned j = layout[leafIdx]; j < layout[leafIdx + 1]; ++j)
    {
        double dx   = xi - x[j];
        double dy   = yi - y[j];
        double dz   = zi - z[j];
        double qij2 = (dx*dx + dy*dy + dz*dz) * hi_sq_inv;
        if (qij2 < sph::radk2)
        {
            double qij = sqrt(qij2), wij, grwij;
            sph::m4_kern(qij, wij, grwij);
            rhoi   += wij;
            gradhi += -qij * grwij - 3.0 * wij;
        }
    }
}

// ---------------------------------------------------------------------------
// GPU kernel: one Newton step per particle.
//
// Performs an iterative depth-first octree traversal (no goto, fixed stack).
// Accumulates the M4 kernel sum and gradient sum for particle i, then applies
// one Newton–Raphson update to h[i].
//
// h[i] is both input (current estimate) and output (updated estimate).
// rho[i]   receives the density (fully normalised with cnormk and pmass).
// gradh[i] receives dρ/dh (fully normalised) — useful for grad-h SPH terms.
// converged[i] is set to 1 if |Δh/h| < HTOL.
// ---------------------------------------------------------------------------
__global__ void sphDensityKernel(const double* __restrict__ x,
                                 const double* __restrict__ y,
                                 const double* __restrict__ z,
                                 double*       h,           // in/out
                                 double*       rho,         // out
                                 double*       gradh,       // out
                                 int*          converged,   // out
                                 int           ngas,
                                 double        pmass,
                                 const TreeNodeIndex* __restrict__ childOffsets,
                                 const TreeNodeIndex* __restrict__ internalToLeaf,
                                 const unsigned*      __restrict__ layout,
                                 const Vec3<double>*  __restrict__ centers,
                                 const Vec3<double>*  __restrict__ sizes)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ngas) return;

    const double xi = x[i];
    const double yi = y[i];
    const double zi = z[i];
    const double hi = h[i];

    const double hi_sq_inv = 1.0 / (hi * hi);
    const double radiusSq  = sph::radk2 * hi * hi;  // (2h)^2

    double rhoi   = 0.0;
    double gradhi = 0.0;   // unnormalised accumulator: sum(-q*grwij - 3*wij)

    // Check root overlap; if the search sphere misses the root entirely,
    // rhoi/gradhi remain zero (particle outside domain — should not happen).
    Vec3<double> target{xi, yi, zi};
    bool rootOverlaps = (norm2(minDistance(target, centers[0], sizes[0])) < radiusSq);

    if (rootOverlaps)
    {
        if (childOffsets[0] == 0)
        {
            // Root is already a leaf (can only happen for tiny test datasets).
            accumulateLeaf(layout, internalToLeaf, 0,
                           x, y, z, xi, yi, zi, hi_sq_inv, rhoi, gradhi);
        }
        else
        {
            // Iterative depth-first traversal with a thread-local stack.
            constexpr TreeNodeIndex STKBOT = -1;
            constexpr int           MAXSTK = 64;
            TreeNodeIndex stack[MAXSTK];
            stack[0]        = STKBOT;
            int stackPos    = 1;
            TreeNodeIndex node = 0;

            do {
                for (int octant = 0; octant < 8; ++octant)
                {
                    TreeNodeIndex child = childOffsets[node] + octant;
                    if (norm2(minDistance(target, centers[child], sizes[child])) >= radiusSq)
                        continue;

                    if (childOffsets[child] == 0)
                    {
                        accumulateLeaf(layout, internalToLeaf, child,
                                       x, y, z, xi, yi, zi, hi_sq_inv, rhoi, gradhi);
                    }
                    else
                    {
                        stack[stackPos++] = child;
                    }
                }
                node = stack[--stackPos];
            } while (node != STKBOT);
        }
    }

    // --- Newton–Raphson update (mirrors Fortran dens.f90 exactly) ---
    const double hi_old = hi;
    const double hi1    = 1.0 / hi;
    const double hi31   = hi1 * hi1 * hi1;
    const double hi41   = hi31 * hi1;

    const double rho_i  = rhoi  * sph::cnormk * pmass * hi31;
    const double grad_i = gradhi * sph::cnormk * pmass * hi41;

    // Guard: if rho_i == 0 (no neighbours), skip update to avoid divide-by-zero.
    if (rho_i == 0.0) { converged[i] = 0; return; }

    const double funci   = pmass * (sph::hfact * hi1) * (sph::hfact * hi1) * (sph::hfact * hi1) - rho_i;
    const double dhdrhoi = -hi / (3.0 * rho_i);
    // omega must use the NORMALISED grad_i (= d(rho)/d(h)), matching Fortran:
    //   gradhi = gradh(i) * cnormk * pmass * hi41
    //   omegai = 1 - dhdrhoi * gradhi
    const double omegai  = 1.0 - dhdrhoi * grad_i;
    const double hi_new_raw = hi - funci * dhdrhoi / omegai;

    // Clamp step to ±20% per iteration (mirrors Fortran finish_cell in dens.F90)
    // to prevent overshoot at steep density gradients (e.g. shock fronts).
    double hi_new;
    if      (hi_new_raw > 1.2 * hi) hi_new = 1.2 * hi;
    else if (hi_new_raw < 0.8 * hi) hi_new = 0.8 * hi;
    else                             hi_new = hi_new_raw;

    rho[i]       = rho_i;
    gradh[i]     = grad_i;
    h[i]         = hi_new;

    const double relChange = fabs((hi_new - hi) / hi_old);
    converged[i] = (relChange < HTOL) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Host function: build tree + run density solve, return timing breakdowns.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// J-leaf list infrastructure.
//
// Instead of each particle doing its own DFS (with a costly register-heavy
// stack), we pre-build one j-leaf list per i-leaf.  With Hilbert ordering
// and warpSize = 64 = BUCKET_SIZE on AMD MI300X, every warp is exactly one
// i-leaf.  All 64 lanes share the same j-leaf list → 64× cache reuse.
// The 156 k DFS traversals needed to build the lists cost < 1 ms.
//
// Fixed-stride layout: jlist[iLeaf * MAX_J_PER_LEAF + k], k ∈ [0, jcount[iLeaf]).
// Geometry: 2h ≈ 3e-3, leaf side ≈ 18e-3 → search sphere spans ≤ 3 leaf widths
// → ≤ 27 j-leaves needed.  64 gives a comfortable 2.4× safety margin.
// ---------------------------------------------------------------------------
// Estimated max j-leaves per i-leaf: (2*(iHalf+leafHalf)/leafDiam)^3 ≈ 172
// for max-jiggle h (3.56e-3). 256 gives a safe margin.
// Memory cost: nLeaves * 256 * 4 = ~160 MB on 10M particles.
static constexpr int MAX_J_PER_LEAF = 256;

// Reverse of internalToLeaf: leaf-CSL index → internal linked-tree node idx.
// Needed to look up center/size of each i-leaf.
__global__ void buildLeafToInternalKernel(
    const TreeNodeIndex* __restrict__ childOffsets,
    const TreeNodeIndex* __restrict__ internalToLeaf,
    int numNodes,
    TreeNodeIndex* __restrict__ leafToInternal)
{
    int node = blockDim.x * blockIdx.x + threadIdx.x;
    if (node >= numNodes) return;
    if (childOffsets[node] == 0)  // childOffsets == 0 marks a leaf
        leafToInternal[internalToLeaf[node]] = node;
}

// For each particle, record which leaf it belongs to.
// With Hilbert order, all particles in layout[iLeaf..iLeaf+1) are adjacent.
__global__ void buildParticleToLeafKernel(
    const unsigned* __restrict__ layout,
    int nLeaves,
    int* __restrict__ particleLeaf)
{
    int iLeaf = blockDim.x * blockIdx.x + threadIdx.x;
    if (iLeaf >= nLeaves) return;
    for (unsigned j = layout[iLeaf]; j < layout[iLeaf + 1]; ++j)
        particleLeaf[j] = iLeaf;
}

// Maximum h in each active leaf — indexed via activeLeaves indirection.
__global__ void computeHmaxLeafKernel(
    const double*    __restrict__ h,
    const unsigned*  __restrict__ layout,
    int              nActiveLeaves,
    const int*       __restrict__ activeLeaves,
    double*          __restrict__ hmax_leaf)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nActiveLeaves) return;
    int iLeaf = activeLeaves[idx];
    double hmax = 0.0;
    for (unsigned j = layout[iLeaf]; j < layout[iLeaf + 1]; ++j)
        if (isfinite(h[j])) hmax = fmax(hmax, h[j]);
    hmax_leaf[iLeaf] = hmax;
}

// Single-pass DFS for each active i-leaf, indexed via activeLeaves indirection.
__global__ void buildJLeafListKernel(
    const TreeNodeIndex* __restrict__ leafToInternal,
    const double*         __restrict__ hmax_leaf,
    const Vec3<double>*   __restrict__ centers,
    const Vec3<double>*   __restrict__ sizes,
    const TreeNodeIndex*  __restrict__ childOffsets,
    const TreeNodeIndex*  __restrict__ internalToLeaf,
    int              nActiveLeaves,
    const int*       __restrict__ activeLeaves,
    int* __restrict__ jlist,
    int* __restrict__ jcount)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nActiveLeaves) return;
    int iLeaf = activeLeaves[idx];

    TreeNodeIndex iNode  = leafToInternal[iLeaf];
    Vec3<double>  iCenter = centers[iNode];
    Vec3<double>  iSize   = sizes[iNode];
    double        hmax    = hmax_leaf[iLeaf];
    // Guard: if all h in this leaf were non-finite, hmax==0 → use iSize only
    // (no real neighbours needed; density kernel will guard the update too).
    if (!isfinite(hmax) || hmax <= 0.0) hmax = 0.0;
    // Expand the i-leaf's bounding box by 2·hmax on every side.
    Vec3<double> iHalf{ iSize[0] + 2.0*hmax,
                        iSize[1] + 2.0*hmax,
                        iSize[2] + 2.0*hmax };

    // DFS traversal — stack depth ≤ 7 × tree_depth ≈ 32 for 10M particles.
    constexpr int MAXSTK = 32;
    TreeNodeIndex stack[MAXSTK];
    stack[0] = -1;
    int stackPos = 1;
    int count    = 0;
    int jBase    = iLeaf * MAX_J_PER_LEAF;

    // Check root overlap.
    if (norm2(minDistance(iCenter, iHalf, centers[0], sizes[0])) > 0.0)
    {
        jcount[iLeaf] = 0;
        return;
    }
    if (childOffsets[0] == 0)  // root is itself a leaf
    {
        if (count < MAX_J_PER_LEAF) jlist[jBase + count] = internalToLeaf[0];
        jcount[iLeaf] = 1;
        return;
    }

    TreeNodeIndex node = 0;
    do {
        for (int oct = 0; oct < 8; ++oct)
        {
            TreeNodeIndex child = childOffsets[node] + oct;
            if (norm2(minDistance(iCenter, iHalf, centers[child], sizes[child])) > 0.0)
                continue;
            if (childOffsets[child] == 0)
            {
                if (count < MAX_J_PER_LEAF)
                    jlist[jBase + count] = internalToLeaf[child];
                ++count;
            }
            else
            {
                if (stackPos < MAXSTK) stack[stackPos++] = child;
            }
        }
        node = stack[--stackPos];
    } while (node != -1);

    jcount[iLeaf] = min(count, MAX_J_PER_LEAF);
}

// ---------------------------------------------------------------------------
// Density kernel using the pre-built j-leaf list.
//
// Indexed via activeParticles[0..nActive): only unconverged particles are
// processed.  No traversal stack → minimal register pressure → max occupancy.
//
// The caller sorts activeParticles by leaf index (d_particleLeaf) before
// each launch so that consecutive threads in a warp cover the same i-leaf
// (or at most a small number of adjacent leaves).  This makes j-particle
// access coalesced regardless of how sparse the active set is — important
// for individual-timestep integrators where active particles may be
// scattered across the domain.
//
// BUCKET_SIZE is a *maximum* leaf population; actual leaf sizes range from
// 1 to BUCKET_SIZE depending on local density.  The sort-by-leaf preparation
// ensures warp coherence even when leaves are partially filled.
// ---------------------------------------------------------------------------
__global__ void sphDensityKernelJList(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    double*       h,           // in/out
    double*       rho,         // out
    double*       gradh,       // out
    int*          converged,   // out
    int           nActive,
    const int*    __restrict__ activeParticles,
    double        pmass,
    const int*       __restrict__ particleLeaf,
    const int*       __restrict__ jcount,
    const int*       __restrict__ jlist,
    const unsigned*  __restrict__ layout)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nActive) return;
    int i = activeParticles[idx];   // actual particle index

    const double xi = x[i], yi = y[i], zi = z[i];
    const double hi        = h[i];
    const double hi_sq_inv = 1.0 / (hi * hi);

    double rhoi  = 0.0;
    double gradhi = 0.0;

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
            double qij2 = (dx*dx + dy*dy + dz*dz) * hi_sq_inv;
            if (qij2 < sph::radk2)
            {
                double qij, wij, grwij;
                qij = sqrt(qij2);
                sph::m4_kern(qij, wij, grwij);
                rhoi  += wij;
                gradhi += -qij * grwij - 3.0 * wij;
            }
        }
    }

    // Newton–Raphson update (identical to sphDensityKernel).
    const double hi_old  = hi;
    const double hi1     = 1.0 / hi;
    const double hi31    = hi1 * hi1 * hi1;
    const double hi41    = hi31 * hi1;
    const double rho_i   = rhoi  * sph::cnormk * pmass * hi31;
    const double grad_i  = gradhi * sph::cnormk * pmass * hi41;

    // Guard: if rhoi==0 (no neighbours found — j-leaf list too small or particle
    // outside domain), skip the Newton update and mark unconverged.  Without this
    // guard, h_new becomes NaN/Inf which cascades into the next iteration's
    // hmax-leaf computation, causing the DFS to visit the entire tree.
    if (!(rho_i > 0.0))
    {
        rho[i]       = 0.0;
        gradh[i]     = 0.0;
        converged[i] = 0;
        return;
    }

    const double funci   = pmass * (sph::hfact*hi1) * (sph::hfact*hi1) * (sph::hfact*hi1) - rho_i;
    const double dhdrhoi = -hi / (3.0 * rho_i);
    // omega uses NORMALISED grad_i, matching Fortran dens.f90.
    const double omegai  = 1.0 - dhdrhoi * grad_i;
    // Guard: avoid sign flip when omegai ≤ 0 (mirrors Fortran finish_cell).
    const double safe_omega = (omegai > 0.0) ? omegai : fabs(omegai + 1e-300);
    const double hi_new_raw = hi - funci * dhdrhoi / safe_omega;
    // Clamp step to ±20% per iteration (mirrors Fortran finish_cell in dens.F90).
    double hi_new;
    if      (hi_new_raw > 1.2 * hi) hi_new = 1.2 * hi;
    else if (hi_new_raw < 0.8 * hi) hi_new = 0.8 * hi;
    else                             hi_new = hi_new_raw;

    rho[i]       = rho_i;
    gradh[i]     = grad_i;
    h[i]         = hi_new;
    converged[i] = (fabs((hi_new - hi) / hi_old) < HTOL) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Alternative density kernel: one GPU block (= one wavefront) per i-leaf.
//
// Block size = BUCKET_SIZE (= warpSize = 64 on AMD MI300X).
// The j-leaf list is loaded cooperatively into shared memory so every lane
// reads j-particle data from the same warp-shared working set.
//
// Trade-offs vs sphDensityKernelJList (flat-particle mode):
//   ADVANTAGES
//     + j-leaf list is in L1/shared memory once per leaf → 64× address reuse
//       over the j-particle inner loop.
//     + j-particle cache lines brought in by one lane are immediately reused
//       by the other 63 lanes → much better L1 hit rate in iteration 1.
//   DISADVANTAGES
//     - Partial leaves (< BUCKET_SIZE particles) waste idle lanes throughout.
//     - In late Newton iterations, a leaf stays active if even ONE particle
//       has not yet converged.  All 64 lanes still run the full j-particle
//       loop for 63 particles that have already converged → severe waste.
//     - This makes it STRICTLY worse than the flat-particle kernel in late
//       iterations, which is why flat-particle tends to win overall on
//       datasets where most particles converge quickly.
//
// The kernel therefore makes sense to benchmark against the flat-particle
// kernel (sphDensityKernelJList) to see where the cross-over lies.
//
// Implementation note: converged particles within an active leaf re-run the
// Newton step but arrive at the same h (fixed point), so converged[i] stays
// 1.  We do not guard on converged[i] before the hot j-loop to avoid warp
// divergence; the redundant rho/gradh writes are harmless.
// ---------------------------------------------------------------------------
__global__ void sphDensityKernelLeafWarp(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    double*       h,           // in/out
    double*       rho,         // out
    double*       gradh,       // out
    int*          converged,   // out
    int           nActiveLeaves,
    const int*    __restrict__ activeLeaves,
    double        pmass,
    const int*       __restrict__ jcount,
    const int*       __restrict__ jlist,
    const unsigned*  __restrict__ layout)
{
    // Shared memory caches the complete j-leaf list for this i-leaf.
    // Cost: MAX_J_PER_LEAF * 4 = 1 KB per block — allows high SM occupancy.
    __shared__ int sJList[MAX_J_PER_LEAF];

    int iLeafIdx = blockIdx.x;
    if (iLeafIdx >= nActiveLeaves) return;
    int iLeaf = activeLeaves[iLeafIdx];

    int lane  = threadIdx.x;           // 0 .. BUCKET_SIZE-1
    int jBase = iLeaf * MAX_J_PER_LEAF;
    int nj    = jcount[iLeaf];

    // Cooperatively load j-leaf list into shared memory.
    // All lanes participate so the load is coalesced.
    for (int k = lane; k < nj; k += blockDim.x)
        sJList[k] = jlist[jBase + k];
    __syncthreads();

    unsigned pStart   = layout[iLeaf];
    unsigned pEnd     = layout[iLeaf + 1];
    unsigned leafSize = pEnd - pStart;

    // Lanes beyond the leaf size are idle.  They still helped with the
    // cooperative j-list load above, so __syncthreads() is safe.
    if ((unsigned)lane >= leafSize) return;

    unsigned i       = pStart + (unsigned)lane;
    double xi        = x[i], yi = y[i], zi = z[i];
    double hi        = h[i];
    double hi_sq_inv = 1.0 / (hi * hi);

    double rhoi = 0.0, gradhi = 0.0;

    for (int jl = 0; jl < nj; ++jl)
    {
        unsigned jLeaf = (unsigned)sJList[jl];
        for (unsigned j = layout[jLeaf]; j < layout[jLeaf + 1]; ++j)
        {
            double dx   = xi - x[j];
            double dy   = yi - y[j];
            double dz   = zi - z[j];
            double qij2 = (dx*dx + dy*dy + dz*dz) * hi_sq_inv;
            if (qij2 < sph::radk2)
            {
                double qij, wij, grwij;
                qij = sqrt(qij2);
                sph::m4_kern(qij, wij, grwij);
                rhoi  += wij;
                gradhi += -qij * grwij - 3.0 * wij;
            }
        }
    }

    // Newton–Raphson update — identical to sphDensityKernelJList.
    const double hi_old  = hi;
    const double hi1     = 1.0 / hi;
    const double hi31    = hi1 * hi1 * hi1;
    const double hi41    = hi31 * hi1;
    const double rho_i   = rhoi  * sph::cnormk * pmass * hi31;
    const double grad_i  = gradhi * sph::cnormk * pmass * hi41;

    if (!(rho_i > 0.0))
    {
        rho[i] = 0.0; gradh[i] = 0.0; converged[i] = 0;
        return;
    }

    const double funci   = pmass * (sph::hfact*hi1) * (sph::hfact*hi1) * (sph::hfact*hi1) - rho_i;
    const double dhdrhoi = -hi / (3.0 * rho_i);
    // omega uses NORMALISED grad_i, matching Fortran dens.f90.
    const double omegai  = 1.0 - dhdrhoi * grad_i;
    // Guard: avoid sign flip when omegai ≤ 0 (mirrors Fortran finish_cell).
    const double safe_omega = (omegai > 0.0) ? omegai : fabs(omegai + 1e-300);
    const double hi_new_raw = hi - funci * dhdrhoi / safe_omega;
    // Clamp step to ±20% per iteration (mirrors Fortran finish_cell in dens.F90).
    double hi_new;
    if      (hi_new_raw > 1.2 * hi) hi_new = 1.2 * hi;
    else if (hi_new_raw < 0.8 * hi) hi_new = 0.8 * hi;
    else                             hi_new = hi_new_raw;

    rho[i]       = rho_i;
    gradh[i]     = grad_i;
    h[i]         = hi_new;
    converged[i] = (fabs((hi_new - hi) / hi_old) < HTOL) ? 1 : 0;
}

// ---------------------------------------------------------------------------

DensTimings solveDensH(// Host input/output
                        std::vector<double>& h_host,
                        std::vector<double>& rho_host,
                        std::vector<double>& gradh_host,
                        // Host input (read-only)
                        const std::vector<double>& x_host,
                        const std::vector<double>& y_host,
                        const std::vector<double>& z_host,
                        double pmass,
                        KernelMode mode)
{
    const int ngas = static_cast<int>(x_host.size());
    DensTimings t{};
    t.kernelMode = mode;

    // CUDA events for fine-grained timing.
    cudaEvent_t evUpload0, evUpload1, evBbox0, evBbox1,
                ev0, ev1, ev2, ev3, evJB0, evJB1, ev4, evDl0, evDl1;
    for (auto* e : {&evUpload0, &evUpload1, &evBbox0, &evBbox1,
                    &ev0, &ev1, &ev2, &ev3, &evJB0, &evJB1, &ev4, &evDl0, &evDl1})
        checkGpuErrors(hipEventCreate(e));

    // -----------------------------------------------------------------------
    // Upload particle data to GPU
    // -----------------------------------------------------------------------
    HIP_CHECK(hipEventRecord(evUpload0));
    thrust::device_vector<double> d_x(x_host), d_y(y_host), d_z(z_host);
    thrust::device_vector<double> d_h(h_host);
    thrust::device_vector<double> d_rho(ngas, 0.0), d_gradh(ngas, 0.0);
    thrust::device_vector<int>    d_converged(ngas, 0);
    HIP_CHECK(hipEventRecord(evUpload1));

    // -----------------------------------------------------------------------
    // Bounding box — compute entirely on GPU using Thrust reductions.
    // Avoids 6 serial host loops over 10M elements.
    // -----------------------------------------------------------------------
    HIP_CHECK(hipEventRecord(evBbox0));
    double xmin, xmax, ymin, ymax, zmin, zmax;
    {
        auto [xlo, xhi] = thrust::minmax_element(thrust::device, d_x.begin(), d_x.end());
        auto [ylo, yhi] = thrust::minmax_element(thrust::device, d_y.begin(), d_y.end());
        auto [zlo, zhi] = thrust::minmax_element(thrust::device, d_z.begin(), d_z.end());
        // Dereferencing device iterators triggers an implicit device→host copy.
        xmin = *xlo; xmax = *xhi;
        ymin = *ylo; ymax = *yhi;
        zmin = *zlo; zmax = *zhi;
    }
    HIP_CHECK(hipEventRecord(evBbox1));

    // Pad box slightly (mirrors Fortran *1.00001 on the largest side).
    double maxSpan = std::max({xmax-xmin, ymax-ymin, zmax-zmin}) * 1.00001;
    double xctr = 0.5*(xmin+xmax), yctr = 0.5*(ymin+ymax), zctr = 0.5*(zmin+zmax);
    double half = 0.5 * maxSpan;
    Box<double> box{xctr - half, xctr + half,
                    yctr - half, yctr + half,
                    zctr - half, zctr + half,
                    BoundaryType::open};

    // -----------------------------------------------------------------------
    // Step 1 — Hilbert keys + GPU sort
    // -----------------------------------------------------------------------
    HIP_CHECK(hipEventRecord(ev0));
    {
        thrust::device_vector<uint64_t> d_keys(ngas);

        constexpr int BLK = 256;
        computeHilbertKeysKernel<<<iceil(ngas, BLK), BLK>>>(
            rawPtr(d_x), rawPtr(d_y), rawPtr(d_z),
            rawPtr(d_keys), ngas, box);
        checkGpuErrors(cudaGetLastError());

        // Sort permutation by Hilbert key, then gather particle data.
        thrust::device_vector<int> d_order(ngas);
        thrust::sequence(d_order.begin(), d_order.end());
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_order.begin());

        thrust::device_vector<double> d_tmp(ngas);
        thrust::gather(d_order.begin(), d_order.end(), d_x.begin(), d_tmp.begin());
        thrust::swap(d_x, d_tmp);
        thrust::gather(d_order.begin(), d_order.end(), d_y.begin(), d_tmp.begin());
        thrust::swap(d_y, d_tmp);
        thrust::gather(d_order.begin(), d_order.end(), d_z.begin(), d_tmp.begin());
        thrust::swap(d_z, d_tmp);
        thrust::gather(d_order.begin(), d_order.end(), d_h.begin(), d_tmp.begin());
        thrust::swap(d_h, d_tmp);

        // -----------------------------------------------------------------------
        // Step 2 — Cornerstone leaf tree + linked tree
        // -----------------------------------------------------------------------
        HIP_CHECK(hipEventRecord(ev1));

        thrust::device_vector<uint64_t>       csTree = std::vector<uint64_t>{0, nodeRange<uint64_t>(0)};
        thrust::device_vector<unsigned>        counts = std::vector<unsigned>{(unsigned)ngas};
        thrust::device_vector<uint64_t>        tmpTree;
        thrust::device_vector<TreeNodeIndex>   workArray;

        // d_keys is already sorted — run update until tree converges.
        while (!updateOctreeGpu(rawPtr(d_keys), rawPtr(d_keys) + ngas,
                                BUCKET_SIZE, csTree, counts, tmpTree, workArray))
        {
            // iterate until stable leaf partition
        }

        OctreeData<uint64_t, GpuTag> octree;
        octree.resize(nNodes(csTree));
        buildLinkedTreeGpu(rawPtr(csTree), octree.data());
        checkGpuErrors(cudaGetLastError());

        // Build particle layout (prefix-sum of counts → first particle per leaf).
        const int nLeaves = (int)nNodes(csTree);
        thrust::device_vector<unsigned> d_layout(nLeaves + 1);
        thrust::exclusive_scan(thrust::device,
                               counts.begin(), counts.end() + 1,
                               d_layout.begin(), 0u);

        // -----------------------------------------------------------------------
        // Step 3 — Node centres and sizes
        // -----------------------------------------------------------------------
        HIP_CHECK(hipEventRecord(ev2));
        {
            int numNodes = octree.numNodes;
            thrust::device_vector<Vec3<double>> d_centers(numNodes), d_sizes(numNodes);

            nodeFpCentersKernel<<<iceil(numNodes, 256), 256>>>(
                rawPtr(octree.prefixes), numNodes,
                rawPtr(d_centers), rawPtr(d_sizes), box);
            checkGpuErrors(cudaGetLastError());

            // -----------------------------------------------------------------------
            // J-leaf list infrastructure.
            // leafToInternal and particleLeaf are stable after tree build.
            // activeLeaves + activeParticles shrink each Newton iteration.
            // -----------------------------------------------------------------------
            thrust::device_vector<TreeNodeIndex> d_leafToInternal(nLeaves, -1);
            buildLeafToInternalKernel<<<iceil(numNodes, 256), 256>>>(
                rawPtr(octree.childOffsets), rawPtr(octree.internalToLeaf),
                numNodes, rawPtr(d_leafToInternal));
            checkGpuErrors(cudaGetLastError());

            thrust::device_vector<int> d_particleLeaf(ngas);
            buildParticleToLeafKernel<<<iceil(nLeaves, 256), 256>>>(
                rawPtr(d_layout), nLeaves, rawPtr(d_particleLeaf));
            checkGpuErrors(cudaGetLastError());

            thrust::device_vector<double> d_hmax_leaf(nLeaves);
            thrust::device_vector<int>    d_jlist(nLeaves * MAX_J_PER_LEAF);
            thrust::device_vector<int>    d_jcount(nLeaves);

            // Active sets — start as all particles / all leaves.
            // After each iteration, compact to unconverged only.
            thrust::device_vector<int> d_activeParticles(ngas);
            thrust::device_vector<int> d_activeTmp(ngas);       // scratch for copy_if
            thrust::device_vector<int> d_activeLeaves(nLeaves);
            thrust::device_vector<int> d_activeLeavesTmp(ngas); // scratch (ngas upper bound)
            thrust::sequence(d_activeParticles.begin(), d_activeParticles.end());
            thrust::sequence(d_activeLeaves.begin(),   d_activeLeaves.end());
            int nActive       = ngas;
            int nActiveLeaves = nLeaves;

            // -----------------------------------------------------------------------
            // Step 4 — Build j-leaf list ONCE before the Newton loop.
            //
            // The j-leaf list is constructed using the initial h values, which are
            // the largest each particle's h will ever be during Newton convergence:
            // the NR update always moves h toward the equilibrium value, which means
            // h can only *decrease* from an over-estimated starting value (e.g. after
            // jiggle) or converge from either side, but never grows beyond the value
            // that caused the leaf's hmax.  The per-particle qij<radk check in the
            // kernel provides the exact spatial cutoff, so a conservative (too-large)
            // j-leaf list is correct — the only cost is extra j-particle inner-loop
            // iterations that fail the distance test.
            //
            // This eliminates nActiveLeaves DFS traversals × (MAX_ITER-1) iterations.
            // For 10M particles with jiggle: saves ~9 × 2ms = 18 ms of j-list build.
            // For the Fortran-equivalent case the walk is also pre-built once.
            //
            // Caveat: if h grows significantly during Newton (e.g. particle starts
            // with h too small relative to its converged value), some true neighbours
            // may be missed on later iterations.  For well-initialised SPH simulations
            // this is rare; where it matters, rebuild the list mid-solve.
            // -----------------------------------------------------------------------
            // -----------------------------------------------------------------------
            // Step 4 — Newton iteration.
            // Each iteration:
            //   a) rebuild hmax + j-leaf list for active leaves only.
            //   b) run density kernel for active particles only.
            //   c) compact active sets to unconverged particles/leaves.
            // -----------------------------------------------------------------------
            HIP_CHECK(hipEventRecord(ev3));
            {
                constexpr int BLK2 = 256;

                for (int iter = 0; iter < MAX_ITER; ++iter)
                {
                    HIP_CHECK(hipEventRecord(evJB0));
                    computeHmaxLeafKernel<<<iceil(nActiveLeaves, 256), 256>>>(
                        rawPtr(d_h), rawPtr(d_layout),
                        nActiveLeaves, rawPtr(d_activeLeaves),
                        rawPtr(d_hmax_leaf));
                    checkGpuErrors(cudaGetLastError());

                    buildJLeafListKernel<<<iceil(nActiveLeaves, 256), 256>>>( 
                        rawPtr(d_leafToInternal), rawPtr(d_hmax_leaf),
                        rawPtr(d_centers), rawPtr(d_sizes),
                        rawPtr(octree.childOffsets), rawPtr(octree.internalToLeaf),
                        nActiveLeaves, rawPtr(d_activeLeaves),
                        rawPtr(d_jlist), rawPtr(d_jcount));
                    checkGpuErrors(cudaGetLastError());
                    HIP_CHECK(hipEventRecord(evJB1));

                    if (mode == KernelMode::WARP_PER_LEAF)
                    {
                        sphDensityKernelLeafWarp<<<nActiveLeaves, BUCKET_SIZE>>>(
                            rawPtr(d_x), rawPtr(d_y), rawPtr(d_z),
                            rawPtr(d_h), rawPtr(d_rho), rawPtr(d_gradh), rawPtr(d_converged),
                            nActiveLeaves, rawPtr(d_activeLeaves),
                            pmass,
                            rawPtr(d_jcount), rawPtr(d_jlist),
                            rawPtr(d_layout));
                    }
                    else
                    {
                        sphDensityKernelJList<<<iceil(nActive, BLK2), BLK2>>>(
                            rawPtr(d_x), rawPtr(d_y), rawPtr(d_z),
                            rawPtr(d_h), rawPtr(d_rho), rawPtr(d_gradh), rawPtr(d_converged),
                            nActive, rawPtr(d_activeParticles),
                            pmass,
                            rawPtr(d_particleLeaf),
                            rawPtr(d_jcount), rawPtr(d_jlist),
                            rawPtr(d_layout));
                    }
                    checkGpuErrors(cudaGetLastError());

                    // Accumulate j-leaf build time.
                    checkGpuErrors(hipEventSynchronize(evJB1));
                    float jms = 0;
                    HIP_CHECK(hipEventElapsedTime(&jms, evJB0, evJB1));
                    t.jleafBuild += jms * 1e-3;

                    // Compact unconverged particles into d_activeTmp.
                    // Predicate reads d_converged[activeParticles[j]] via permutation iterator.
                    auto pred = thrust::logical_not<int>();
                    auto permIt = thrust::make_permutation_iterator(
                        d_converged.begin(), d_activeParticles.begin());
                    auto newEnd = thrust::copy_if(
                        thrust::device,
                        d_activeParticles.begin(),
                        d_activeParticles.begin() + nActive,
                        permIt,
                        d_activeTmp.begin(),
                        pred);
                    t.itersRun = iter + 1;
                    nActive    = static_cast<int>(newEnd - d_activeTmp.begin());
                    thrust::swap(d_activeParticles, d_activeTmp);

                    if (nActive == 0) break;

                    // Build the new active-leaf set:
                    // map active particles → their leaves, sort+unique.
                    thrust::gather(thrust::device,
                        d_activeParticles.begin(), d_activeParticles.begin() + nActive,
                        d_particleLeaf.begin(), d_activeLeavesTmp.begin());
                    thrust::sort(thrust::device,
                        d_activeLeavesTmp.begin(), d_activeLeavesTmp.begin() + nActive);
                    auto leafEnd = thrust::unique(thrust::device,
                        d_activeLeavesTmp.begin(), d_activeLeavesTmp.begin() + nActive);
                    nActiveLeaves = static_cast<int>(leafEnd - d_activeLeavesTmp.begin());
                    thrust::copy(thrust::device,
                        d_activeLeavesTmp.begin(), d_activeLeavesTmp.begin() + nActiveLeaves,
                        d_activeLeaves.begin());
                }
            }
            HIP_CHECK(hipEventRecord(ev4));
            checkGpuErrors(hipEventSynchronize(ev4));

            // Download results — use hipMemcpy so transfers land on the default
            // stream and are correctly bracketed by the HIP events.
            // Results are in Hilbert-sorted order; scatter back to original
            // particle order using d_order (which maps sorted index → original index).
            HIP_CHECK(hipEventRecord(evDl0));
            {
                thrust::device_vector<double> d_out(ngas);
                thrust::scatter(d_h.begin(),    d_h.end(),    d_order.begin(), d_out.begin());
                HIP_CHECK(hipMemcpy(h_host.data(),     rawPtr(d_out), ngas*sizeof(double), hipMemcpyDeviceToHost));
                thrust::scatter(d_rho.begin(),  d_rho.end(),  d_order.begin(), d_out.begin());
                HIP_CHECK(hipMemcpy(rho_host.data(),   rawPtr(d_out), ngas*sizeof(double), hipMemcpyDeviceToHost));
                thrust::scatter(d_gradh.begin(),d_gradh.end(),d_order.begin(), d_out.begin());
                HIP_CHECK(hipMemcpy(gradh_host.data(), rawPtr(d_out), ngas*sizeof(double), hipMemcpyDeviceToHost));
            }
            HIP_CHECK(hipEventRecord(evDl1));
            checkGpuErrors(hipEventSynchronize(evDl1));
        }
    }

    // Collect timings (ms -> s).
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, evUpload0, evUpload1)); t.upload       = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, evBbox0,   evBbox1));   t.bboxAndSetup = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, ev0,       ev1));        t.keysAndSort  = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, ev1,       ev2));        t.treeBuild    = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, ev2,       ev3));        t.nodeCenters  = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, ev3,       ev4));        t.densKernel   = ms * 1e-3 - t.jleafBuild;
    HIP_CHECK(hipEventElapsedTime(&ms, evDl0,     evDl1));      t.download     = ms * 1e-3;

    for (auto* e : {&evUpload0, &evUpload1, &evBbox0, &evBbox1,
                    &ev0, &ev1, &ev2, &ev3, &evJB0, &evJB1, &ev4, &evDl0, &evDl1})
        HIP_CHECK(hipEventDestroy(*e));

    return t;
}
