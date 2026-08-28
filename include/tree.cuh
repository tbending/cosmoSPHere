/*
 * tree.cuh — Cornerstone octree construction and the neighbour-leaf walk.
 *
 * Everything here is about the TREE, not about SPH physics: build the Hilbert
 * ordering, build the octree, derive node geometry, and walk it to find which leaves
 * can hold a neighbour.  The density solve (density_base.cu) and the force pass
 * (force.cu) both sit on top of it.
 *
 * The non-template kernels are `static`: a plain __global__ defined in a header is
 * emitted in every translation unit that includes it and the definitions collide at
 * link time.  static gives each TU its own copy — a few KB of fatbin, no collision.
 * Cornerstone's own *_gpu.cuh headers have the same problem and are NOT included
 * here for that reason; tree.cu includes them directly.
 *
 * buildJLeafListKernel stays a template in a header because density needs <false>
 * and force <true>; in a .cu that would take explicit instantiation naming both.
 */

#pragma once

#include <cstdint>

#include <thrust/device_vector.h>

#include "util/annotation.hpp"
#include "util/cuda_utils.hpp"
#include "sfc/box.hpp"
#include "sfc/hilbert.hpp"
#include "tree/csarray.hpp"
#include "tree/octree.hpp"

#include "gpu_check.hpp"
#include "gpu_state.hpp"

using namespace cstone;

// Maximum particles per octree leaf node.
// Smaller = finer tree (more nodes, shorter j-loops per node).
// 64 is a good balance for O(50) neighbours; tune if needed.
static constexpr unsigned BUCKET_SIZE = 64;

// ---------------------------------------------------------------------------
// GPU kernel: compute Hilbert keys for all particles
// ---------------------------------------------------------------------------
static __global__ void computeHilbertKeysKernel(const double* __restrict__ x,
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
static __global__ void nodeFpCentersKernel(const uint64_t* __restrict__ prefixes,
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
// for max-jiggle h (3.56e-3). 256 was enough for near-uniform h, but when a
// particle's h grows during Newton (rarefied/post-shock gas) the search sphere
// can overlap far more leaves; silently truncating the list under-counts
// neighbours -> rho too low -> Newton grows h further -> runaway feedback.
// 1024 + overflow COUNTING (see d_overflow) instead of silent truncation.
// Memory cost: nLeaves * 1024 * 4 = ~640 MB on 10M particles (fits A100/MI300).
static constexpr int MAX_J_PER_LEAF = 1024;
// Reverse of internalToLeaf: leaf-CSL index → internal linked-tree node idx.
// Needed to look up center/size of each i-leaf.
static __global__ void buildLeafToInternalKernel(
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
static __global__ void buildParticleToLeafKernel(
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
static __global__ void computeHmaxLeafKernel(
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

// Largest h anywhere beneath each node.  Launched once per tree level, DEEPEST
// FIRST, so a node's 8 children are final when it is read.  Leaves occur at every
// level, hence the childOffsets==0 branch rather than a separate leaf pass.
static __global__ void hmaxUpsweepKernel(TreeNodeIndex first,
                                         TreeNodeIndex last,
                                         const TreeNodeIndex* __restrict__ childOffsets,
                                         const TreeNodeIndex* __restrict__ internalToLeaf,
                                         const double*        __restrict__ hmax_leaf,
                                         double*              __restrict__ hmax_node)
{
    TreeNodeIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) return;

    TreeNodeIndex c = childOffsets[i];
    if (c == 0)                                  // leaf at this level
    {
        double hm = hmax_leaf[internalToLeaf[i]];
        hmax_node[i] = (isfinite(hm) && hm > 0.0) ? hm : 0.0;
        return;
    }

    double m = 0.0;
    for (int o = 0; o < 8; ++o) m = fmax(m, hmax_node[c + o]);
    hmax_node[i] = m;
}

/*! @brief Can node @p node hold a particle interacting with one in the i-leaf?
 *
 * Symmetric=false (density): radius 2*hmax_i, applied by inflating the i-box.
 * Symmetric=true  (force):   radius 2*max(hmax_i, hmax_node).  Depends on the
 *     candidate, so the i-box cannot be pre-inflated; tested as a Euclidean distance
 *     between the raw boxes, which is tighter than inflate-and-overlap and costs the
 *     same.
 */
template<bool Symmetric>
__device__ inline bool nodeInRange(const Vec3<double>& iCenter,
                                   const Vec3<double>& iSize,
                                   const Vec3<double>& iHalf,
                                   double              twoHi,
                                   const double* __restrict__ hmax_node,
                                   TreeNodeIndex node,
                                   const Vec3<double>* __restrict__ centers,
                                   const Vec3<double>* __restrict__ sizes)
{
    if constexpr (Symmetric)
    {
        const double rs = fmax(twoHi, 2.0 * hmax_node[node]);
        return norm2(minDistance(iCenter, iSize, centers[node], sizes[node])) <= rs * rs;
    }
    else
    {
        return norm2(minDistance(iCenter, iHalf, centers[node], sizes[node])) <= 0.0;
    }
}

// Single-pass DFS for each active i-leaf, indexed via activeLeaves indirection.
// Symmetric=false: gather only (density); hmax_node may be nullptr.
// Symmetric=true : gather + scatter (force); hmax_node must cover all numNodes,
//                  i.e. hmaxUpsweepKernel must have run over the whole tree.
template<bool Symmetric>
__global__ void buildJLeafListKernel(
    const TreeNodeIndex* __restrict__ leafToInternal,
    const double*         __restrict__ hmax_leaf,
    const double*         __restrict__ hmax_node,
    const Vec3<double>*   __restrict__ centers,
    const Vec3<double>*   __restrict__ sizes,
    const TreeNodeIndex*  __restrict__ childOffsets,
    const TreeNodeIndex*  __restrict__ internalToLeaf,
    int              nActiveLeaves,
    const int*       __restrict__ activeLeaves,
    int* __restrict__ jlist,
    int* __restrict__ jcount,
    int* __restrict__ overflow)   // [0] += jlist truncations, [1] += stack drops
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
    const double twoHi = 2.0 * hmax;
    // Gather only: the i-leaf's box expanded by 2·hmax.  The symmetric radius depends
    // on the candidate node, so that path cannot pre-inflate and leaves this unused.
    Vec3<double> iHalf{ iSize[0] + twoHi,
                        iSize[1] + twoHi,
                        iSize[2] + twoHi };

    // DFS traversal.  The old bound (7 x tree_depth ~ 32) assumed a NARROW
    // search sphere; once hmax grows the sphere overlaps many subtrees and the
    // pending-node count is no longer depth-limited.  Dropping subtrees on
    // stack overflow silently loses neighbours (same runaway feedback as the
    // jlist cap), so use a generous stack and COUNT any overflow.
    constexpr int MAXSTK = 192;
    TreeNodeIndex stack[MAXSTK];
    stack[0] = -1;
    int stackPos = 1;
    int count    = 0;
    int jBase    = iLeaf * MAX_J_PER_LEAF;

    // Check root overlap.
    if (!nodeInRange<Symmetric>(iCenter, iSize, iHalf, twoHi, hmax_node, 0, centers, sizes))
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
            if (!nodeInRange<Symmetric>(iCenter, iSize, iHalf, twoHi, hmax_node, child,
                                        centers, sizes))
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
                else                   atomicAdd(&overflow[1], 1);  // subtree DROPPED
            }
        }
        node = stack[--stackPos];
    } while (node != -1);

    if (count > MAX_J_PER_LEAF) atomicAdd(&overflow[0], count - MAX_J_PER_LEAF);
    jcount[iLeaf] = min(count, MAX_J_PER_LEAF);
}

//! @brief Per-phase cost of one tree build, in seconds.
struct TreeTimings
{
    double bbox     = 0.0;
    double keysSort = 0.0;
    double build    = 0.0;
    double nodes    = 0.0;
};

/*! @brief Build the octree for the particles currently in @p s, in place.
 *
 * Sorts s.x/y/z/h into Hilbert order (and anything in @p alsoSort alongside them),
 * fills s.order, s.octree, s.box, s.centers, s.sizes, s.leafToInternal, s.layout,
 * s.particleLeaf, s.nLeaves and s.numNodes, and sizes s.jlist/s.jcount/s.hmax_leaf.
 *
 * Allocations are reused across calls; the tree contents are rebuilt every time,
 * because the particles have moved.
 */
void buildTree(GpuState& s,
               const std::vector<thrust::device_vector<double>*>& alsoSort,
               TreeTimings& tt);
