/*
 * gpu_state.hpp — device buffers that outlive a single solve.
 *
 * WHY THIS EXISTS
 * ---------------
 * Reusing the allocations is worth ~8 ms of a measured 33.7 ms solve: it removes the
 * Thrust allocate/free churn that used to happen on every call, and the pageable
 * upload/download shrink with warm buffers.
 *
 * It also lets a second pass reuse the tree instead of rebuilding it.  Phantom runs
 * density and force as separate calls (deriv.F90 :139 and :195), so a force pass that
 * had to re-upload, re-sort and rebuild would repeat ~10.8 ms of work every step.
 *
 * NOTE it does NOT skip the rebuild.  Positions move every step, so the tree is
 * rebuilt each density call; what persists is the storage, not the contents.
 *
 * PARTICLE ORDERING
 * -----------------
 * Everything here is in HILBERT-SORTED order, which is not phantom's order.
 * `order` maps sorted index -> original phantom index.  Each C API entry point
 * gathers inputs in and scatters outputs back, so phantom never sees the sorted
 * ordering.  Removing that round trip means keeping particles resident across steps
 * — deliberately not attempted yet.  It is worth ~0.5 ms; pinning the host buffers
 * is worth ~3.7 ms and is the better next target.
 *
 * Single device, single state.  No multi-GPU, no concurrent solves.
 *
 * A second pass reading this must be sure a solve actually ran for the current
 * positions; nothing here enforces that yet.
 */

#pragma once

#include <thrust/device_vector.h>

#include "sfc/box.hpp"
#include "tree/octree.hpp"

struct GpuState
{
    // ---- particle data, Hilbert-sorted, length ngas ----
    thrust::device_vector<double> x, y, z;
    thrust::device_vector<double> h;             // converged smoothing length
    thrust::device_vector<double> rho, gradh;    // evaluated at the converged h
    thrust::device_vector<double> vx, vy, vz;    // velocity

    // ---- ordering ----
    thrust::device_vector<int> order;            // sorted index -> phantom index

    // ---- the octree ----
    cstone::OctreeData<uint64_t, cstone::GpuTag> octree;
    cstone::Box<double> box{0., 1., cstone::BoundaryType::open};
    thrust::device_vector<cstone::Vec3<double>> centers, sizes;   // per node, geometric
    thrust::device_vector<cstone::TreeNodeIndex> leafToInternal;
    thrust::device_vector<unsigned> layout;      // leaf L owns particles [layout[L], layout[L+1])
    thrust::device_vector<int> particleLeaf;     // particle -> its leaf

    // ---- per-leaf smoothing length ----
    // Largest h in each leaf; the walk uses it to size the i-leaf's search radius.
    thrust::device_vector<double> hmax_leaf;

    // ---- j-leaf lists ----
    // Fixed stride: leaf L occupies jlist[L*MAX_J_PER_LEAF ... +jcount[L]).
    thrust::device_vector<int> jlist, jcount;
    thrust::device_vector<int> overflow;         // [0] list truncations, [1] stack drops

    int ngas     = 0;
    int nLeaves  = 0;
    int numNodes = 0;
};

//! @brief The one state shared by the density and force entry points (gpu_state.cu).
GpuState& gpuState();
