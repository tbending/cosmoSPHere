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
 */

#pragma once

#include <cstdint>
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
    // csTree/counts are the cornerstone leaf tree and its per-leaf particle counts, the
    // in/out pair updateOctreeGpu iterates on; tmpTree and workArray are its scratch.
    // These were function locals in buildTree, so all four were allocated and freed on
    // every density solve -- exactly the thrust churn this struct exists to remove.
    // Only the STORAGE persists: buildTree re-seeds csTree and counts every call, so the
    // tree is still rebuilt from scratch (see the note at the top of this file).
    thrust::device_vector<uint64_t>              csTree, tmpTree;
    thrust::device_vector<unsigned>              counts;
    thrust::device_vector<cstone::TreeNodeIndex> workArray;
    cstone::OctreeData<uint64_t, cstone::GpuTag> octree;
    cstone::Box<double> box{0., 1., cstone::BoundaryType::open};
    thrust::device_vector<cstone::Vec3<double>> centers, sizes;   // per node, geometric
    thrust::device_vector<cstone::TreeNodeIndex> leafToInternal;
    thrust::device_vector<unsigned> layout;      // leaf L owns particles [layout[L], layout[L+1])
    thrust::device_vector<int> particleLeaf;     // particle -> its leaf

    // ---- smoothing length per leaf, and per node ----
    // hmax_node is force-only: the gather walk only ever needs h for its own i-leaf.
    thrust::device_vector<double> hmax_leaf, hmax_node;

    // ---- j-leaf lists ----
    // CSR: leaf L occupies jlist[jOffset[L] ... jOffset[L]+jcount[L]), and jOffset has
    // nLeaves+1 entries, the last being the total.  See buildJLeafListsCSR.
    // Gather-only after the density solve; SYMMETRIC after buildForceJLeafList.
    thrust::device_vector<int> jlist, jcount, jOffset;
    thrust::device_vector<int> allLeaves;        // 0..nLeaves-1, for whole-tree list builds
    thrust::device_vector<int> overflow;         // [0] list truncations, [1] stack drops

    // ---- density solve and tree build scratch, resized per call and reused ----
    thrust::device_vector<uint64_t> keys;        // Hilbert keys
    thrust::device_vector<double> sortTmp;       // gather target, swapped with each sorted array
    thrust::device_vector<double> ax, ay, az;    // acceleration, gradient sweep only
    thrust::device_vector<int> converged, activeParticles, activeTmp, activeLeaves, activeLeavesTmp;
    thrust::device_vector<double> divv, ddivvdt, xi;     // gradient sweep outputs, Hilbert order
    thrust::device_vector<double> dStage;                // phantom-order staging for downloads

    // ---- force pass buffers, sized to ngas and reused across calls ----
    thrust::device_vector<double> pro2, spsound, alphaAV, u;         // inputs, Hilbert order
    // per-particle factors of the force sum, formed once per pass (forcePrepKernel)
    thrust::device_vector<double> hsqinv, hinv, rhoh, rho1, grkfac, pres, auterm, divfac;
    thrust::device_vector<double> fx, fy, fz, f4, vsigmax, divvF;   // outputs, Hilbert order
    thrust::device_vector<double> fStage;        // one array in phantom order, either direction

    int ngas     = 0;
    double hfact = 0.0;   // h-rho relation of the last density solve, for the force pass
    // Live particles (h > 0) form the prefix [0, nAlive) of the Hilbert order; dead
    // ones sit after it, belong to no leaf, never enter a j-list, and keep the values
    // phantom gave them (in particular their negative h).
    int nAlive   = 0;
    int nLeaves  = 0;
    int numNodes = 0;

    // Bumped by each completed density solve; 0 = none has run.  NOT consumed by
    // force: deriv.f90:105-109 (icall=2) reuses the tree on purpose, so force runs
    // more often than density.
    uint64_t token = 0;

    // The token jlist holds the SYMMETRIC list for.  Density overwrites jlist and
    // bumps token, so a mismatch is exactly "needs rebuild" — nothing to invalidate.
    uint64_t jlistToken = 0;

    // The token of the last force pass.  The first force pass after a solve can use
    // the velocities the solve uploaded -- derivs passes the same array to both --
    // but a later pass on the same tree is the leapfrog corrector, whose velocities
    // have changed, so it uploads them again.
    uint64_t forceToken = 0;

    bool readyForForce(int n) const { return ngas == n && token != 0; }
};

//! @brief The one state shared by the density and force entry points (gpu_state.cu).
GpuState& gpuState();
