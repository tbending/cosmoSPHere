/*
 * tree.cu — build the Cornerstone octree for the current particle positions.
 *
 * Lifted verbatim out of solveDensH: same order of operations, same Thrust calls,
 * same arithmetic.  What changed is only where the results go — into GpuState, so the
 * force pass can use the tree without rebuilding it.
 */

#include "tree.cuh"

// Cornerstone GPU tree builders — kept out of tree.cuh so their non-static
// __global__s are emitted in exactly one translation unit.
#include "tree/csarray_gpu.cuh"
#include "tree/octree_gpu.cuh"

#include <algorithm>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/swap.h>

void buildTree(GpuState& s,
               const std::vector<thrust::device_vector<double>*>& alsoSort,
               TreeTimings& tt)
{
    const int ngas = s.ngas;

    cudaEvent_t e0, e1, e2, e3, e4;
    for (auto* e : {&e0, &e1, &e2, &e3, &e4}) checkGpuErrors(hipEventCreate(e));

    // -----------------------------------------------------------------------
    // Bounding box — entirely on the GPU.  Avoids 6 serial host loops over 10M
    // elements.
    // -----------------------------------------------------------------------
    HIP_CHECK(hipEventRecord(e0));
    double xmin, xmax, ymin, ymax, zmin, zmax;
    {
        auto [xlo, xhi] = thrust::minmax_element(thrust::device, s.x.begin(), s.x.end());
        auto [ylo, yhi] = thrust::minmax_element(thrust::device, s.y.begin(), s.y.end());
        auto [zlo, zhi] = thrust::minmax_element(thrust::device, s.z.begin(), s.z.end());
        // Dereferencing device iterators triggers an implicit device→host copy.
        xmin = *xlo; xmax = *xhi;
        ymin = *ylo; ymax = *yhi;
        zmin = *zlo; zmax = *zhi;
    }
    HIP_CHECK(hipEventRecord(e1));

    // Pad box slightly (mirrors Fortran *1.00001 on the largest side).
    double maxSpan = std::max({xmax-xmin, ymax-ymin, zmax-zmin}) * 1.00001;
    double xctr = 0.5*(xmin+xmax), yctr = 0.5*(ymin+ymax), zctr = 0.5*(zmin+zmax);
    double half = 0.5 * maxSpan;
    s.box = Box<double>{xctr - half, xctr + half,
                        yctr - half, yctr + half,
                        zctr - half, zctr + half,
                        BoundaryType::open};

    // -----------------------------------------------------------------------
    // Hilbert keys + GPU sort
    // -----------------------------------------------------------------------
    thrust::device_vector<uint64_t> d_keys(ngas);

    constexpr int BLK = 256;
    computeHilbertKeysKernel<<<iceil(ngas, BLK), BLK>>>(
        rawPtr(s.x), rawPtr(s.y), rawPtr(s.z),
        rawPtr(d_keys), ngas, s.box);
    checkGpuErrors(cudaGetLastError());

    // Sort permutation by Hilbert key, then gather particle data.
    s.order.resize(ngas);
    thrust::sequence(s.order.begin(), s.order.end());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), s.order.begin());

    thrust::device_vector<double> d_tmp(ngas);
    for (auto* v : {&s.x, &s.y, &s.z, &s.h})
    {
        thrust::gather(s.order.begin(), s.order.end(), v->begin(), d_tmp.begin());
        thrust::swap(*v, d_tmp);
    }
    for (auto* v : alsoSort)
    {
        thrust::gather(s.order.begin(), s.order.end(), v->begin(), d_tmp.begin());
        thrust::swap(*v, d_tmp);
    }
    HIP_CHECK(hipEventRecord(e2));

    // -----------------------------------------------------------------------
    // Cornerstone leaf tree + fully linked internal tree
    // -----------------------------------------------------------------------
    thrust::device_vector<uint64_t>      csTree = std::vector<uint64_t>{0, nodeRange<uint64_t>(0)};
    thrust::device_vector<unsigned>      counts = std::vector<unsigned>{(unsigned)ngas};
    thrust::device_vector<uint64_t>      tmpTree;
    thrust::device_vector<TreeNodeIndex> workArray;

    // d_keys is already sorted — run update until the leaf partition is stable.
    while (!updateOctreeGpu(rawPtr(d_keys), rawPtr(d_keys) + ngas,
                            BUCKET_SIZE, csTree, counts, tmpTree, workArray))
    {
        // iterate until stable leaf partition
    }

    s.octree.resize(nNodes(csTree));
    buildLinkedTreeGpu(rawPtr(csTree), s.octree.data());
    checkGpuErrors(cudaGetLastError());

    // Particle layout (prefix-sum of counts → first particle of each leaf).
    s.nLeaves  = (int)nNodes(csTree);
    s.numNodes = s.octree.numNodes;
    s.layout.resize(s.nLeaves + 1);
    thrust::exclusive_scan(thrust::device,
                           counts.begin(), counts.end() + 1,
                           s.layout.begin(), 0u);
    HIP_CHECK(hipEventRecord(e3));

    // -----------------------------------------------------------------------
    // Node geometry and the leaf/particle maps.
    //
    // Centres and half-sizes come straight from each node's SFC prefix — no particle
    // data, valid at every level.  That is why the neighbour test costs nothing
    // geometrically, and why hmax (which cannot be derived from a key) needs its own
    // upsweep.
    // -----------------------------------------------------------------------
    s.centers.resize(s.numNodes);
    s.sizes.resize(s.numNodes);
    nodeFpCentersKernel<<<iceil(s.numNodes, 256), 256>>>(
        rawPtr(s.octree.prefixes), s.numNodes,
        rawPtr(s.centers), rawPtr(s.sizes), s.box);
    checkGpuErrors(cudaGetLastError());

    s.leafToInternal.assign(s.nLeaves, -1);
    buildLeafToInternalKernel<<<iceil(s.numNodes, 256), 256>>>(
        rawPtr(s.octree.childOffsets), rawPtr(s.octree.internalToLeaf),
        s.numNodes, rawPtr(s.leafToInternal));
    checkGpuErrors(cudaGetLastError());

    s.particleLeaf.resize(ngas);
    buildParticleToLeafKernel<<<iceil(s.nLeaves, 256), 256>>>(
        rawPtr(s.layout), s.nLeaves, rawPtr(s.particleLeaf));
    checkGpuErrors(cudaGetLastError());

    // Storage for the walk. Reused across calls; every entry is written before use.
    s.hmax_leaf.resize(s.nLeaves);
    s.jlist.resize((size_t)s.nLeaves * MAX_J_PER_LEAF);
    s.jcount.resize(s.nLeaves);
    s.overflow.assign(2, 0);
    HIP_CHECK(hipEventRecord(e4));
    checkGpuErrors(hipEventSynchronize(e4));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); tt.bbox     = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, e1, e2)); tt.keysSort = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, e2, e3)); tt.build    = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, e3, e4)); tt.nodes    = ms * 1e-3;
    for (auto* e : {&e0, &e1, &e2, &e3, &e4}) HIP_CHECK(hipEventDestroy(*e));
}
