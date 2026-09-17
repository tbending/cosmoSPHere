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

#include <thrust/binary_search.h>
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
    // Keys and the gather scratch live in `s`, so a tree build allocates no particle-
    // sized buffers.  The sorted arrays are swapped with sortTmp, both persistent.
    auto& d_keys = s.keys;
    d_keys.resize(ngas);

    constexpr int BLK = 256;
    computeHilbertKeysKernel<<<iceil(ngas, BLK), BLK>>>(
        rawPtr(s.x), rawPtr(s.y), rawPtr(s.z), rawPtr(s.h),
        rawPtr(d_keys), ngas, s.box);
    checkGpuErrors(cudaGetLastError());

    // Sort permutation by Hilbert key, then gather particle data.  Dead particles carry
    // the maximum key, so this same sort parks them at the end.
    s.order.resize(ngas);
    thrust::sequence(s.order.begin(), s.order.end());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), s.order.begin());

    // Live prefix: everything before the first maximum key.  The gathers below still
    // cover all ngas, so the dead tail keeps its own data; only the TREE is restricted.
    s.nAlive = (int)(thrust::lower_bound(thrust::device, d_keys.begin(), d_keys.end(),
                                         ~uint64_t(0)) - d_keys.begin());

    auto& d_tmp = s.sortTmp;
    d_tmp.resize(ngas);
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
    thrust::device_vector<unsigned>      counts = std::vector<unsigned>{(unsigned)s.nAlive};
    thrust::device_vector<uint64_t>      tmpTree;
    thrust::device_vector<TreeNodeIndex> workArray;

    // d_keys is already sorted — run update until the leaf partition is stable.
    while (!updateOctreeGpu(rawPtr(d_keys), rawPtr(d_keys) + s.nAlive,
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
    // layout[0] = 0, layout[L+1] = counts[0] + ... + counts[L].  counts has exactly
    // nLeaves entries: scanning to counts.end() + 1, as this used to, read one element
    // past the end on the device -- harmless while the next page happened to be mapped,
    // an illegal-address fault when it was not.
    s.layout.resize(s.nLeaves + 1);
    s.layout[0] = 0u;
    thrust::inclusive_scan(thrust::device,
                           counts.begin(), counts.end(),
                           s.layout.begin() + 1);
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

    s.particleLeaf.resize(ngas);   // entries past nAlive are never set and never read
    buildParticleToLeafKernel<<<iceil(s.nLeaves, 256), 256>>>(
        rawPtr(s.layout), s.nLeaves, rawPtr(s.particleLeaf));
    checkGpuErrors(cudaGetLastError());

    // Storage for the walk. Reused across calls; every entry is written before use.
    // jlist itself is sized by buildJLeafListsCSR, from the counts.
    s.hmax_leaf.resize(s.nLeaves);
    s.jcount.resize(s.nLeaves);
    s.jOffset.resize(s.nLeaves + 1);
    s.allLeaves.resize(s.nLeaves);
    thrust::sequence(s.allLeaves.begin(), s.allLeaves.end());
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
