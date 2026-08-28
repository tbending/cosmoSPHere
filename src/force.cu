/*
 * force.cu — the symmetric neighbour walk, and the GPU force kernel.
 *
 * The tree walk itself is shared with the density solve (tree.cuh); what differs is
 * one line of the accept test, selected by the Symmetric template parameter.
 */

#include "force.hpp"
#include "tree.cuh"
#include "kernel.hpp"

#include <cstdio>
#include <vector>

#include <thrust/sequence.h>

void buildForceJLeafList(GpuState& s, ForceTimings& ft)
{
    // Already symmetric for this tree.  phantom calls force more often than density
    // (deriv.f90:105-109 — icall=2 reuses the tree because positions have not moved),
    // and the list is a materialised array in the state, so the second call is free.
    if (s.jlistToken == s.token) return;

    cudaEvent_t e0, e1, e2;
    for (auto* e : {&e0, &e1, &e2}) checkGpuErrors(hipEventCreate(e));
    HIP_CHECK(hipEventRecord(e0));

    // -----------------------------------------------------------------------
    // Propagate hmax to every node, deepest level first.
    //
    // The gather walk only ever needs hmax for the i-leaf — one number, from the
    // particles the thread already holds.  The symmetric test needs it for the FAR
    // side, including internal nodes it is trying to reject without descending into,
    // and unlike node geometry it cannot be derived from the SFC key.  Hence an
    // upsweep.
    //
    // levelRange lives on the device and is tiny (maxTreeLevel+2 = 23 entries for
    // 64-bit keys).  Only ~5 levels are populated at this problem size — octree depth
    // is log8(npart/bucket) — so most iterations are skipped outright.
    // -----------------------------------------------------------------------
    s.hmax_node.assign(s.numNodes, 0.0);

    std::vector<TreeNodeIndex> h_levelRange(s.octree.levelRange.size());
    HIP_CHECK(hipMemcpy(h_levelRange.data(), rawPtr(s.octree.levelRange),
                        h_levelRange.size() * sizeof(TreeNodeIndex),
                        hipMemcpyDeviceToHost));

    for (int lvl = int(maxTreeLevel<uint64_t>{}); lvl >= 0; --lvl)
    {
        TreeNodeIndex first = h_levelRange[lvl];
        TreeNodeIndex last  = h_levelRange[lvl + 1];
        if (last <= first) continue;                  // empty level
        hmaxUpsweepKernel<<<iceil(last - first, 256), 256>>>(
            first, last,
            rawPtr(s.octree.childOffsets), rawPtr(s.octree.internalToLeaf),
            rawPtr(s.hmax_leaf), rawPtr(s.hmax_node));
        checkGpuErrors(cudaGetLastError());
    }
    HIP_CHECK(hipEventRecord(e1));

    // -----------------------------------------------------------------------
    // The walk, over every leaf.  Identical traversal to the density one; the only
    // difference is that the accept radius becomes 2*max(hmax_i, hmax_node) instead
    // of 2*hmax_i, tested as a Euclidean distance between the raw boxes rather than
    // inflate-and-overlap.  Overwrites the gather lists.
    // -----------------------------------------------------------------------
    thrust::device_vector<int> d_allLeaves(s.nLeaves);
    thrust::sequence(d_allLeaves.begin(), d_allLeaves.end());

    buildJLeafListKernel<true><<<iceil(s.nLeaves, 256), 256>>>(
        rawPtr(s.leafToInternal), rawPtr(s.hmax_leaf), rawPtr(s.hmax_node),
        rawPtr(s.centers), rawPtr(s.sizes),
        rawPtr(s.octree.childOffsets), rawPtr(s.octree.internalToLeaf),
        s.nLeaves, rawPtr(d_allLeaves),
        rawPtr(s.jlist), rawPtr(s.jcount),
        rawPtr(s.overflow));
    checkGpuErrors(cudaGetLastError());

    HIP_CHECK(hipEventRecord(e2));
    checkGpuErrors(hipEventSynchronize(e2));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); ft.hmaxUpsweep = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, e1, e2)); ft.jleafBuild  = ms * 1e-3;
    for (auto* e : {&e0, &e1, &e2}) HIP_CHECK(hipEventDestroy(*e));

    s.jlistToken = s.token;

    // The symmetric radius makes the lists longer, so a truncation that never fired
    // for gather could fire here.  Silent truncation loses neighbours — make it loud.
    int ovf[2] = {0, 0};
    HIP_CHECK(hipMemcpy(ovf, rawPtr(s.overflow), 2*sizeof(int), hipMemcpyDeviceToHost));
    if (ovf[0] > 0 || ovf[1] > 0)
        std::fprintf(stderr,
            "WARNING! buildForceJLeafList: jlist_trunc=%d stack_drops=%d\n",
            ovf[0], ovf[1]);
}

