/*
 * force.cu — the symmetric neighbour walk, and the GPU force kernel.
 *
 * The tree walk itself is shared with the density solve (tree.cuh); what differs is
 * one line of the accept test, selected by the Symmetric template parameter.
 */

#include "force.hpp"
#include "tree.cuh"
#include "kernel.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
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
    buildJLeafListsCSR<true>(s, rawPtr(s.hmax_node));

    HIP_CHECK(hipEventRecord(e2));
    checkGpuErrors(hipEventSynchronize(e2));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); ft.hmaxUpsweep = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, e1, e2)); ft.jleafBuild  = ms * 1e-3;
    for (auto* e : {&e0, &e1, &e2}) HIP_CHECK(hipEventDestroy(*e));

    s.jlistToken = s.token;

    // Under CSR a list cannot truncate, and the stack can only drop subtrees on a
    // pathological tree; either would lose neighbours silently, so make it loud.
    int ovf[2] = {0, 0};
    HIP_CHECK(hipMemcpy(ovf, rawPtr(s.overflow), 2*sizeof(int), hipMemcpyDeviceToHost));
    if (ovf[0] > 0 || ovf[1] > 0)
        std::fprintf(stderr,
            "WARNING! buildForceJLeafList: jlist_trunc=%d stack_drops=%d\n",
            ovf[0], ovf[1]);
}

// SPH force on one particle per thread, over the symmetric j-leaf list.  Threads index
// Hilbert-sorted particles.
// DiscVisc selects phantom's disc_viscosity form of the artificial viscosity (see below).
template<bool Periodic, bool DiscVisc>
__global__ void sphForceKernelJList(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ vz,
    const double* __restrict__ h,
    const double* __restrict__ gradh,
    const double* __restrict__ pro2,
    const double* __restrict__ spsound,
    const double* __restrict__ alphaAV,
    const double* __restrict__ u,
    double* __restrict__ fx,
    double* __restrict__ fy,
    double* __restrict__ fz,
    double* __restrict__ f4,
    double* __restrict__ vsigmax,
    double* __restrict__ divv,
    int n,
    double pmass,
    double beta,
    double alphau,
    const int* __restrict__ particleLeaf,
    const int* __restrict__ jOffset,
    const int* __restrict__ jcount,
    const int* __restrict__ jlist,
    const unsigned* __restrict__ layout,
    Box<double> box)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    const double hi = h[i];

    // Dead particles (h <= 0): no force, and skip the neighbour loop.
    if (!(hi > 0.0))
    {
        fx[i]      = 0.0;
        fy[i]      = 0.0;
        fz[i]      = 0.0;
        f4[i]      = 0.0;
        vsigmax[i] = 0.0;
        divv[i]    = 0.0;
        return;
    }

    const double xi = x[i], yi = y[i], zi = z[i];
    const double hi_sq_inv = 1.0 / (hi * hi);
    const double hi_4_inv  = hi_sq_inv * hi_sq_inv;

    // rhoh(hi), as in phantom's force.F90, not the summed density
    const double hfoh_i = sph::hfact / hi;
    const double rhoi   = pmass * hfoh_i * hfoh_i * hfoh_i;

    const double grad_i      = gradh[i];   // d(rho)/dh at fixed q, from the density solve
    const double dhdrhoi     = -hi / (3.0 * rhoi);
    const double omegai      = 1.0 - dhdrhoi * grad_i;
    const double omega_inv_i = 1 / omegai;
    const double hfacgrkerni = hi_4_inv * sph::cnormk * omega_inv_i;   // hfacgrkern in force.F90

    const double pro2i   = pro2[i];
    const double vwavei  = spsound[i];
    const double alphai  = alphaAV[i];
    const double eni     = u[i];
    const double rho1i   = 1.0 / rhoi;
    const double pri     = pro2i * rhoi * rhoi;
    const double autermi = 0.5 * pmass * rho1i * alphau;

    double itermx = 0.0;   // force from neighbours inside h_i
    double itermy = 0.0;
    double itermz = 0.0;
    double jtermx = 0.0;   // force from neighbours inside h_j
    double jtermy = 0.0;
    double jtermz = 0.0;

    double f4sum     = 0.0;   // du/dt
    double vsigmax_i = 0.0;
    double divv_s    = 0.0;

    const int iLeaf = particleLeaf[i];
    const int jBase = jOffset[iLeaf];
    const int nj    = jcount[iLeaf];

    for (int jl = 0; jl < nj; ++jl)
    {
        const int jLeaf = jlist[jBase + jl];
        for (unsigned j = layout[jLeaf]; j < layout[jLeaf + 1]; ++j)
        {
            double dx  = xi - x[j];
            double dy  = yi - y[j];
            double dz  = zi - z[j];
            nearestImage<Periodic>(dx, dy, dz, box);
            double dr2 = dx*dx + dy*dy + dz*dz;
            if (!(dr2 > 0.0)) continue;
            double qij2 = dr2 * hi_sq_inv;

            const double hj        = h[j];
            const double hj_sq_inv = 1.0 / (hj * hj);
            const double hj_4_inv  = hj_sq_inv * hj_sq_inv;

            double qj_ij2 = dr2 * hj_sq_inv;

            double dr    = sqrt(dr2);
            double runix = dx / dr;   // unit vector (r_i - r_j) / |r_i - r_j|
            double runiy = dy / dr;
            double runiz = dz / dr;
            const double dvxij = vx[i] - vx[j];
            const double dvyij = vy[i] - vy[j];
            const double dvzij = vz[i] - vz[j];
            const double projv = dvxij * runix + dvyij * runiy + dvzij * runiz;

            // rhoh(hj)
            const double hfoh_j = sph::hfact / hj;
            const double rhoj   = pmass * hfoh_j * hfoh_j * hfoh_j;

            const double rho1j   = 1.0 / rhoj;
            const double pro2j   = pro2[j];
            const double vwavej  = spsound[j];
            const double alphaj  = alphaAV[j];
            const double enj     = u[j];

            const double prj     = pro2j * rhoj * rhoj;
            const double autermj = 0.5 * pmass * rho1j * alphau;

            // signal speeds: vsig* with alpha = 1 for the timestep,
            // vsigav* with the particle's alpha for the viscosity
            const double vsigi   = fmax(vwavei - beta * projv, 0.0);
            const double vsigavi = fmax(alphai * vwavei - beta * projv, 0.0);
            const double vsigj   = fmax(vwavej - beta * projv, 0.0);
            const double vsigavj = fmax(alphaj * vwavej - beta * projv, 0.0);

            const double pair_vsigmax = fmax(vsigi, vsigj);

            double qrho2i = 0.0;
            double qrho2j = 0.0;

            const double denij  = eni - enj;
            const double rhoav1 = 2.0 / (rhoi + rhoj);
            const double vsigu  = sqrt(fabs(pri - prj) * rhoav1);

            if constexpr (DiscVisc)
            {
                // phantom's disc_viscosity (force.F90), how a Shakura-Sunyaev alpha is
                // modelled with the artificial viscosity: scaled by h/r_ij, and applied to
                // receding pairs too, without the beta term
                const double hor_i = hi / dr, hor_j = hj / dr;
                if (projv < 0.0)
                {
                    qrho2i = -0.5 * rho1i * (alphai * vwavei - beta * projv) * hor_i * projv;
                    qrho2j = -0.5 * rho1j * (alphaj * vwavej - beta * projv) * hor_j * projv;
                }
                else
                {
                    qrho2i = -0.5 * rho1i * alphai * vwavei * hor_i * projv;
                    qrho2j = -0.5 * rho1j * alphaj * vwavej * hor_j * projv;
                }
            }
            else if (projv < 0.0)
            {
                qrho2i = -0.5 * rho1i * vsigavi * projv;
                qrho2j = -0.5 * rho1j * vsigavj * projv;
            }

            if (qij2 < sph::radk2)
            {
                vsigmax_i = fmax(vsigmax_i, pair_vsigmax);

                double qij, wij, grwij;
                qij = sqrt(qij2);
                sph::m4_kern(qij, wij, grwij);

                // div v: projv is already (v_i - v_j) . r_ij / |r_ij|
                divv_s += pmass * grwij * projv;

                const double gradkerni = grwij * hfacgrkerni;   // F_ij(h_i) / omega_i

                const double gradpi = pmass * (pro2i + qrho2i) * gradkerni;

                itermx += -gradpi * runix;
                itermy += -gradpi * runiy;
                itermz += -gradpi * runiz;

                // du/dt: p dV work, viscous heating, conductivity
                const double pdvtermi     = pmass * pro2i * projv * gradkerni;
                // disc viscosity heats with its own form, as in force.F90
                const double dudtdissi    = DiscVisc
                    ? -0.5 * pmass * rho1i * alphai * vwavei * (hi / dr) * projv * projv * gradkerni
                    : pmass * qrho2i * projv * gradkerni;
                const double dendisstermi = vsigu * denij * autermi * gradkerni;

                f4sum += pdvtermi;
                f4sum += dudtdissi;
                f4sum += dendisstermi;
            }

            if (qj_ij2 < sph::radk2)
            {
                vsigmax_i = fmax(vsigmax_i, pair_vsigmax);

                double qj_ij, wj_ij, grwj_ij;
                qj_ij = sqrt(qj_ij2);
                sph::m4_kern(qj_ij, wj_ij, grwj_ij);

                const double dhdrhoj     = -hj / (3.0 * rhoj);
                const double grad_j      = gradh[j];
                const double omegaj      = 1.0 - dhdrhoj * grad_j;
                const double omega_inv_j = 1 / omegaj;

                double hfacgrkernj = hj_4_inv * sph::cnormk * omega_inv_j;
                double gradkernj   = grwj_ij * hfacgrkernj;   // F_ij(h_j) / omega_j

                const double gradpj = pmass * (pro2j + qrho2j) * gradkernj;

                jtermx -= gradpj * runix;
                jtermy -= gradpj * runiy;
                jtermz -= gradpj * runiz;

                const double dendisstermj = vsigu * denij * autermj * gradkernj;

                f4sum += dendisstermj;
            }
        }
    }

    fx[i]      = itermx + jtermx;
    fy[i]      = itermy + jtermy;
    fz[i]      = itermz + jtermz;
    f4[i]      = f4sum;
    vsigmax[i] = vsigmax_i;

    const double term_divv = sph::cnormk * omega_inv_i * hi_4_inv / rhoi;

    divv[i] = -divv_s * term_divv;
}

void computeForces(GpuState& s, const ForceFields& f, double pmass, double beta,
                   double alphau, bool discViscosity, ForceTimings& ft)
{
    const int n = s.ngas;

    buildForceJLeafList(s, ft);

    // Phase boundaries on the device timeline, like buildForceJLeafList: recording
    // an event does not synchronise, so timing does not perturb what it measures.
    cudaEvent_t e0, e1, e2, e3;
    for (auto* e : {&e0, &e1, &e2, &e3}) checkGpuErrors(hipEventCreate(e));
    HIP_CHECK(hipEventRecord(e0));

    // Every device buffer lives in GpuState and is resized to n, a no-op after the first
    // call, so a force pass allocates nothing on the device.
    const size_t nbytes = static_cast<size_t>(n) * sizeof(double);
    s.fStage.resize(n);

    // phantom order -> Hilbert order (s.order maps sorted index -> phantom index)
    auto upload = [&](const double* host, thrust::device_vector<double>& sorted)
    {
        HIP_CHECK(hipMemcpy(rawPtr(s.fStage), host, nbytes, hipMemcpyHostToDevice));
        sorted.resize(n);
        thrust::gather(s.order.begin(), s.order.end(), s.fStage.begin(), sorted.begin());
    };
    // Positions and h: the solve's Hilbert-sorted copies are exactly what phantom holds
    // -- the positions it uploaded and the converged h it stored back -- and phantom
    // only calls force again on the same tree when positions have not moved.
    // Velocities: the solve's copies serve the first force pass after it; a later one is
    // the corrector, with new velocities (see GpuState::forceToken).
    const bool refreshV = (s.forceToken == s.token) || (int)s.vx.size() != n;
    if (refreshV)
    {
        upload(f.vx, s.vx);
        upload(f.vy, s.vy);
        upload(f.vz, s.vz);
    }
    s.forceToken = s.token;

    upload(f.pro2,    s.pro2);
    upload(f.spsound, s.spsound);
    upload(f.alphaAV, s.alphaAV);
    upload(f.u,       s.u);

    // Not zeroed: the kernel writes all n entries of every output, dead particles included.
    for (auto* v : {&s.fx, &s.fy, &s.fz, &s.f4, &s.vsigmax, &s.divvF}) v->resize(n);
    HIP_CHECK(hipEventRecord(e1));

    // Periodic or not is whatever the density solve built this tree with.
    auto launch = [&](auto periodic, auto disc) {
        sphForceKernelJList<decltype(periodic)::value, decltype(disc)::value><<<iceil(n, 256), 256>>>(
            rawPtr(s.x), rawPtr(s.y), rawPtr(s.z),
            rawPtr(s.vx), rawPtr(s.vy), rawPtr(s.vz),
            rawPtr(s.h), rawPtr(s.gradh),
            rawPtr(s.pro2), rawPtr(s.spsound), rawPtr(s.alphaAV), rawPtr(s.u),
            rawPtr(s.fx), rawPtr(s.fy), rawPtr(s.fz), rawPtr(s.f4),
            rawPtr(s.vsigmax), rawPtr(s.divvF),
            n, pmass, beta, alphau,
            rawPtr(s.particleLeaf), rawPtr(s.jOffset), rawPtr(s.jcount), rawPtr(s.jlist),
            rawPtr(s.layout), s.box);
    };
    dispatchPeriodic(s.box, [&](auto periodic) {
        if (discViscosity) launch(periodic, std::true_type{});
        else               launch(periodic, std::false_type{});
    });
    checkGpuErrors(cudaGetLastError());
    HIP_CHECK(hipEventRecord(e2));
    HIP_CHECK(hipDeviceSynchronize());

    // Hilbert order -> phantom order, then to the host
    auto download = [&](const thrust::device_vector<double>& sorted, double* host)
    {
        thrust::scatter(sorted.begin(), sorted.end(), s.order.begin(), s.fStage.begin());
        HIP_CHECK(hipMemcpy(host, rawPtr(s.fStage), nbytes, hipMemcpyDeviceToHost));
    };
    download(s.fx, f.fx);
    download(s.fy, f.fy);
    download(s.fz, f.fz);
    download(s.f4, f.f4);
    download(s.vsigmax, f.vsigmax);
    download(s.divvF, f.divv);

    HIP_CHECK(hipEventRecord(e3));
    checkGpuErrors(hipEventSynchronize(e3));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); ft.upload   = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, e1, e2)); ft.kernel   = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, e2, e3)); ft.download = ms * 1e-3;
    for (auto* e : {&e0, &e1, &e2, &e3}) HIP_CHECK(hipEventDestroy(*e));
}
