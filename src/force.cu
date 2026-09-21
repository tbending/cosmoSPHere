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
    // difference is that the accept radius becomes radkernel*max(hmax_i, hmax_node) instead
    // of radkernel*hmax_i, tested as a Euclidean distance between the raw boxes rather than
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

// Per-particle factors of the force sum, formed once per pass instead of once per pair.
// The expressions are those the kernel used to evaluate in its loop, in the same order.
__global__ void forcePrepKernel(
    const double* __restrict__ h,
    const double* __restrict__ gradh,
    const double* __restrict__ pro2,
    double* __restrict__ hsqinv,   // 1/h^2
    double* __restrict__ hinv,     // 1/h
    double* __restrict__ rhoh,     // rho(h), as phantom's rhoh, not the summed density
    double* __restrict__ rho1,     // 1/rho
    double* __restrict__ grkfac,   // cnormk h^-4 / Omega: grad W's factor, F_ij(h)/Omega
    double* __restrict__ pres,     // P = pro2 rho^2
    double* __restrict__ auterm,   // conductivity factor 0.5 m alphau / rho
    double* __restrict__ divfac,   // factor turning the div v sum into div v
    int n,
    double pmass,
    double alphau,
    double hfact)             // rho = pmass (hfact/h)^3, as the density solve used
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    const double hi = h[i];
    if (!(hi > 0.0))   // dead: never a neighbour, and its own force is set to zero
    {
        hsqinv[i] = 0.0; hinv[i] = 0.0; rhoh[i] = 0.0; rho1[i] = 0.0; grkfac[i] = 0.0;
        pres[i] = 0.0; auterm[i] = 0.0; divfac[i] = 0.0;
        return;
    }
    const double h_sq_inv = 1.0 / (hi * hi);
    const double h_4_inv  = h_sq_inv * h_sq_inv;
    const double hfoh     = hfact / hi;
    const double rho      = pmass * hfoh * hfoh * hfoh;
    const double dhdrho   = -hi / (3.0 * rho);
    const double omega    = 1.0 - dhdrho * gradh[i];
    // as the gradient kernel: a non-positive omega (possible where h changes sharply) falls
    // back to 1 rather than dividing by it
    const double omega_inv = (omega > 0.0) ? (1 / omega) : 1.0;

    hsqinv[i] = h_sq_inv;
    hinv[i]   = 1.0 / hi;
    rhoh[i]   = rho;
    rho1[i]   = 1.0 / rho;
    grkfac[i] = h_4_inv * sph::cnormk * omega_inv;
    pres[i]   = pro2[i] * rho * rho;
    auterm[i] = 0.5 * pmass * (1.0 / rho) * alphau;
    divfac[i] = sph::cnormk * omega_inv * h_4_inv / rho;
}

// SPH force on one particle per thread, over the symmetric j-leaf list.  Threads index
// Hilbert-sorted particles.
// DiscVisc selects phantom's disc_viscosity form of the artificial viscosity (see below).
// FullHeating is the usual case, p dV work and shock heating both in du/dt; otherwise the
// runtime pdvHeating and shockHeating choose.  A template parameter so the usual case
// compiles exactly as it did before the switches existed.
template<bool Periodic, bool DiscVisc, bool FullHeating>
__global__ void sphForceKernelJList(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ vz,
    const double* __restrict__ h,
    const double* __restrict__ pro2,
    const double* __restrict__ spsound,
    const double* __restrict__ alphaAV,
    const double* __restrict__ u,
    const double* __restrict__ hsqinv,
    const double* __restrict__ hinv,
    const double* __restrict__ rhoh,
    const double* __restrict__ rho1,
    const double* __restrict__ grkfac,
    const double* __restrict__ pres,
    const double* __restrict__ auterm,
    const double* __restrict__ divfac,
    double* __restrict__ fx,
    double* __restrict__ fy,
    double* __restrict__ fz,
    double* __restrict__ f4,
    double* __restrict__ vsigmax,
    double* __restrict__ divv,
    int n,
    double pmass,
    double beta,
    bool pdvHeating,          // phantom's ipdv_heating
    bool shockHeating,        // phantom's ishock_heating
    const int* __restrict__ particleLeaf,
    const int* __restrict__ jOffset,
    const int* __restrict__ jcount,
    const int* __restrict__ jlist,
    const unsigned* __restrict__ layout,
    Box<double> box,
    double hfact)             // rho = pmass (hfact/h)^3, as the density solve used
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
    const double hi_sq_inv   = hsqinv[i];
    const double rhoi        = rhoh[i];
    const double hfacgrkerni = grkfac[i];   // hfacgrkern in force.F90

    const double pro2i   = pro2[i];
    const double vwavei  = spsound[i];
    const double alphai  = alphaAV[i];
    const double eni     = u[i];
    const double rho1i   = rho1[i];
    const double pri     = pres[i];
    const double autermi = auterm[i];

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

            // Cheap reject before any division or square root: ~95% of candidates are
            // outside both kernels.  The margin keeps it conservative, so the exact tests
            // below still decide every pair as before.
            const double hj = h[j];
            {
                const double rmax2 = sph::radk2 * fmax(hi * hi, hj * hj) * (1.0 + 1e-10);
                if (dr2 >= rmax2) continue;
            }

            double qij2 = dr2 * hi_sq_inv;
            double qj_ij2 = dr2 * hsqinv[j];

            // as force.F90: one division for the unit vector, and q = r/h from it
            const double dr   = sqrt(dr2);
            const double rij1 = 1.0 / dr;
            const double runix = dx * rij1;   // unit vector (r_i - r_j) / |r_i - r_j|
            const double runiy = dy * rij1;
            const double runiz = dz * rij1;
            const double dvxij = vx[i] - vx[j];
            const double dvyij = vy[i] - vy[j];
            const double dvzij = vz[i] - vz[j];
            const double projv = dvxij * runix + dvyij * runiy + dvzij * runiz;

            const double rhoj    = rhoh[j];
            const double rho1j   = rho1[j];
            const double pro2j   = pro2[j];
            const double vwavej  = spsound[j];
            const double alphaj  = alphaAV[j];
            const double enj     = u[j];

            const double prj     = pres[j];
            const double autermj = auterm[j];

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
            const double vsigu  = sqrt(fabs(pri - prj) * (2.0 / (rhoi + rhoj)));

            if constexpr (DiscVisc)
            {
                // phantom's disc_viscosity (force.F90), how a Shakura-Sunyaev alpha is
                // modelled with the artificial viscosity: scaled by h/r_ij, and applied to
                // receding pairs too, without the beta term
                const double hor_i = hi * rij1, hor_j = hj * rij1;
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

                double wij, grwij;
                const double qij = dr * hinv[i];
                sph::kern(qij2, qij, wij, grwij);

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
                    ? -0.5 * pmass * rho1i * alphai * vwavei * (hi * rij1) * projv * projv * gradkerni
                    : pmass * qrho2i * projv * gradkerni;
                const double dendisstermi = vsigu * denij * autermi * gradkerni;

                if constexpr (FullHeating)
                {
                    f4sum += pdvtermi;
                    f4sum += dudtdissi;
                }
                else
                {
                    if (pdvHeating)   f4sum += pdvtermi;
                    if (shockHeating) f4sum += dudtdissi;
                }
                f4sum += dendisstermi;
            }

            if (qj_ij2 < sph::radk2)
            {
                vsigmax_i = fmax(vsigmax_i, pair_vsigmax);

                double wj_ij, grwj_ij;
                const double qj_ij = dr * hinv[j];
                sph::kern(qj_ij2, qj_ij, wj_ij, grwj_ij);

                double gradkernj   = grwj_ij * grkfac[j];   // F_ij(h_j) / omega_j

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

    divv[i] = -divv_s * divfac[i];
}

void computeForces(GpuState& s, const ForceFields& f, double pmass, double beta,
                   double alphau, bool discViscosity, bool pdvHeating,
                   bool shockHeating, ForceTimings& ft)
{
    const int n = s.ngas;

    buildForceJLeafList(s, ft);

    // Phase boundaries on the device timeline, like buildForceJLeafList: recording
    // an event does not synchronise, so timing does not perturb what it measures.
    cudaEvent_t e0, e1, ep, e2, e3;
    for (auto* e : {&e0, &e1, &ep, &e2, &e3}) checkGpuErrors(hipEventCreate(e));
    HIP_CHECK(hipEventRecord(e0));

    // Every device buffer lives in GpuState, sized once by cosmo_arrays_init, so a force
    // pass allocates nothing on the device.
    const size_t nbytes = static_cast<size_t>(n) * sizeof(double);

    // phantom order -> Hilbert order (s.order maps sorted index -> phantom index)
    auto upload = [&](const double* host, thrust::device_vector<double>& sorted)
    {
        HIP_CHECK(hipMemcpy(rawPtr(s.fStage), host, nbytes, hipMemcpyHostToDevice));
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
    HIP_CHECK(hipEventRecord(e1));

    forcePrepKernel<<<iceil(n, 256), 256>>>(
        rawPtr(s.h), rawPtr(s.gradh), rawPtr(s.pro2),
        rawPtr(s.hsqinv), rawPtr(s.hinv), rawPtr(s.rhoh), rawPtr(s.rho1), rawPtr(s.grkfac),
        rawPtr(s.pres), rawPtr(s.auterm), rawPtr(s.divfac),
        n, pmass, alphau, s.hfact);
    checkGpuErrors(cudaGetLastError());
    HIP_CHECK(hipEventRecord(ep));

    // Periodic or not is whatever the density solve built this tree with.
    const bool fullHeating = pdvHeating && shockHeating;
    auto launch = [&](auto periodic, auto disc) {
        auto run = [&](auto heat) {
            sphForceKernelJList<decltype(periodic)::value, decltype(disc)::value,
                                decltype(heat)::value><<<iceil(n, 256), 256>>>(
                rawPtr(s.x), rawPtr(s.y), rawPtr(s.z),
                rawPtr(s.vx), rawPtr(s.vy), rawPtr(s.vz),
                rawPtr(s.h),
                rawPtr(s.pro2), rawPtr(s.spsound), rawPtr(s.alphaAV), rawPtr(s.u),
                rawPtr(s.hsqinv), rawPtr(s.hinv), rawPtr(s.rhoh), rawPtr(s.rho1), rawPtr(s.grkfac),
                rawPtr(s.pres), rawPtr(s.auterm), rawPtr(s.divfac),
                rawPtr(s.fx), rawPtr(s.fy), rawPtr(s.fz), rawPtr(s.f4),
                rawPtr(s.vsigmax), rawPtr(s.divvF),
                n, pmass, beta, pdvHeating, shockHeating,
                rawPtr(s.particleLeaf), rawPtr(s.jOffset), rawPtr(s.jcount), rawPtr(s.jlist),
                rawPtr(s.layout), s.box, s.hfact);
        };
        if (fullHeating) run(std::true_type{});
        else             run(std::false_type{});
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
    HIP_CHECK(hipEventElapsedTime(&ms, e1, ep)); ft.prep     = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, ep, e2)); ft.kernel   = ms * 1e-3;
    HIP_CHECK(hipEventElapsedTime(&ms, e2, e3)); ft.download = ms * 1e-3;
    for (auto* e : {&e0, &e1, &ep, &e2, &e3}) HIP_CHECK(hipEventDestroy(*e));
}
