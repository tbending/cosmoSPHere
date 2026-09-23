/*
 * main.cu — standalone driver for the density solver.
 *
 * Builds build/density_hip.  Runs the solver without any of phantom around it, which is
 * the quickest way to try the code or to time a kernel change on its own.
 *
 *   density_hip lattice <npartx>   a close-packed lattice in a unit box, generated here
 *   density_hip <datafile>         xyzh from a file (see io.hpp for the format)
 *
 * Add "jiggle" to perturb h on ~1% of particles first.  A converged dump otherwise
 * solves in one Newton iteration, which is not what a timestep looks like.
 *
 * The solver keeps its data on the device and the caller moves it, so a solve here is
 * the same four steps phantom takes: size the device arrays, send the inputs, compute,
 * fetch the results.  solve() below is that sequence; everything else is setup and
 * reporting.
 */

/*
 * main.cu — Driver for the Cornerstone GPU-based SPH density solver.
 *
 * Usage:
 *   ./density_gpu <datafile> [output_h_file]
 *
 *   datafile      — Fortran unformatted binary written by cosmoSPHere
 *   output_h_file — optional; if given writes one h per line (default: h_cpp.txt)
 *                   pass 'no_output' to suppress
 *
 * Outputs timing lines in the same format as cosmoSPHere so you can
 * paste them side-by-side for comparison.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "io.hpp"
#include "arrays.hpp"
#include "density.hpp"

// Mirror the Fortran jiggle logic exactly:
// ~1% of particles have h perturbed by up to ±50%, strongly peaked near 0.
static void applyJiggle(std::vector<double>& h, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    int nJiggled = 0;
    for (auto& hi : h)
    {
        if (dist(rng) < 0.99) continue;
        double rval  = dist(rng);
        double delta = std::copysign(std::pow(rval, 30.0), rval - 0.5);
        hi *= (1.0 + 0.5 * delta);
        ++nJiggled;
    }
    std::printf(" Jiggled %d particles (%.2f%%)\n", nJiggled,
                100.0 * nJiggled / static_cast<double>(h.size()));
}

static double wallTime()
{
    using clock = std::chrono::steady_clock;
    using dur   = std::chrono::duration<double>;
    return dur(clock::now().time_since_epoch()).count();
}



// ---------------------------------------------------------------------------
// A close-packed lattice in the unit box, the same arrangement phantom's sedov setup
// uses, at a comparable resolution: npartx = 50 gives 173,850 particles and npartx = 100
// gives 1,403,000.  Those are close to but NOT the same as phantom's 174,000 and
// 1,368,000 -- it fits the lattice to a periodic box and this does not -- so this is for
// trying the solver and timing kernel changes, not for reproducing a published number.
//
// h starts at hfact times the particle spacing, which is close enough that the Newton
// iteration converges in a few passes, as it does from a phantom dump.
// ---------------------------------------------------------------------------
static io::ParticleData makeLattice(int npartx)
{
    if (npartx < 4) throw std::runtime_error("npartx must be at least 4");
    const double dx = 1.0 / npartx;
    const double dy = dx * std::sqrt(3.0) / 2.0;
    const double dz = dx * std::sqrt(6.0) / 3.0;
    const int ny = (int)(1.0 / dy);
    const int nz = (int)(1.0 / dz);

    io::ParticleData pd;
    pd.x.reserve((size_t)npartx * ny * nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < npartx; ++i)
            {
                const double xo = ((j + k) % 2) * 0.5 * dx;
                const double yo = (k % 2) * dy / 3.0;
                pd.x.push_back(i * dx + xo);
                pd.y.push_back(j * dy + yo);
                pd.z.push_back(k * dz);
            }
    pd.ngas  = (int)pd.x.size();
    pd.pmass = 1.0 / pd.ngas;                 // unit total mass in a unit box
    pd.h.assign(pd.ngas, 1.2 * dx);           // hfact_default * spacing
    return pd;
}

// ---------------------------------------------------------------------------
// One solve, with the transfers the caller is responsible for.
//
// A slot arrives as ncomp contiguous runs of n, so positions go up as one x|y|z block
// and rho/gradh come back as one rho|gradh block; the packing and unpacking here is
// only to meet that layout from separate std::vectors.
// ---------------------------------------------------------------------------
static DensTimings solve(std::vector<double>& x, std::vector<double>& y,
                         std::vector<double>& z, std::vector<double>& h,
                         std::vector<double>& rho, std::vector<double>& gradh,
                         double pmass, KernelMode mode)
{
    const int n = (int)x.size();
    cosmo_arrays_init(n);

    std::vector<double> pos(3 * (size_t)n);
    std::copy(x.begin(), x.end(), pos.begin());
    std::copy(y.begin(), y.end(), pos.begin() + n);
    std::copy(z.begin(), z.end(), pos.begin() + 2 * (size_t)n);
    cosmo_upload(COSMO_POS,  pos.data(), n);
    cosmo_upload(COSMO_HSML, h.data(),   n);

    DensTimings t = solveDensH(n, pmass, mode);

    cosmo_download(COSMO_HSML, h.data(), n);
    std::vector<double> densout(2 * (size_t)n);
    cosmo_download(COSMO_DENS_OUT, densout.data(), n);
    std::copy(densout.begin(), densout.begin() + n, rho.begin());
    std::copy(densout.begin() + n, densout.end(), gradh.begin());
    return t;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "Usage: %s lattice <npartx> [output_h_file] [jiggle]\n", argv[0]);
        std::fprintf(stderr, "       %s <datafile> [output_h_file] [jiggle]\n", argv[0]);
        std::fprintf(stderr, "\n  lattice 50 gives 173,850 particles, 100 gives 1,403,000.\n");
        std::fprintf(stderr, "  output_h_file may be 'no_output'.\n");
        std::fprintf(stderr, "\n  jiggle perturbs h on ~1%% of particles before solving, as the\n"
                             "  Fortran version did.  Without it a converged dump needs only one\n"
                             "  Newton iteration, so the solve is not representative of a timestep,\n"
                             "  where h has moved and 4-5 iterations are usual.\n");
        return 1;
    }

    // "jiggle" is a flag and may appear anywhere; everything else is positional.
    std::vector<std::string> pos;
    bool doJiggle = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "jiggle") doJiggle = true;
        else                                  pos.push_back(argv[i]);
    }
    const std::string dataFile  = pos.empty() ? "" : pos[0];
    const bool        doLattice = (dataFile == "lattice");
    const int         npartx    = doLattice ? (pos.size() >= 2 ? std::atoi(pos[1].c_str()) : 50) : 0;
    const size_t      iout      = doLattice ? 2 : 1;
    const std::string outFile   = (pos.size() > iout) ? pos[iout] : "h_cpp.txt";
    const bool        doOutput  = (outFile != "no_output");

    std::printf("cosmoSPHere standalone density solver\n");

    double t0 = wallTime();
    io::ParticleData pd;
    if (doLattice)
    {
        std::printf("Input     : close-packed lattice, npartx = %d\n", npartx);
        pd = makeLattice(npartx);
    }
    else
    {
        std::printf("Data file : %s\n", dataFile.c_str());
        try {
            pd = io::readCosmoFile(dataFile);
        }
        catch (const std::exception& e) {
            std::fprintf(stderr, "Error reading file: %s\n", e.what());
            return 1;
        }
    }
    double t1 = wallTime();

    std::printf("Finished read, ngas = %d\n", pd.ngas);
    std::printf(" pmass = %.6e\n", pd.pmass);
    std::printf(" maxh  = %.6e   minh = %.6e\n",
                *std::max_element(pd.h.begin(), pd.h.end()),
                *std::min_element(pd.h.begin(), pd.h.end()));
    std::printf(" Time reading file: %.4f s\n", t1 - t0);

    if (doJiggle)
    {
        std::printf("\nApplying jiggle to initial h...\n");
        applyJiggle(pd.h);
        std::printf(" maxh  = %.6e   minh = %.6e  (after jiggle)\n\n",
                    *std::max_element(pd.h.begin(), pd.h.end()),
                    *std::min_element(pd.h.begin(), pd.h.end()));
    }
    else
    {
        std::printf("\n");
    }

    // ------------------------------------------------------------------
    // Allocate output arrays
    // ------------------------------------------------------------------
    const int ngas = pd.ngas;
    std::vector<double> rho(ngas, 0.0), gradh(ngas, 0.0);

    // Save initial h so we can restore it for the second call.
    std::vector<double> h_init = pd.h;

    // Helper lambda to print one timing block.
    auto printTimings = [&](const char* label, const DensTimings& tm, double wall)
    {
        const char* modeName = (tm.kernelMode == KernelMode::WARP_PER_LEAF)
                               ? "warp-per-leaf" : "flat-particle";
        std::printf("%s  [kernel: %s]\n", label, modeName);
        std::printf(" Time upload (host -> device)  :  %.4f s\n", tm.upload);
        std::printf(" Time bounding box (GPU)       :  %.4f s\n", tm.bboxAndSetup);
        std::printf(" Time Hilbert keys + GPU sort  :  %.4f s\n", tm.keysAndSort);
        std::printf(" Time tree build (cs + linked) :  %.4f s\n", tm.treeBuild);
        std::printf(" Time node centres             :  %.4f s\n", tm.nodeCenters);
        std::printf(" Time j-leaf list build        :  %.4f s  (%d iter)\n", tm.jleafBuild, tm.itersRun);
        std::printf(" Time density kernel           :  %.4f s  (%d iter)\n", tm.densKernel,  tm.itersRun);
        std::printf(" Time download (device -> host):  %.4f s\n", tm.download);
        std::printf(" Total time building+solving   :  %.4f s\n", wall);
    };

    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // Warm-up call — runs the full pipeline once to:
    //   • bring HIP runtime and driver to steady state
    //   • warm GPU HBM pages (avoids page-fault penalties in call 1)
    //   • avoid JIT overhead on first kernel launch
    // Results are discarded; h is restored from h_init afterwards.
    // ------------------------------------------------------------------
    std::printf("Warm-up call (results discarded)...\n");
    {
        std::vector<double> h_wu = h_init;
        std::vector<double> rho_wu(ngas, 0.0), gradh_wu(ngas, 0.0);
        try {
            solve(pd.x, pd.y, pd.z, h_wu, rho_wu, gradh_wu,
                  pd.pmass, KernelMode::FLAT_PARTICLE);
        }
        catch (const std::exception& e) {
            std::fprintf(stderr, "Warm-up error: %s\n", e.what());
            return 1;
        }
    }
    std::printf("Warm-up complete.\n\n");

    // ------------------------------------------------------------------
    // First call — flat-particle kernel (Fortran-style active list).
    // ------------------------------------------------------------------
    std::printf("Building tree and solving density (ngas = %d)...\n", ngas);
    double tSolveStart = wallTime();
    DensTimings timing;
    try {
        timing = solve(pd.x, pd.y, pd.z, pd.h, rho, gradh,
                       pd.pmass, KernelMode::FLAT_PARTICLE);
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Solver error: %s\n", e.what());
        return 1;
    }
    double tSolveEnd = wallTime();

    std::printf("\n");
    printTimings("--- Call 1 (flat-particle) ---", timing, tSolveEnd - tSolveStart);
    {
        double hmin = pd.h[0], hmax = pd.h[0];
        double rmin = rho[0],  rmax = rho[0];
        for (int i = 1; i < ngas; ++i) {
            hmin = std::min(hmin, pd.h[i]); hmax = std::max(hmax, pd.h[i]);
            rmin = std::min(rmin, rho[i]);  rmax = std::max(rmax, rho[i]);
        }
        std::printf(" Output h   range: [%.6e, %.6e]\n", hmin, hmax);
        std::printf(" Output rho range: [%.6e, %.6e]\n\n", rmin, rmax);
    }

    // ------------------------------------------------------------------
    // Second call — warp-per-leaf kernel (shared-memory j-list).
    // Restore initial h so iteration count matches call 1.
    // ------------------------------------------------------------------
    pd.h = h_init;
    std::fill(rho.begin(), rho.end(), 0.0);
    std::fill(gradh.begin(), gradh.end(), 0.0);

    tSolveStart = wallTime();
    DensTimings timing2;
    try {
        timing2 = solve(pd.x, pd.y, pd.z, pd.h, rho, gradh,
                        pd.pmass, KernelMode::WARP_PER_LEAF);
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Solver error (call 2): %s\n", e.what());
        return 1;
    }
    tSolveEnd = wallTime();

    printTimings("--- Call 2 (warp-per-leaf) ---", timing2, tSolveEnd - tSolveStart);
    {
        double hmin = pd.h[0], hmax = pd.h[0];
        double rmin = rho[0],  rmax = rho[0];
        for (int i = 1; i < ngas; ++i) {
            hmin = std::min(hmin, pd.h[i]); hmax = std::max(hmax, pd.h[i]);
            rmin = std::min(rmin, rho[i]);  rmax = std::max(rmax, rho[i]);
        }
        std::printf(" Output h   range: [%.6e, %.6e]\n", hmin, hmax);
        std::printf(" Output rho range: [%.6e, %.6e]\n\n", rmin, rmax);
    }

    // ------------------------------------------------------------------
    // Write h output (uses result from second call)
    // ------------------------------------------------------------------
    if (doOutput)
    {
        try {
            io::writeH(outFile, pd.h);
            std::printf(" h written to: %s\n", outFile.c_str());
        }
        catch (const std::exception& e) {
            std::fprintf(stderr, "Warning: could not write output: %s\n", e.what());
        }
    }

    return 0;
}
