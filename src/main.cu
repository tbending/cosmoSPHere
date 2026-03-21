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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "io.hpp"
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

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "Usage: %s <datafile> [output_h_file]\n", argv[0]);
        return 1;
    }

    const std::string dataFile  = argv[1];
    const std::string outFile   = (argc >= 3) ? argv[2] : "h_cpp.txt";
    const std::string jiggleArg = (argc >= 4) ? argv[3] : "";
    const bool        doOutput  = (outFile != "no_output");
    const bool        doJiggle  = (jiggleArg == "jiggle");

    std::printf("density-cpp (Cornerstone GPU density solver)\n");
    std::printf("Data file : %s\n", dataFile.c_str());

    // ------------------------------------------------------------------
    // Read input data
    // ------------------------------------------------------------------
    double t0 = wallTime();
    io::ParticleData pd;
    try {
        pd = io::readCosmoFile(dataFile);
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Error reading file: %s\n", e.what());
        return 1;
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
            solveDensH(h_wu, rho_wu, gradh_wu,
                       pd.x, pd.y, pd.z,
                       pd.pmass,
                       KernelMode::FLAT_PARTICLE);
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
        timing = solveDensH(pd.h, rho, gradh,
                             pd.x, pd.y, pd.z,
                             pd.pmass,
                             KernelMode::FLAT_PARTICLE);
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
        timing2 = solveDensH(pd.h, rho, gradh,
                              pd.x, pd.y, pd.z,
                              pd.pmass,
                              KernelMode::WARP_PER_LEAF);
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
