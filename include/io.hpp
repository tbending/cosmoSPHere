/*
 * io.hpp — Read the Fortran unformatted sequential binary written by cosmoSPHere.
 *
 * Fortran unformatted sequential format wraps every WRITE statement with a 4-byte
 * record-length marker before and after the data payload:
 *
 *   [int32 len] [data bytes] [int32 len]
 *
 * The three records written by cosmoSPHere are:
 *   1.  ngas          (int32,  4 bytes)
 *   2.  pmass         (float64, 8 bytes)
 *   3.  xyzh(:,1:n)  (float64, 4*ngas*8 bytes, Fortran column-major)
 *
 * Column-major xyzh(4,ngas) lays out in memory as:
 *   x1 y1 z1 h1  x2 y2 z2 h2  ...  xN yN zN hN
 * which is the AoS interleaved order; we de-interleave into four separate arrays.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace io
{

// Read a single Fortran record: check markers match, fill buf.
static void readFortranRecord(std::FILE* fp, void* buf, size_t expectedBytes)
{
    int32_t before = 0, after = 0;

    if (std::fread(&before, sizeof(int32_t), 1, fp) != 1)
        throw std::runtime_error("io: failed reading record start marker");

    if (static_cast<size_t>(before) != expectedBytes)
        throw std::runtime_error("io: record start marker mismatch: got " +
                                 std::to_string(before) + " expected " +
                                 std::to_string(expectedBytes));

    if (std::fread(buf, 1, expectedBytes, fp) != expectedBytes)
        throw std::runtime_error("io: failed reading record payload");

    if (std::fread(&after, sizeof(int32_t), 1, fp) != 1)
        throw std::runtime_error("io: failed reading record end marker");

    if (before != after)
        throw std::runtime_error("io: record end marker mismatch");
}

// Result struct so caller gets everything in one call.
struct ParticleData
{
    int    ngas;
    double pmass;
    std::vector<double> x, y, z, h;
};

ParticleData readCosmoFile(const std::string& filename)
{
    std::FILE* fp = std::fopen(filename.c_str(), "rb");
    if (!fp)
        throw std::runtime_error("io: cannot open '" + filename + "'");

    ParticleData pd;

    // Record 1: ngas (Fortran default integer = int32)
    {
        int32_t ngas_raw = 0;
        readFortranRecord(fp, &ngas_raw, sizeof(int32_t));
        pd.ngas = static_cast<int>(ngas_raw);
    }

    // Record 2: pmass
    readFortranRecord(fp, &pd.pmass, sizeof(double));

    // Record 3: xyzh(4, ngas) — column-major, de-interleave on the fly.
    const size_t ngas = static_cast<size_t>(pd.ngas);
    const size_t recBytes = 4 * ngas * sizeof(double);

    // Check the start marker before allocating.
    int32_t before = 0;
    if (std::fread(&before, sizeof(int32_t), 1, fp) != 1)
        throw std::runtime_error("io: failed reading xyzh record start marker");
    if (static_cast<size_t>(before) != recBytes)
        throw std::runtime_error("io: xyzh record marker mismatch");

    pd.x.resize(ngas);
    pd.y.resize(ngas);
    pd.z.resize(ngas);
    pd.h.resize(ngas);

    // Read in chunks to avoid a 320 MB temporary buffer.
    // Each Fortran element is a column xyzh(:,i) = {x,y,z,h} contiguous in file.
    constexpr size_t CHUNK = 65536;
    std::vector<double> buf(4 * CHUNK);

    size_t done = 0;
    while (done < ngas)
    {
        const size_t batch = std::min(CHUNK, ngas - done);
        const size_t toRead = 4 * batch * sizeof(double);
        if (std::fread(buf.data(), 1, toRead, fp) != toRead)
            throw std::runtime_error("io: failed reading xyzh payload");
        for (size_t k = 0; k < batch; ++k)
        {
            pd.x[done + k] = buf[4 * k + 0];
            pd.y[done + k] = buf[4 * k + 1];
            pd.z[done + k] = buf[4 * k + 2];
            pd.h[done + k] = buf[4 * k + 3];
        }
        done += batch;
    }

    // Read and check end marker.
    int32_t after = 0;
    if (std::fread(&after, sizeof(int32_t), 1, fp) != 1)
        throw std::runtime_error("io: failed reading xyzh record end marker");
    if (before != after)
        throw std::runtime_error("io: xyzh record end marker mismatch");

    std::fclose(fp);
    return pd;
}

// Write smoothing lengths, one per line — matches cosmoSPHere's h.txt format.
void writeH(const std::string& filename, const std::vector<double>& h)
{
    std::FILE* fp = std::fopen(filename.c_str(), "w");
    if (!fp)
        throw std::runtime_error("io: cannot open '" + filename + "' for writing");
    for (double hi : h)
        std::fprintf(fp, "%.15e\n", hi);
    std::fclose(fp);
}

} // namespace io
