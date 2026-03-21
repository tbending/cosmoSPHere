# cosmoSPHere

GPU-accelerated SPH density and smoothing-length solver using the
[Cornerstone](https://github.com/exafmm/cornerstone-octree) octree library.

Designed to plug into [Phantom](https://github.com/danieljprice/phantom) as a
drop-in replacement for its CPU tree, but also usable standalone for benchmarking.

## Requirements

- AMD GPU with ROCm / `hipcc` (tested on MI300X, architecture `gfx942`)
- Cornerstone octree source tree (see below)

## Getting the code

cosmoSPHere is intended to be used as a submodule inside a parent project that
also provides Cornerstone:

```bash
git clone --recurse-submodules https://github.com/tbending/phantom
```

For standalone use, clone cosmoSPHere and obtain Cornerstone separately:

```bash
git clone https://github.com/tbending/cosmoSPHere
git clone https://github.com/exafmm/cornerstone-octree octree-miniapp
```

## Building

```bash
cd cosmoSPHere
make CORNERSTONE_DIR=../cornerstone
```

This produces two binaries in `build/`:

| Binary | Description |
|---|---|
| `build/density_hip` | Scalar inner j-loop |
| `build/density_hip_unrolled` | 4× manually unrolled inner j-loop |

### Makefile variables

| Variable | Default | Description |
|---|---|---|
| `CORNERSTONE_DIR` | `../octree-miniapp` | Path to the octree-miniapp directory |
| `HIP_ARCH` | `gfx942` | AMD GPU target architecture |
| `HIPCC` | `hipcc` | HIP compiler |

## Running

```bash
build/density_hip_unrolled  <datafile.cosmo>  [no_output]  [jiggle]
```

- `no_output` — suppress writing `h_cpp.txt`
- `jiggle` — perturb 1% of smoothing lengths by up to 35% before solving
  (stress-tests the Newton–Raphson iteration)

### Input file format

Fortran unformatted binary containing:

```fortran
write(unit) ngas          ! integer
write(unit) pmass         ! real(8)
write(unit) xyzh(4,ngas)  ! real(8) — x,y,z,h per particle
```

## Repository structure

```
cosmoSPHere/
├── Makefile
├── LICENSE
├── README.md
├── include/
│   ├── cuda_runtime.h    — HIP/CUDA compatibility shim
│   ├── density.hpp       — DensTimings struct, KernelMode enum
│   ├── io.hpp            — input file reader
│   └── kernel.hpp        — M4 cubic spline kernel (W, dW/dq)
├── src/
│   ├── main.cu           — standalone test driver
│   ├── density_base.cu   — GPU solver, scalar inner j-loop
│   └── density_unrolled.cu — GPU solver, 4× unrolled inner j-loop
└── build/                — created by make, gitignored
```

## Algorithm

1. Compute 64-bit Hilbert keys for all particles on the GPU
2. Sort particles into Hilbert order (Thrust radix sort)
3. Build adaptive Cornerstone leaf tree (GPU, iterative)
4. Build fully-linked internal octree
5. Compute floating-point node centres
6. Newton–Raphson loop (up to 10 iterations):
   - DFS to build a j-leaf list for each active i-leaf
   - Accumulate density `ρ` and gradient `∂ρ/∂h` over neighbours
   - Update `h` via NR; compact unconverged particles; repeat

Convergence criterion: `|Δh/h| < 1e-4`

## Linking with Phantom

See `phantom/src/main/cosmoSPHere_utils.f90` in the
[tbending/phantom](https://github.com/tbending/phantom) fork.
Build phantom with `COSMOSPHERE=yes`.

## Licence

GNU General Public License v3 — see [LICENSE](LICENSE).
