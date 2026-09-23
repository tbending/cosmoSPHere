# cosmoSPHere

GPU-accelerated SPH density and smoothing-length solver using the
[Cornerstone](https://github.com/exafmm/cornerstone-octree) octree library.

Designed to plug into [Phantom](https://github.com/danieljprice/phantom) as a
drop-in replacement for its CPU tree, but also usable standalone for benchmarking.

## Requirements

- An NVIDIA GPU with CUDA / `nvcc` (the default backend; tested on A100 and GH200), or an
  AMD GPU with ROCm / `hipcc` (`GPU_BACKEND=hip`; tested on MI300A, architecture `gfx942`)
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

Nothing else is needed: the standalone driver generates its own particles.

## Building

```bash
cd cosmoSPHere
make CORNERSTONE_DIR=../cornerstone                    # CUDA
make CORNERSTONE_DIR=../cornerstone GPU_BACKEND=hip    # AMD
```

Phantom builds the library itself (`make GPU=yes GPU_TARGET=...`), passing the backend,
architecture and kernel.

`make` produces `build/density_hip`, the standalone solver; `make lib` produces
`build/libcosmoSPHere.a`, which is what Phantom links.

```bash
./build/density_hip lattice 50        # 173,850 particles on a close-packed lattice
./build/density_hip lattice 100       # 1,403,000
./build/density_hip <datafile>        # xyzh from a file, see include/io.hpp
```

It reports the solve broken down by phase — upload, Hilbert sort, tree build, j-leaf
lists, density kernel, download — and runs both the flat-particle and warp-per-leaf
kernels so their results and timings can be compared. No Phantom, no input file.

### Makefile variables

| Variable | Default | Description |
|---|---|---|
| `CORNERSTONE_DIR` | `../octree-miniapp` | Path to the octree-miniapp directory |
| `GPU_BACKEND` | `cuda` | `cuda` or `hip` |
| `CUDA_ARCH` | `80` | CUDA compute capabilities, space-separated |
| `HIP_ARCH` | `gfx942` | AMD GPU target architecture |
| `GPUCC` | `nvcc` / `hipcc` | Compiler override |
| `KERNEL` | `cubic` | `cubic` or `quintic` |

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
│   ├── main.cu           — standalone driver (build/density_hip)
│   ├── arrays_c_api.cu   — sizing and the host/device transfers
│   ├── density_base.cu   — GPU solver
│   └── density_unrolled.cu — NOT BUILT: a 4× unrolled variant, kept as a record
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
