# cosmoSPHere Makefile
#
# Targets:
#   all (default)      — the standalone density solver, build/density_hip
#   lib                — the static library phantom links against
#   clean              — remove build products
#
# src/density_unrolled.cu is NOT built.  It is a second copy of the solver with a
# 4x-unrolled inner j-loop, an experiment that was not carried through; it still calls
# solveDensH with the signature it had before the host took over the transfers, so it
# no longer compiles.  That is deliberate.  Reviving it means giving it the current
# signature and fetching results with cosmo_download, as src/main.cu does.
#
# Configurable variables (override on command line or environment):
#   CORNERSTONE_DIR  — path to cornerstone-octree source tree
#                      default: ../octree-miniapp
#   GPU_BACKEND      — cuda (default) | hip
#   CUDA_ARCH        — space-separated SM list for the CUDA backend -> fat binary
#                      (default: 80 = A100/A30). e.g. "80 61" for A100 + P2000.
#   HIP_ARCH         — AMD GPU architecture for the hip backend (default: gfx942)
#   KERNEL           — cubic (default) | quintic
#   GPUCC            — compiler override (default: nvcc for cuda, hipcc for hip)

CORNERSTONE_DIR ?= ../octree-miniapp
GPU_BACKEND     ?= cuda
# name of the Phantom GPU_TARGET profile this was built for (informational only)
GPU_TARGET      ?=
BUILDDIR := build
# Ensure the build output directory exists (a fresh clone has no build/).
$(shell mkdir -p $(BUILDDIR))

# Force a rebuild when the GPU backend/arch changes. The .cu -> .o mapping is identical
# for the cuda and hip backends, so without this a cuda<->hip switch silently reuses the
# wrong-backend objects (and then fails to link, e.g. cuda .o linked with -lamdhip64).
# Stamp the active config into a file, rewriting it ONLY when it changes so its mtime
# bumps and the objects below (which list $(TAGFILE) as a prerequisite) recompile.
# Smoothing kernel, as phantom's KERNEL: cubic (default) or quintic.  Part of the build
# tag, so switching kernel recompiles everything.
KERNEL ?= cubic
ifeq ($(KERNEL),quintic)
    KERNEL_FLAGS := -DCOSMO_KERNEL_QUINTIC
else ifeq ($(KERNEL),cubic)
    KERNEL_FLAGS :=
else
    $(error KERNEL=$(KERNEL) is not implemented in cosmoSPHere -- use cubic or quintic)
endif

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
INCLUDES := -Iinclude -I$(CORNERSTONE_DIR) $(KERNEL_FLAGS)

ifeq ($(GPU_BACKEND),cuda)
    # Native CUDA build (nvcc). The sources use HIP names; hip_to_cuda.h is
    # force-included to map them onto the real CUDA runtime. Cornerstone's
    # <cuda_runtime.h> resolves to the genuine header (no shim on the path).
    GPUCC     ?= nvcc
    # CUDA_ARCH is a space-separated list of SM numbers -> one fat binary.
    CUDA_ARCH ?= 80
    GENCODE   := $(foreach a,$(CUDA_ARCH),-gencode arch=compute_$(a),code=sm_$(a))
    GPU_FLAGS := -std=c++17 -O3 $(INCLUDES) $(GENCODE)              \
                 -Xcompiler -fopenmp                               \
                 -DUSE_CUDA                                        \
                 -include $(CURDIR)/include/compat/hip_to_cuda.h
else ifeq ($(GPU_BACKEND),hip)
    # AMD HIP build (hipcc). compat/hipcc/ is placed first so the forward shim
    # cuda_runtime.h intercepts Cornerstone's #include <cuda_runtime.h> and
    # redirects CUDA names onto HIP before any system cuda path.
    GPUCC    ?= hipcc
    HIP_ARCH ?= gfx942
    GPU_FLAGS := -std=c++17 -O3 -Iinclude/compat/hipcc $(INCLUDES) \
                 --offload-arch=$(HIP_ARCH)                        \
                 -fopenmp                                          \
                 -DUSE_CUDA
else
    $(error Unknown GPU_BACKEND=$(GPU_BACKEND) -- use 'cuda' or 'hip')
endif

# The build tag, after the backend branches have set their architecture defaults, so a
# change to a default architecture also invalidates the objects.
BUILD_TAG := $(GPU_BACKEND) $(CUDA_ARCH) $(HIP_ARCH) $(KERNEL)
TAGFILE   := $(BUILDDIR)/.build_tag
$(shell [ "$$(cat $(TAGFILE) 2>/dev/null)" = "$(BUILD_TAG)" ] || printf '%s' "$(BUILD_TAG)" > $(TAGFILE))

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all lib clean info

all: $(BUILDDIR)/density_hip

# ---------------------------------------------------------------------------
# Static library target — used when linking against Phantom (GPU=yes).
# Contains the density solver core + the Fortran-callable C API wrapper.
# Phantom links with: -L<cosmoSPHere>/build -lcosmoSPHere
# ---------------------------------------------------------------------------
lib: announce $(BUILDDIR)/libcosmoSPHere.a

.PHONY: announce
announce:
	@echo ""
	@echo "Compiling cosmoSPHere for $(GPU_TARGET) system..........."
	@echo ""
	@echo "Using $(GPU_BACKEND) backend"
	@echo "Using $(KERNEL) kernel"
ifeq ($(GPU_BACKEND),cuda)
	@echo "CUDA_ARCH is $(CUDA_ARCH)"
else
	@echo "HIP_ARCH is $(HIP_ARCH)"
endif
	@echo "Using the Cornerstone octree from octree-miniapp (header-only: compiled as part of cosmoSPHere)"
	@echo ""

$(BUILDDIR)/libcosmoSPHere.a: $(BUILDDIR)/density_base.o $(BUILDDIR)/tree.o $(BUILDDIR)/force.o $(BUILDDIR)/gpu_state.o $(BUILDDIR)/arrays_c_api.o $(BUILDDIR)/dens_c_api.o $(BUILDDIR)/force_c_api.o $(BUILDDIR)/pin_c_api.o
	ar rcs $@ $^

$(BUILDDIR)/pin_c_api.o: src/pin_c_api.cu $(TAGFILE)
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/pin_c_api.d -c -o $@ $<

$(BUILDDIR)/dens_c_api.o: src/dens_c_api.cu $(TAGFILE)
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/dens_c_api.d -c -o $@ $<

$(BUILDDIR)/arrays_c_api.o: src/arrays_c_api.cu $(TAGFILE)
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/arrays_c_api.d -c -o $@ $<

# Standalone density solver: the same objects phantom links, plus a driver.
$(BUILDDIR)/density_hip: $(BUILDDIR)/main.o $(BUILDDIR)/density_base.o $(BUILDDIR)/tree.o \
                         $(BUILDDIR)/gpu_state.o $(BUILDDIR)/arrays_c_api.o
	$(GPUCC) $(GPU_FLAGS) -o $@ $^

$(BUILDDIR)/main.o: src/main.cu $(TAGFILE)
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/main.d -c -o $@ $<

# Compile rules: each .cu in src/ becomes a .o in build/
$(BUILDDIR)/tree.o: src/tree.cu $(TAGFILE)
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/tree.d -c -o $@ $<

$(BUILDDIR)/force.o: src/force.cu $(TAGFILE)
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/force.d -c -o $@ $<

$(BUILDDIR)/gpu_state.o: src/gpu_state.cu $(TAGFILE)
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/gpu_state.d -c -o $@ $<

$(BUILDDIR)/force_c_api.o: src/force_c_api.cu $(TAGFILE)
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/force_c_api.d -c -o $@ $<

$(BUILDDIR)/density_base.o: src/density_base.cu $(TAGFILE)
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/density_base.d -c -o $@ $<

-include $(BUILDDIR)/*.d

clean:
	rm -f $(BUILDDIR)/*.o $(BUILDDIR)/*.d \
	      $(BUILDDIR)/density_hip $(BUILDDIR)/density_hip_unrolled \
	      $(BUILDDIR)/libcosmoSPHere.a

info:
	@$(GPUCC) --version
	@echo "GPU_BACKEND     = $(GPU_BACKEND)"
	@echo "CUDA_ARCH       = $(CUDA_ARCH)"
	@echo "HIP_ARCH        = $(HIP_ARCH)"
	@echo "CORNERSTONE_DIR = $(CORNERSTONE_DIR)"
