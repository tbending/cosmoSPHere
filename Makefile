# cosmoSPHere Makefile
#
# Targets:
#   all                — build both standalone test binaries (default)
#   build/density_hip          — scalar inner j-loop
#   build/density_hip_unrolled — 4x-unrolled inner j-loop
#   clean              — remove build products
#
# Configurable variables (override on command line or environment):
#   CORNERSTONE_DIR  — path to cornerstone-octree source tree
#                      default: ../octree-miniapp
#   GPU_BACKEND      — cuda (default) | hip
#   CUDA_ARCH        — space-separated SM list for the CUDA backend -> fat binary
#                      (default: 80 = A100/A30). e.g. "80 61" for A100 + P2000.
#   HIP_ARCH         — AMD GPU architecture for the hip backend (default: gfx942)
#   GPUCC            — compiler override (default: nvcc for cuda, hipcc for hip)

CORNERSTONE_DIR ?= ../octree-miniapp
GPU_BACKEND     ?= cuda          # cuda (default) | hip

BUILDDIR := build
# Ensure the build output directory exists (a fresh clone has no build/).
$(shell mkdir -p $(BUILDDIR))

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
INCLUDES := -Iinclude -I$(CORNERSTONE_DIR)

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

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all lib clean info

all: $(BUILDDIR)/density_hip $(BUILDDIR)/density_hip_unrolled

# ---------------------------------------------------------------------------
# Static library target — used when linking against Phantom (GPU=yes).
# Contains the density solver core + the Fortran-callable C API wrapper.
# Phantom links with: -L<cosmoSPHere>/build -lcosmoSPHere
# ---------------------------------------------------------------------------
lib: $(BUILDDIR)/libcosmoSPHere.a

$(BUILDDIR)/libcosmoSPHere.a: $(BUILDDIR)/density_base.o $(BUILDDIR)/dens_c_api.o
	ar rcs $@ $^

$(BUILDDIR)/dens_c_api.o: src/dens_c_api.cu
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/dens_c_api.d -c -o $@ $<

# Scalar (base) binary
$(BUILDDIR)/density_hip: $(BUILDDIR)/main.base.o $(BUILDDIR)/density_base.o
	$(GPUCC) $(GPU_FLAGS) -o $@ $^

# 4x-unrolled binary
$(BUILDDIR)/density_hip_unrolled: $(BUILDDIR)/main.unrolled.o $(BUILDDIR)/density_unrolled.o
	$(GPUCC) $(GPU_FLAGS) -o $@ $^

# Compile rules: each .cu in src/ becomes a .o in build/
$(BUILDDIR)/density_base.o: src/density_base.cu
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/density_base.d -c -o $@ $<

$(BUILDDIR)/density_unrolled.o: src/density_unrolled.cu
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/density_unrolled.d -c -o $@ $<

# main.cu compiled twice — once for each binary — so the banner can show the
# correct kernel name.  For now both are identical; we use the same .cu file.
$(BUILDDIR)/main.base.o: src/main.cu
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/main.base.d -c -o $@ $<

$(BUILDDIR)/main.unrolled.o: src/main.cu
	$(GPUCC) $(GPU_FLAGS) -MMD -MP -MF $(BUILDDIR)/main.unrolled.d -c -o $@ $<

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
