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
#                      default: ../cornerstone  (submodule checkout next to this repo)
#   HIP_ARCH         — AMD GPU architecture  (default: gfx942 = MI300X / MI300A)
#   HIPCC            — hipcc compiler        (default: hipcc, must be in PATH)

CORNERSTONE_DIR ?= ../octree-miniapp
HIPCC           ?= hipcc
HIP_ARCH        ?= gfx942

BUILDDIR := build

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
# include/ must come first: our cuda_runtime.h redirect intercepts
# Cornerstone's #include <cuda_runtime.h> before any system cuda path.
INCLUDES := -Iinclude -I$(CORNERSTONE_DIR)

HIP_FLAGS := -std=c++17 -O3 $(INCLUDES)  \
             --offload-arch=$(HIP_ARCH)   \
             -fopenmp                     \
             -DUSE_CUDA

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
	$(HIPCC) $(HIP_FLAGS) -MMD -MP -MF $(BUILDDIR)/dens_c_api.d -c -o $@ $<

# Scalar (base) binary
$(BUILDDIR)/density_hip: $(BUILDDIR)/main.base.o $(BUILDDIR)/density_base.o
	$(HIPCC) $(HIP_FLAGS) -o $@ $^

# 4x-unrolled binary
$(BUILDDIR)/density_hip_unrolled: $(BUILDDIR)/main.unrolled.o $(BUILDDIR)/density_unrolled.o
	$(HIPCC) $(HIP_FLAGS) -o $@ $^

# Compile rules: each .cu in src/ becomes a .o in build/
$(BUILDDIR)/density_base.o: src/density_base.cu
	$(HIPCC) $(HIP_FLAGS) -MMD -MP -MF $(BUILDDIR)/density_base.d -c -o $@ $<

$(BUILDDIR)/density_unrolled.o: src/density_unrolled.cu
	$(HIPCC) $(HIP_FLAGS) -MMD -MP -MF $(BUILDDIR)/density_unrolled.d -c -o $@ $<

# main.cu compiled twice — once for each binary — so the banner can show the
# correct kernel name.  For now both are identical; we use the same .cu file.
$(BUILDDIR)/main.base.o: src/main.cu
	$(HIPCC) $(HIP_FLAGS) -MMD -MP -MF $(BUILDDIR)/main.base.d -c -o $@ $<

$(BUILDDIR)/main.unrolled.o: src/main.cu
	$(HIPCC) $(HIP_FLAGS) -MMD -MP -MF $(BUILDDIR)/main.unrolled.d -c -o $@ $<

-include $(BUILDDIR)/*.d

clean:
	rm -f $(BUILDDIR)/*.o $(BUILDDIR)/*.d \
	      $(BUILDDIR)/density_hip $(BUILDDIR)/density_hip_unrolled \
	      $(BUILDDIR)/libcosmoSPHere.a

info:
	@$(HIPCC) --version
	@echo "HIP_ARCH        = $(HIP_ARCH)"
	@echo "CORNERSTONE_DIR = $(CORNERSTONE_DIR)"
