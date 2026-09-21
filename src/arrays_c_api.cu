/*
 * arrays_c_api.cu — the array interface: sizing the device side.
 *
 * The host owns the data and decides the footprint; the device is asked to compute.  So
 * the particle-length device arrays are sized here, once per particle count, from the
 * host's gpu_arrays_init, rather than by whichever kernel first happened to touch one.
 *
 * The tree-shaped arrays are not covered — their lengths follow nLeaves and numNodes,
 * which do not exist until the tree has been built.  See sizeParticleArrays.
 */

#include "gpu_check.hpp"
#include "gpu_state.hpp"

//! @brief Size the device's particle-length arrays for n particles.  Idempotent.
extern "C" void cosmo_arrays_init(int n)
{
    if (n <= 0) return;
    sizeParticleArrays(gpuState(), n);
}
