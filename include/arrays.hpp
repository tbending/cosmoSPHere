/*
 * arrays.hpp — the slots the host moves data in and out by.
 *
 * These ids are shared with phantom's gpu_arrays (its ibun_* parameters) and the two
 * lists must stay in step: the host names a slot, this side turns that into the device
 * arrays it stands for.  A slot's components are numbered in the same order on both
 * sides, and arrive as one contiguous host block of ncomp*n doubles, because the host
 * arena is component-major within a bundle.
 */

#pragma once

#include <cstddef>

enum CosmoSlot
{
    COSMO_POS       = 1,   // x, y, z
    COSMO_HSML      = 2,   // h
    COSMO_VEL       = 3,   // vx, vy, vz
    COSMO_ACCEL     = 4,   // ax, ay, az
    COSMO_DENS_OUT  = 5,   // rho, d(rho)/d(h)
    COSMO_GRAD_OUT  = 6,   // div v, xi, d(div v)/dt
    COSMO_THERMO    = 7,   // p/rho^2, c_s, alpha_AV, u
    COSMO_FORCE_OUT = 8    // fx, fy, fz, du/dt, vsigmax, div v
};

// ---------------------------------------------------------------------------
// The transfer interface.  The caller owns the data and decides what moves; these are
// the only ways particle data crosses to the device or back.
//
// `host` is one contiguous block of ncomp * n doubles: the slot's components end to end,
// each n long, in the order given above.
// ---------------------------------------------------------------------------
extern "C" {

//! @brief Size the device's particle-length arrays for n particles.  Idempotent.
//! Must be called before any transfer, and again if n changes.
void cosmo_arrays_init(int n);

//! @brief How many components this side gives a slot; 0 if it does not know it.
int cosmo_slot_ncomp(int slot);

//! @brief Send a slot in the caller's particle order.  For use before the tree for this
//! particle set exists: the solve sorts it as part of building the tree.
void cosmo_upload(int slot, const double* host, int n);

//! @brief Send a slot and put it in the device's own particle order.  For use once a
//! solve has built the tree; aborts if none has.
void cosmo_upload_sorted(int slot, const double* host, int n);

//! @brief Fetch a slot, back in the caller's particle order.
void cosmo_download(int slot, double* host, int n);

//! @brief Page-lock a host buffer so transfers are not slowed by on-demand paging.
void cosmo_pin_host(void* ptr, std::size_t nbytes);
void cosmo_unpin_host(void* ptr);

//! @brief The support radius, in units of h, of the kernel this was built with.
double cosmo_kernel_radius(void);

}
