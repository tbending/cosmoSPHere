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
