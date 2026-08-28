/*
 * force.hpp — GPU force pass.
 *
 * Phantom runs density and force as two passes (deriv.F90 :139 and :195), so this is
 * a separate entry point rather than a tail on the density solve.  It reuses the tree
 * and the sorted particles left in GpuState by solveDensH; rebuilding them here would
 * cost ~10.8 ms of a 33.7 ms solve, every step, for nothing.
 */

#pragma once

#include "gpu_check.hpp"
#include "gpu_state.hpp"

//! @brief Per-phase cost of one force pass, in seconds.
struct ForceTimings
{
    double hmaxUpsweep = 0.0;
    double jleafBuild  = 0.0;
};

/*! @brief Rebuild the j-leaf lists with the SYMMETRIC (gather + scatter) criterion.
 *
 * Density asks only "is j inside MY kernel?" — radius 2*hmax_i.  The force sum is
 * symmetric (force.F90:1287 takes a pair when q2i < radkern2 .OR. q2j < radkern2), so
 * it also needs the SCATTER neighbours: j with r_ij < 2*h_j but r_ij >= 2*h_i.  Those
 * contribute to the force on i but never to its density, and without them the pair
 * force is not antisymmetric, so momentum is not conserved.
 *
 * Requires s.hmax_leaf valid for EVERY leaf at the converged h — the full-tree pass
 * at the top of solveDensH's gradient sweep leaves it that way.  Propagates it to
 * every node (hmax cannot be derived from an SFC key the way node geometry can) and
 * overwrites s.jlist / s.jcount.
 *
 * Measured on sedov (176900 particles, 41 dumps to t=0.1): 0% larger than the gather
 * list at uniform ICs, ~7% once the blast develops, peaking at 8.3%.  It stays small
 * because leaf size and h are both set by local density, so 2h is ~1 leaf width
 * everywhere and a big-h leaf is also a big leaf.  Longest list seen: 433 against
 * MAX_J_PER_LEAF = 1024.
 */
void buildForceJLeafList(GpuState& s, ForceTimings& ft);
