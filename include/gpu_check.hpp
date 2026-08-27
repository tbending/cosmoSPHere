/*
 * gpu_check.hpp — HIP_CHECK, the abort-on-error wrapper for runtime API calls.
 *
 * Was in density.hpp, which no longer makes sense now that tree.cu and force.cu use
 * it too and neither is about density.  Cornerstone's own checkGpuErrors (from
 * util/cuda_utils.hpp) does the same job for kernel-launch checks; both are in use.
 */

#pragma once

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t _e = (call);                                                \
        if (_e != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP error %s:%d  %s\n",                      \
                         __FILE__, __LINE__, hipGetErrorString(_e));           \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
