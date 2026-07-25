/*
 * cuda_runtime.h — thin redirect for hipcc builds.
 *
 * When Cornerstone's cuda_utils.hpp (compiled via hipcc with -DUSE_CUDA)
 * does  #include <cuda_runtime.h>  it finds this file first because
 * density-cpp/include/ is first on the include path.  We pull in the
 * real ROCm runtime and map every CUDA name that our code uses to the
 * equivalent HIP name.
 *
 * This file deliberately lives in density-cpp/include/ so the miniapp
 * sources are never touched.
 */
#pragma once

#include <hip/hip_runtime.h>

/* --- Error handling --- */
typedef hipError_t                          cudaError_t;
#define cudaSuccess                         hipSuccess
#define cudaGetErrorName                    hipGetErrorName
#define cudaGetErrorString                  hipGetErrorString
#define cudaGetLastError                    hipGetLastError
#define cudaMemcpyFromSymbol                hipMemcpyFromSymbol
#define cudaMemcpyKind                      hipMemcpyKind
#define cudaMemcpyDeviceToHost              hipMemcpyDeviceToHost

/* --- Events (used for timing in density.cuh) --- */
typedef hipEvent_t                          cudaEvent_t;
#define cudaEventCreate(e)                  hipEventCreate(e)
#define cudaEventDestroy(e)                 hipEventDestroy(e)
#define cudaEventRecord(e, ...)             hipEventRecord(e, ##__VA_ARGS__)
#define cudaEventSynchronize(e)             hipEventSynchronize(e)
#define cudaEventElapsedTime(ms, s, e)      hipEventElapsedTime(ms, s, e)

/* --- Device management --- */
#define cudaDeviceSynchronize               hipDeviceSynchronize
#define cudaSetDevice                       hipSetDevice
