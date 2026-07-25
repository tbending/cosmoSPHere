#pragma once
/*
 * hip_to_cuda.h — reverse compatibility shim for native CUDA (nvcc) builds.
 *
 * The cosmoSPHere sources are written with HIP names (hipEventRecord,
 * hipMemcpy, hipError_t, the HIP_CHECK macro, ...).  When building with the
 * CUDA backend we compile them with nvcc and force-include this header
 * (nvcc -include).  It pulls in the *real* <cuda_runtime.h> and maps every
 * HIP name the sources use onto its CUDA equivalent.
 *
 * Cornerstone's own #include <cuda_runtime.h> is satisfied directly by the
 * real CUDA header — this file is NOT named cuda_runtime.h, so it never
 * shadows it (contrast the HIP backend, which uses the forward shim in
 * compat/hipcc/cuda_runtime.h to redirect CUDA names onto HIP).
 */
#include <cuda_runtime.h>

/* --- Types --- */
typedef cudaError_t  hipError_t;
typedef cudaEvent_t  hipEvent_t;

/* --- Error handling --- */
#define hipSuccess              cudaSuccess
#define hipGetErrorString       cudaGetErrorString
#define hipGetErrorName         cudaGetErrorName
#define hipGetLastError         cudaGetLastError

/* --- Device management --- */
#define hipDeviceSynchronize    cudaDeviceSynchronize
#define hipSetDevice            cudaSetDevice

/* --- Events (timing) --- */
#define hipEventCreate          cudaEventCreate
#define hipEventRecord          cudaEventRecord
#define hipEventSynchronize     cudaEventSynchronize
#define hipEventElapsedTime     cudaEventElapsedTime
#define hipEventDestroy         cudaEventDestroy

/* --- Memory --- */
#define hipMemcpy               cudaMemcpy
#define hipMemcpyKind           cudaMemcpyKind
#define hipMemcpyFromSymbol     cudaMemcpyFromSymbol
#define hipMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define hipMemcpyHostToDevice   cudaMemcpyHostToDevice
