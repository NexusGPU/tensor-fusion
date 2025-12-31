/*
 * Copyright 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DRIVER_MOCK_H
#define DRIVER_MOCK_H

#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// HIP-like API types
typedef int hipError_t;
typedef int hipDevice_t;
typedef void* hipDevicePtr_t;

// Error codes
#define hipSuccess 0
#define hipErrorInvalidValue 1
#define hipErrorOutOfMemory 2

// Shared memory structure for tracking processes
#define MAX_TRACKED_PROCESSES 256
#define MAX_ALLOCATIONS_PER_PROCESS 1024
#define SHM_FILE_NAME "tmp.example_accelerator.bin"

// Allocation record for tracking VRAM allocations
typedef struct {
    hipDevicePtr_t ptr;
    size_t size;
} AllocationRecord;

typedef struct {
    pid_t processId;
    uint64_t memoryAllocatedBytes;      // VRAM allocated in bytes
    uint64_t kernelLaunchCount;
    uint64_t lastKernelLaunchTimeMs;
    double gpuUtilizationPercent;       // Current GPU utilization for this process (0-100)
    char deviceUUID[64];
    AllocationRecord allocations[MAX_ALLOCATIONS_PER_PROCESS];  // Track individual allocations
    size_t allocationCount;             // Number of active allocations
} ProcessRecord;

// Device-level metrics (internal to driver_mock, not to be confused with accelerator.h DeviceMetrics)
typedef struct {
    char deviceUUID[64];
    double gpuUtilizationPercent;       // Total device GPU utilization (0-100)
    uint64_t totalVRAMBytes;             // Total VRAM capacity
    uint64_t usedVRAMBytes;              // Total VRAM used across all processes
    uint64_t kernelLaunchesLastSecond;  // Kernel launches in the last second
    uint64_t lastSecondStartTimeMs;     // Start time of current second window
} MockDeviceMetrics;

typedef struct {
    uint32_t magic;  // Magic number to verify structure
    uint32_t version;
    size_t processCount;
    ProcessRecord processes[MAX_TRACKED_PROCESSES];
    MockDeviceMetrics device;                // Single device metrics (mock device)
    pthread_mutex_t mutex;  // For synchronization
} SharedMemoryHeader;

// HIP-like API functions
hipError_t hipInit(unsigned int flags);
hipError_t hipDeviceGetCount(int* count);
hipError_t hipDeviceGet(hipDevice_t* device, int deviceId);
hipError_t hipMalloc(hipDevicePtr_t* ptr, size_t size);
hipError_t hipFree(hipDevicePtr_t ptr);
hipError_t hipLaunchKernel(const void* func, 
                          uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                          uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                          uint32_t sharedMemBytes, void* stream,
                          void** kernelParams, void** extra);

// Internal functions for shared memory management
int driver_mock_init_shm(void);
void driver_mock_cleanup_shm(void);
int driver_mock_register_process(pid_t pid);
int driver_mock_unregister_process(pid_t pid);
int driver_mock_update_memory(pid_t pid, int64_t bytesDiff);
int driver_mock_record_kernel_launch(pid_t pid, uint32_t gridSize);

// Metrics API functions (HIP-like)
typedef struct {
    pid_t processId;
    char deviceUUID[64];
    double utilizationPercent;  // GPU utilization (0-100)
} ProcessUtilization;

typedef struct {
    pid_t processId;
    char deviceUUID[64];
    uint64_t usedBytes;          // VRAM used in bytes
    uint64_t reservedBytes;      // VRAM reserved in bytes (same as used for mock)
} ProcessVRAMUsage;

typedef struct {
    char deviceUUID[64];
    double utilizationPercent;   // Total GPU utilization (0-100)
} DeviceUtilization;

typedef struct {
    char deviceUUID[64];
    uint64_t totalBytes;         // Total VRAM capacity
    uint64_t usedBytes;          // Total VRAM used
    uint64_t freeBytes;          // Total VRAM free
    double utilizationPercent;    // VRAM utilization (0-100)
} DeviceVRAMUsage;

// Get process GPU utilization
hipError_t hipGetProcessUtilization(ProcessUtilization* utilizations, size_t maxCount, size_t* count);

// Get process VRAM usage
hipError_t hipGetProcessVRAMUsage(ProcessVRAMUsage* usages, size_t maxCount, size_t* count);

// Get device GPU utilization
hipError_t hipGetDeviceUtilization(DeviceUtilization* utilization);

// Get device VRAM usage
hipError_t hipGetDeviceVRAMUsage(DeviceVRAMUsage* usage);

#ifdef __cplusplus
}
#endif

#endif // DRIVER_MOCK_H

