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

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include "driver_mock.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

#define SHM_MAGIC 0x4D4F434B  // "MOCK"
#define SHM_VERSION 2  // Updated version for new structure

// GPU utilization constants
#define GPU_UTIL_PER_KERNEL_LAUNCH 1.0  // Each kernel launch = 1% GPU utilization
#define MAX_KERNEL_LAUNCHES_PER_SECOND 100  // Max launches/second before hitting 100% utilization
#define MOCK_DEVICE_TOTAL_VRAM_BYTES (16ULL * 1024 * 1024 * 1024)  // 16GB VRAM

static SharedMemoryHeader* g_shm = NULL;
#define DeviceMetrics MockDeviceMetrics  // Use MockDeviceMetrics internally
static int g_shm_fd = -1;
static int g_shm_initialized = 0;

// Initialize shared memory
int driver_mock_init_shm(void) {
    if (g_shm_initialized) {
        return 0;
    }

    size_t shm_size = sizeof(SharedMemoryHeader);
    const char* shm_path = SHM_FILE_NAME;

    // Try to open existing shared memory file
    g_shm_fd = open(shm_path, O_RDWR);
    if (g_shm_fd < 0) {
        // File doesn't exist, create it
        g_shm_fd = open(shm_path, O_RDWR | O_CREAT, 0666);
        if (g_shm_fd < 0) {
            fprintf(stderr, "Failed to create shared memory file: %s\n", strerror(errno));
            return -1;
        }
        // Truncate to required size
        if (ftruncate(g_shm_fd, shm_size) < 0) {
            fprintf(stderr, "Failed to truncate shared memory file: %s\n", strerror(errno));
            close(g_shm_fd);
            return -1;
        }
    }

    // Map shared memory
    g_shm = (SharedMemoryHeader*)mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
    if (g_shm == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap shared memory: %s\n", strerror(errno));
        close(g_shm_fd);
        return -1;
    }

    // Initialize if this is the first process or version mismatch
    if (g_shm->magic != SHM_MAGIC || g_shm->version != SHM_VERSION) {
        memset(g_shm, 0, shm_size);
        g_shm->magic = SHM_MAGIC;
        g_shm->version = SHM_VERSION;
        g_shm->processCount = 0;
        
        // Initialize device metrics
        snprintf(g_shm->device.deviceUUID, sizeof(g_shm->device.deviceUUID), "mock-device-0");
        g_shm->device.gpuUtilizationPercent = 0.0;
        g_shm->device.totalVRAMBytes = MOCK_DEVICE_TOTAL_VRAM_BYTES;
        g_shm->device.usedVRAMBytes = 0;
        g_shm->device.kernelLaunchesLastSecond = 0;
        g_shm->device.lastSecondStartTimeMs = 0;
        
        // Initialize mutex with shared attribute
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&g_shm->mutex, &attr);
        pthread_mutexattr_destroy(&attr);
    }

    g_shm_initialized = 1;
    return 0;
}

// Cleanup shared memory
void driver_mock_cleanup_shm(void) {
    if (g_shm && g_shm != MAP_FAILED) {
        munmap(g_shm, sizeof(SharedMemoryHeader));
        g_shm = NULL;
    }
    if (g_shm_fd >= 0) {
        close(g_shm_fd);
        g_shm_fd = -1;
    }
    g_shm_initialized = 0;
}

// Register a process in shared memory
int driver_mock_register_process(pid_t pid) {
    if (!g_shm_initialized && driver_mock_init_shm() < 0) {
        return -1;
    }

    pthread_mutex_lock(&g_shm->mutex);

    // Check if process already exists
    for (size_t i = 0; i < g_shm->processCount; i++) {
        if (g_shm->processes[i].processId == pid) {
            pthread_mutex_unlock(&g_shm->mutex);
            return 0;  // Already registered
        }
    }

    // Add new process
    if (g_shm->processCount >= MAX_TRACKED_PROCESSES) {
        pthread_mutex_unlock(&g_shm->mutex);
        return -1;  // Too many processes
    }

    ProcessRecord* record = &g_shm->processes[g_shm->processCount];
    record->processId = pid;
    record->memoryAllocatedBytes = 0;
    record->kernelLaunchCount = 0;
    record->lastKernelLaunchTimeMs = 0;
    record->gpuUtilizationPercent = 0.0;
    record->allocationCount = 0;
    memset(record->allocations, 0, sizeof(record->allocations));
    snprintf(record->deviceUUID, sizeof(record->deviceUUID), "mock-device-0");

    g_shm->processCount++;
    pthread_mutex_unlock(&g_shm->mutex);
    return 0;
}

// Unregister a process
int driver_mock_unregister_process(pid_t pid) {
    if (!g_shm_initialized) {
        return -1;
    }

    pthread_mutex_lock(&g_shm->mutex);
    for (size_t i = 0; i < g_shm->processCount; i++) {
        if (g_shm->processes[i].processId == pid) {
            // Move last element to this position
            if (i < g_shm->processCount - 1) {
                g_shm->processes[i] = g_shm->processes[g_shm->processCount - 1];
            }
            g_shm->processCount--;
            pthread_mutex_unlock(&g_shm->mutex);
            return 0;
        }
    }
    pthread_mutex_unlock(&g_shm->mutex);
    return -1;
}

// Update memory allocation for a process
int driver_mock_update_memory(pid_t pid, int64_t bytesDiff) {
    if (!g_shm_initialized) {
        return -1;
    }

    pthread_mutex_lock(&g_shm->mutex);
    for (size_t i = 0; i < g_shm->processCount; i++) {
        if (g_shm->processes[i].processId == pid) {
            if (bytesDiff > 0) {
                uint64_t newTotal = g_shm->processes[i].memoryAllocatedBytes + (uint64_t)bytesDiff;
                // Check if allocation exceeds device capacity
                if (newTotal > MOCK_DEVICE_TOTAL_VRAM_BYTES) {
                    pthread_mutex_unlock(&g_shm->mutex);
                    return -1;  // Out of VRAM
                }
                g_shm->processes[i].memoryAllocatedBytes = newTotal;
                g_shm->device.usedVRAMBytes += (uint64_t)bytesDiff;
            } else {
                uint64_t diff = (uint64_t)(-bytesDiff);
                if (g_shm->processes[i].memoryAllocatedBytes >= diff) {
                    g_shm->processes[i].memoryAllocatedBytes -= diff;
                    if (g_shm->device.usedVRAMBytes >= diff) {
                        g_shm->device.usedVRAMBytes -= diff;
                    } else {
                        g_shm->device.usedVRAMBytes = 0;
                    }
                } else {
                    g_shm->device.usedVRAMBytes -= g_shm->processes[i].memoryAllocatedBytes;
                    g_shm->processes[i].memoryAllocatedBytes = 0;
                }
            }
            pthread_mutex_unlock(&g_shm->mutex);
            return 0;
        }
    }
    pthread_mutex_unlock(&g_shm->mutex);
    return -1;
}

// Record kernel launch and update GPU utilization (pub-sub pattern)
int driver_mock_record_kernel_launch(pid_t pid, uint32_t gridSize __attribute__((unused))) {
    if (!g_shm_initialized) {
        return -1;
    }

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t currentTimeMs = (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;

    pthread_mutex_lock(&g_shm->mutex);
    
    // Reset second window if needed (sliding 1-second window)
    if (g_shm->device.lastSecondStartTimeMs == 0 || 
        (currentTimeMs - g_shm->device.lastSecondStartTimeMs) >= 1000) {
        // Reset the window
        g_shm->device.kernelLaunchesLastSecond = 0;
        g_shm->device.lastSecondStartTimeMs = currentTimeMs;
        g_shm->device.gpuUtilizationPercent = 0.0;
    }
    
    // Check rate limit: if >= 100 launches in current second, block and show 100% utilization
    if (g_shm->device.kernelLaunchesLastSecond >= MAX_KERNEL_LAUNCHES_PER_SECOND) {
        // Device is at 100% utilization, block this launch
        g_shm->device.gpuUtilizationPercent = 100.0;
        pthread_mutex_unlock(&g_shm->mutex);
        return -1;  // Blocked due to 100% utilization
    }
    
    // Increment launch count for current second
    g_shm->device.kernelLaunchesLastSecond++;
    
    // Update device-level utilization (simple: launches/sec * 1% per launch)
    double deviceUtil = (double)g_shm->device.kernelLaunchesLastSecond * GPU_UTIL_PER_KERNEL_LAUNCH;
    if (deviceUtil > 100.0) {
        deviceUtil = 100.0;
    }
    g_shm->device.gpuUtilizationPercent = deviceUtil;
    
    // Find and update process record
    for (size_t i = 0; i < g_shm->processCount; i++) {
        if (g_shm->processes[i].processId == pid) {
            g_shm->processes[i].kernelLaunchCount++;
            g_shm->processes[i].lastKernelLaunchTimeMs = currentTimeMs;
            
            // Update process-level utilization (distribute device utilization proportionally)
            // Simple approach: each process gets utilization based on its contribution
            // For now, distribute evenly among active processes
            size_t activeProcesses = 0;
            for (size_t j = 0; j < g_shm->processCount; j++) {
                if (g_shm->processes[j].lastKernelLaunchTimeMs > 0 &&
                    (currentTimeMs - g_shm->processes[j].lastKernelLaunchTimeMs) < 2000) {
                    activeProcesses++;
                }
            }
            if (activeProcesses > 0) {
                g_shm->processes[i].gpuUtilizationPercent = deviceUtil / (double)activeProcesses;
            } else {
                g_shm->processes[i].gpuUtilizationPercent = GPU_UTIL_PER_KERNEL_LAUNCH;
            }
            
            pthread_mutex_unlock(&g_shm->mutex);
            return 0;
        }
    }
    pthread_mutex_unlock(&g_shm->mutex);
    return -1;
}

// HIP-like API implementations
hipError_t hipInit(unsigned int flags) {
    (void)flags;
    
    pid_t pid = getpid();
    if (driver_mock_init_shm() < 0) {
        return hipErrorInvalidValue;
    }
    
    if (driver_mock_register_process(pid) < 0) {
        return hipErrorInvalidValue;
    }
    
    return hipSuccess;
}

hipError_t hipDeviceGetCount(int* count) {
    if (!count) {
        return hipErrorInvalidValue;
    }
    *count = 1;  // Mock: always 1 device
    return hipSuccess;
}

hipError_t hipDeviceGet(hipDevice_t* device, int deviceId) {
    if (!device || deviceId != 0) {
        return hipErrorInvalidValue;
    }
    *device = 0;
    return hipSuccess;
}

hipError_t hipMalloc(hipDevicePtr_t* ptr, size_t size) {
    if (!ptr || size == 0) {
        return hipErrorInvalidValue;
    }
    
    // Mock VRAM allocation: just track in shared memory, return a fake pointer
    // No actual host memory allocation - this is GPU VRAM
    pid_t pid = getpid();
    
    if (!g_shm_initialized && driver_mock_init_shm() < 0) {
        return hipErrorInvalidValue;
    }
    
    pthread_mutex_lock(&g_shm->mutex);
    
    // Find process record
    ProcessRecord* record = NULL;
    for (size_t i = 0; i < g_shm->processCount; i++) {
        if (g_shm->processes[i].processId == pid) {
            record = &g_shm->processes[i];
            break;
        }
    }
    
    if (!record) {
        pthread_mutex_unlock(&g_shm->mutex);
        return hipErrorInvalidValue;
    }
    
    // Check if allocation would exceed device capacity
    if (g_shm->device.usedVRAMBytes + size > MOCK_DEVICE_TOTAL_VRAM_BYTES) {
        pthread_mutex_unlock(&g_shm->mutex);
        return hipErrorOutOfMemory;
    }
    
    // Check if we have space for allocation record
    if (record->allocationCount >= MAX_ALLOCATIONS_PER_PROCESS) {
        pthread_mutex_unlock(&g_shm->mutex);
        return hipErrorOutOfMemory;
    }
    
    // Generate a fake pointer (use process ID and allocation count for uniqueness)
    // This ensures uniqueness across processes
    uint64_t fakePtrValue = ((uint64_t)pid << 32) | (uint64_t)record->allocationCount;
    hipDevicePtr_t fakePtr = (hipDevicePtr_t)(uintptr_t)fakePtrValue;
    
    // Record allocation
    AllocationRecord* alloc = &record->allocations[record->allocationCount];
    alloc->ptr = fakePtr;
    alloc->size = size;
    record->allocationCount++;
    
    // Update memory tracking
    record->memoryAllocatedBytes += size;
    g_shm->device.usedVRAMBytes += size;
    
    *ptr = fakePtr;
    pthread_mutex_unlock(&g_shm->mutex);
    
    return hipSuccess;
}

hipError_t hipFree(hipDevicePtr_t ptr) {
    if (!ptr) {
        return hipErrorInvalidValue;
    }
    
    pid_t pid = getpid();
    
    if (!g_shm_initialized) {
        return hipErrorInvalidValue;
    }
    
    pthread_mutex_lock(&g_shm->mutex);
    
    // Find process record
    ProcessRecord* record = NULL;
    for (size_t i = 0; i < g_shm->processCount; i++) {
        if (g_shm->processes[i].processId == pid) {
            record = &g_shm->processes[i];
            break;
        }
    }
    
    if (!record) {
        pthread_mutex_unlock(&g_shm->mutex);
        return hipErrorInvalidValue;
    }
    
    // Find allocation by pointer
    size_t allocIndex = SIZE_MAX;
    for (size_t i = 0; i < record->allocationCount; i++) {
        if (record->allocations[i].ptr == ptr) {
            allocIndex = i;
            break;
        }
    }
    
    if (allocIndex == SIZE_MAX) {
        pthread_mutex_unlock(&g_shm->mutex);
        return hipErrorInvalidValue;  // Pointer not found
    }
    
    // Get size before removing
    size_t size = record->allocations[allocIndex].size;
    
    // Remove allocation (move last to this position)
    if (allocIndex < record->allocationCount - 1) {
        record->allocations[allocIndex] = record->allocations[record->allocationCount - 1];
    }
    record->allocationCount--;
    
    // Update memory tracking
    if (record->memoryAllocatedBytes >= size) {
        record->memoryAllocatedBytes -= size;
    } else {
        record->memoryAllocatedBytes = 0;
    }
    
    if (g_shm->device.usedVRAMBytes >= size) {
        g_shm->device.usedVRAMBytes -= size;
    } else {
        g_shm->device.usedVRAMBytes = 0;
    }
    
    pthread_mutex_unlock(&g_shm->mutex);
    
    return hipSuccess;
}

hipError_t hipLaunchKernel(const void* func,
                          uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                          uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                          uint32_t sharedMemBytes, void* stream,
                          void** kernelParams, void** extra) {
    (void)func;
    (void)gridDimY;
    (void)gridDimZ;
    (void)blockDimX;
    (void)blockDimY;
    (void)blockDimZ;
    (void)sharedMemBytes;
    (void)stream;
    (void)kernelParams;
    (void)extra;
    
    // Calculate total grid size
    uint32_t gridSize = gridDimX * gridDimY * gridDimZ;
    
    // Record kernel launch
    pid_t pid = getpid();
    driver_mock_record_kernel_launch(pid, gridSize);
    
    return hipSuccess;
}

// ============================================================================
// Metrics API Implementations
// ============================================================================

hipError_t hipGetProcessUtilization(ProcessUtilization* utilizations, size_t maxCount, size_t* count) {
    if (!utilizations || !count || maxCount == 0) {
        return hipErrorInvalidValue;
    }
    
    if (!g_shm_initialized && driver_mock_init_shm() < 0) {
        return hipErrorInvalidValue;
    }
    
    pthread_mutex_lock(&g_shm->mutex);
    
    size_t actualCount = g_shm->processCount;
    if (actualCount > maxCount) {
        actualCount = maxCount;
    }
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t currentTimeMs = (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
    
    for (size_t i = 0; i < actualCount; i++) {
        ProcessRecord* proc = &g_shm->processes[i];
        // Only include processes that were active in the last 2 seconds
        if (proc->lastKernelLaunchTimeMs > 0 && 
            (currentTimeMs - proc->lastKernelLaunchTimeMs) < 2000) {
            utilizations[i].processId = proc->processId;
            strncpy(utilizations[i].deviceUUID, proc->deviceUUID, sizeof(utilizations[i].deviceUUID) - 1);
            utilizations[i].deviceUUID[sizeof(utilizations[i].deviceUUID) - 1] = '\0';
            utilizations[i].utilizationPercent = proc->gpuUtilizationPercent;
        } else {
            utilizations[i].processId = proc->processId;
            strncpy(utilizations[i].deviceUUID, proc->deviceUUID, sizeof(utilizations[i].deviceUUID) - 1);
            utilizations[i].deviceUUID[sizeof(utilizations[i].deviceUUID) - 1] = '\0';
            utilizations[i].utilizationPercent = 0.0;
        }
    }
    
    *count = actualCount;
    pthread_mutex_unlock(&g_shm->mutex);
    
    return hipSuccess;
}

hipError_t hipGetProcessVRAMUsage(ProcessVRAMUsage* usages, size_t maxCount, size_t* count) {
    if (!usages || !count || maxCount == 0) {
        return hipErrorInvalidValue;
    }
    
    if (!g_shm_initialized && driver_mock_init_shm() < 0) {
        return hipErrorInvalidValue;
    }
    
    pthread_mutex_lock(&g_shm->mutex);
    
    size_t actualCount = g_shm->processCount;
    if (actualCount > maxCount) {
        actualCount = maxCount;
    }
    
    for (size_t i = 0; i < actualCount; i++) {
        ProcessRecord* proc = &g_shm->processes[i];
        usages[i].processId = proc->processId;
        strncpy(usages[i].deviceUUID, proc->deviceUUID, sizeof(usages[i].deviceUUID) - 1);
        usages[i].deviceUUID[sizeof(usages[i].deviceUUID) - 1] = '\0';
        usages[i].usedBytes = proc->memoryAllocatedBytes;
        usages[i].reservedBytes = proc->memoryAllocatedBytes;  // For mock, reserved = used
    }
    
    *count = actualCount;
    pthread_mutex_unlock(&g_shm->mutex);
    
    return hipSuccess;
}

hipError_t hipGetDeviceUtilization(DeviceUtilization* utilization) {
    if (!utilization) {
        return hipErrorInvalidValue;
    }
    
    if (!g_shm_initialized && driver_mock_init_shm() < 0) {
        return hipErrorInvalidValue;
    }
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t currentTimeMs = (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
    
    pthread_mutex_lock(&g_shm->mutex);
    
    // Check if window needs reset (utilization should decay if no launches in last second)
    if (g_shm->device.lastSecondStartTimeMs > 0 &&
        (currentTimeMs - g_shm->device.lastSecondStartTimeMs) >= 1000) {
        // Window expired, reset utilization
        g_shm->device.kernelLaunchesLastSecond = 0;
        g_shm->device.gpuUtilizationPercent = 0.0;
        g_shm->device.lastSecondStartTimeMs = 0;  // Mark as needing reset
    }
    
    strncpy(utilization->deviceUUID, g_shm->device.deviceUUID, sizeof(utilization->deviceUUID) - 1);
    utilization->deviceUUID[sizeof(utilization->deviceUUID) - 1] = '\0';
    utilization->utilizationPercent = g_shm->device.gpuUtilizationPercent;
    pthread_mutex_unlock(&g_shm->mutex);
    
    return hipSuccess;
}

hipError_t hipGetDeviceVRAMUsage(DeviceVRAMUsage* usage) {
    if (!usage) {
        return hipErrorInvalidValue;
    }
    
    if (!g_shm_initialized && driver_mock_init_shm() < 0) {
        return hipErrorInvalidValue;
    }
    
    pthread_mutex_lock(&g_shm->mutex);
    strncpy(usage->deviceUUID, g_shm->device.deviceUUID, sizeof(usage->deviceUUID) - 1);
    usage->deviceUUID[sizeof(usage->deviceUUID) - 1] = '\0';
    usage->totalBytes = g_shm->device.totalVRAMBytes;
    usage->usedBytes = g_shm->device.usedVRAMBytes;
    usage->freeBytes = (g_shm->device.totalVRAMBytes > g_shm->device.usedVRAMBytes) ?
                       (g_shm->device.totalVRAMBytes - g_shm->device.usedVRAMBytes) : 0;
    if (g_shm->device.totalVRAMBytes > 0) {
        usage->utilizationPercent = ((double)g_shm->device.usedVRAMBytes / (double)g_shm->device.totalVRAMBytes) * 100.0;
    } else {
        usage->utilizationPercent = 0.0;
    }
    pthread_mutex_unlock(&g_shm->mutex);
    
    return hipSuccess;
}
