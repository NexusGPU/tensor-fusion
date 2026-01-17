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
static int g_shm_fd = -1;
static int g_shm_initialized = 0;

// Helper: Validate processor handle (mock accepts NULL or (void*)0)
static inline int validate_processor_handle(amdsmi_processor_handle handle) {
    return (handle == NULL || handle == (void*)0) ? 0 : -1;
}

// Helper: Ensure shared memory is initialized
static inline int ensure_shm_init(void) {
    return (!g_shm_initialized && driver_mock_init_shm() < 0) ? -1 : 0;
}

// Helper: Find process record by PID
static ProcessRecord* find_process_record(pid_t pid) {
    for (size_t i = 0; i < g_shm->processCount; i++) {
        if (g_shm->processes[i].processId == pid) {
            return &g_shm->processes[i];
        }
    }
    return NULL;
}

// Helper: Get current time in milliseconds
static inline uint64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

// Helper: Calculate scaled value (base + (max - base) * percent / 100)
static inline double scale_value(double base, double max, double percent) {
    if (percent > 100.0) percent = 100.0;
    return base + (max - base) * percent / 100.0;
}

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

int driver_mock_register_process(pid_t pid) {
    if (ensure_shm_init() < 0) return -1;

    pthread_mutex_lock(&g_shm->mutex);
    if (find_process_record(pid)) {
        pthread_mutex_unlock(&g_shm->mutex);
        return 0;  // Already registered
    }

    if (g_shm->processCount >= MAX_TRACKED_PROCESSES) {
        pthread_mutex_unlock(&g_shm->mutex);
        return -1;
    }

    ProcessRecord* record = &g_shm->processes[g_shm->processCount++];
    memset(record, 0, sizeof(ProcessRecord));
    record->processId = pid;
    snprintf(record->deviceUUID, sizeof(record->deviceUUID), "mock-device-0");
    pthread_mutex_unlock(&g_shm->mutex);
    return 0;
}

int driver_mock_unregister_process(pid_t pid) {
    if (!g_shm_initialized) return -1;

    pthread_mutex_lock(&g_shm->mutex);
    for (size_t i = 0; i < g_shm->processCount; i++) {
        if (g_shm->processes[i].processId == pid) {
            if (i < --g_shm->processCount) {
                g_shm->processes[i] = g_shm->processes[g_shm->processCount];
            }
            pthread_mutex_unlock(&g_shm->mutex);
            return 0;
        }
    }
    pthread_mutex_unlock(&g_shm->mutex);
    return -1;
}

int driver_mock_update_memory(pid_t pid, int64_t bytesDiff) {
    if (!g_shm_initialized) return -1;

    pthread_mutex_lock(&g_shm->mutex);
    ProcessRecord* record = find_process_record(pid);
    if (!record) {
        pthread_mutex_unlock(&g_shm->mutex);
        return -1;
    }

    if (bytesDiff > 0) {
        uint64_t newTotal = record->memoryAllocatedBytes + (uint64_t)bytesDiff;
        if (newTotal > MOCK_DEVICE_TOTAL_VRAM_BYTES) {
            pthread_mutex_unlock(&g_shm->mutex);
            return -1;
        }
        record->memoryAllocatedBytes = newTotal;
        g_shm->device.usedVRAMBytes += (uint64_t)bytesDiff;
    } else {
        uint64_t diff = (uint64_t)(-bytesDiff);
        if (record->memoryAllocatedBytes > diff) {
            record->memoryAllocatedBytes -= diff;
        } else {
            diff = record->memoryAllocatedBytes;
            record->memoryAllocatedBytes = 0;
        }
        if (g_shm->device.usedVRAMBytes > diff) {
            g_shm->device.usedVRAMBytes -= diff;
        } else {
            g_shm->device.usedVRAMBytes = 0;
        }
    }
    pthread_mutex_unlock(&g_shm->mutex);
    return 0;
}

int driver_mock_record_kernel_launch(pid_t pid, uint32_t gridSize __attribute__((unused))) {
    if (!g_shm_initialized) return -1;

    uint64_t currentTimeMs = get_time_ms();
    pthread_mutex_lock(&g_shm->mutex);
    
    // Reset 1-second window if expired
    if (g_shm->device.lastSecondStartTimeMs == 0 || 
        (currentTimeMs - g_shm->device.lastSecondStartTimeMs) >= 1000) {
        g_shm->device.kernelLaunchesLastSecond = 0;
        g_shm->device.lastSecondStartTimeMs = currentTimeMs;
        g_shm->device.gpuUtilizationPercent = 0.0;
    }
    
    if (g_shm->device.kernelLaunchesLastSecond >= MAX_KERNEL_LAUNCHES_PER_SECOND) {
        g_shm->device.gpuUtilizationPercent = 100.0;
        pthread_mutex_unlock(&g_shm->mutex);
        return -1;  // Rate limited
    }
    
    g_shm->device.kernelLaunchesLastSecond++;
    double deviceUtil = (double)g_shm->device.kernelLaunchesLastSecond * GPU_UTIL_PER_KERNEL_LAUNCH;
    if (deviceUtil > 100.0) deviceUtil = 100.0;
    g_shm->device.gpuUtilizationPercent = deviceUtil;
    
    ProcessRecord* record = find_process_record(pid);
    if (!record) {
        pthread_mutex_unlock(&g_shm->mutex);
        return -1;
    }
    
    record->kernelLaunchCount++;
    record->lastKernelLaunchTimeMs = currentTimeMs;
    
    // Count active processes (launched in last 2 seconds)
    size_t activeProcesses = 0;
    for (size_t j = 0; j < g_shm->processCount; j++) {
        if (g_shm->processes[j].lastKernelLaunchTimeMs > 0 &&
            (currentTimeMs - g_shm->processes[j].lastKernelLaunchTimeMs) < 2000) {
            activeProcesses++;
        }
    }
    record->gpuUtilizationPercent = activeProcesses > 0 ? deviceUtil / (double)activeProcesses : GPU_UTIL_PER_KERNEL_LAUNCH;
    
    pthread_mutex_unlock(&g_shm->mutex);
    return 0;
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

hipError_t hipGetDeviceCount(int* count) {
    if (!count) {
        return hipErrorInvalidValue;
    }
    *count = 1;  // Mock: always 1 device
    return hipSuccess;
}

hipError_t hipGetDevice(int* deviceId) {
    if (!deviceId) {
        return hipErrorInvalidValue;
    }
    *deviceId = 0;  // Mock device always returns device 0
    return hipSuccess;
}

hipError_t hipMalloc(void** ptr, size_t size) {
    if (!ptr || size == 0) return hipErrorInvalidValue;
    if (ensure_shm_init() < 0) return hipErrorInvalidValue;

    pid_t pid = getpid();
    pthread_mutex_lock(&g_shm->mutex);
    
    ProcessRecord* record = find_process_record(pid);
    if (!record || record->allocationCount >= MAX_ALLOCATIONS_PER_PROCESS ||
        g_shm->device.usedVRAMBytes + size > MOCK_DEVICE_TOTAL_VRAM_BYTES) {
        pthread_mutex_unlock(&g_shm->mutex);
        return record ? hipErrorOutOfMemory : hipErrorInvalidValue;
    }
    
    uint64_t fakePtrValue = ((uint64_t)pid << 32) | (uint64_t)record->allocationCount;
    void* fakePtr = (void*)(uintptr_t)fakePtrValue;
    
    AllocationRecord* alloc = &record->allocations[record->allocationCount++];
    alloc->ptr = (hipDevicePtr_t)fakePtr;
    alloc->size = size;
    record->memoryAllocatedBytes += size;
    g_shm->device.usedVRAMBytes += size;
    
    *ptr = fakePtr;
    pthread_mutex_unlock(&g_shm->mutex);
    return hipSuccess;
}

hipError_t hipFree(void* ptr) {
    if (!ptr || !g_shm_initialized) return hipErrorInvalidValue;

    pid_t pid = getpid();
    pthread_mutex_lock(&g_shm->mutex);
    
    ProcessRecord* record = find_process_record(pid);
    if (!record) {
        pthread_mutex_unlock(&g_shm->mutex);
        return hipErrorInvalidValue;
    }
    
    size_t allocIndex = SIZE_MAX;
    for (size_t i = 0; i < record->allocationCount; i++) {
        if (record->allocations[i].ptr == ptr) {
            allocIndex = i;
            break;
        }
    }
    
    if (allocIndex == SIZE_MAX) {
        pthread_mutex_unlock(&g_shm->mutex);
        return hipErrorInvalidValue;
    }
    
    size_t size = record->allocations[allocIndex].size;
    if (allocIndex < --record->allocationCount) {
        record->allocations[allocIndex] = record->allocations[record->allocationCount];
    }
    
    if (record->memoryAllocatedBytes > size) {
        record->memoryAllocatedBytes -= size;
    } else {
        record->memoryAllocatedBytes = 0;
    }
    
    if (g_shm->device.usedVRAMBytes > size) {
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
    (void)func; (void)gridDimY; (void)gridDimZ; (void)blockDimX; (void)blockDimY;
    (void)blockDimZ; (void)sharedMemBytes; (void)stream; (void)kernelParams; (void)extra;
    
    return driver_mock_record_kernel_launch(getpid(), gridDimX * gridDimY * gridDimZ) == 0 
           ? hipSuccess : hipErrorInvalidValue;
}

// Mock AMD SMI API Implementations
amdsmi_status_t amdsmi_get_gpu_process_list(
    amdsmi_processor_handle processor_handle __attribute__((unused)),
    uint32_t* max_processes,
    amdsmi_proc_info_t* list
) {
    if (!max_processes || !list || ensure_shm_init() < 0) {
        return AMDSMI_STATUS_INVAL;
    }
    
    pthread_mutex_lock(&g_shm->mutex);
    uint32_t actualCount = (uint32_t)g_shm->processCount;
    
    if (*max_processes == 0) {
        *max_processes = actualCount;
        pthread_mutex_unlock(&g_shm->mutex);
        return AMDSMI_STATUS_SUCCESS;
    }
    
    uint32_t countToReturn = (actualCount > *max_processes) ? *max_processes : actualCount;
    
    for (uint32_t i = 0; i < countToReturn; i++) {
        ProcessRecord* proc = &g_shm->processes[i];
        amdsmi_proc_info_t* info = &list[i];
        memset(info, 0, sizeof(amdsmi_proc_info_t));
        
        snprintf(info->name, sizeof(info->name), "mock-process-%d", (int)proc->processId);
        info->pid = (uint32_t)proc->processId;
        info->mem = proc->memoryAllocatedBytes;
        info->engine_usage.gfx = proc->kernelLaunchCount * 1000000ULL;  // 1ms per launch in ns
        info->memory_usage.vram_mem = proc->memoryAllocatedBytes;
        info->cu_occupancy = proc->gpuUtilizationPercent > 0.0 
            ? (uint32_t)((proc->gpuUtilizationPercent / 100.0) * 108.0) : 0;
        if (info->cu_occupancy == 0 && proc->gpuUtilizationPercent > 0.0) {
            info->cu_occupancy = 1;
        }
    }
    
    *max_processes = actualCount;
    amdsmi_status_t status = (actualCount > countToReturn) ? AMDSMI_STATUS_OUT_OF_RESOURCES : AMDSMI_STATUS_SUCCESS;
    pthread_mutex_unlock(&g_shm->mutex);
    return status;
}

amdsmi_status_t amdsmi_get_gpu_activity(
    amdsmi_processor_handle processor_handle __attribute__((unused)),
    amdsmi_engine_usage_t* info
) {
    if (!info || ensure_shm_init() < 0) return AMDSMI_STATUS_INVAL;

    uint64_t currentTimeMs = get_time_ms();
    pthread_mutex_lock(&g_shm->mutex);
    
    if (g_shm->device.lastSecondStartTimeMs > 0 &&
        (currentTimeMs - g_shm->device.lastSecondStartTimeMs) >= 1000) {
        g_shm->device.kernelLaunchesLastSecond = 0;
        g_shm->device.gpuUtilizationPercent = 0.0;
        g_shm->device.lastSecondStartTimeMs = 0;
    }
    
    uint32_t utilPercent = (uint32_t)g_shm->device.gpuUtilizationPercent;
    if (utilPercent > 100) utilPercent = 100;
    
    info->gfx_activity = utilPercent;
    info->umc_activity = info->mm_activity = 0;
    memset(info->reserved, 0, sizeof(info->reserved));
    
    pthread_mutex_unlock(&g_shm->mutex);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_vram_usage(
    amdsmi_processor_handle processor_handle __attribute__((unused)),
    amdsmi_vram_usage_t* info
) {
    if (!info || ensure_shm_init() < 0) return AMDSMI_STATUS_INVAL;

    pthread_mutex_lock(&g_shm->mutex);
    uint64_t totalMB = g_shm->device.totalVRAMBytes / (1024 * 1024);
    uint64_t usedMB = g_shm->device.usedVRAMBytes / (1024 * 1024);
    info->vram_total = (totalMB > UINT32_MAX) ? UINT32_MAX : (uint32_t)totalMB;
    info->vram_used = (usedMB > UINT32_MAX) ? UINT32_MAX : (uint32_t)usedMB;
    memset(info->reserved, 0, sizeof(info->reserved));
    pthread_mutex_unlock(&g_shm->mutex);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_power_info(
    amdsmi_processor_handle processor_handle,
    amdsmi_power_info_t* info
) {
    if (!info || validate_processor_handle(processor_handle) < 0 || ensure_shm_init() < 0) {
        return AMDSMI_STATUS_INVAL;
    }
    
    pthread_mutex_lock(&g_shm->mutex);
    uint64_t socketPower = (uint64_t)scale_value(150.0, 300.0, g_shm->device.gpuUtilizationPercent);
    
    info->socket_power = socketPower;
    info->current_socket_power = info->average_socket_power = (uint32_t)socketPower;
    info->gfx_voltage = 1150;  // 1.15V in mV
    info->soc_voltage = 1000;  // 1.0V in mV
    info->mem_voltage = 1200;  // 1.2V in mV
    info->power_limit = 300;  // 300W
    memset(info->reserved, 0, sizeof(info->reserved));
    pthread_mutex_unlock(&g_shm->mutex);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_temp_metric(
    amdsmi_processor_handle processor_handle,
    amdsmi_temperature_type_t sensor_type,
    amdsmi_temperature_metric_t metric,
    int64_t* temperature
) {
    if (!temperature || validate_processor_handle(processor_handle) < 0 || ensure_shm_init() < 0) {
        return AMDSMI_STATUS_INVAL;
    }
    
    if (metric != AMDSMI_TEMP_CURRENT) return AMDSMI_STATUS_NOT_SUPPORTED;
    if (sensor_type != AMDSMI_TEMPERATURE_TYPE_EDGE && 
        sensor_type != AMDSMI_TEMPERATURE_TYPE_HOTSPOT &&
        sensor_type != AMDSMI_TEMPERATURE_TYPE_JUNCTION) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }
    
    pthread_mutex_lock(&g_shm->mutex);
    int64_t tempCelsius = (int64_t)scale_value(40.0, 80.0, g_shm->device.gpuUtilizationPercent);
    if (sensor_type == AMDSMI_TEMPERATURE_TYPE_HOTSPOT || 
        sensor_type == AMDSMI_TEMPERATURE_TYPE_JUNCTION) {
        tempCelsius += 8;  // Hotspot offset
    }
    *temperature = tempCelsius;
    pthread_mutex_unlock(&g_shm->mutex);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_pci_throughput(
    amdsmi_processor_handle processor_handle,
    uint64_t* sent,
    uint64_t* received,
    uint64_t* max_pkt_sz
) {
    if (!sent || !received || !max_pkt_sz || 
        validate_processor_handle(processor_handle) < 0 || ensure_shm_init() < 0) {
        return AMDSMI_STATUS_INVAL;
    }
    
    pthread_mutex_lock(&g_shm->mutex);
    double throughputMBps = scale_value(100.0, 1000.0, g_shm->device.gpuUtilizationPercent);
    uint64_t throughputBytes = (uint64_t)(throughputMBps * 1024.0 * 1024.0);
    *sent = throughputBytes * 2;  // TX = 2x RX
    *received = throughputBytes;
    *max_pkt_sz = 4096;  // 4KB
    pthread_mutex_unlock(&g_shm->mutex);
    return AMDSMI_STATUS_SUCCESS;
}
