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

// Mock HIP and AMD SMI APIs matching real API signatures
// References: ROCm amdsmi.h and hip_runtime_api.h

// HIP API types
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

// Device-level metrics (internal to driver_mock)
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

// HIP-like API functions (AMD style)
hipError_t hipInit(unsigned int flags);
hipError_t hipGetDeviceCount(int* count);
hipError_t hipGetDevice(int* deviceId);
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
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

// AMD SMI API types (matching real API)
#define AMDSMI_MAX_STRING_LENGTH 256

typedef void* amdsmi_processor_handle;
typedef int amdsmi_status_t;
#define AMDSMI_STATUS_SUCCESS 0
#define AMDSMI_STATUS_INVAL 1
#define AMDSMI_STATUS_NOT_SUPPORTED 2
#define AMDSMI_STATUS_OUT_OF_RESOURCES 15

typedef uint32_t amdsmi_process_handle_t;

typedef struct {
    uint32_t gfx_activity;  //!< In %
    uint32_t umc_activity;  //!< In %
    uint32_t mm_activity;   //!< In %
    uint32_t reserved[13];
} amdsmi_engine_usage_t;

typedef struct {
    char name[AMDSMI_MAX_STRING_LENGTH];
    amdsmi_process_handle_t pid;
    uint64_t mem;  //!< In Bytes
    struct engine_usage_ {
        uint64_t gfx;  //!< In nano-secs
        uint64_t enc;  //!< In nano-secs
        uint32_t reserved[12];
    } engine_usage; //!< time the process spends using these engines in ns
    struct memory_usage_ {
        uint64_t gtt_mem;   //!< In Bytes
        uint64_t cpu_mem;   //!< In Bytes
        uint64_t vram_mem;  //!< In Bytes
        uint32_t reserved[10];
    } memory_usage;  //!< In Bytes
    char container_name[AMDSMI_MAX_STRING_LENGTH];
    uint32_t cu_occupancy;  //!< Num CUs utilized
    uint32_t evicted_time;    //!< Time that queues are evicted on a GPU in milliseconds
    uint32_t reserved[10];
} amdsmi_proc_info_t;

typedef struct {
    uint32_t vram_total;  //!< In MB
    uint32_t vram_used;   //!< In MB
    uint32_t reserved[2];
} amdsmi_vram_usage_t;

typedef enum {
    AMDSMI_TEMPERATURE_TYPE_EDGE = 0,
    AMDSMI_TEMPERATURE_TYPE_FIRST = AMDSMI_TEMPERATURE_TYPE_EDGE,
    AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
    AMDSMI_TEMPERATURE_TYPE_JUNCTION = AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
    AMDSMI_TEMPERATURE_TYPE_VRAM,
    AMDSMI_TEMPERATURE_TYPE__MAX
} amdsmi_temperature_type_t;

typedef enum {
    AMDSMI_TEMP_CURRENT = 0x0,
    AMDSMI_TEMP_FIRST = AMDSMI_TEMP_CURRENT,
    AMDSMI_TEMP_MAX,
    AMDSMI_TEMP_MIN,
    AMDSMI_TEMP_LAST
} amdsmi_temperature_metric_t;

typedef struct {
    uint64_t socket_power;          //!< Socket power in W
    uint32_t current_socket_power;  //!< Current socket power in W, Mi 300+ Series cards
    uint32_t average_socket_power;  //!< Average socket power in W, Navi + Mi 200 and earlier Series cards
    uint64_t gfx_voltage;           //!< GFX voltage measurement in mV
    uint64_t soc_voltage;           //!< SOC voltage measurement in mV
    uint64_t mem_voltage;           //!< MEM voltage measurement in mV
    uint32_t power_limit;           //!< The power limit in W
    uint64_t reserved[18];
} amdsmi_power_info_t;

amdsmi_status_t amdsmi_get_gpu_process_list(
    amdsmi_processor_handle processor_handle,
    uint32_t* max_processes,
    amdsmi_proc_info_t* list
);

amdsmi_status_t amdsmi_get_gpu_activity(
    amdsmi_processor_handle processor_handle,
    amdsmi_engine_usage_t* info
);

amdsmi_status_t amdsmi_get_gpu_vram_usage(
    amdsmi_processor_handle processor_handle,
    amdsmi_vram_usage_t* info
);

amdsmi_status_t amdsmi_get_power_info(
    amdsmi_processor_handle processor_handle,
    amdsmi_power_info_t* info
);

amdsmi_status_t amdsmi_get_temp_metric(
    amdsmi_processor_handle processor_handle,
    amdsmi_temperature_type_t sensor_type,
    amdsmi_temperature_metric_t metric,
    int64_t* temperature
);

amdsmi_status_t amdsmi_get_gpu_pci_throughput(
    amdsmi_processor_handle processor_handle,
    uint64_t* sent,
    uint64_t* received,
    uint64_t* max_pkt_sz
);

#ifdef __cplusplus
}
#endif

#endif // DRIVER_MOCK_H

