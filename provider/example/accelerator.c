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

// Feature test macros for POSIX functions (required on Linux)
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include "../accelerator.h"

// Define Result type for limiter.h compatibility (before including limiter.h)
typedef enum {
    RESULT_SUCCESS = ACCEL_SUCCESS,
    RESULT_ERROR_INVALID_PARAM = ACCEL_ERROR_INVALID_PARAM,
    RESULT_ERROR_NOT_FOUND = ACCEL_ERROR_NOT_FOUND,
    RESULT_ERROR_NOT_SUPPORTED = ACCEL_ERROR_NOT_SUPPORTED,
    RESULT_ERROR_RESOURCE_EXHAUSTED = ACCEL_ERROR_RESOURCE_EXHAUSTED,
    RESULT_ERROR_OPERATION_FAILED = ACCEL_ERROR_OPERATION_FAILED,
    RESULT_ERROR_INTERNAL = ACCEL_ERROR_INTERNAL
} Result;

#include "../limiter.h"
#include "device_mock/driver_mock.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>

// ============================================================================
// Global Variables for Limiter Thread
// ============================================================================

static const char* g_processId = "example-process-0";
static _Atomic uint64_t g_lastComputeCallTimeMs = 0; // Last call time in milliseconds
static pthread_t g_limiterThread;
static volatile int g_threadRunning = 0;

// ============================================================================
// Log Callback
// ============================================================================

// Log callback storage (declared early so it can be used throughout the file)
static LogCallbackFunc Log = NULL;

// Helper function to safely call the log callback
static void logMessage(const char* level, const char* message) {
    if (Log != NULL) {
        Log(level, message);
    }
}

// ============================================================================
// Limiter Thread Function
// ============================================================================

static void* limiterThreadFunc(void* arg __attribute__((unused))) {
    // Get first device UUID for testing
    // Use heap allocation instead of stack to avoid stack overflow
    // ExtendedDeviceInfo is ~21KB, 256 of them would be ~5.4MB which exceeds thread stack size
    ExtendedDeviceInfo* devices = (ExtendedDeviceInfo*)malloc(4 * sizeof(ExtendedDeviceInfo));
    if (!devices) {
        return NULL;
    }
    size_t deviceCount = 0;
    char deviceUUID[64] = {0};
    
    if (AccelGetAllDevices(devices, 4, &deviceCount) != ACCEL_SUCCESS || deviceCount == 0) {
        free(devices);
        return NULL;
    }
    snprintf(deviceUUID, sizeof(deviceUUID), "%s", devices[0].basic.uuid);
    free(devices);  // Free after copying UUID

    // Add worker process to limiter tracking
    AddWorkerProcess(deviceUUID, g_processId);

    // Call CheckAndRecordMemoryOps once
    MemoryOpRecord memRecord;
    CheckAndRecordMemoryOps(g_processId, deviceUUID, 0, &memRecord);

    // Call CheckAndRecordComputeOps every second
    while (g_threadRunning) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t currentTimeMs = (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
        
        ComputeOpRecord computeRecord;
        CheckAndRecordComputeOps(g_processId, deviceUUID, 1000, &computeRecord);
        
        // Update global variable
        g_lastComputeCallTimeMs = currentTimeMs;
        
        // Sleep for 1 second
        sleep(1);
    }

    return NULL;
}

// ============================================================================
// Constructor - Initialize Limiter Thread
// ============================================================================

__attribute__((constructor))
static void initLimiterThread(void) {
    g_threadRunning = 1;
    if (pthread_create(&g_limiterThread, NULL, limiterThreadFunc, NULL) != 0) {
        fprintf(stderr, "Failed to create limiter thread\n");
        return;
    }
    pthread_detach(g_limiterThread);
}

// ============================================================================
// Destructor - Cleanup Limiter Thread
// ============================================================================

__attribute__((destructor))
static void cleanupLimiterThread(void) {
    g_threadRunning = 0;
    // Thread will exit on next iteration
}

// ============================================================================
// Example Implementation - Limiter APIs
// Note: These APIs are defined in limiter.h, not accelerator.h
// They are kept here for the example implementation to work with limiter.so
// ============================================================================

Result AddWorkerProcess(const char* deviceUUID, const char* processId) {
    (void)deviceUUID;  
    (void)processId;   
    return RESULT_SUCCESS;
}

Result CheckAndRecordMemoryOps(const char* processId, const char* deviceUUID, int64_t bytesDiff, MemoryOpRecord* record) {
    (void)processId;   
    (void)deviceUUID;  
    (void)bytesDiff;   
    
    if (!record) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    // Example: always allow, set available bytes to a large value
    record->shouldBlock = false;
    record->availableBytes = 16ULL * 1024 * 1024 * 1024; // 16GB
    return RESULT_SUCCESS;
}

Result CheckAndRecordComputeOps(const char* processId, const char* deviceUUID, uint64_t computeTokens, ComputeOpRecord* record) {
    (void)processId;      
    (void)deviceUUID;     
    (void)computeTokens;  
    
    if (!record) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    // Example: always allow, set available tokens to a large value
    record->shouldBlock = false;
    record->availableTokens = 1000000; // Large token pool
    return RESULT_SUCCESS;
}

Result FreezeWorker(const char* workerId, WorkerFreezeState* state) {
    (void)workerId;  
    if (!state) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    state->isFrozen = false;
    state->freezeTimeMs = 0;
    return RESULT_SUCCESS;
}

Result ResumeWorker(const char* workerId, WorkerFreezeState* state) {
    (void)workerId;  
    if (!state) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    state->isFrozen = false;
    state->freezeTimeMs = 0;
    return RESULT_SUCCESS;
}

Result AutoFreeze(const char* workerId, const char* deviceUUID, const char* resourceType) {
    (void)workerId;      
    (void)deviceUUID;   
    (void)resourceType; 
    return RESULT_SUCCESS;
}

Result AutoResume(const char* workerId, const char* deviceUUID, const char* resourceType) {
    (void)workerId;      
    (void)deviceUUID;   
    (void)resourceType; 
    return RESULT_SUCCESS;
}

// ============================================================================
// Example Implementation - DeviceInfo APIs
// ============================================================================

AccelResult AccelInit(void) {
    logMessage("INFO", "AccelInit called from provider example");
    return ACCEL_SUCCESS;
}

AccelResult AccelShutdown(void) {
    logMessage("INFO", "AccelShutdown called from provider example");
    return ACCEL_SUCCESS;
}

AccelResult AccelGetDeviceCount(size_t* deviceCount) {
    if (!deviceCount) {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Example: return 4 devices
    *deviceCount = 4;
    return ACCEL_SUCCESS;
}

// Helper function to initialize a single device info
static void initDeviceInfo(ExtendedDeviceInfo* info, int32_t deviceIndex) {
    // Initialize basic info
    snprintf(info->basic.uuid, sizeof(info->basic.uuid), "example-device-%d", deviceIndex);
    snprintf(info->basic.vendor, sizeof(info->basic.vendor), "STUB");
    snprintf(info->basic.model, sizeof(info->basic.model), "Example-GPU-Model");
    snprintf(info->basic.driverVersion, sizeof(info->basic.driverVersion), "1.0.0-example");
    snprintf(info->basic.firmwareVersion, sizeof(info->basic.firmwareVersion), "1.0.0-example");
    info->basic.index = deviceIndex;
    info->basic.numaNode = deviceIndex % 2;
    info->basic.totalMemoryBytes = 16ULL * 1024 * 1024 * 1024; // 16GB
    info->basic.totalComputeUnits = 108;
    info->basic.maxTflops = 312.0;
    info->basic.pcieGen = 4;
    info->basic.pcieWidth = 16;

    // Initialize properties as key-value pairs
    size_t propCount = 0;
    
    if (propCount < MAX_DEVICE_PROPERTIES) {
        snprintf(info->props.properties[propCount].key, sizeof(info->props.properties[propCount].key), "clockGraphics");
        snprintf(info->props.properties[propCount].value, sizeof(info->props.properties[propCount].value), "1410");
        propCount++;
    }
    
    if (propCount < MAX_DEVICE_PROPERTIES) {
        snprintf(info->props.properties[propCount].key, sizeof(info->props.properties[propCount].key), "clockSM");
        snprintf(info->props.properties[propCount].value, sizeof(info->props.properties[propCount].value), "1410");
        propCount++;
    }
    
    if (propCount < MAX_DEVICE_PROPERTIES) {
        snprintf(info->props.properties[propCount].key, sizeof(info->props.properties[propCount].key), "clockMem");
        snprintf(info->props.properties[propCount].value, sizeof(info->props.properties[propCount].value), "1215");
        propCount++;
    }
    
    if (propCount < MAX_DEVICE_PROPERTIES) {
        snprintf(info->props.properties[propCount].key, sizeof(info->props.properties[propCount].key), "powerLimit");
        snprintf(info->props.properties[propCount].value, sizeof(info->props.properties[propCount].value), "400");
        propCount++;
    }
    
    if (propCount < MAX_DEVICE_PROPERTIES) {
        snprintf(info->props.properties[propCount].key, sizeof(info->props.properties[propCount].key), "temperatureThreshold");
        snprintf(info->props.properties[propCount].value, sizeof(info->props.properties[propCount].value), "83");
        propCount++;
    }
    
    if (propCount < MAX_DEVICE_PROPERTIES) {
        snprintf(info->props.properties[propCount].key, sizeof(info->props.properties[propCount].key), "eccEnabled");
        snprintf(info->props.properties[propCount].value, sizeof(info->props.properties[propCount].value), "true");
        propCount++;
    }
    
    if (propCount < MAX_DEVICE_PROPERTIES) {
        snprintf(info->props.properties[propCount].key, sizeof(info->props.properties[propCount].key), "persistenceModeEnabled");
        snprintf(info->props.properties[propCount].value, sizeof(info->props.properties[propCount].value), "false");
        propCount++;
    }
    
    if (propCount < MAX_DEVICE_PROPERTIES) {
        snprintf(info->props.properties[propCount].key, sizeof(info->props.properties[propCount].key), "computeCapability");
        snprintf(info->props.properties[propCount].value, sizeof(info->props.properties[propCount].value), "8.0");
        propCount++;
    }
    
    if (propCount < MAX_DEVICE_PROPERTIES) {
        snprintf(info->props.properties[propCount].key, sizeof(info->props.properties[propCount].key), "chipType");
        snprintf(info->props.properties[propCount].value, sizeof(info->props.properties[propCount].value), "STUB");
        propCount++;
    }
    
    info->props.count = propCount;

    // Initialize virtualization capabilities
    info->virtualizationCapabilities.supportsPartitioning = true;
    info->virtualizationCapabilities.supportsSoftIsolation = true;
    info->virtualizationCapabilities.supportsHardIsolation = true;
    info->virtualizationCapabilities.supportsSnapshot = true;
    info->virtualizationCapabilities.supportsMetrics = true;
    info->virtualizationCapabilities.supportsRemoting = false;
    info->virtualizationCapabilities.maxPartitions = 7;
    info->virtualizationCapabilities.maxWorkersPerDevice = 16;

    // Initialize device node mappings
    info->deviceNodes.count = 1;
    snprintf(info->deviceNodes.nodes[0].hostPath, sizeof(info->deviceNodes.nodes[0].hostPath),
             "/dev/example%d", deviceIndex);
    snprintf(info->deviceNodes.nodes[0].guestPath, sizeof(info->deviceNodes.nodes[0].guestPath),
             "/dev/example%d", deviceIndex);
}

AccelResult AccelGetAllDevices(ExtendedDeviceInfo* devices, size_t maxCount, size_t* deviceCount) {
    if (!devices || !deviceCount || maxCount == 0) {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Example: return 4 devices (but not more than maxCount)
    size_t actualCount = 4;
    if (actualCount > maxCount) {
        actualCount = maxCount;
    }
    *deviceCount = actualCount;

    // Initialize each device
    for (size_t i = 0; i < actualCount; i++) {
        initDeviceInfo(&devices[i], (int32_t)i);
    }
    logMessage("INFO", "AccelGetAllDevices called from provider example");
    return ACCEL_SUCCESS;
}

AccelResult AccelGetAllDevicesTopology(ExtendedDeviceTopology* topology) {
    if (!topology) {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Get device count first
    size_t deviceCount = 0;
    if (AccelGetDeviceCount(&deviceCount) != ACCEL_SUCCESS || deviceCount == 0) {
        topology->deviceCount = 0;
        return ACCEL_ERROR_INTERNAL;
    }

    if (deviceCount > MAX_TOPOLOGY_DEVICES) {
        deviceCount = MAX_TOPOLOGY_DEVICES;
    }
    topology->deviceCount = deviceCount;

    // Get all devices to populate topology
    ExtendedDeviceInfo devices[MAX_TOPOLOGY_DEVICES];
    size_t actualDeviceCount = 0;
    if (AccelGetAllDevices(devices, deviceCount, &actualDeviceCount) != ACCEL_SUCCESS) {
        topology->deviceCount = 0;
        return ACCEL_ERROR_INTERNAL;
    }

    // Initialize each device topology
    for (size_t i = 0; i < actualDeviceCount && i < MAX_TOPOLOGY_DEVICES; i++) {
        DeviceTopologyInfo* dt = &topology->devices[i];
        snprintf(dt->deviceUUID, sizeof(dt->deviceUUID), "%.63s", devices[i].basic.uuid);
        dt->deviceIndex = devices[i].basic.index;
        dt->numaNode = devices[i].basic.numaNode;
        dt->peerCount = 0;

        // Initialize peer topology (simplified: all devices are at system level)
        for (size_t j = 0; j < actualDeviceCount && j < MAX_TOPOLOGY_DEVICES; j++) {
            if (i != j) {
                if (dt->peerCount < MAX_TOPOLOGY_DEVICES) {
                    DeviceTopoNode* peer = &dt->peers[dt->peerCount];
                    snprintf(peer->peerUUID, sizeof(peer->peerUUID), "%.63s", devices[j].basic.uuid);
                    peer->peerIndex = devices[j].basic.index;
                    // Same NUMA node = NUMA level, different = system level
                    if (devices[i].basic.numaNode == devices[j].basic.numaNode && devices[i].basic.numaNode >= 0) {
                        peer->topoLevel = TOPO_LEVEL_NUMA_NODE;
                    } else {
                        peer->topoLevel = TOPO_LEVEL_SYSTEM;
                    }
                    dt->peerCount++;
                }
            }
        }
    }

    logMessage("INFO", "AccelGetAllDevicesTopology called from provider example");
    return ACCEL_SUCCESS;
}

// ============================================================================
// Example Implementation - Virtualization APIs - Partitioned Isolation
// ============================================================================

// Counter for generating unique partition IDs
static int partitionCounter = 0;

AccelResult AccelAssignPartition(const char* templateId, const char* deviceUUID, PartitionResult* partitionResult) {
    if (!templateId || !deviceUUID || !partitionResult) {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    if (templateId[0] == '\0' || deviceUUID[0] == '\0') {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Example: set partition result type to environment variable
    partitionResult->type = PARTITION_TYPE_ENVIRONMENT_VARIABLE;
    partitionResult->deviceNodes.count = 0;
    // Generate unique partition UUID by appending counter to device UUID
    snprintf(partitionResult->deviceUUID, sizeof(partitionResult->deviceUUID), 
             "%s-partition-%d", deviceUUID, partitionCounter++);

    // Set example environment variables
    snprintf(partitionResult->envVars[0], sizeof(partitionResult->envVars[0]), "CUDA_VISIBLE_DEVICES=0");
    snprintf(partitionResult->envVars[1], sizeof(partitionResult->envVars[1]), "GPU_UUID=%s", deviceUUID);
    snprintf(partitionResult->envVars[2], sizeof(partitionResult->envVars[2]), "PARTITION_TEMPLATE=%s", templateId);

    logMessage("INFO", "AccelAssignPartition called from provider example");
    return ACCEL_SUCCESS;
}

AccelResult AccelRemovePartition(const char* templateId, const char* deviceUUID) {
    if (!templateId || !deviceUUID) {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Example: always succeed
    logMessage("INFO", "AccelRemovePartition called from provider example");
    return ACCEL_SUCCESS;
}

// ============================================================================
// Example Implementation - Virtualization APIs - Hard Isolation
// ============================================================================

AccelResult AccelSetMemHardLimit(const char* deviceUUID, uint64_t memoryLimitBytes) {
    if (!deviceUUID || memoryLimitBytes == 0) {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Example: always succeed
    logMessage("INFO", "AccelSetMemHardLimit called from provider example");
    return ACCEL_SUCCESS;
}

AccelResult AccelSetComputeUnitHardLimit(const char* deviceUUID, uint32_t computeUnitLimit) {
    if (!deviceUUID || computeUnitLimit == 0 || computeUnitLimit > 100) {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Example: always succeed
    logMessage("INFO", "AccelSetComputeUnitHardLimit called from provider example");
    return ACCEL_SUCCESS;
}

// ============================================================================
// Example Implementation - Virtualization APIs - Device Snapshot/Migration
// ============================================================================

AccelResult AccelSnapshot(SnapshotContext* context) {
    if (!context) {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Support both process-level and device-level snapshots
    if (context->processIds != NULL && context->processCount > 0) {
        // Process-level snapshot
        if (context->processCount > MAX_PROCESSES) {
            return ACCEL_ERROR_INVALID_PARAM;
        }

        // Example: verify processes exist (basic check)
        for (size_t i = 0; i < context->processCount; i++) {
            if (kill(context->processIds[i], 0) != 0) {
                // Process doesn't exist or no permission
                return ACCEL_ERROR_NOT_FOUND;
            }
        }
        logMessage("INFO", "AccelSnapshot (process-level) called from provider example");
    } else if (context->deviceUUID != NULL) {
        // Device-level snapshot
        logMessage("INFO", "AccelSnapshot (device-level) called from provider example");
    } else {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Example: always succeed (no actual snapshot implementation)
    return ACCEL_SUCCESS;
}

AccelResult AccelResume(SnapshotContext* context) {
    if (!context) {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Support both process-level and device-level resume
    if (context->processIds != NULL && context->processCount > 0) {
        // Process-level resume
        if (context->processCount > MAX_PROCESSES) {
            return ACCEL_ERROR_INVALID_PARAM;
        }
        logMessage("INFO", "AccelResume (process-level) called from provider example");
    } else if (context->deviceUUID != NULL) {
        // Device-level resume
        logMessage("INFO", "AccelResume (device-level) called from provider example");
    } else {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Example: always succeed (no actual resume implementation)
    return ACCEL_SUCCESS;
}

// ============================================================================
// Example Implementation - Metrics APIs
// ============================================================================

AccelResult AccelGetProcessInformation(
    ProcessInformation* processInfos,
    size_t maxCount,
    size_t* processInfoCount
) {
    if (!processInfos || !processInfoCount || maxCount == 0) {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Use mock AMD SMI-like API to get process information
    // This follows AMD SMI style: amdsmi_get_gpu_process_list
    amdsmi_proc_info_t procInfos[256];
    uint32_t maxProcs = 256;
    amdsmi_status_t status = amdsmi_get_gpu_process_list(0, &maxProcs, procInfos);
    if (status != AMDSMI_STATUS_SUCCESS && status != AMDSMI_STATUS_OUT_OF_RESOURCES) {
        *processInfoCount = 0;
        return ACCEL_SUCCESS;  // Return empty if driver_mock not initialized
    }

    // Convert mock AMD SMI process info to ProcessInformation.
    // amdsmi_get_gpu_process_list may return maxProcs > local procInfos capacity
    // (e.g. OUT_OF_RESOURCES semantics), so always clamp to the fetched buffer size.
    size_t actualCount = (size_t)maxProcs;
    const size_t fetchedCapacity = sizeof(procInfos) / sizeof(procInfos[0]);
    if (actualCount > fetchedCapacity) {
        actualCount = fetchedCapacity;
    }
    if (actualCount > maxCount) {
        actualCount = maxCount;
    }

    // Get one device info sample to calculate percentages.
    // Do not allocate a large ExtendedDeviceInfo array on stack:
    // each entry is large and can overflow the thread stack.
    ExtendedDeviceInfo device;
    size_t deviceCount = 0;
    uint64_t totalMemoryBytes = 16ULL * 1024 * 1024 * 1024; // Default 16GB
    uint64_t totalCUs = 108; // Default 108 CUs
    
    if (AccelGetAllDevices(&device, 1, &deviceCount) == ACCEL_SUCCESS && deviceCount > 0) {
        totalMemoryBytes = device.basic.totalMemoryBytes;
        totalCUs = device.basic.totalComputeUnits;
    }

    for (size_t i = 0; i < actualCount; i++) {
        ProcessInformation* info = &processInfos[i];
        memset(info, 0, sizeof(ProcessInformation));
        
        // Process ID
        snprintf(info->processId, sizeof(info->processId), "%d", (int)procInfos[i].pid);
        
        // Device UUID - try to get from device info, fallback to mock-device-0
        if (deviceCount > 0) {
            snprintf(info->deviceUUID, sizeof(info->deviceUUID), "%.63s", device.basic.uuid);
        } else {
            snprintf(info->deviceUUID, sizeof(info->deviceUUID), "%.63s", "mock-device-0");
        }
        
        // Compute utilization from CU occupancy (AMD SMI style)
        info->activeSMs = procInfos[i].cu_occupancy;
        info->totalSMs = totalCUs;
        if (totalCUs > 0) {
            info->computeUtilizationPercent = ((double)procInfos[i].cu_occupancy / (double)totalCUs) * 100.0;
        } else {
            info->computeUtilizationPercent = 0.0;
        }
        
        // Memory utilization (AMD SMI style: memory_usage.used and memory_usage.reserved)
        // Use vram_mem from memory_usage (real API has gtt_mem, cpu_mem, vram_mem)
        info->memoryUsedBytes = procInfos[i].memory_usage.vram_mem;
        info->memoryReservedBytes = 0;  // Reserved is an array in real API, not a single value
        if (totalMemoryBytes > 0) {
            info->memoryUtilizationPercent = ((double)procInfos[i].memory_usage.vram_mem / (double)totalMemoryBytes) * 100.0;
        } else {
            info->memoryUtilizationPercent = 0.0;
        }
    }

    *processInfoCount = actualCount;
    logMessage("INFO", "AccelGetProcessInformation called from provider example (AMD SMI style)");
    return ACCEL_SUCCESS;
}

AccelResult AccelGetDeviceMetrics(
    const char** deviceUUIDs,
    size_t deviceCount,
    DeviceMetrics* metrics
) {
    if (!deviceUUIDs || deviceCount == 0 || !metrics) {
        return ACCEL_ERROR_INVALID_PARAM;
    }

    // Try to get real metrics from driver_mock for first device using AMD SMI APIs
    amdsmi_engine_usage_t gpuActivity;
    amdsmi_vram_usage_t vramUsage;
    amdsmi_power_info_t powerInfo;
    int64_t temperature;
    uint64_t pcieSent, pcieReceived, pcieMaxPktSz;
    amdsmi_status_t statusActivity = amdsmi_get_gpu_activity(NULL, &gpuActivity);
    amdsmi_status_t statusVRAM = amdsmi_get_gpu_vram_usage(NULL, &vramUsage);
    amdsmi_status_t statusPower = amdsmi_get_power_info(NULL, &powerInfo);
    amdsmi_status_t statusTemp = amdsmi_get_temp_metric(NULL, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, &temperature);
    amdsmi_status_t statusPCIe = amdsmi_get_gpu_pci_throughput(NULL, &pcieSent, &pcieReceived, &pcieMaxPktSz);
    bool hasMockData = (statusActivity == AMDSMI_STATUS_SUCCESS && statusVRAM == AMDSMI_STATUS_SUCCESS &&
                       statusPower == AMDSMI_STATUS_SUCCESS && statusTemp == AMDSMI_STATUS_SUCCESS &&
                       statusPCIe == AMDSMI_STATUS_SUCCESS);

    // Fill metrics for all requested devices
    for (size_t i = 0; i < deviceCount; i++) {
        DeviceMetrics* dm = &metrics[i];
        // Copy UUID from string pointer
        if (deviceUUIDs[i] != NULL) {
            strncpy(dm->deviceUUID, deviceUUIDs[i], sizeof(dm->deviceUUID) - 1);
            dm->deviceUUID[sizeof(dm->deviceUUID) - 1] = '\0';
        } else {
            dm->deviceUUID[0] = '\0';
        }
        
        // Use mock data if available, otherwise use example data
        if (hasMockData && deviceUUIDs[i] != NULL) {
            // Use GFX utilization from AMD SMI activity counter (gfx_activity is in %)
            dm->utilizationPercent = gpuActivity.gfx_activity;
            // Convert MB to bytes (real API returns MB, but DeviceMetrics expects bytes)
            dm->memoryUsedBytes = (uint64_t)vramUsage.vram_used * 1024ULL * 1024ULL;
            // Use power from AMD SMI (socket_power is in W, convert to double)
            dm->powerUsageWatts = (double)powerInfo.socket_power;
            // Use temperature from AMD SMI (temperature is in Celsius, convert to double)
            dm->temperatureCelsius = (double)temperature;
            // Use PCIe throughput from AMD SMI (received = RX, sent = TX, both in bytes)
            dm->pcieRxBytes = pcieReceived;
            dm->pcieTxBytes = pcieSent;
        } else {
            dm->utilizationPercent = 0;
            dm->memoryUsedBytes = 0;
            dm->powerUsageWatts = 200.0 + (i * 10.0); // Example: 200-300W
            dm->temperatureCelsius = 45.0 + (i * 5.0); // Example: 45-50C
            dm->pcieRxBytes = 1024ULL * 1024 * 1024 * (i + 1); // Example: 1-4GB
            dm->pcieTxBytes = 512ULL * 1024 * 1024 * (i + 1); // Example: 0.5-2GB
        }

        // Fill extra metrics (using fixed-size array)
        size_t extraCount = 0;
        const size_t maxExtraMetrics = MAX_EXTRA_METRICS;

        // Add some example extra metrics for example
        if (extraCount < maxExtraMetrics) {
            snprintf(dm->extraMetrics[extraCount].key, sizeof(dm->extraMetrics[extraCount].key), "memoryBandwidthMBps");
            dm->extraMetrics[extraCount].value = 800.0 + (i * 50.0); // Example: 800-1200 MB/s
            extraCount++;
        }

        if (extraCount < maxExtraMetrics) {
            snprintf(dm->extraMetrics[extraCount].key, sizeof(dm->extraMetrics[extraCount].key), "tensorCoreUsagePercent");
            dm->extraMetrics[extraCount].value = 30.0 + (i * 5.0); // Example: 30-50%
            extraCount++;
        }

        if (extraCount < maxExtraMetrics) {
            snprintf(dm->extraMetrics[extraCount].key, sizeof(dm->extraMetrics[extraCount].key), "encoderUtilization");
            dm->extraMetrics[extraCount].value = 10.0 + (i * 2.0); // Example: 10-20%
            extraCount++;
        }

        if (extraCount < maxExtraMetrics) {
            snprintf(dm->extraMetrics[extraCount].key, sizeof(dm->extraMetrics[extraCount].key), "decoderUtilization");
            dm->extraMetrics[extraCount].value = 15.0 + (i * 3.0); // Example: 15-30%
            extraCount++;
        }

        dm->extraMetricsCount = extraCount;
    }

    logMessage("INFO", "AccelGetDeviceMetrics called from provider example");
    return ACCEL_SUCCESS;
}

AccelResult AccelGetVendorMountLibs(MountPath* mounts, size_t maxCount, size_t* mountCount) {
    if (!mounts || maxCount == 0 || !mountCount) {
        return ACCEL_ERROR_INVALID_PARAM;
    }
    *mountCount = 0;
    logMessage("INFO", "AccelGetVendorMountLibs called from provider example");
    return ACCEL_SUCCESS;
}

AccelResult AccelRegisterLogCallback(LogCallbackFunc callback) {
    Log = callback;
    return ACCEL_SUCCESS;
}
