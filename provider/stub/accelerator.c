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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <pthread.h>
#include <time.h>

// ============================================================================
// Global Variables for Limiter Thread
// ============================================================================

static const char* g_processId = "stub-process-0";
static _Atomic uint64_t g_lastComputeCallTimeMs = 0; // Last call time in milliseconds
static pthread_t g_limiterThread;
static int g_threadCreated = 0;
static volatile int g_threadRunning = 0;

// ============================================================================
// Limiter Thread Function
// ============================================================================

static void* limiterThreadFunc(void* arg __attribute__((unused))) {
    // Get first device UUID for testing
    ExtendedDeviceInfo devices[256]; // Stack-allocated buffer
    size_t deviceCount = 0;
    char deviceUUID[64] = {0};
    
    if (GetAllDevices(devices, 256, &deviceCount) != RESULT_SUCCESS || deviceCount == 0) {
        return NULL;
    }
    snprintf(deviceUUID, sizeof(deviceUUID), "%s", devices[0].basic.uuid);

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
        g_threadRunning = 0;
        return;
    }
    g_threadCreated = 1;
}

// ============================================================================
// Destructor - Cleanup Limiter Thread
// ============================================================================

__attribute__((destructor))
static void cleanupLimiterThread(void) {
    g_threadRunning = 0;
    if (g_threadCreated) {
        // Thread will exit on next iteration; join to avoid executing code after dlclose.
        pthread_join(g_limiterThread, NULL);
        g_threadCreated = 0;
    }
}

// ============================================================================
// Stub Implementation - Limiter APIs
// ============================================================================

Result AddWorkerProcess(const char* deviceUUID, const char* processId) {
    (void)deviceUUID;  // Unused in stub
    (void)processId;   // Unused in stub
    return RESULT_SUCCESS;
}

Result CheckAndRecordMemoryOps(const char* processId, const char* deviceUUID, int64_t bytesDiff, MemoryOpRecord* record) {
    (void)processId;   // Unused in stub
    (void)deviceUUID;  // Unused in stub
    (void)bytesDiff;   // Unused in stub
    
    if (!record) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    // Stub: always allow, set available bytes to a large value
    record->shouldBlock = false;
    record->availableBytes = 16ULL * 1024 * 1024 * 1024; // 16GB
    return RESULT_SUCCESS;
}

Result CheckAndRecordComputeOps(const char* processId, const char* deviceUUID, uint64_t computeTokens, ComputeOpRecord* record) {
    (void)processId;      // Unused in stub
    (void)deviceUUID;     // Unused in stub
    (void)computeTokens;  // Unused in stub
    
    if (!record) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    // Stub: always allow, set available tokens to a large value
    record->shouldBlock = false;
    record->availableTokens = 1000000; // Large token pool
    return RESULT_SUCCESS;
}

Result FreezeWorker(const char* workerId, WorkerFreezeState* state) {
    (void)workerId;  // Unused in stub
    if (!state) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    state->isFrozen = false;
    state->freezeTimeMs = 0;
    return RESULT_SUCCESS;
}

Result ResumeWorker(const char* workerId, WorkerFreezeState* state) {
    (void)workerId;  // Unused in stub
    if (!state) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    state->isFrozen = false;
    state->freezeTimeMs = 0;
    return RESULT_SUCCESS;
}

Result AutoFreeze(const char* workerId, const char* deviceUUID, const char* resourceType) {
    (void)workerId;      // Unused in stub
    (void)deviceUUID;   // Unused in stub
    (void)resourceType; // Unused in stub
    return RESULT_SUCCESS;
}

Result AutoResume(const char* workerId, const char* deviceUUID, const char* resourceType) {
    (void)workerId;      // Unused in stub
    (void)deviceUUID;   // Unused in stub
    (void)resourceType; // Unused in stub
    return RESULT_SUCCESS;
}

// ============================================================================
// Stub Implementation - DeviceInfo APIs
// ============================================================================

Result GetDeviceCount(size_t* deviceCount) {
    if (!deviceCount) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Stub: return 4 devices
    *deviceCount = 4;
    return RESULT_SUCCESS;
}

// Helper function to initialize a single device info
static void initDeviceInfo(ExtendedDeviceInfo* info, int32_t deviceIndex) {
    // Initialize basic info
    snprintf(info->basic.uuid, sizeof(info->basic.uuid), "stub-device-%d", deviceIndex);
    snprintf(info->basic.vendor, sizeof(info->basic.vendor), "STUB");
    snprintf(info->basic.model, sizeof(info->basic.model), "Stub-GPU-Model");
    snprintf(info->basic.driverVersion, sizeof(info->basic.driverVersion), "1.0.0-stub");
    snprintf(info->basic.firmwareVersion, sizeof(info->basic.firmwareVersion), "1.0.0-stub");
    info->basic.index = deviceIndex;
    info->basic.numaNode = deviceIndex % 2; // Stub: alternate NUMA nodes
    info->basic.totalMemoryBytes = 16ULL * 1024 * 1024 * 1024; // 16GB
    info->basic.totalComputeUnits = 108; // Stub: 108 SMs
    info->basic.maxTflops = 312.0; // Stub: 312 TFLOPS
    info->basic.pcieGen = 4;
    info->basic.pcieWidth = 16;

    // Initialize properties
    info->props.clockGraphics = 1410; // MHz
    info->props.clockSM = 1410; // MHz
    info->props.clockMem = 1215; // MHz
    info->props.powerLimit = 400; // W
    info->props.temperatureThreshold = 83; // C
    info->props.eccEnabled = true;
    info->props.persistenceModeEnabled = false;
    snprintf(info->props.computeCapability, sizeof(info->props.computeCapability), "8.0");
    info->props.clockAI = 0; // Not applicable for stub
    snprintf(info->props.chipType, sizeof(info->props.chipType), "STUB");

    // Initialize capabilities
    info->capabilities.supportsPartitioning = true;
    info->capabilities.supportsSoftIsolation = true;
    info->capabilities.supportsHardIsolation = true;
    info->capabilities.supportsSnapshot = true;
    info->capabilities.supportsMetrics = true;
    info->capabilities.maxPartitions = 7;
    info->capabilities.maxWorkersPerDevice = 16;

    // Initialize topology links (stub: no links)
    info->links = NULL;
    info->linkCount = 0;
}

Result GetAllDevices(ExtendedDeviceInfo* devices, size_t maxCount, size_t* deviceCount) {
    if (!devices || !deviceCount || maxCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Stub: return 4 devices (but not more than maxCount)
    size_t actualCount = 4;
    if (actualCount > maxCount) {
        actualCount = maxCount;
    }
    *deviceCount = actualCount;

    // Initialize each device
    for (size_t i = 0; i < actualCount; i++) {
        initDeviceInfo(&devices[i], (int32_t)i);
    }

    return RESULT_SUCCESS;
}

// GetPartitionTemplates was removed from the ABI

Result GetDeviceTopology(int32_t* deviceIndexArray, size_t deviceCount, ExtendedDeviceTopology* topology, size_t maxConnectionsPerDevice) {
    if (!deviceIndexArray || deviceCount == 0 || !topology || maxConnectionsPerDevice == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Note: topology->devices must be pre-allocated by caller with size >= deviceCount
    // topology->devices[i].links must be pre-allocated by caller with size >= maxConnectionsPerDevice
    if (!topology->devices) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    topology->deviceCount = deviceCount;

    // Initialize each device topology
    for (size_t i = 0; i < deviceCount; i++) {
        DeviceTopology* dt = &topology->devices[i];
        snprintf(dt->deviceUUID, sizeof(dt->deviceUUID), "stub-device-%d", deviceIndexArray[i]);
        dt->numaNode = deviceIndexArray[i] % 2;

        // Stub: create links to other devices
        size_t linkCount = (deviceCount > 1) ? (deviceCount - 1) : 0;
        if (linkCount > maxConnectionsPerDevice) {
            linkCount = maxConnectionsPerDevice;
        }

        if (linkCount > 0 && dt->links) {
            dt->linkCount = linkCount;

            size_t connIdx = 0;
            for (size_t j = 0; j < deviceCount && connIdx < linkCount; j++) {
                if (j != i) {
                    TopologyLink* link = &dt->links[connIdx];
                    snprintf(link->deviceUUID, sizeof(link->deviceUUID), "stub-device-%d", deviceIndexArray[j]);
                    link->linkType = INTERCONNECT_NVLINK;
                    snprintf(link->linkName, sizeof(link->linkName), "nvlink%zu", connIdx);
                    link->version = 3;  // NVLink 3.0
                    link->widthLanes = 4;
                    link->bandwidthMBps = 600000; // 600 GB/s (stub)
                    link->latencyNs = 100; // 100ns (stub)
                    connIdx++;
                }
            }
        } else {
            dt->links = NULL;
            dt->linkCount = 0;
        }
    }

    // Set extended topology info
    topology->primaryInterconnect = INTERCONNECT_NVLINK;
    snprintf(topology->fabricLabel, sizeof(topology->fabricLabel), "NVLink");

    return RESULT_SUCCESS;
}

// ============================================================================
// Stub Implementation - Virtualization APIs - Partitioned Isolation
// ============================================================================

bool AssignPartition(PartitionAssignment* assignment) {
    if (!assignment || assignment->templateId[0] == '\0' || assignment->deviceUUID[0] == '\0') {
        return false;
    }

    // Stub: generate a partition UUID
    // Limit string lengths to ensure output fits in 64-byte buffer:
    // "partition-" (9) + templateId (26) + "-" (1) + deviceUUID (26) + null (1) = 63 bytes
    snprintf(assignment->partitionUUID, sizeof(assignment->partitionUUID),
             "partition-%.26s-%.26s", assignment->templateId, assignment->deviceUUID);

    return true;
}

bool RemovePartition(const char* templateId, const char* deviceUUID) {
    if (!templateId || !deviceUUID) {
        return false;
    }

    // Stub: always succeed
    return true;
}

// ============================================================================
// Stub Implementation - Virtualization APIs - Hard Isolation
// ============================================================================

Result SetMemHardLimit(const char* workerId, const char* deviceUUID, uint64_t memoryLimitBytes) {
    if (!workerId || !deviceUUID || memoryLimitBytes == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Stub: always succeed
    return RESULT_SUCCESS;
}

Result SetComputeUnitHardLimit(const char* workerId, const char* deviceUUID, uint32_t computeUnitLimit) {
    if (!workerId || !deviceUUID || computeUnitLimit == 0 || computeUnitLimit > 100) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Stub: always succeed
    return RESULT_SUCCESS;
}

// ============================================================================
// Stub Implementation - Virtualization APIs - Device Snapshot/Migration
// ============================================================================

Result Snapshot(ProcessArray* processes) {
    if (!processes || !processes->processIds || processes->processCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Stub: verify processes exist (basic check)
    for (size_t i = 0; i < processes->processCount; i++) {
        if (kill(processes->processIds[i], 0) != 0) {
            // Process doesn't exist or no permission
            return RESULT_ERROR_NOT_FOUND;
        }
    }

    // Stub: always succeed (no actual snapshot implementation)
    return RESULT_SUCCESS;
}

Result Resume(ProcessArray* processes) {
    if (!processes || !processes->processIds || processes->processCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Stub: always succeed (no actual resume implementation)
    return RESULT_SUCCESS;
}

// ============================================================================
// Stub Implementation - Metrics APIs
// ============================================================================

Result GetProcessComputeUtilization(
    ComputeUtilization* utilizations,
    size_t maxCount,
    size_t maxEnginesPerProcess,
    size_t* utilizationCount
) {
    if (!utilizations || !utilizationCount || maxCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    (void)maxEnginesPerProcess;  // Unused in stub

    // TODO: Get actual device and process list from limiter
    // For now, stub implementation returns empty
    // The actual implementation should query limiter for all tracked processes
    *utilizationCount = 0;
    return RESULT_SUCCESS;
}

Result GetProcessMemoryUtilization(
    MemoryUtilization* utilizations,
    size_t maxCount,
    size_t* utilizationCount
) {
    if (!utilizations || !utilizationCount || maxCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // TODO: Get actual device and process list from limiter
    // For now, stub implementation returns empty
    // The actual implementation should query limiter for all tracked processes
    *utilizationCount = 0;
    return RESULT_SUCCESS;
}

Result GetDeviceMetrics(
    const char** deviceUUIDArray,
    size_t deviceCount,
    DeviceMetrics* metrics,
    size_t maxComputeEnginesPerDevice,
    size_t maxMemoryPoolsPerDevice,
    size_t maxExtraMetricsPerDevice
) {
    if (!deviceUUIDArray || deviceCount == 0 || !metrics) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Fill stub data
    for (size_t i = 0; i < deviceCount; i++) {
        DeviceMetrics* dm = &metrics[i];
        snprintf(dm->deviceUUID, sizeof(dm->deviceUUID), "%s", deviceUUIDArray[i]);
        dm->powerUsageWatts = 200.0 + (i * 10.0); // Stub: 200-300W
        dm->temperatureCelsius = 45.0 + (i * 5.0); // Stub: 45-50C

        // Populate compute engine utilizations
        size_t computeCount = 0;
        if (dm->compute && maxComputeEnginesPerDevice > 0) {
            // SM (general compute)
            if (computeCount < maxComputeEnginesPerDevice) {
                ComputeEngineUtilization* eng = &dm->compute[computeCount];
                eng->engineType = COMPUTE_ENGINE_GENERAL;
                eng->engineName = ENGINE_NAME_SM;
                eng->precision = COMPUTE_PRECISION_FP32;
                eng->utilizationPercent = 50.0 + (i * 10.0);  // Stub: 50-90%
                eng->activeUnits = 54 + (i * 10);
                eng->totalUnits = 108;
                eng->throughputOpsPerSec = 0;
                computeCount++;
            }
            // Tensor Core (matrix)
            if (computeCount < maxComputeEnginesPerDevice) {
                ComputeEngineUtilization* eng = &dm->compute[computeCount];
                eng->engineType = COMPUTE_ENGINE_MATRIX;
                eng->engineName = ENGINE_NAME_TENSOR_CORE;
                eng->precision = COMPUTE_PRECISION_FP16;
                eng->utilizationPercent = 30.0 + (i * 5.0);  // Stub: 30-50%
                eng->activeUnits = 200 + (i * 20);
                eng->totalUnits = 432;
                eng->throughputOpsPerSec = 0;
                computeCount++;
            }
        }
        dm->computeCount = computeCount;

        // Populate memory pool metrics
        size_t memPoolCount = 0;
        if (dm->memoryPools && maxMemoryPoolsPerDevice > 0) {
            MemoryPoolMetrics* pool = &dm->memoryPools[memPoolCount];
            pool->poolType = MEMORY_POOL_DEVICE_HBM;
            snprintf(pool->poolName, sizeof(pool->poolName), "HBM0");
            pool->totalBytes = 16ULL * 1024 * 1024 * 1024;  // 16GB
            pool->usedBytes = 8ULL * 1024 * 1024 * 1024;    // 8GB
            pool->reservedBytes = 0;
            pool->readBytesPerSec = 800ULL * 1024 * 1024 * 1024;  // 800 GB/s
            pool->writeBytesPerSec = 700ULL * 1024 * 1024 * 1024; // 700 GB/s
            pool->utilizationPercent = 50.0;
            memPoolCount++;
        }
        dm->memoryPoolCount = memPoolCount;

        // Populate extra metrics if requested
        if (dm->extraMetrics && maxExtraMetricsPerDevice > 0) {
            size_t extraCount = 0;
            if (extraCount < maxExtraMetricsPerDevice) {
                snprintf(dm->extraMetrics[extraCount].key, sizeof(dm->extraMetrics[extraCount].key), "stub_metric_1");
                dm->extraMetrics[extraCount].value = 42.0;
                extraCount++;
            }
            if (extraCount < maxExtraMetricsPerDevice) {
                snprintf(dm->extraMetrics[extraCount].key, sizeof(dm->extraMetrics[extraCount].key), "stub_metric_2");
                dm->extraMetrics[extraCount].value = 99.5;
                extraCount++;
            }
            dm->extraMetricsCount = extraCount;
        } else {
            dm->extraMetricsCount = 0;
        }
    }

    return RESULT_SUCCESS;
}

Result GetExtendedDeviceMetrics(
    const char** deviceUUIDArray,
    size_t deviceCount,
    ExtendedDeviceMetrics* metrics,
    size_t maxInterconnectPerDevice
) {
    if (!deviceUUIDArray || deviceCount == 0 || !metrics) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Fill stub data
    for (size_t i = 0; i < deviceCount; i++) {
        ExtendedDeviceMetrics* edm = &metrics[i];
        snprintf(edm->deviceUUID, sizeof(edm->deviceUUID), "%s", deviceUUIDArray[i]);

        size_t interconnectCount = 0;
        if (edm->interconnects && maxInterconnectPerDevice > 0) {
            // Add NVLink connections (stub: 6 links)
            for (size_t j = 0; j < 6 && interconnectCount < maxInterconnectPerDevice; j++) {
                InterconnectMetrics* link = &edm->interconnects[interconnectCount];
                link->linkType = INTERCONNECT_NVLINK;
                snprintf(link->linkName, sizeof(link->linkName), "nvlink%zu", j);
                link->version = 3;  // NVLink 3.0
                link->widthLanes = 4;
                link->rxBytes = 50ULL * 1024 * 1024 * 1024;  // 50 GB (stub)
                link->txBytes = 48ULL * 1024 * 1024 * 1024;  // 48 GB (stub)
                link->utilizationPercent = 60.0 + (j * 5.0);  // 60-90%
                interconnectCount++;
            }

            // Add PCIe link (stub: 1 link)
            if (interconnectCount < maxInterconnectPerDevice) {
                InterconnectMetrics* link = &edm->interconnects[interconnectCount];
                link->linkType = INTERCONNECT_PCIE;
                snprintf(link->linkName, sizeof(link->linkName), "pcie0");
                link->version = 4;  // PCIe Gen4
                link->widthLanes = 16;
                link->rxBytes = 5ULL * 1024 * 1024 * 1024;   // 5 GB (stub)
                link->txBytes = 4ULL * 1024 * 1024 * 1024;   // 4 GB (stub)
                link->utilizationPercent = 25.0;
                interconnectCount++;
            }
        }
        edm->interconnectCount = interconnectCount;
    }

    return RESULT_SUCCESS;
}

// Static paths to avoid allocation (Go side doesn't free)
static const char* gStubDriverPath = "/usr/lib/x86_64-linux-gnu";
static const char* gStubDcmiPath = "/usr/local/dcmi";

Result GetVendorMountLibs(Mount* mounts, size_t maxCount, size_t* mountCount) {
    if (!mounts || !mountCount || maxCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Static mount entries - no malloc needed
    static Mount stubMounts[2];
    stubMounts[0].hostPath = (char*)gStubDriverPath;
    stubMounts[0].guestPath = (char*)gStubDriverPath;
    stubMounts[1].hostPath = (char*)gStubDcmiPath;
    stubMounts[1].guestPath = (char*)gStubDcmiPath;

    size_t available = 2;
    size_t count = (maxCount < available) ? maxCount : available;

    for (size_t i = 0; i < count; i++) {
        mounts[i].hostPath = stubMounts[i].hostPath;
        mounts[i].guestPath = stubMounts[i].guestPath;
    }

    *mountCount = count;
    return RESULT_SUCCESS;
}

// ============================================================================
// Stub Implementation - Utility APIs
// ============================================================================

Result Log(const char* level, const char* message) {
    if (!level || !message) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Stub: print to stderr
    fprintf(stderr, "[%s] %s\n", level, message);
    fflush(stderr);

    return RESULT_SUCCESS;
}
