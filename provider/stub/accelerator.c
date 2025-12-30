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
    logMessage("INFO", "GetAllDevices called from provider stub");
    return RESULT_SUCCESS;
}

Result GetPartitionTemplates(int32_t deviceIndex __attribute__((unused)), PartitionTemplate* templates, size_t maxCount, size_t* templateCount) {
    if (!templates || !templateCount || maxCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Stub: return 3 example templates (but not more than maxCount)
    size_t actualCount = 3;
    if (actualCount > maxCount) {
        actualCount = maxCount;
    }
    *templateCount = actualCount;

    // Template 1: 1/7 slice
    if (actualCount > 0) {
        PartitionTemplate* t1 = &templates[0];
        snprintf(t1->templateId, sizeof(t1->templateId), "mig-1g.7gb");
        snprintf(t1->name, sizeof(t1->name), "1/7 GPU Slice");
        t1->memoryBytes = 7ULL * 1024 * 1024 * 1024; // 7GB
        t1->computeUnits = 14; // 1/7 of 108 SMs
        t1->tflops = 312.0 / 7.0; // ~44.6 TFLOPS
        t1->sliceCount = 1;
        t1->isDefault = false;
        snprintf(t1->description, sizeof(t1->description), "1/7 GPU slice with 7GB memory");
    }

    // Template 2: 2/7 slice
    if (actualCount > 1) {
        PartitionTemplate* t2 = &templates[1];
        snprintf(t2->templateId, sizeof(t2->templateId), "mig-2g.14gb");
        snprintf(t2->name, sizeof(t2->name), "2/7 GPU Slice");
        t2->memoryBytes = 14ULL * 1024 * 1024 * 1024; // 14GB
        t2->computeUnits = 28; // 2/7 of 108 SMs
        t2->tflops = 312.0 * 2.0 / 7.0; // ~89.1 TFLOPS
        t2->sliceCount = 2;
        t2->isDefault = true;
        snprintf(t2->description, sizeof(t2->description), "2/7 GPU slice with 14GB memory");
    }

    // Template 3: 3/7 slice
    if (actualCount > 2) {
        PartitionTemplate* t3 = &templates[2];
        snprintf(t3->templateId, sizeof(t3->templateId), "mig-3g.21gb");
        snprintf(t3->name, sizeof(t3->name), "3/7 GPU Slice");
        t3->memoryBytes = 21ULL * 1024 * 1024 * 1024; // 21GB (stub, exceeds total)
        t3->computeUnits = 42; // 3/7 of 108 SMs
        t3->tflops = 312.0 * 3.0 / 7.0; // ~133.7 TFLOPS
        t3->sliceCount = 3;
        t3->isDefault = false;
        snprintf(t3->description, sizeof(t3->description), "3/7 GPU slice with 21GB memory");
    }

    logMessage("INFO", "GetPartitionTemplates called from provider stub");

    return RESULT_SUCCESS;
}

Result GetDeviceTopology(int32_t* deviceIndexArray, size_t deviceCount, ExtendedDeviceTopology* topology) {
    if (!deviceIndexArray || deviceCount == 0 || !topology) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    if (deviceCount > MAX_TOPOLOGY_DEVICES) {
        deviceCount = MAX_TOPOLOGY_DEVICES;
    }
    topology->deviceCount = deviceCount;

    // Initialize each device topology
    for (size_t i = 0; i < deviceCount; i++) {
        DeviceTopology* dt = &topology->devices[i];
        snprintf(dt->deviceUUID, sizeof(dt->deviceUUID), "stub-device-%d", deviceIndexArray[i]);
        dt->numaNode = deviceIndexArray[i] % 2;
    }

    // Set extended topology info
    snprintf(topology->topologyType, sizeof(topology->topologyType), "NVLink");

    logMessage("INFO", "GetDeviceTopology called from provider stub");
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

    logMessage("INFO", "AssignPartition called from provider stub");
    return true;
}

bool RemovePartition(const char* templateId, const char* deviceUUID) {
    if (!templateId || !deviceUUID) {
        return false;
    }

    // Stub: always succeed
    logMessage("INFO", "RemovePartition called from provider stub");
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
    logMessage("INFO", "SetMemHardLimit called from provider stub");
    return RESULT_SUCCESS;
}

Result SetComputeUnitHardLimit(const char* workerId, const char* deviceUUID, uint32_t computeUnitLimit) {
    if (!workerId || !deviceUUID || computeUnitLimit == 0 || computeUnitLimit > 100) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Stub: always succeed
    logMessage("INFO", "SetComputeUnitHardLimit called from provider stub");
    return RESULT_SUCCESS;
}

// ============================================================================
// Stub Implementation - Virtualization APIs - Device Snapshot/Migration
// ============================================================================

Result Snapshot(ProcessArray* processes) {
    if (!processes || processes->processCount == 0 || processes->processCount > MAX_PROCESSES) {
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
    logMessage("INFO", "Snapshot called from provider stub");
    return RESULT_SUCCESS;
}

Result Resume(ProcessArray* processes) {
    if (!processes || processes->processCount == 0 || processes->processCount > MAX_PROCESSES) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Stub: always succeed (no actual resume implementation)
    logMessage("INFO", "Resume called from provider stub");
    return RESULT_SUCCESS;
}

// ============================================================================
// Stub Implementation - Metrics APIs
// ============================================================================

Result GetProcessComputeUtilization(
    ComputeUtilization* utilizations,
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
    logMessage("INFO", "GetProcessComputeUtilization called from provider stub");
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
    logMessage("INFO", "GetProcessMemoryUtilization called from provider stub");
    return RESULT_SUCCESS;
}

Result GetDeviceMetrics(
    const DeviceUUIDEntry* deviceUUIDs,
    size_t deviceCount,
    DeviceMetrics* metrics
) {
    if (!deviceUUIDs || deviceCount == 0 || !metrics) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Fill stub data
    for (size_t i = 0; i < deviceCount; i++) {
        DeviceMetrics* dm = &metrics[i];
        // Copy UUID from DeviceUUIDEntry
        strncpy(dm->deviceUUID, deviceUUIDs[i].uuid, sizeof(dm->deviceUUID) - 1);
        dm->deviceUUID[sizeof(dm->deviceUUID) - 1] = '\0';
        
        dm->powerUsageWatts = 200.0 + (i * 10.0); // Stub: 200-300W
        dm->temperatureCelsius = 45.0 + (i * 5.0); // Stub: 45-50C
        dm->pcieRxBytes = 1024ULL * 1024 * 1024 * (i + 1); // Stub: 1-4GB
        dm->pcieTxBytes = 512ULL * 1024 * 1024 * (i + 1); // Stub: 0.5-2GB
        dm->utilizationPercent = 50 + (i * 10); // Stub: 50-90%
        dm->memoryUsedBytes = 8ULL * 1024 * 1024 * 1024; // Stub: 8GB

        // Fill extra metrics (using fixed-size array)
        size_t extraCount = 0;
        const size_t maxExtraMetrics = MAX_EXTRA_METRICS;

        // Add some example extra metrics
        if (extraCount < maxExtraMetrics) {
            snprintf(dm->extraMetrics[extraCount].key, sizeof(dm->extraMetrics[extraCount].key), "gpuUtilization");
            dm->extraMetrics[extraCount].value = 75.0 + (i * 5.0); // Stub: 75-95%
            extraCount++;
        }

        if (extraCount < maxExtraMetrics) {
            snprintf(dm->extraMetrics[extraCount].key, sizeof(dm->extraMetrics[extraCount].key), "memoryBandwidthMBps");
            dm->extraMetrics[extraCount].value = 800.0 + (i * 50.0); // Stub: 800-1200 MB/s
            extraCount++;
        }

        if (extraCount < maxExtraMetrics) {
            snprintf(dm->extraMetrics[extraCount].key, sizeof(dm->extraMetrics[extraCount].key), "tensorCoreUsagePercent");
            dm->extraMetrics[extraCount].value = 30.0 + (i * 5.0); // Stub: 30-50%
            extraCount++;
        }

        if (extraCount < maxExtraMetrics) {
            snprintf(dm->extraMetrics[extraCount].key, sizeof(dm->extraMetrics[extraCount].key), "encoderUtilization");
            dm->extraMetrics[extraCount].value = 10.0 + (i * 2.0); // Stub: 10-20%
            extraCount++;
        }

        if (extraCount < maxExtraMetrics) {
            snprintf(dm->extraMetrics[extraCount].key, sizeof(dm->extraMetrics[extraCount].key), "decoderUtilization");
            dm->extraMetrics[extraCount].value = 15.0 + (i * 3.0); // Stub: 15-30%
            extraCount++;
        }

        dm->extraMetricsCount = extraCount;
    }

    logMessage("INFO", "GetDeviceMetrics called from provider stub");
    return RESULT_SUCCESS;
}


Result GetVendorMountLibs(Mount* mounts, size_t maxCount, size_t* mountCount) {
    if (!mounts || maxCount == 0 || !mountCount) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    *mountCount = 0;
    logMessage("INFO", "GetVendorMountLibs called from provider stub");
    return RESULT_SUCCESS;
}

Result RegisterLogCallback(LogCallbackFunc callback) {
    Log = callback;
    return RESULT_SUCCESS;
}
