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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../accelerator.h"

// Test result tracking
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(condition, message) \
    do { \
        tests_run++; \
        if (condition) { \
            tests_passed++; \
            printf("  ✓ %s\n", message); \
        } else { \
            tests_failed++; \
            printf("  ✗ %s\n", message); \
        } \
    } while (0)

// Test VGPUInit
void test_virtualGPUInit() {
    printf("\n=== Testing VGPUInit ===\n");
    
    AccelResult result = VGPUInit();
    TEST_ASSERT(result == ACCEL_SUCCESS, "VGPUInit returns success");
    
    result = VGPUShutdown();
    TEST_ASSERT(result == ACCEL_SUCCESS, "VGPUShutdown returns success");
}

// Test GetDeviceCount
void test_getDeviceCount() {
    printf("\n=== Testing GetDeviceCount ===\n");
    
    size_t deviceCount = 0;
    AccelResult result = GetDeviceCount(&deviceCount);
    
    TEST_ASSERT(result == ACCEL_SUCCESS, "GetDeviceCount returns success");
    TEST_ASSERT(deviceCount > 0, "Device count > 0");
    
    // Test NULL parameter
    result = GetDeviceCount(NULL);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL parameter returns error");
}

// Test GetAllDevices
void test_getAllDevices() {
    printf("\n=== Testing GetAllDevices ===\n");
    
    ExtendedDeviceInfo devices[256];
    size_t deviceCount = 0;
    
    AccelResult result = GetAllDevices(devices, 256, &deviceCount);
    
    TEST_ASSERT(result == ACCEL_SUCCESS, "GetAllDevices returns success");
    TEST_ASSERT(deviceCount > 0, "Device count > 0");
    
    if (deviceCount > 0) {
        TEST_ASSERT(strlen(devices[0].basic.uuid) > 0, "Device UUID is not empty");
        TEST_ASSERT(strlen(devices[0].basic.vendor) > 0, "Vendor is not empty");
        TEST_ASSERT(strlen(devices[0].basic.model) > 0, "Model is not empty");
        TEST_ASSERT(devices[0].basic.totalMemoryBytes > 0, "Total memory > 0");
        TEST_ASSERT(devices[0].basic.totalComputeUnits > 0, "Total compute units > 0");
        TEST_ASSERT(devices[0].basic.maxTflops > 0, "Max TFLOPS > 0");
        TEST_ASSERT(devices[0].virtualizationCapabilities.maxPartitions > 0, "Max partitions > 0");
    }
    
    // Test invalid parameters
    result = GetAllDevices(NULL, 256, &deviceCount);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL devices returns error");
    
    result = GetAllDevices(devices, 0, &deviceCount);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "Zero maxCount returns error");
}

// Test GetAllDevicesTopology
void test_getDeviceTopology() {
    printf("\n=== Testing GetAllDevicesTopology ===\n");
    
    ExtendedDeviceTopology topology;
    
    AccelResult result = GetAllDevicesTopology(&topology);
    
    TEST_ASSERT(result == ACCEL_SUCCESS, "GetAllDevicesTopology returns success");
    TEST_ASSERT(topology.deviceCount > 0, "Device count > 0");
    
    if (topology.deviceCount > 0) {
        TEST_ASSERT(strlen(topology.devices[0].deviceUUID) > 0, "First device has UUID");
    }
    
    // Test invalid parameters
    result = GetAllDevicesTopology(NULL);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL topology returns error");
}

// Test AssignPartition
void test_assignPartition() {
    printf("\n=== Testing AssignPartition ===\n");
    
    PartitionResult partitionResult;
    memset(&partitionResult, 0, sizeof(partitionResult));
    
    AccelResult result = AssignPartition("mig-1g.7gb", "stub-device-0", &partitionResult);
    
    TEST_ASSERT(result == ACCEL_SUCCESS, "AssignPartition returns success");
    TEST_ASSERT(strlen(partitionResult.deviceUUID) > 0, "Device UUID is set");
    TEST_ASSERT(partitionResult.type == PARTITION_TYPE_ENVIRONMENT_VARIABLE || 
                partitionResult.type == PARTITION_TYPE_DEVICE_NODE, "Partition type is valid");
    
    // Test invalid input
    result = AssignPartition(NULL, "stub-device-0", &partitionResult);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL templateId returns error");
    
    result = AssignPartition("mig-1g.7gb", NULL, &partitionResult);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL deviceUUID returns error");
    
    result = AssignPartition("mig-1g.7gb", "stub-device-0", NULL);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL partitionResult returns error");
}

// Test RemovePartition
void test_removePartition() {
    printf("\n=== Testing RemovePartition ===\n");
    
    AccelResult result = RemovePartition("mig-1g.7gb", "stub-device-0");
    TEST_ASSERT(result == ACCEL_SUCCESS, "RemovePartition returns success");
    
    result = RemovePartition(NULL, "stub-device-0");
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL templateId returns error");
    
    result = RemovePartition("mig-1g.7gb", NULL);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL deviceUUID returns error");
}

// Test SetMemHardLimit
void test_setMemHardLimit() {
    printf("\n=== Testing SetMemHardLimit ===\n");
    
    AccelResult result = SetMemHardLimit("stub-device-0", 4ULL * 1024 * 1024 * 1024);
    TEST_ASSERT(result == ACCEL_SUCCESS, "SetMemHardLimit returns success");
    
    result = SetMemHardLimit(NULL, 4ULL * 1024 * 1024 * 1024);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL deviceUUID returns error");
    
    result = SetMemHardLimit("stub-device-0", 0);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "Zero memory limit returns error");
}

// Test SetComputeUnitHardLimit
void test_setComputeUnitHardLimit() {
    printf("\n=== Testing SetComputeUnitHardLimit ===\n");
    
    AccelResult result = SetComputeUnitHardLimit("stub-device-0", 50);
    TEST_ASSERT(result == ACCEL_SUCCESS, "SetComputeUnitHardLimit returns success");
    
    result = SetComputeUnitHardLimit("stub-device-0", 150);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "Invalid limit > 100 returns error");
    
    result = SetComputeUnitHardLimit(NULL, 50);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL deviceUUID returns error");
    
    result = SetComputeUnitHardLimit("stub-device-0", 0);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "Zero limit returns error");
}

// Test GetProcessInformation (combines compute and memory utilization)
void test_getProcessInformation() {
    printf("\n=== Testing GetProcessInformation ===\n");
    
    ProcessInformation processInfos[256];
    size_t processInfoCount = 0;
    
    AccelResult result = GetProcessInformation(processInfos, 256, &processInfoCount);
    
    TEST_ASSERT(result == ACCEL_SUCCESS, "GetProcessInformation returns success");
    
    if (processInfoCount > 0) {
        // Test compute utilization fields
        TEST_ASSERT(processInfos[0].computeUtilizationPercent >= 0 && 
                   processInfos[0].computeUtilizationPercent <= 100, 
                   "Compute utilization percent in valid range");
        TEST_ASSERT(processInfos[0].activeSMs <= processInfos[0].totalSMs,
                   "Active SMs <= Total SMs");
        
        // Test memory utilization fields
        TEST_ASSERT(processInfos[0].memoryUtilizationPercent >= 0 && 
                   processInfos[0].memoryUtilizationPercent <= 100,
                   "Memory utilization percent in valid range");
    }
    
    // Test invalid parameters
    result = GetProcessInformation(NULL, 256, &processInfoCount);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL processInfos returns error");
    
    result = GetProcessInformation(processInfos, 0, &processInfoCount);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "Zero maxCount returns error");
}

// Test GetDeviceMetrics
void test_getDeviceMetrics() {
    printf("\n=== Testing GetDeviceMetrics ===\n");
    
    const char* deviceUUIDs[] = {"stub-device-0"};
    DeviceMetrics metrics[1];
    
    AccelResult result = GetDeviceMetrics(deviceUUIDs, 1, metrics);
    
    TEST_ASSERT(result == ACCEL_SUCCESS, "GetDeviceMetrics returns success");
    TEST_ASSERT(strlen(metrics[0].deviceUUID) > 0, "Device UUID is not empty");
    TEST_ASSERT(metrics[0].powerUsageWatts >= 0, "Power usage >= 0");
    TEST_ASSERT(metrics[0].temperatureCelsius >= 0, "Temperature >= 0");
    
    // Test invalid parameters
    result = GetDeviceMetrics(NULL, 1, metrics);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL deviceUUIDs returns error");
    
    result = GetDeviceMetrics(deviceUUIDs, 0, metrics);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "Zero deviceCount returns error");
}

// Test GetVendorMountLibs
void test_getVendorMountLibs() {
    printf("\n=== Testing GetVendorMountLibs ===\n");
    
    MountPath mounts[64];
    size_t mountCount = 0;
    
    AccelResult result = GetVendorMountLibs(mounts, 64, &mountCount);
    
    TEST_ASSERT(result == ACCEL_SUCCESS, "GetVendorMountLibs returns success");
    // mountCount can be 0 for example implementation
    
    // Test invalid parameters
    result = GetVendorMountLibs(NULL, 64, &mountCount);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "NULL mounts returns error");
    
    result = GetVendorMountLibs(mounts, 0, &mountCount);
    TEST_ASSERT(result == ACCEL_ERROR_INVALID_PARAM, "Zero maxCount returns error");
}

// Test RegisterLogCallback
void test_registerLogCallback() {
    printf("\n=== Testing RegisterLogCallback ===\n");
    
    // Test with NULL callback (unregister)
    AccelResult result = RegisterLogCallback(NULL);
    TEST_ASSERT(result == ACCEL_SUCCESS, "RegisterLogCallback with NULL returns success");
}

// Main test runner
int main() {
    printf("========================================\n");
    printf("Accelerator Library Test Suite\n");
    printf("========================================\n");
    
    test_virtualGPUInit();
    test_getDeviceCount();
    test_getAllDevices();
    test_getDeviceTopology();
    test_assignPartition();
    test_removePartition();
    test_setMemHardLimit();
    test_setComputeUnitHardLimit();
    test_getProcessInformation();
    test_getDeviceMetrics();
    test_getVendorMountLibs();
    test_registerLogCallback();
    
    printf("\n========================================\n");
    printf("Test Summary\n");
    printf("========================================\n");
    printf("Total tests:  %d\n", tests_run);
    printf("Passed:       %d\n", tests_passed);
    printf("Failed:       %d\n", tests_failed);
    printf("========================================\n");
    
    if (tests_failed == 0) {
        printf("All tests passed! ✓\n");
        return 0;
    } else {
        printf("Some tests failed! ✗\n");
        return 1;
    }
}
