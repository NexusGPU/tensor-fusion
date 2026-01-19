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

// Test VirtualGPUInit
void test_virtualGPUInit() {
    printf("\n=== Testing VirtualGPUInit ===\n");
    
    Result result = VirtualGPUInit();
    TEST_ASSERT(result == RESULT_SUCCESS, "VirtualGPUInit returns success");
}

// Test GetDeviceCount
void test_getDeviceCount() {
    printf("\n=== Testing GetDeviceCount ===\n");
    
    size_t deviceCount = 0;
    Result result = GetDeviceCount(&deviceCount);
    
    TEST_ASSERT(result == RESULT_SUCCESS, "GetDeviceCount returns success");
    TEST_ASSERT(deviceCount > 0, "Device count > 0");
    
    // Test NULL parameter
    result = GetDeviceCount(NULL);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "NULL parameter returns error");
}

// Test GetAllDevices
void test_getAllDevices() {
    printf("\n=== Testing GetAllDevices ===\n");
    
    ExtendedDeviceInfo devices[256];
    size_t deviceCount = 0;
    
    Result result = GetAllDevices(devices, 256, &deviceCount);
    
    TEST_ASSERT(result == RESULT_SUCCESS, "GetAllDevices returns success");
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
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "NULL devices returns error");
    
    result = GetAllDevices(devices, 0, &deviceCount);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "Zero maxCount returns error");
}

// Test GetDeviceTopology
void test_getDeviceTopology() {
    printf("\n=== Testing GetDeviceTopology ===\n");
    
    int32_t deviceIndices[] = {0, 1};
    size_t deviceCount = 2;
    ExtendedDeviceTopology topology;
    
    Result result = GetDeviceTopology(deviceIndices, deviceCount, &topology);
    
    TEST_ASSERT(result == RESULT_SUCCESS, "GetDeviceTopology returns success");
    TEST_ASSERT(topology.deviceCount == deviceCount, "Device count matches");
    
    if (topology.deviceCount > 0) {
        TEST_ASSERT(strlen(topology.devices[0].deviceUUID) > 0, "First device has UUID");
    }
    
    // Test invalid parameters
    result = GetDeviceTopology(NULL, deviceCount, &topology);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "NULL deviceIndexArray returns error");
    
    result = GetDeviceTopology(deviceIndices, 0, &topology);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "Zero deviceCount returns error");
}

// Test AssignPartition
void test_assignPartition() {
    printf("\n=== Testing AssignPartition ===\n");
    
    PartitionAssignment assignment;
    memset(&assignment, 0, sizeof(assignment));
    snprintf(assignment.templateId, sizeof(assignment.templateId), "mig-1g.7gb");
    snprintf(assignment.deviceUUID, sizeof(assignment.deviceUUID), "stub-device-0");
    
    bool result = AssignPartition(&assignment);
    
    TEST_ASSERT(result == true, "AssignPartition returns true");
    TEST_ASSERT(strlen(assignment.partitionUUID) > 0, "Partition UUID is assigned");
    
    // Test invalid input
    PartitionAssignment invalid;
    memset(&invalid, 0, sizeof(invalid));
    result = AssignPartition(&invalid);
    TEST_ASSERT(result == false, "Empty assignment returns false");
    
    result = AssignPartition(NULL);
    TEST_ASSERT(result == false, "NULL assignment returns false");
}

// Test RemovePartition
void test_removePartition() {
    printf("\n=== Testing RemovePartition ===\n");
    
    bool result = RemovePartition("mig-1g.7gb", "stub-device-0");
    TEST_ASSERT(result == true, "RemovePartition returns true");
    
    result = RemovePartition(NULL, "stub-device-0");
    TEST_ASSERT(result == false, "NULL templateId returns false");
    
    result = RemovePartition("mig-1g.7gb", NULL);
    TEST_ASSERT(result == false, "NULL deviceUUID returns false");
}

// Test SetMemHardLimit
void test_setMemHardLimit() {
    printf("\n=== Testing SetMemHardLimit ===\n");
    
    Result result = SetMemHardLimit("worker-1", "stub-device-0", 4ULL * 1024 * 1024 * 1024);
    TEST_ASSERT(result == RESULT_SUCCESS, "SetMemHardLimit returns success");
    
    result = SetMemHardLimit(NULL, "stub-device-0", 4ULL * 1024 * 1024 * 1024);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "NULL workerId returns error");
    
    result = SetMemHardLimit("worker-1", NULL, 4ULL * 1024 * 1024 * 1024);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "NULL deviceUUID returns error");
}

// Test SetComputeUnitHardLimit
void test_setComputeUnitHardLimit() {
    printf("\n=== Testing SetComputeUnitHardLimit ===\n");
    
    Result result = SetComputeUnitHardLimit("worker-1", "stub-device-0", 50);
    TEST_ASSERT(result == RESULT_SUCCESS, "SetComputeUnitHardLimit returns success");
    
    result = SetComputeUnitHardLimit("worker-1", "stub-device-0", 150);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "Invalid limit > 100 returns error");
    
    result = SetComputeUnitHardLimit(NULL, "stub-device-0", 50);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "NULL workerId returns error");
}

// Test GetProcessInformation (combines compute and memory utilization)
void test_getProcessInformation() {
    printf("\n=== Testing GetProcessInformation ===\n");
    
    ProcessInformation processInfos[256];
    size_t processInfoCount = 0;
    
    Result result = GetProcessInformation(processInfos, 256, &processInfoCount);
    
    TEST_ASSERT(result == RESULT_SUCCESS, "GetProcessInformation returns success");
    
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
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "NULL processInfos returns error");
    
    result = GetProcessInformation(processInfos, 0, &processInfoCount);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "Zero maxCount returns error");
}

// Test GetDeviceMetrics
void test_getDeviceMetrics() {
    printf("\n=== Testing GetDeviceMetrics ===\n");
    
    const char* deviceUUIDs[] = {"stub-device-0"};
    DeviceMetrics metrics[1];
    
    Result result = GetDeviceMetrics(deviceUUIDs, 1, metrics);
    
    TEST_ASSERT(result == RESULT_SUCCESS, "GetDeviceMetrics returns success");
    TEST_ASSERT(strlen(metrics[0].deviceUUID) > 0, "Device UUID is not empty");
    TEST_ASSERT(metrics[0].powerUsageWatts >= 0, "Power usage >= 0");
    TEST_ASSERT(metrics[0].temperatureCelsius >= 0, "Temperature >= 0");
    
    // Test invalid parameters
    result = GetDeviceMetrics(NULL, 1, metrics);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "NULL deviceUUIDs returns error");
    
    result = GetDeviceMetrics(deviceUUIDs, 0, metrics);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "Zero deviceCount returns error");
}

// Test GetVendorMountLibs
void test_getVendorMountLibs() {
    printf("\n=== Testing GetVendorMountLibs ===\n");
    
    Mount mounts[64];
    size_t mountCount = 0;
    
    Result result = GetVendorMountLibs(mounts, 64, &mountCount);
    
    TEST_ASSERT(result == RESULT_SUCCESS, "GetVendorMountLibs returns success");
    // mountCount can be 0 for example implementation
    
    // Test invalid parameters
    result = GetVendorMountLibs(NULL, 64, &mountCount);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "NULL mounts returns error");
    
    result = GetVendorMountLibs(mounts, 0, &mountCount);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "Zero maxCount returns error");
}

// Test RegisterLogCallback
void test_registerLogCallback() {
    printf("\n=== Testing RegisterLogCallback ===\n");
    
    // Test with NULL callback (unregister)
    Result result = RegisterLogCallback(NULL);
    TEST_ASSERT(result == RESULT_SUCCESS, "RegisterLogCallback with NULL returns success");
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
