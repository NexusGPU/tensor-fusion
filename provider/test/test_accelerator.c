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

// Test getDeviceInfo
void test_getDeviceInfo() {
    printf("\n=== Testing getDeviceInfo ===\n");
    
    ExtendedDeviceInfo info;
    Result result = getDeviceInfo(0, &info);
    
    TEST_ASSERT(result == RESULT_SUCCESS, "getDeviceInfo returns success");
    TEST_ASSERT(strlen(info.basic.uuid) > 0, "Device UUID is not empty");
    TEST_ASSERT(strlen(info.basic.vendor) > 0, "Vendor is not empty");
    TEST_ASSERT(strlen(info.basic.model) > 0, "Model is not empty");
    TEST_ASSERT(info.basic.totalMemoryBytes > 0, "Total memory > 0");
    TEST_ASSERT(info.basic.totalComputeUnits > 0, "Total compute units > 0");
    TEST_ASSERT(info.basic.maxTflops > 0, "Max TFLOPS > 0");
    TEST_ASSERT(info.capabilities.maxPartitions > 0, "Max partitions > 0");
    
    // Test invalid device index
    result = getDeviceInfo(-1, &info);
    TEST_ASSERT(result != RESULT_SUCCESS, "Invalid device index returns error");
    
    // Cleanup
    freeExtendedDeviceInfo(&info);
}

// Test getPartitionTemplates
void test_getPartitionTemplates() {
    printf("\n=== Testing getPartitionTemplates ===\n");
    
    PartitionTemplate* templates = NULL;
    size_t templateCount = 0;
    Result result = getPartitionTemplates(0, &templates, &templateCount);
    
    TEST_ASSERT(result == RESULT_SUCCESS, "getPartitionTemplates returns success");
    TEST_ASSERT(templates != NULL, "Templates array is not NULL");
    TEST_ASSERT(templateCount > 0, "Template count > 0");
    
    if (templates && templateCount > 0) {
        TEST_ASSERT(strlen(templates[0].templateId) > 0, "First template has ID");
        TEST_ASSERT(strlen(templates[0].name) > 0, "First template has name");
        TEST_ASSERT(templates[0].memoryBytes > 0, "First template has memory");
        TEST_ASSERT(templates[0].computeUnits > 0, "First template has compute units");
    }
    
    // Cleanup
    freePartitionTemplates(templates, templateCount);
}

// Test getDeviceTopology
void test_getDeviceTopology() {
    printf("\n=== Testing getDeviceTopology ===\n");
    
    int32_t deviceIndices[] = {0, 1};
    size_t deviceCount = 2;
    ExtendedDeviceTopology topology;
    
    Result result = getDeviceTopology(deviceIndices, deviceCount, &topology);
    
    TEST_ASSERT(result == RESULT_SUCCESS, "getDeviceTopology returns success");
    TEST_ASSERT(topology.devices != NULL, "Devices array is not NULL");
    TEST_ASSERT(topology.deviceCount == deviceCount, "Device count matches");
    
    if (topology.devices && topology.deviceCount > 0) {
        TEST_ASSERT(strlen(topology.devices[0].deviceUUID) > 0, "First device has UUID");
    }
    
    // Cleanup
    freeExtendedDeviceTopology(&topology);
}

// Test assignPartition
void test_assignPartition() {
    printf("\n=== Testing assignPartition ===\n");
    
    PartitionAssignment assignment;
    snprintf(assignment.templateId, sizeof(assignment.templateId), "mig-1g.7gb");
    snprintf(assignment.deviceUUID, sizeof(assignment.deviceUUID), "stub-device-0");
    
    bool result = assignPartition(&assignment);
    
    TEST_ASSERT(result == true, "assignPartition returns true");
    TEST_ASSERT(strlen(assignment.partitionUUID) > 0, "Partition UUID is assigned");
    TEST_ASSERT(assignment.partitionOverheadBytes > 0, "Partition overhead > 0");
    
    // Test invalid input
    PartitionAssignment invalid;
    invalid.templateId[0] = '\0';
    invalid.deviceUUID[0] = '\0';
    result = assignPartition(&invalid);
    TEST_ASSERT(result == false, "Invalid assignment returns false");
}

// Test removePartition
void test_removePartition() {
    printf("\n=== Testing removePartition ===\n");
    
    bool result = removePartition("mig-1g.7gb", "stub-device-0");
    TEST_ASSERT(result == true, "removePartition returns true");
    
    result = removePartition(NULL, "stub-device-0");
    TEST_ASSERT(result == false, "NULL templateId returns false");
}

// Test setMemHardLimit
void test_setMemHardLimit() {
    printf("\n=== Testing setMemHardLimit ===\n");
    
    Result result = setMemHardLimit("worker-1", "stub-device-0", 4ULL * 1024 * 1024 * 1024);
    TEST_ASSERT(result == RESULT_SUCCESS, "setMemHardLimit returns success");
    
    result = setMemHardLimit(NULL, "stub-device-0", 4ULL * 1024 * 1024 * 1024);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "NULL workerId returns error");
}

// Test setComputeUnitHardLimit
void test_setComputeUnitHardLimit() {
    printf("\n=== Testing setComputeUnitHardLimit ===\n");
    
    Result result = setComputeUnitHardLimit("worker-1", "stub-device-0", 50);
    TEST_ASSERT(result == RESULT_SUCCESS, "setComputeUnitHardLimit returns success");
    
    result = setComputeUnitHardLimit("worker-1", "stub-device-0", 150);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "Invalid limit > 100 returns error");
}

// Test getProcessComputeUtilization
void test_getProcessComputeUtilization() {
    printf("\n=== Testing getProcessComputeUtilization ===\n");
    
    const char* deviceUUIDs[] = {"stub-device-0"};
    const char* processIds[] = {"12345"};
    ComputeUtilization* utilizations = NULL;
    size_t utilizationCount = 0;
    
    Result result = getProcessComputeUtilization(
        deviceUUIDs, 1,
        processIds, 1,
        &utilizations, &utilizationCount
    );
    
    TEST_ASSERT(result == RESULT_SUCCESS, "getProcessComputeUtilization returns success");
    TEST_ASSERT(utilizations != NULL, "Utilizations array is not NULL");
    TEST_ASSERT(utilizationCount > 0, "Utilization count > 0");
    
    if (utilizations && utilizationCount > 0) {
        TEST_ASSERT(utilizations[0].utilizationPercent >= 0 && 
                   utilizations[0].utilizationPercent <= 100, 
                   "Utilization percent in valid range");
    }
    
    freeComputeUtilizations(utilizations, utilizationCount);
}

// Test getProcessMemoryUtilization
void test_getProcessMemoryUtilization() {
    printf("\n=== Testing getProcessMemoryUtilization ===\n");
    
    const char* deviceUUIDs[] = {"stub-device-0"};
    const char* processIds[] = {"12345"};
    MemoryUtilization* utilizations = NULL;
    size_t utilizationCount = 0;
    
    Result result = getProcessMemoryUtilization(
        deviceUUIDs, 1,
        processIds, 1,
        &utilizations, &utilizationCount
    );
    
    TEST_ASSERT(result == RESULT_SUCCESS, "getProcessMemoryUtilization returns success");
    TEST_ASSERT(utilizations != NULL, "Utilizations array is not NULL");
    TEST_ASSERT(utilizationCount > 0, "Utilization count > 0");
    
    if (utilizations && utilizationCount > 0) {
        TEST_ASSERT(utilizations[0].usedBytes > 0, "Used bytes > 0");
    }
    
    freeMemoryUtilizations(utilizations, utilizationCount);
}

// Test getDeviceMetrics
void test_getDeviceMetrics() {
    printf("\n=== Testing getDeviceMetrics ===\n");
    
    const char* deviceUUIDs[] = {"stub-device-0"};
    DeviceMetrics* metrics = NULL;
    
    Result result = getDeviceMetrics(deviceUUIDs, 1, &metrics);
    
    TEST_ASSERT(result == RESULT_SUCCESS, "getDeviceMetrics returns success");
    TEST_ASSERT(metrics != NULL, "Metrics array is not NULL");
    
    if (metrics) {
        TEST_ASSERT(strlen(metrics[0].deviceUUID) > 0, "Device UUID is not empty");
        TEST_ASSERT(metrics[0].powerUsageWatts >= 0, "Power usage >= 0");
        TEST_ASSERT(metrics[0].temperatureCelsius >= 0, "Temperature >= 0");
    }
    
    freeDeviceMetrics(metrics, 1);
}

// Test getExtendedDeviceMetrics
void test_getExtendedDeviceMetrics() {
    printf("\n=== Testing getExtendedDeviceMetrics ===\n");
    
    const char* deviceUUIDs[] = {"stub-device-0"};
    ExtendedDeviceMetrics* metrics = NULL;
    
    Result result = getExtendedDeviceMetrics(deviceUUIDs, 1, &metrics);
    
    TEST_ASSERT(result == RESULT_SUCCESS, "getExtendedDeviceMetrics returns success");
    TEST_ASSERT(metrics != NULL, "Metrics array is not NULL");
    
    if (metrics) {
        TEST_ASSERT(strlen(metrics[0].deviceUUID) > 0, "Device UUID is not empty");
        TEST_ASSERT(metrics[0].nvlinkCount > 0, "NVLink count > 0");
    }
    
    freeExtendedDeviceMetrics(metrics, 1);
}

// Main test runner
int main() {
    printf("========================================\n");
    printf("Accelerator Library Test Suite\n");
    printf("========================================\n");
    
    test_getDeviceInfo();
    test_getPartitionTemplates();
    test_getDeviceTopology();
    test_assignPartition();
    test_removePartition();
    test_setMemHardLimit();
    test_setComputeUnitHardLimit();
    test_getProcessComputeUtilization();
    test_getProcessMemoryUtilization();
    test_getDeviceMetrics();
    test_getExtendedDeviceMetrics();
    
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

