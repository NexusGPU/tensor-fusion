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

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "../accelerator.h"

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

static void test_device_enumeration(void) {
    printf("\n=== Device enumeration ===\n");
    size_t count = 0;
    Result r = GetDeviceCount(&count);
    TEST_ASSERT(r == RESULT_SUCCESS, "GetDeviceCount succeeds");
    TEST_ASSERT(count > 0, "Device count > 0");

    ExtendedDeviceInfo infos[8];
    size_t returned = 0;
    r = GetAllDevices(infos, 8, &returned);
    TEST_ASSERT(r == RESULT_SUCCESS, "GetAllDevices succeeds");
    TEST_ASSERT(returned == count || returned == 8, "Returned count matches buffer limit");
    TEST_ASSERT(strlen(infos[0].basic.uuid) > 0, "First device UUID not empty");
}

// GetPartitionTemplates was removed from ABI

static void test_device_topology(void) {
    printf("\n=== Device topology ===\n");
    int32_t indices[] = {0, 1};
    TopologyLink linkBuf[2][2];
    DeviceTopology topoDevices[2];
    memset(topoDevices, 0, sizeof(topoDevices));
    topoDevices[0].links = linkBuf[0];
    topoDevices[1].links = linkBuf[1];
    ExtendedDeviceTopology topo = {
        .devices = topoDevices,
    };

    Result r = GetDeviceTopology(indices, 2, &topo, 2);
    TEST_ASSERT(r == RESULT_SUCCESS, "GetDeviceTopology succeeds");
    TEST_ASSERT(topo.deviceCount == 2, "Topology deviceCount matches");
    TEST_ASSERT(topo.devices[0].linkCount <= 2, "Link count within bound");
}

static void test_partition_lifecycle(void) {
    printf("\n=== Partition lifecycle ===\n");
    PartitionAssignment assignment = {0};
    snprintf(assignment.templateId, sizeof(assignment.templateId), "mig-1g.7gb");
    snprintf(assignment.deviceUUID, sizeof(assignment.deviceUUID), "stub-device-0");

    bool ok = AssignPartition(&assignment);
    TEST_ASSERT(ok == true, "AssignPartition succeeds");
    TEST_ASSERT(strlen(assignment.partitionUUID) > 0, "Partition UUID set");

    ok = RemovePartition("mig-1g.7gb", "stub-device-0");
    TEST_ASSERT(ok == true, "RemovePartition succeeds");

    PartitionAssignment invalid = {0};
    ok = AssignPartition(&invalid);
    TEST_ASSERT(ok == false, "Invalid assign rejected");
}

static void test_hard_limits(void) {
    printf("\n=== Hard limits ===\n");
    Result r = SetMemHardLimit("worker-1", "stub-device-0", 4ULL * 1024 * 1024 * 1024);
    TEST_ASSERT(r == RESULT_SUCCESS, "SetMemHardLimit succeeds for stub");
    r = SetMemHardLimit(NULL, "stub-device-0", 1);
    TEST_ASSERT(r == RESULT_ERROR_INVALID_PARAM, "SetMemHardLimit validates input");

    r = SetComputeUnitHardLimit("worker-1", "stub-device-0", 50);
    TEST_ASSERT(r == RESULT_SUCCESS, "SetComputeUnitHardLimit succeeds for stub");
    r = SetComputeUnitHardLimit("worker-1", "stub-device-0", 150);
    TEST_ASSERT(r == RESULT_ERROR_INVALID_PARAM, "Compute limit validated");
}

static void test_process_metrics(void) {
    printf("\n=== Process metrics ===\n");
    ComputeUtilization cu[4];
    ComputeEngineUtilization engineBuf[4][4];
    memset(cu, 0, sizeof(cu));
    for (int i = 0; i < 4; i++) {
        cu[i].engines = engineBuf[i];
        cu[i].engineCount = 0;
    }
    size_t cuCount = 0;
    Result r = GetProcessComputeUtilization(cu, 4, 4, &cuCount);
    TEST_ASSERT(r == RESULT_SUCCESS, "GetProcessComputeUtilization succeeds");
    TEST_ASSERT(cuCount == 0, "No processes tracked in stub");

    MemoryUtilization mu[4];
    size_t muCount = 0;
    r = GetProcessMemoryUtilization(mu, 4, &muCount);
    TEST_ASSERT(r == RESULT_SUCCESS, "GetProcessMemoryUtilization succeeds");
    TEST_ASSERT(muCount == 0, "No processes tracked in stub");
}

static void test_device_metrics(void) {
    printf("\n=== Device metrics ===\n");
    const char* deviceUUIDs[] = {"stub-device-0"};

    // Allocate buffers for compute engines, memory pools, and extra metrics
    ComputeEngineUtilization computeBuf[4];
    MemoryPoolMetrics memPoolBuf[2];
    ExtraMetric extraBuf[4];

    DeviceMetrics dm[1];
    memset(dm, 0, sizeof(dm));
    dm[0].compute = computeBuf;
    dm[0].computeCount = 0;
    dm[0].memoryPools = memPoolBuf;
    dm[0].memoryPoolCount = 0;
    dm[0].extraMetrics = extraBuf;
    dm[0].extraMetricsCount = 0;

    Result r = GetDeviceMetrics(deviceUUIDs, 1, dm, 4, 2, 4);
    TEST_ASSERT(r == RESULT_SUCCESS, "GetDeviceMetrics succeeds");
    TEST_ASSERT(strlen(dm[0].deviceUUID) > 0, "Metrics device UUID set");
    TEST_ASSERT(dm[0].computeCount > 0, "Compute engines populated");
    TEST_ASSERT(dm[0].memoryPoolCount > 0, "Memory pools populated");

    // Test extended device metrics
    InterconnectMetrics interconnectBuf[8];
    ExtendedDeviceMetrics edm[1];
    memset(edm, 0, sizeof(edm));
    edm[0].interconnects = interconnectBuf;
    edm[0].interconnectCount = 0;

    r = GetExtendedDeviceMetrics(deviceUUIDs, 1, edm, 8);
    TEST_ASSERT(r == RESULT_SUCCESS, "GetExtendedDeviceMetrics succeeds");
    TEST_ASSERT(edm[0].interconnectCount > 0, "Interconnect count populated");
}

int main(void) {
    printf("========================================\n");
    printf("Accelerator Library Test Suite\n");
    printf("========================================\n");

    test_device_enumeration();
    // test_partition_templates removed from ABI
    test_device_topology();
    test_partition_lifecycle();
    test_hard_limits();
    test_process_metrics();
    test_device_metrics();

    printf("\n========================================\n");
    printf("Test Summary\n");
    printf("========================================\n");
    printf("Total tests:  %d\n", tests_run);
    printf("Passed:       %d\n", tests_passed);
    printf("Failed:       %d\n", tests_failed);
    printf("========================================\n");

    return (tests_failed == 0) ? 0 : 1;
}
