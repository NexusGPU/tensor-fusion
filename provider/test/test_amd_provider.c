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

#include "../accelerator.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

// Test log callback
static void testLogCallback(const char* level, const char* message) {
    printf("[%s] %s\n", level, message);
}

// Test result tracking
static int g_testsPassed = 0;
static int g_testsFailed = 0;

#define TEST_ASSERT(condition, message) \
    do { \
        if (condition) { \
            printf("  ✓ PASS: %s\n", message); \
            g_testsPassed++; \
        } else { \
            printf("  ✗ FAIL: %s\n", message); \
            g_testsFailed++; \
        } \
    } while(0)

void test_initialization() {
    printf("\n=== Test: Initialization ===\n");
    
    Result result = VirtualGPUInit();
    TEST_ASSERT(result == RESULT_SUCCESS, "VirtualGPUInit should succeed");
    
    // Test idempotent initialization
    result = VirtualGPUInit();
    TEST_ASSERT(result == RESULT_SUCCESS, "VirtualGPUInit should be idempotent");
}

void test_device_count() {
    printf("\n=== Test: Device Count ===\n");
    
    size_t deviceCount = 0;
    Result result = GetDeviceCount(&deviceCount);
    TEST_ASSERT(result == RESULT_SUCCESS, "GetDeviceCount should succeed");
    
    printf("  Found %zu AMD GPU(s)\n", deviceCount);
    TEST_ASSERT(deviceCount >= 0, "Device count should be non-negative");
    
    // Test with NULL parameter
    result = GetDeviceCount(NULL);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "GetDeviceCount with NULL should return invalid param");
}

void test_device_discovery() {
    printf("\n=== Test: Device Discovery ===\n");
    
    ExtendedDeviceInfo devices[32];
    size_t deviceCount = 0;
    
    Result result = GetAllDevices(devices, 32, &deviceCount);
    TEST_ASSERT(result == RESULT_SUCCESS, "GetAllDevices should succeed");
    
    printf("  Discovered %zu device(s)\n", deviceCount);
    
    for (size_t i = 0; i < deviceCount; i++) {
        printf("\n  Device %zu:\n", i);
        printf("    UUID: %s\n", devices[i].basic.uuid);
        printf("    Vendor: %s\n", devices[i].basic.vendor);
        printf("    Model: %s\n", devices[i].basic.model);
        printf("    Driver: %s\n", devices[i].basic.driverVersion);
        printf("    Memory: %lu bytes (%.2f GB)\n", 
               devices[i].basic.totalMemoryBytes,
               devices[i].basic.totalMemoryBytes / (1024.0 * 1024.0 * 1024.0));
        printf("    Compute Units: %lu\n", devices[i].basic.totalComputeUnits);
        printf("    Max TFLOPS: %.2f\n", devices[i].basic.maxTflops);
        printf("    PCIe: Gen%u x%u\n", devices[i].basic.pcieGen, devices[i].basic.pcieWidth);
        printf("    NUMA Node: %d\n", devices[i].basic.numaNode);
        
        TEST_ASSERT(strcmp(devices[i].basic.vendor, "AMD") == 0, "Vendor should be AMD");
        TEST_ASSERT(strlen(devices[i].basic.uuid) > 0, "UUID should not be empty");
        TEST_ASSERT(devices[i].basic.index == (int32_t)i, "Device index should match");
        
        // Check virtualization capabilities
        printf("    Capabilities:\n");
        printf("      Partitioning: %s\n", devices[i].virtualizationCapabilities.supportsPartitioning ? "Yes" : "No");
        printf("      Soft Isolation: %s\n", devices[i].virtualizationCapabilities.supportsSoftIsolation ? "Yes" : "No");
        printf("      Hard Isolation: %s\n", devices[i].virtualizationCapabilities.supportsHardIsolation ? "Yes" : "No");
        printf("      Snapshot: %s\n", devices[i].virtualizationCapabilities.supportsSnapshot ? "Yes" : "No");
        printf("      Metrics: %s\n", devices[i].virtualizationCapabilities.supportsMetrics ? "Yes" : "No");
        printf("      Remoting: %s\n", devices[i].virtualizationCapabilities.supportsRemoting ? "Yes" : "No");
        
        TEST_ASSERT(devices[i].virtualizationCapabilities.supportsMetrics == true, "Should support metrics");
        TEST_ASSERT(devices[i].virtualizationCapabilities.supportsRemoting == true, "Should support remoting");
    }
    
    // Test with NULL parameters
    result = GetAllDevices(NULL, 32, &deviceCount);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "GetAllDevices with NULL devices should fail");
    
    result = GetAllDevices(devices, 32, NULL);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "GetAllDevices with NULL count should fail");
}

void test_device_metrics() {
    printf("\n=== Test: Device Metrics ===\n");
    
    // First get devices
    ExtendedDeviceInfo devices[32];
    size_t deviceCount = 0;
    Result result = GetAllDevices(devices, 32, &deviceCount);
    
    if (result != RESULT_SUCCESS || deviceCount == 0) {
        printf("  Skipping metrics test - no devices found\n");
        return;
    }
    
    // Prepare UUID array
    const char* deviceUUIDs[32];
    for (size_t i = 0; i < deviceCount; i++) {
        deviceUUIDs[i] = devices[i].basic.uuid;
    }
    
    // Get metrics
    DeviceMetrics metrics[32];
    result = GetDeviceMetrics(deviceUUIDs, deviceCount, metrics);
    TEST_ASSERT(result == RESULT_SUCCESS, "GetDeviceMetrics should succeed");
    
    for (size_t i = 0; i < deviceCount; i++) {
        printf("\n  Metrics for device %zu (%s):\n", i, metrics[i].deviceUUID);
        printf("    Utilization: %u%%\n", metrics[i].utilizationPercent);
        printf("    Memory Used: %lu bytes (%.2f GB)\n", 
               metrics[i].memoryUsedBytes,
               metrics[i].memoryUsedBytes / (1024.0 * 1024.0 * 1024.0));
        printf("    Power: %.2f W\n", metrics[i].powerUsageWatts);
        printf("    Temperature: %.2f °C\n", metrics[i].temperatureCelsius);
        printf("    PCIe RX: %lu bytes\n", metrics[i].pcieRxBytes);
        printf("    PCIe TX: %lu bytes\n", metrics[i].pcieTxBytes);
        
        TEST_ASSERT(metrics[i].utilizationPercent <= 100, "Utilization should be <= 100%");
        TEST_ASSERT(metrics[i].powerUsageWatts >= 0, "Power should be non-negative");
    }
    
    // Test with NULL parameters
    result = GetDeviceMetrics(NULL, deviceCount, metrics);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "GetDeviceMetrics with NULL UUIDs should fail");
}

void test_process_information() {
    printf("\n=== Test: Process Information ===\n");
    
    ProcessInformation processInfos[1024];
    size_t processInfoCount = 0;
    
    Result result = GetProcessInformation(processInfos, 1024, &processInfoCount);
    TEST_ASSERT(result == RESULT_SUCCESS, "GetProcessInformation should succeed");
    
    printf("  Found %zu process(es) using GPU\n", processInfoCount);
    
    for (size_t i = 0; i < processInfoCount && i < 10; i++) {
        printf("    Process %s on device %s: %lu bytes memory\n",
               processInfos[i].processId,
               processInfos[i].deviceUUID,
               processInfos[i].memoryUsedBytes);
    }
    
    // Test with NULL parameters
    result = GetProcessInformation(NULL, 1024, &processInfoCount);
    TEST_ASSERT(result == RESULT_ERROR_INVALID_PARAM, "GetProcessInformation with NULL should fail");
}

void test_device_topology() {
    printf("\n=== Test: Device Topology ===\n");
    
    size_t deviceCount = 0;
    Result result = GetDeviceCount(&deviceCount);
    
    if (result != RESULT_SUCCESS || deviceCount == 0) {
        printf("  Skipping topology test - no devices found\n");
        return;
    }
    
    int32_t deviceIndices[32];
    for (size_t i = 0; i < deviceCount && i < 32; i++) {
        deviceIndices[i] = i;
    }
    
    ExtendedDeviceTopology topology;
    result = GetDeviceTopology(deviceIndices, deviceCount, &topology);
    TEST_ASSERT(result == RESULT_SUCCESS, "GetDeviceTopology should succeed");
    
    printf("  Topology type: %s\n", topology.topologyType);
    printf("  Devices in topology: %zu\n", topology.deviceCount);
    
    for (size_t i = 0; i < topology.deviceCount; i++) {
        printf("    Device %s: NUMA node %d\n",
               topology.devices[i].deviceUUID,
               topology.devices[i].numaNode);
    }
}

void test_vendor_mount_libs() {
    printf("\n=== Test: Vendor Mount Libs ===\n");
    
    Mount mounts[16];
    size_t mountCount = 0;
    
    Result result = GetVendorMountLibs(mounts, 16, &mountCount);
    TEST_ASSERT(result == RESULT_SUCCESS, "GetVendorMountLibs should succeed");
    
    printf("  Mount count: %zu\n", mountCount);
    for (size_t i = 0; i < mountCount; i++) {
        printf("    %s -> %s\n", mounts[i].hostPath, mounts[i].guestPath);
    }
    
    TEST_ASSERT(mountCount >= 2, "Should have at least 2 mounts (lib and bin)");
    
    // Check that ROCm paths are included
    bool hasRocmLib = false;
    bool hasRocmBin = false;
    for (size_t i = 0; i < mountCount; i++) {
        if (strstr(mounts[i].hostPath, "rocm/lib") != NULL) {
            hasRocmLib = true;
        }
        if (strstr(mounts[i].hostPath, "rocm/bin") != NULL) {
            hasRocmBin = true;
        }
    }
    
    TEST_ASSERT(hasRocmLib, "Should have ROCm lib mount");
    TEST_ASSERT(hasRocmBin, "Should have ROCm bin mount");
}

void test_unsupported_features() {
    printf("\n=== Test: Unsupported Features ===\n");
    
    // Test partition assignment (not supported for AMD)
    PartitionAssignment assignment = {0};
    bool partResult = AssignPartition(&assignment);
    TEST_ASSERT(partResult == false, "AssignPartition should return false (not supported)");
    
    partResult = RemovePartition("test", "test-uuid");
    TEST_ASSERT(partResult == false, "RemovePartition should return false (not supported)");
    
    // Test hard limits (not yet implemented)
    Result result = SetMemHardLimit("worker1", "dev-uuid", 1024);
    TEST_ASSERT(result == RESULT_ERROR_NOT_SUPPORTED, "SetMemHardLimit should return not supported");
    
    result = SetComputeUnitHardLimit("worker1", "dev-uuid", 50);
    TEST_ASSERT(result == RESULT_ERROR_NOT_SUPPORTED, "SetComputeUnitHardLimit should return not supported");
    
    // Test snapshot/resume (not supported)
    ProcessArray processes = {0};
    result = Snapshot(&processes);
    TEST_ASSERT(result == RESULT_ERROR_NOT_SUPPORTED, "Snapshot should return not supported");
    
    result = Resume(&processes);
    TEST_ASSERT(result == RESULT_ERROR_NOT_SUPPORTED, "Resume should return not supported");
}

void test_log_callback() {
    printf("\n=== Test: Log Callback ===\n");
    
    Result result = RegisterLogCallback(testLogCallback);
    TEST_ASSERT(result == RESULT_SUCCESS, "RegisterLogCallback should succeed");
    
    // Trigger some operations that should log
    size_t deviceCount = 0;
    GetDeviceCount(&deviceCount);
    
    // Unregister callback
    result = RegisterLogCallback(NULL);
    TEST_ASSERT(result == RESULT_SUCCESS, "Unregistering callback should succeed");
}

int main() {
    printf("========================================\n");
    printf("AMD Provider Test Suite\n");
    printf("========================================\n");
    
    // Register log callback for all tests
    RegisterLogCallback(testLogCallback);
    
    // Run tests
    test_initialization();
    test_device_count();
    test_device_discovery();
    test_device_metrics();
    test_process_information();
    test_device_topology();
    test_vendor_mount_libs();
    test_unsupported_features();
    test_log_callback();
    
    // Print summary
    printf("\n========================================\n");
    printf("Test Summary\n");
    printf("========================================\n");
    printf("Passed: %d\n", g_testsPassed);
    printf("Failed: %d\n", g_testsFailed);
    printf("Total:  %d\n", g_testsPassed + g_testsFailed);
    
    if (g_testsFailed == 0) {
        printf("\n✓ All tests passed!\n");
        return 0;
    } else {
        printf("\n✗ Some tests failed!\n");
        return 1;
    }
}
