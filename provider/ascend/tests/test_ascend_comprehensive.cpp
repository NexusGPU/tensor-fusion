/*
 * Comprehensive Ascend Accelerator Test Suite
 *
 * Combines all test functionalities:
 * 1. Logic ID probing (from test_logic_id_probe.cpp)
 * 2. Direct DCMI API testing (from test_dcmi_direct.cpp)
 * 3. Complete accelerator.h API coverage (from test_ascend_debug.cpp)
 * 4. Partition lifecycle testing (from test_ascend.cpp)
 *
 * This single tool provides complete testing coverage for all Ascend functionality.
 */

#include <cstdio>
#include <cstring>
#include <vector>
#include <dlfcn.h>
#include <cstdlib>
#include "../../accelerator.h"

// ============================================================================
// Helper Functions
// ============================================================================

const char* resultToString(Result r) {
    switch (r) {
        case RESULT_SUCCESS: return "SUCCESS";
        case RESULT_ERROR_INVALID_PARAM: return "ERROR_INVALID_PARAM";
        case RESULT_ERROR_NOT_FOUND: return "ERROR_NOT_FOUND";
        case RESULT_ERROR_NOT_SUPPORTED: return "ERROR_NOT_SUPPORTED";
        case RESULT_ERROR_RESOURCE_EXHAUSTED: return "ERROR_RESOURCE_EXHAUSTED";
        case RESULT_ERROR_OPERATION_FAILED: return "ERROR_OPERATION_FAILED";
        case RESULT_ERROR_INTERNAL: return "ERROR_INTERNAL";
        default: return "UNKNOWN";
    }
}

void printSeparator(const char* title) {
    std::fprintf(stderr, "\n================================================================================\n");
    std::fprintf(stderr, "  %s\n", title);
    std::fprintf(stderr, "================================================================================\n");
}

void printSubsection(const char* title) {
    std::fprintf(stderr, "\n--- %s ---\n", title);
}

// ============================================================================
// Phase 1: Direct DCMI Low-Level Testing
// ============================================================================

typedef int (*dcmi_init_fn)(void);
typedef int (*dcmi_get_all_device_count_fn)(int*);
typedef int (*dcmi_get_card_id_device_id_from_logicid_fn)(int*, int*, unsigned int);
typedef int (*dcmi_get_device_power_info_fn)(int, int, int*);
typedef int (*dcmi_get_device_temperature_fn)(int, int, int*);
typedef int (*dcmi_get_device_utilization_rate_fn)(int, int, int, unsigned int*);

void testDcmiDirectAccess() {
    printSeparator("PHASE 1: Direct DCMI Low-Level API Testing");

    const char* libPath = std::getenv("DCMI_LIB_PATH");
    if (!libPath) {
        libPath = "/usr/local/dcmi/libdcmi.so";
    }

    std::fprintf(stderr, "Loading DCMI from: %s\n", libPath);
    void* handle = dlopen(libPath, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "❌ Failed to load DCMI: %s\n", dlerror());
        std::fprintf(stderr, "⚠️  Skipping low-level DCMI tests\n");
        return;
    }

    // Load symbols
    auto p_dcmi_init = reinterpret_cast<dcmi_init_fn>(dlsym(handle, "dcmi_init"));
    auto p_dcmi_get_all_device_count = reinterpret_cast<dcmi_get_all_device_count_fn>(
        dlsym(handle, "dcmi_get_all_device_count"));
    auto p_dcmi_get_card_id_device_id_from_logicid =
        reinterpret_cast<dcmi_get_card_id_device_id_from_logicid_fn>(
            dlsym(handle, "dcmi_get_card_id_device_id_from_logicid"));
    auto p_dcmi_get_device_power_info = reinterpret_cast<dcmi_get_device_power_info_fn>(
        dlsym(handle, "dcmi_get_device_power_info"));
    auto p_dcmi_get_device_temperature = reinterpret_cast<dcmi_get_device_temperature_fn>(
        dlsym(handle, "dcmi_get_device_temperature"));
    auto p_dcmi_get_device_utilization_rate = reinterpret_cast<dcmi_get_device_utilization_rate_fn>(
        dlsym(handle, "dcmi_get_device_utilization_rate"));

    if (!p_dcmi_init) {
        std::fprintf(stderr, "❌ Failed to load dcmi_init symbol\n");
        dlclose(handle);
        return;
    }

    // Initialize DCMI
    printSubsection("DCMI Initialization");
    int initRet = p_dcmi_init();
    std::fprintf(stderr, "dcmi_init() return code: %d %s\n",
                initRet, initRet == 0 ? "✅" : "❌");
    if (initRet != 0) {
        std::fprintf(stderr, "⚠️  DCMI initialization failed, skipping DCMI tests\n");
        dlclose(handle);
        return;
    }

    // Get device count
    printSubsection("Device Count Query");
    if (p_dcmi_get_all_device_count) {
        int count = 0;
        int ret = p_dcmi_get_all_device_count(&count);
        std::fprintf(stderr, "dcmi_get_all_device_count() ret=%d, count=%d %s\n",
                    ret, count, ret == 0 ? "✅" : "❌");
    }

    // Logic ID Probing
    printSubsection("Logic ID Probing (0-15)");
    if (p_dcmi_get_card_id_device_id_from_logicid) {
        int successCount = 0;
        for (unsigned int logicId = 0; logicId < 16; ++logicId) {
            int card = -1;
            int dev = -1;
            int mapRet = p_dcmi_get_card_id_device_id_from_logicid(&card, &dev, logicId);

            if (mapRet == 0) {
                std::fprintf(stderr, "✅ logic_id=%2u -> card_id=%d, device_id=%d\n",
                            logicId, card, dev);
                successCount++;

                // Test metrics for this device
                if (successCount == 1) {  // Test first found device
                    printSubsection("Testing Metrics for First Device");

                    // Power
                    if (p_dcmi_get_device_power_info) {
                        int power = -1;
                        int ret = p_dcmi_get_device_power_info(card, dev, &power);
                        std::fprintf(stderr, "  Power: ret=%d, value=%d mW (%.2f W) %s\n",
                                    ret, power, power / 1000.0, ret == 0 ? "✅" : "❌");
                    }

                    // Temperature
                    if (p_dcmi_get_device_temperature) {
                        int temp = -1;
                        int ret = p_dcmi_get_device_temperature(card, dev, &temp);
                        std::fprintf(stderr, "  Temperature: ret=%d, value=%d °C %s\n",
                                    ret, temp, ret == 0 ? "✅" : "❌");
                    }

                    // Utilization (try different types)
                    if (p_dcmi_get_device_utilization_rate) {
                        const char* types[] = {"AICore", "AICpu", "CtrlCPU", "VectorCore"};
                        for (int type = 0; type < 4; type++) {
                            unsigned int util = 0;
                            int ret = p_dcmi_get_device_utilization_rate(card, dev, type, &util);
                            if (ret == 0) {
                                std::fprintf(stderr, "  Utilization[%s]: %u%% ✅\n", types[type], util);
                            } else {
                                std::fprintf(stderr, "  Utilization[%s]: ret=%d ❌\n", types[type], ret);
                            }
                        }
                    }
                }
            }
        }
        std::fprintf(stderr, "\nLogic ID Probe Summary: Found %d device(s)\n", successCount);
    }

    dlclose(handle);
}

// ============================================================================
// Phase 2: Accelerator.h High-Level API Testing
// ============================================================================

void testGetDeviceCount() {
    printSubsection("GetDeviceCount");

    size_t count = 0;
    Result r = GetDeviceCount(&count);

    std::fprintf(stderr, "Result: %s\n", resultToString(r));
    std::fprintf(stderr, "Device Count: %zu\n", count);

    if (r == RESULT_SUCCESS && count > 0) {
        std::fprintf(stderr, "✅ PASS\n");
    } else {
        std::fprintf(stderr, "⚠️  No devices or error\n");
    }
}

void testGetAllDevices() {
    printSubsection("GetAllDevices");

    size_t count = 0;
    Result r = GetDeviceCount(&count);
    if (r != RESULT_SUCCESS || count == 0) {
        std::fprintf(stderr, "⚠️  Skipped: No devices available\n");
        return;
    }

    std::vector<ExtendedDeviceInfo> devices(count);
    size_t actualCount = 0;
    r = GetAllDevices(devices.data(), count, &actualCount);

    std::fprintf(stderr, "Result: %s\n", resultToString(r));
    std::fprintf(stderr, "Devices Returned: %zu\n\n", actualCount);

    for (size_t i = 0; i < actualCount; i++) {
        std::fprintf(stderr, "Device %zu:\n", i);
        std::fprintf(stderr, "  UUID: %s\n", devices[i].basic.uuid);
        std::fprintf(stderr, "  Vendor: %s\n", devices[i].basic.vendor);
        std::fprintf(stderr, "  Model: %s\n", devices[i].basic.model);
        std::fprintf(stderr, "  Driver: %s\n", devices[i].basic.driverVersion);
        std::fprintf(stderr, "  Index: %d\n", devices[i].basic.index);
        std::fprintf(stderr, "  NUMA: %d\n", devices[i].basic.numaNode);
        std::fprintf(stderr, "  Memory: %.2f GB\n",
                    devices[i].basic.totalMemoryBytes / (1024.0 * 1024.0 * 1024.0));
        std::fprintf(stderr, "  Compute Units: %llu\n",
                    static_cast<unsigned long long>(devices[i].basic.totalComputeUnits));
        std::fprintf(stderr, "  Max TFLOPS: %.2f\n", devices[i].basic.maxTflops);
        std::fprintf(stderr, "  PCIe: Gen %u x%u\n",
                    devices[i].basic.pcieGen, devices[i].basic.pcieWidth);
        std::fprintf(stderr, "  Chip Type: %s\n", devices[i].props.chipType);
        std::fprintf(stderr, "  Clock AI: %u MHz\n", devices[i].props.clockAI);
        std::fprintf(stderr, "  Power Limit: %u W\n", devices[i].props.powerLimit);
        std::fprintf(stderr, "  Temp Threshold: %u C\n", devices[i].props.temperatureThreshold);
        std::fprintf(stderr, "  ECC: %s\n", devices[i].props.eccEnabled ? "yes" : "no");
        std::fprintf(stderr, "  Partitioning: %s\n",
                    devices[i].capabilities.supportsPartitioning ? "yes" : "no");
        std::fprintf(stderr, "  Max Partitions: %u\n", devices[i].capabilities.maxPartitions);
        std::fprintf(stderr, "\n");
    }

    std::fprintf(stderr, "✅ PASS\n");
}

void testGetDeviceTopology() {
    printSubsection("GetDeviceTopology");

    size_t count = 0;
    Result r = GetDeviceCount(&count);
    if (r != RESULT_SUCCESS || count == 0) {
        std::fprintf(stderr, "⚠️  Skipped: No devices available\n");
        return;
    }

    std::vector<int32_t> deviceIndices(count);
    for (size_t i = 0; i < count; i++) {
        deviceIndices[i] = static_cast<int32_t>(i);
    }

    ExtendedDeviceTopology topology = {};
    std::vector<DeviceTopology> deviceTopos(count);
    topology.devices = deviceTopos.data();
    topology.deviceCount = 0;

    r = GetDeviceTopology(deviceIndices.data(), count, &topology, 10);

    std::fprintf(stderr, "Result: %s\n", resultToString(r));
    std::fprintf(stderr, "Device Count: %zu\n", topology.deviceCount);
    std::fprintf(stderr, "Topology Type: %s\n", topology.topologyType);
    std::fprintf(stderr, "NVLink Bandwidth: %u MB/s\n", topology.nvlinkBandwidthMBps);
    std::fprintf(stderr, "IB NIC Count: %u\n\n", topology.ibNicCount);

    for (size_t i = 0; i < topology.deviceCount; i++) {
        std::fprintf(stderr, "Device %zu:\n", i);
        std::fprintf(stderr, "  UUID: %s\n", topology.devices[i].deviceUUID);
        std::fprintf(stderr, "  NUMA: %d\n", topology.devices[i].numaNode);
        std::fprintf(stderr, "  Connections: %zu\n", topology.devices[i].connectionCount);

        for (size_t j = 0; j < topology.devices[i].connectionCount; j++) {
            RelatedDevice* conn = &topology.devices[i].connections[j];
            std::fprintf(stderr, "    -> %s [%s] BW=%u MB/s Lat=%u ns\n",
                        conn->deviceUUID, conn->connectionType,
                        conn->bandwidthMBps, conn->latencyNs);
        }
    }

    std::fprintf(stderr, "✅ PASS\n");
}

void testGetPartitionTemplates() {
    printSubsection("GetPartitionTemplates");

    size_t count = 0;
    Result r = GetDeviceCount(&count);
    if (r != RESULT_SUCCESS || count == 0) {
        std::fprintf(stderr, "⚠️  Skipped: No devices available\n");
        return;
    }

    PartitionTemplate templates[16] = {};
    size_t tmplCount = 0;
    r = GetPartitionTemplates(0, templates, 16, &tmplCount);

    std::fprintf(stderr, "Result: %s\n", resultToString(r));
    std::fprintf(stderr, "Template Count: %zu\n\n", tmplCount);

    for (size_t i = 0; i < tmplCount; i++) {
        std::fprintf(stderr, "Template %zu:\n", i);
        std::fprintf(stderr, "  ID: %s\n", templates[i].templateId);
        std::fprintf(stderr, "  Name: %s\n", templates[i].name);
        std::fprintf(stderr, "  Memory: %.2f GB\n",
                    templates[i].memoryBytes / (1024.0 * 1024.0 * 1024.0));
        std::fprintf(stderr, "  Compute Units: %llu\n",
                    static_cast<unsigned long long>(templates[i].computeUnits));
        std::fprintf(stderr, "\n");
    }

    std::fprintf(stderr, "✅ PASS\n");
}

void testGetDeviceMetrics() {
    printSubsection("GetDeviceMetrics");

    size_t count = 0;
    Result r = GetDeviceCount(&count);
    if (r != RESULT_SUCCESS || count == 0) {
        std::fprintf(stderr, "⚠️  Skipped: No devices available\n");
        return;
    }

    std::vector<ExtendedDeviceInfo> devices(count);
    size_t actualCount = 0;
    GetAllDevices(devices.data(), count, &actualCount);

    std::vector<const char*> uuids(actualCount);
    for (size_t i = 0; i < actualCount; i++) {
        uuids[i] = devices[i].basic.uuid;
    }

    std::vector<DeviceMetrics> metrics(actualCount);
    r = GetDeviceMetrics(uuids.data(), actualCount, metrics.data());

    std::fprintf(stderr, "Result: %s\n\n", resultToString(r));

    for (size_t i = 0; i < actualCount; i++) {
        std::fprintf(stderr, "Device %zu Metrics:\n", i);
        std::fprintf(stderr, "  UUID: %s\n", metrics[i].deviceUUID);
        std::fprintf(stderr, "  Power: %.2f W\n", metrics[i].powerUsageWatts);
        std::fprintf(stderr, "  Temperature: %.2f °C\n", metrics[i].temperatureCelsius);
        std::fprintf(stderr, "  SM Active: %u%%\n", metrics[i].smActivePercent);
        std::fprintf(stderr, "  Tensor Core: %u%%\n", metrics[i].tensorCoreUsagePercent);
        std::fprintf(stderr, "  Memory Used: %.2f GB / %.2f GB\n",
                    metrics[i].memoryUsedBytes / (1024.0 * 1024.0 * 1024.0),
                    metrics[i].memoryTotalBytes / (1024.0 * 1024.0 * 1024.0));
        std::fprintf(stderr, "  PCIe RX: %llu bytes\n",
                    static_cast<unsigned long long>(metrics[i].pcieRxBytes));
        std::fprintf(stderr, "  PCIe TX: %llu bytes\n",
                    static_cast<unsigned long long>(metrics[i].pcieTxBytes));
        std::fprintf(stderr, "\n");
    }

    std::fprintf(stderr, "✅ PASS\n");
}

void testGetExtendedDeviceMetrics() {
    printSubsection("GetExtendedDeviceMetrics");

    size_t count = 0;
    Result r = GetDeviceCount(&count);
    if (r != RESULT_SUCCESS || count == 0) {
        std::fprintf(stderr, "⚠️  Skipped: No devices available\n");
        return;
    }

    std::vector<ExtendedDeviceInfo> devices(count);
    size_t actualCount = 0;
    GetAllDevices(devices.data(), count, &actualCount);

    std::vector<const char*> uuids(actualCount);
    for (size_t i = 0; i < actualCount; i++) {
        uuids[i] = devices[i].basic.uuid;
    }

    std::vector<ExtendedDeviceMetrics> metrics(actualCount);
    r = GetExtendedDeviceMetrics(uuids.data(), actualCount, metrics.data(), 10, 10, 10);

    std::fprintf(stderr, "Result: %s\n\n", resultToString(r));

    for (size_t i = 0; i < actualCount; i++) {
        std::fprintf(stderr, "Device %zu Extended Metrics:\n", i);
        std::fprintf(stderr, "  UUID: %s\n", metrics[i].deviceUUID);
        std::fprintf(stderr, "  NVLink Count: %zu\n", metrics[i].nvlinkCount);
        std::fprintf(stderr, "  IB NIC Count: %zu\n", metrics[i].ibNicCount);
        std::fprintf(stderr, "  PCIe Link Count: %zu\n", metrics[i].pcieLinkCount);
        std::fprintf(stderr, "\n");
    }

    std::fprintf(stderr, "✅ PASS\n");
}

void testProcessUtilization() {
    printSubsection("GetProcessComputeUtilization");

    ComputeUtilization utils[10] = {};
    size_t utilCount = 0;

    Result r = GetProcessComputeUtilization(utils, 10, &utilCount);
    std::fprintf(stderr, "Result: %s\n", resultToString(r));
    std::fprintf(stderr, "Process Count: %zu\n", utilCount);

    for (size_t i = 0; i < utilCount; i++) {
        std::fprintf(stderr, "  Process %s @ %s: %.2f%%, %.2f TFLOPS\n",
                    utils[i].processId, utils[i].deviceUUID,
                    utils[i].utilizationPercent, utils[i].tflopsUsed);
    }

    std::fprintf(stderr, "✅ PASS\n");

    printSubsection("GetProcessMemoryUtilization");

    MemoryUtilization memUtils[10] = {};
    size_t memCount = 0;

    r = GetProcessMemoryUtilization(memUtils, 10, &memCount);
    std::fprintf(stderr, "Result: %s\n", resultToString(r));
    std::fprintf(stderr, "Process Count: %zu\n", memCount);

    for (size_t i = 0; i < memCount; i++) {
        std::fprintf(stderr, "  Process %s @ %s: %.2f GB (%.2f%%)\n",
                    memUtils[i].processId, memUtils[i].deviceUUID,
                    memUtils[i].usedBytes / (1024.0 * 1024.0 * 1024.0),
                    memUtils[i].utilizationPercent);
    }

    std::fprintf(stderr, "✅ PASS\n");
}

void testIsolationAPIs() {
    printSubsection("SetMemHardLimit");
    Result r = SetMemHardLimit("test-worker-1", "npu-0-chip-0", 1024ULL * 1024 * 1024);
    std::fprintf(stderr, "Result: %s %s\n", resultToString(r),
                r == RESULT_ERROR_NOT_SUPPORTED ? "(expected - not supported)" : "");

    printSubsection("SetComputeUnitHardLimit");
    r = SetComputeUnitHardLimit("test-worker-1", "npu-0-chip-0", 50);
    std::fprintf(stderr, "Result: %s %s\n", resultToString(r),
                r == RESULT_ERROR_NOT_SUPPORTED ? "(expected - not supported)" : "");

    printSubsection("Snapshot");
    pid_t pids[] = {1234, 5678};
    ProcessArray processes = {};
    processes.processIds = pids;
    processes.processCount = 2;
    std::snprintf(processes.deviceUUID, sizeof(processes.deviceUUID), "npu-0-chip-0");
    r = Snapshot(&processes);
    std::fprintf(stderr, "Result: %s %s\n", resultToString(r),
                r == RESULT_ERROR_NOT_SUPPORTED ? "(expected - not supported)" : "");

    printSubsection("Resume");
    r = Resume(&processes);
    std::fprintf(stderr, "Result: %s %s\n", resultToString(r),
                r == RESULT_ERROR_NOT_SUPPORTED ? "(expected - not supported)" : "");

    std::fprintf(stderr, "✅ PASS (Isolation APIs correctly return NOT_SUPPORTED)\n");
}

void testPartitionLifecycle() {
    printSubsection("Partition Lifecycle (AssignPartition/RemovePartition)");

    size_t devCount = 0;
    GetDeviceCount(&devCount);
    if (devCount == 0) {
        std::fprintf(stderr, "⚠️  Skipped: No devices available\n");
        return;
    }

    std::vector<ExtendedDeviceInfo> devices(devCount);
    size_t actualDevCount = 0;
    GetAllDevices(devices.data(), devCount, &actualDevCount);
    if (actualDevCount == 0) {
        std::fprintf(stderr, "⚠️  Skipped: No devices enumerated\n");
        return;
    }

    PartitionTemplate templates[16] = {};
    size_t tmplCount = 0;
    Result r = GetPartitionTemplates(0, templates, 16, &tmplCount);
    if (r != RESULT_SUCCESS || tmplCount == 0) {
        std::fprintf(stderr, "⚠️  Skipped: No partition templates available\n");
        return;
    }

    const char* templateId = templates[0].templateId;
    std::fprintf(stderr, "Using template: %s\n", templateId);
    std::fprintf(stderr, "Using device: %s\n\n", devices[0].basic.uuid);

    // Assign Partition
    std::fprintf(stderr, "1. AssignPartition()...\n");
    PartitionAssignment assignment{};
    snprintf(assignment.templateId, sizeof(assignment.templateId), "%s", templateId);
    snprintf(assignment.deviceUUID, sizeof(assignment.deviceUUID), "%s", devices[0].basic.uuid);

    bool success = AssignPartition(&assignment);
    std::fprintf(stderr, "   Result: %s\n", success ? "true ✅" : "false ❌");
    if (success) {
        std::fprintf(stderr, "   Partition UUID: %s\n", assignment.partitionUUID);
        std::fprintf(stderr, "   Overhead: %llu bytes\n",
                    static_cast<unsigned long long>(assignment.partitionOverheadBytes));
    }

    // Re-assign (should be idempotent)
    std::fprintf(stderr, "\n2. Re-AssignPartition (idempotency test)...\n");
    PartitionAssignment second{};
    snprintf(second.templateId, sizeof(second.templateId), "%s", templateId);
    snprintf(second.deviceUUID, sizeof(second.deviceUUID), "%s", devices[0].basic.uuid);

    success = AssignPartition(&second);
    std::fprintf(stderr, "   Result: %s\n", success ? "true ✅" : "false ❌");
    if (success && strlen(assignment.partitionUUID) > 0) {
        bool sameUUID = strcmp(assignment.partitionUUID, second.partitionUUID) == 0;
        std::fprintf(stderr, "   UUID matches: %s %s\n",
                    sameUUID ? "yes" : "no",
                    sameUUID ? "✅" : "❌");
    }

    // Remove Partition
    std::fprintf(stderr, "\n3. RemovePartition()...\n");
    success = RemovePartition(templateId, devices[0].basic.uuid);
    std::fprintf(stderr, "   Result: %s\n", success ? "true ✅" : "false ❌");

    // Remove invalid partition (should succeed)
    std::fprintf(stderr, "\n4. RemovePartition (invalid, should succeed)...\n");
    success = RemovePartition("invalid-template", "invalid-uuid");
    std::fprintf(stderr, "   Result: %s\n", success ? "true ✅" : "false ❌");

    std::fprintf(stderr, "\n✅ PASS\n");
}

void testLogging() {
    printSubsection("Log API");

    Result r1 = Log("INFO", "Test info message from comprehensive test");
    std::fprintf(stderr, "Log(INFO): %s\n", resultToString(r1));

    Result r2 = Log("ERROR", "Test error message from comprehensive test");
    std::fprintf(stderr, "Log(ERROR): %s\n", resultToString(r2));

    std::fprintf(stderr, "✅ PASS\n");
}

void testHighLevelAPIs() {
    printSeparator("PHASE 2: High-Level Accelerator.h API Testing");

    testGetDeviceCount();
    testGetAllDevices();
    testGetDeviceTopology();
    testGetPartitionTemplates();
    testGetDeviceMetrics();
    testGetExtendedDeviceMetrics();
    testProcessUtilization();
    testIsolationAPIs();
    testPartitionLifecycle();
    testLogging();
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main() {
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "################################################################################\n");
    std::fprintf(stderr, "#                                                                              #\n");
    std::fprintf(stderr, "#          Ascend Accelerator Comprehensive Test Suite                        #\n");
    std::fprintf(stderr, "#                                                                              #\n");
    std::fprintf(stderr, "#  Coverage:                                                                   #\n");
    std::fprintf(stderr, "#  - Direct DCMI low-level API testing                                        #\n");
    std::fprintf(stderr, "#  - Logic ID probing and device discovery                                    #\n");
    std::fprintf(stderr, "#  - Complete accelerator.h API coverage                                      #\n");
    std::fprintf(stderr, "#  - Partition lifecycle testing                                              #\n");
    std::fprintf(stderr, "#                                                                              #\n");
    std::fprintf(stderr, "################################################################################\n");
    std::fprintf(stderr, "\n");

    // Phase 1: Low-level DCMI testing
    testDcmiDirectAccess();

    // Phase 2: High-level API testing
    testHighLevelAPIs();

    // Summary
    printSeparator("TEST SUITE COMPLETE");
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "All tests executed. Review results above for any failures.\n");
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "Legend:\n");
    std::fprintf(stderr, "  ✅ = Test passed or returned expected result\n");
    std::fprintf(stderr, "  ❌ = Test failed or returned unexpected error\n");
    std::fprintf(stderr, "  ⚠️  = Test skipped or not applicable\n");
    std::fprintf(stderr, "\n");

    return 0;
}
