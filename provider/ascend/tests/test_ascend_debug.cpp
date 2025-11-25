/*
 * Debug tool to test all Ascend accelerator APIs (except AssignPartition/RemovePartition)
 * and print detailed return values.
 */

#include <cstdio>
#include <cstring>
#include <vector>
#include "../../accelerator.h"

// Helper to print Result enum
const char* resultToString(Result r) {
    switch (r) {
        case RESULT_SUCCESS: return "RESULT_SUCCESS";
        case RESULT_ERROR_INVALID_PARAM: return "RESULT_ERROR_INVALID_PARAM";
        case RESULT_ERROR_NOT_FOUND: return "RESULT_ERROR_NOT_FOUND";
        case RESULT_ERROR_NOT_SUPPORTED: return "RESULT_ERROR_NOT_SUPPORTED";
        case RESULT_ERROR_RESOURCE_EXHAUSTED: return "RESULT_ERROR_RESOURCE_EXHAUSTED";
        case RESULT_ERROR_OPERATION_FAILED: return "RESULT_ERROR_OPERATION_FAILED";
        case RESULT_ERROR_INTERNAL: return "RESULT_ERROR_INTERNAL";
        default: return "UNKNOWN";
    }
}

void printSeparator(const char* title) {
    std::fprintf(stderr, "\n========================================\n");
    std::fprintf(stderr, "%s\n", title);
    std::fprintf(stderr, "========================================\n");
}

void testGetDeviceCount() {
    printSeparator("TEST: GetDeviceCount");

    size_t count = 0;
    Result r = GetDeviceCount(&count);

    std::fprintf(stderr, "Result: %s\n", resultToString(r));
    std::fprintf(stderr, "Device Count: %zu\n", count);
}

void testGetAllDevices() {
    printSeparator("TEST: GetAllDevices");

    size_t count = 0;
    Result r = GetDeviceCount(&count);
    if (r != RESULT_SUCCESS || count == 0) {
        std::fprintf(stderr, "Skipped: No devices available\n");
        return;
    }

    std::vector<ExtendedDeviceInfo> devices(count);
    size_t actualCount = 0;
    r = GetAllDevices(devices.data(), count, &actualCount);

    std::fprintf(stderr, "Result: %s\n", resultToString(r));
    std::fprintf(stderr, "Devices Returned: %zu\n", actualCount);

    for (size_t i = 0; i < actualCount; i++) {
        std::fprintf(stderr, "\nDevice %zu:\n", i);
        std::fprintf(stderr, "  UUID: %s\n", devices[i].basic.uuid);
        std::fprintf(stderr, "  Vendor: %s\n", devices[i].basic.vendor);
        std::fprintf(stderr, "  Model: %s\n", devices[i].basic.model);
        std::fprintf(stderr, "  Driver Version: %s\n", devices[i].basic.driverVersion);
        std::fprintf(stderr, "  Index: %d\n", devices[i].basic.index);
        std::fprintf(stderr, "  NUMA Node: %d\n", devices[i].basic.numaNode);
        std::fprintf(stderr, "  Total Memory: %llu bytes (%.2f GB)\n",
                    devices[i].basic.totalMemoryBytes,
                    devices[i].basic.totalMemoryBytes / (1024.0 * 1024.0 * 1024.0));
        std::fprintf(stderr, "  Compute Units: %llu\n", devices[i].basic.totalComputeUnits);
        std::fprintf(stderr, "  Max TFLOPS: %.2f\n", devices[i].basic.maxTflops);
        std::fprintf(stderr, "  PCIe: Gen %u x%u\n", devices[i].basic.pcieGen, devices[i].basic.pcieWidth);

        std::fprintf(stderr, "  Properties:\n");
        std::fprintf(stderr, "    Compute Capability: %s\n", devices[i].props.computeCapability);
        std::fprintf(stderr, "    Chip Type: %s\n", devices[i].props.chipType);
        std::fprintf(stderr, "    Clock AI: %u MHz\n", devices[i].props.clockAI);
        std::fprintf(stderr, "    Power Limit: %u W\n", devices[i].props.powerLimit);
        std::fprintf(stderr, "    Temperature Threshold: %u C\n", devices[i].props.temperatureThreshold);
        std::fprintf(stderr, "    ECC Enabled: %s\n", devices[i].props.eccEnabled ? "yes" : "no");

        std::fprintf(stderr, "  Capabilities:\n");
        std::fprintf(stderr, "    Supports Partitioning: %s\n", devices[i].capabilities.supportsPartitioning ? "yes" : "no");
        std::fprintf(stderr, "    Supports Soft Isolation: %s\n", devices[i].capabilities.supportsSoftIsolation ? "yes" : "no");
        std::fprintf(stderr, "    Supports Hard Isolation: %s\n", devices[i].capabilities.supportsHardIsolation ? "yes" : "no");
        std::fprintf(stderr, "    Supports Snapshot: %s\n", devices[i].capabilities.supportsSnapshot ? "yes" : "no");
        std::fprintf(stderr, "    Supports Metrics: %s\n", devices[i].capabilities.supportsMetrics ? "yes" : "no");
        std::fprintf(stderr, "    Max Partitions: %u\n", devices[i].capabilities.maxPartitions);
        std::fprintf(stderr, "    Max Workers Per Device: %u\n", devices[i].capabilities.maxWorkersPerDevice);
    }
}

void testGetDeviceTopology() {
    printSeparator("TEST: GetDeviceTopology");

    size_t count = 0;
    Result r = GetDeviceCount(&count);
    if (r != RESULT_SUCCESS || count == 0) {
        std::fprintf(stderr, "Skipped: No devices available\n");
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
    std::fprintf(stderr, "NVLink Bandwidth: %u MB/s\n", topology.nvlinkBandwidthMBps);
    std::fprintf(stderr, "IB NIC Count: %u\n", topology.ibNicCount);
    std::fprintf(stderr, "Topology Type: %s\n", topology.topologyType);

    for (size_t i = 0; i < topology.deviceCount; i++) {
        std::fprintf(stderr, "\nDevice %zu Topology:\n", i);
        std::fprintf(stderr, "  UUID: %s\n", topology.devices[i].deviceUUID);
        std::fprintf(stderr, "  NUMA Node: %d\n", topology.devices[i].numaNode);
        std::fprintf(stderr, "  Connections: %zu\n", topology.devices[i].connectionCount);
    }
}

void testSetMemHardLimit() {
    printSeparator("TEST: SetMemHardLimit");

    Result r = SetMemHardLimit("test-worker-1", "npu-0-chip-0", 1024ULL * 1024 * 1024);
    std::fprintf(stderr, "Result: %s\n", resultToString(r));
}

void testSetComputeUnitHardLimit() {
    printSeparator("TEST: SetComputeUnitHardLimit");

    Result r = SetComputeUnitHardLimit("test-worker-1", "npu-0-chip-0", 50);
    std::fprintf(stderr, "Result: %s\n", resultToString(r));
}

void testSnapshot() {
    printSeparator("TEST: Snapshot");

    pid_t pids[] = {1234, 5678};
    ProcessArray processes = {};
    processes.processIds = pids;
    processes.processCount = 2;
    std::snprintf(processes.deviceUUID, sizeof(processes.deviceUUID), "npu-0-chip-0");

    Result r = Snapshot(&processes);
    std::fprintf(stderr, "Result: %s\n", resultToString(r));
}

void testResume() {
    printSeparator("TEST: Resume");

    pid_t pids[] = {1234, 5678};
    ProcessArray processes = {};
    processes.processIds = pids;
    processes.processCount = 2;
    std::snprintf(processes.deviceUUID, sizeof(processes.deviceUUID), "npu-0-chip-0");

    Result r = Resume(&processes);
    std::fprintf(stderr, "Result: %s\n", resultToString(r));
}

void testGetProcessComputeUtilization() {
    printSeparator("TEST: GetProcessComputeUtilization");

    ComputeUtilization utils[10] = {};
    size_t utilCount = 0;

    Result r = GetProcessComputeUtilization(utils, 10, &utilCount);
    std::fprintf(stderr, "Result: %s\n", resultToString(r));
    std::fprintf(stderr, "Utilization Count: %zu\n", utilCount);

    for (size_t i = 0; i < utilCount; i++) {
        std::fprintf(stderr, "\nProcess %zu:\n", i);
        std::fprintf(stderr, "  Process ID: %s\n", utils[i].processId);
        std::fprintf(stderr, "  Device UUID: %s\n", utils[i].deviceUUID);
        std::fprintf(stderr, "  Utilization: %.2f%%\n", utils[i].utilizationPercent);
        std::fprintf(stderr, "  Active SMs: %llu / %llu\n", utils[i].activeSMs, utils[i].totalSMs);
        std::fprintf(stderr, "  TFLOPS Used: %.2f\n", utils[i].tflopsUsed);
    }
}

void testGetProcessMemoryUtilization() {
    printSeparator("TEST: GetProcessMemoryUtilization");

    MemoryUtilization utils[10] = {};
    size_t utilCount = 0;

    Result r = GetProcessMemoryUtilization(utils, 10, &utilCount);
    std::fprintf(stderr, "Result: %s\n", resultToString(r));
    std::fprintf(stderr, "Utilization Count: %zu\n", utilCount);

    for (size_t i = 0; i < utilCount; i++) {
        std::fprintf(stderr, "\nProcess %zu:\n", i);
        std::fprintf(stderr, "  Process ID: %s\n", utils[i].processId);
        std::fprintf(stderr, "  Device UUID: %s\n", utils[i].deviceUUID);
        std::fprintf(stderr, "  Used: %llu bytes (%.2f GB)\n",
                    utils[i].usedBytes,
                    utils[i].usedBytes / (1024.0 * 1024.0 * 1024.0));
        std::fprintf(stderr, "  Reserved: %llu bytes (%.2f GB)\n",
                    utils[i].reservedBytes,
                    utils[i].reservedBytes / (1024.0 * 1024.0 * 1024.0));
        std::fprintf(stderr, "  Utilization: %.2f%%\n", utils[i].utilizationPercent);
    }
}

void testGetDeviceMetrics() {
    printSeparator("TEST: GetDeviceMetrics");

    size_t count = 0;
    Result r = GetDeviceCount(&count);
    if (r != RESULT_SUCCESS || count == 0) {
        std::fprintf(stderr, "Skipped: No devices available\n");
        return;
    }

    // Get device UUIDs
    std::vector<ExtendedDeviceInfo> devices(count);
    size_t actualCount = 0;
    GetAllDevices(devices.data(), count, &actualCount);

    std::vector<const char*> uuids(actualCount);
    for (size_t i = 0; i < actualCount; i++) {
        uuids[i] = devices[i].basic.uuid;
    }

    std::vector<DeviceMetrics> metrics(actualCount);
    r = GetDeviceMetrics(uuids.data(), actualCount, metrics.data());

    std::fprintf(stderr, "Result: %s\n", resultToString(r));

    for (size_t i = 0; i < actualCount; i++) {
        std::fprintf(stderr, "\nDevice %zu Metrics:\n", i);
        std::fprintf(stderr, "  UUID: %s\n", metrics[i].deviceUUID);
        std::fprintf(stderr, "  Power Usage: %.2f W\n", metrics[i].powerUsageWatts);
        std::fprintf(stderr, "  Temperature: %.2f C\n", metrics[i].temperatureCelsius);
        std::fprintf(stderr, "  PCIe RX: %llu bytes\n", metrics[i].pcieRxBytes);
        std::fprintf(stderr, "  PCIe TX: %llu bytes\n", metrics[i].pcieTxBytes);
        std::fprintf(stderr, "  SM Active: %u%%\n", metrics[i].smActivePercent);
        std::fprintf(stderr, "  Tensor Core Usage: %u%%\n", metrics[i].tensorCoreUsagePercent);
        std::fprintf(stderr, "  Memory Used: %llu bytes (%.2f GB)\n",
                    metrics[i].memoryUsedBytes,
                    metrics[i].memoryUsedBytes / (1024.0 * 1024.0 * 1024.0));
        std::fprintf(stderr, "  Memory Total: %llu bytes (%.2f GB)\n",
                    metrics[i].memoryTotalBytes,
                    metrics[i].memoryTotalBytes / (1024.0 * 1024.0 * 1024.0));
    }
}

void testGetExtendedDeviceMetrics() {
    printSeparator("TEST: GetExtendedDeviceMetrics");

    size_t count = 0;
    Result r = GetDeviceCount(&count);
    if (r != RESULT_SUCCESS || count == 0) {
        std::fprintf(stderr, "Skipped: No devices available\n");
        return;
    }

    // Get device UUIDs
    std::vector<ExtendedDeviceInfo> devices(count);
    size_t actualCount = 0;
    GetAllDevices(devices.data(), count, &actualCount);

    std::vector<const char*> uuids(actualCount);
    for (size_t i = 0; i < actualCount; i++) {
        uuids[i] = devices[i].basic.uuid;
    }

    std::vector<ExtendedDeviceMetrics> metrics(actualCount);
    r = GetExtendedDeviceMetrics(uuids.data(), actualCount, metrics.data(), 10, 10, 10);

    std::fprintf(stderr, "Result: %s\n", resultToString(r));

    for (size_t i = 0; i < actualCount; i++) {
        std::fprintf(stderr, "\nDevice %zu Extended Metrics:\n", i);
        std::fprintf(stderr, "  UUID: %s\n", metrics[i].deviceUUID);
        std::fprintf(stderr, "  NVLink Count: %zu\n", metrics[i].nvlinkCount);
        std::fprintf(stderr, "  IB NIC Count: %zu\n", metrics[i].ibNicCount);
        std::fprintf(stderr, "  PCIe Link Count: %zu\n", metrics[i].pcieLinkCount);
    }
}

void testLog() {
    printSeparator("TEST: Log");

    Result r1 = Log("INFO", "Test info message");
    std::fprintf(stderr, "Log(INFO) Result: %s\n", resultToString(r1));

    Result r2 = Log("ERROR", "Test error message");
    std::fprintf(stderr, "Log(ERROR) Result: %s\n", resultToString(r2));
}

int main() {
    std::fprintf(stderr, "======================================\n");
    std::fprintf(stderr, "Ascend Accelerator API Debug Tool\n");
    std::fprintf(stderr, "======================================\n");

    testGetDeviceCount();
    testGetAllDevices();
    testGetDeviceTopology();
    testSetMemHardLimit();
    testSetComputeUnitHardLimit();
    testSnapshot();
    testResume();
    testGetProcessComputeUtilization();
    testGetProcessMemoryUtilization();
    testGetDeviceMetrics();
    testGetExtendedDeviceMetrics();
    testLog();

    printSeparator("All Tests Complete");

    return 0;
}
