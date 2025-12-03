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

#ifndef ACCELERATOR_H
#define ACCELERATOR_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Common Types
// ============================================================================

typedef enum {
    RESULT_SUCCESS = 0,
    RESULT_ERROR_INVALID_PARAM = 1,
    RESULT_ERROR_NOT_FOUND = 2,
    RESULT_ERROR_NOT_SUPPORTED = 3,
    RESULT_ERROR_RESOURCE_EXHAUSTED = 4,
    RESULT_ERROR_OPERATION_FAILED = 5,
    RESULT_ERROR_INTERNAL = 6
} Result;

typedef enum {
    ISOLATION_MODE_SHARED = 0,      // Timeslicing, no resource control
    ISOLATION_MODE_SOFT = 1,        // Hook-based, token-based limiting
    ISOLATION_MODE_HARD = 2,        // One-time resource limits
    ISOLATION_MODE_PARTITIONED = 3  // Hardware/driver-level partitioning (MIG)
} IsolationMode;

// ============================================================================
// DeviceInfo Types
// ============================================================================

// Device capabilities
typedef struct {
    bool supportsPartitioning;      // e.g., MIG support
    bool supportsSoftIsolation;     // Hook-based isolation support
    bool supportsHardIsolation;     // One-time limit support
    bool supportsSnapshot;          // Process snapshot/resume support
    bool supportsMetrics;           // Metrics collection support
    uint32_t maxPartitions;         // Maximum number of partitions
    uint32_t maxWorkersPerDevice;   // Maximum workers per device
} DeviceCapabilities;

// Basic device information
typedef struct {
    char uuid[64];                  // Device UUID
    char vendor[32];                // Vendor name (e.g., "NVIDIA", "AMD")
    char model[128];                // Model name (e.g., "A100", "H100")
    char driverVersion[64];          // Driver version
    char firmwareVersion[64];       // Firmware version
    int32_t index;                  // Device index
    int32_t numaNode;               // NUMA node ID (-1 if not assigned)
    uint64_t totalMemoryBytes;      // Total memory in bytes
    uint64_t totalComputeUnits;     // Total compute units (e.g., SMs for NVIDIA)
    double maxTflops;               // Maximum TFLOPS
    uint32_t pcieGen;               // PCIe generation
    uint32_t pcieWidth;             // PCIe width (lanes)
} DeviceBasicInfo;

// Device properties
typedef struct {
    uint32_t clockGraphics;         // Graphics clock (MHz)
    uint32_t clockSM;                // SM clock (MHz) - for NVIDIA
    uint32_t clockMem;               // Memory clock (MHz)
    uint32_t clockAI;                // AI core clock (MHz) - for Ascend
    uint32_t powerLimit;             // Power limit (W)
    uint32_t temperatureThreshold;  // Temperature threshold (C)
    bool eccEnabled;                 // ECC enabled
    bool persistenceModeEnabled;    // Persistence mode
    char computeCapability[16];      // Compute capability (e.g., "8.0", "9.0" for NVIDIA, "Ascend310" for Ascend)
    char chipType[32];               // Chip type (e.g., "NVIDIA", "Ascend", "AMD")
} DeviceProperties;

// Related device information (for topology)
typedef struct {
    char deviceUUID[64];             // Related device UUID
    char connectionType[32];         // Connection type (e.g., "NVLink", "PCIe", "IB")
    uint32_t bandwidthMBps;          // Bandwidth in MB/s
    uint32_t latencyNs;              // Latency in nanoseconds
} RelatedDevice;

// Extended device information
typedef struct {
    DeviceBasicInfo basic;
    DeviceProperties props;
    RelatedDevice* relatedDevices;   // Array of related devices
    size_t relatedDeviceCount;       // Number of related devices
    DeviceCapabilities capabilities;
} ExtendedDeviceInfo;

// Partition template for hardware partitioning (e.g., MIG)
typedef struct {
    char templateId[64];             // Template identifier
    char name[128];                  // Human-readable name
    uint64_t memoryBytes;            // Memory allocated to partition
    uint64_t computeUnits;           // Compute units allocated
    double tflops;                   // TFLOPS for this partition
    uint32_t sliceCount;              // Number of slices (for MIG)
    bool isDefault;                  // Is this a default template
    char description[256];           // Description
} PartitionTemplate;

// Device topology information
typedef struct {
    char deviceUUID[64];             // Device UUID
    int32_t numaNode;                // NUMA node
    RelatedDevice* connections;      // Array of connections
    size_t connectionCount;          // Number of connections
} DeviceTopology;

// Extended topology (includes NVLink, IB NIC, etc.)
typedef struct {
    DeviceTopology* devices;         // Array of device topologies
    size_t deviceCount;              // Number of devices
    uint32_t nvlinkBandwidthMBps;    // NVLink total bandwidth
    uint32_t ibNicCount;             // InfiniBand NIC count
    char topologyType[32];           // Topology type (e.g., "NVLink", "PCIe")
} ExtendedDeviceTopology;

// ============================================================================
// Virtualization Types
// ============================================================================

// Partition assignment request
typedef struct {
    char templateId[64];             // Template ID to use
    char deviceUUID[64];             // Target device UUID
    char partitionUUID[64];         // Output: assigned partition UUID
} PartitionAssignment;

// Worker information for isolation
typedef struct {
    char workerId[64];               // Worker identifier
    char deviceUUID[64];             // Device UUID
    pid_t processId;                 // Process ID
    uint64_t memoryLimitBytes;       // Memory limit (for hard isolation)
    uint32_t computeUnitLimit;       // Compute unit limit (for hard isolation)
    IsolationMode isolationMode;     // Isolation mode
} WorkerInfo;

// Process array for snapshot/resume
typedef struct {
    pid_t* processIds;               // Array of process IDs
    size_t processCount;             // Number of processes
    char deviceUUID[64];             // Device UUID
} ProcessArray;

// ============================================================================
// Metrics Types
// ============================================================================

// Extra metric key-value pair
typedef struct {
    char key[64];              // Metric key name
    double value;              // Metric value
} ExtraMetric;

// Compute utilization
typedef struct {
    char processId[32];              // Process ID as string
    char deviceUUID[64];             // Device UUID
    double utilizationPercent;      // Utilization percentage (0-100)
    uint64_t activeSMs;              // Active SMs/Compute Units
    uint64_t totalSMs;               // Total SMs/Compute Units
    double tflopsUsed;               // TFLOPS currently used
} ComputeUtilization;

// Memory utilization
typedef struct {
    char processId[32];              // Process ID as string
    char deviceUUID[64];             // Device UUID
    uint64_t usedBytes;              // Memory used in bytes
    uint64_t reservedBytes;          // Memory reserved in bytes
    double utilizationPercent;      // Utilization percentage (0-100)
} MemoryUtilization;

// Basic device metrics
typedef struct {
    char deviceUUID[64];             // Device UUID
    double powerUsageWatts;           // Current power usage (W)
    double temperatureCelsius;        // Temperature (C)
    uint64_t pcieRxBytes;             // PCIe RX bytes
    uint64_t pcieTxBytes;             // PCIe TX bytes
    uint32_t smActivePercent;        // SM active percentage
    uint32_t tensorCoreUsagePercent; // Tensor Core usage percentage
    uint64_t memoryUsedBytes;         // Memory used
    uint64_t memoryTotalBytes;        // Memory total
    ExtraMetric* extraMetrics;        // Array of extra metrics (key-value pairs)
    size_t extraMetricsCount;         // Number of extra metrics
} DeviceMetrics;

// Extended device metrics (NVLink, etc.)
typedef struct {
    char deviceUUID[64];             // Device UUID
    uint32_t* nvlinkBandwidthMBps;   // NVLink bandwidth per link (MB/s)
    size_t nvlinkCount;              // Number of NVLink connections
    uint64_t* ibNicBandwidthMBps;    // IB NIC bandwidth per NIC (MB/s)
    size_t ibNicCount;               // Number of IB NICs
    uint32_t* pcieBandwidthMBps;     // PCIe bandwidth per link (MB/s)
    size_t pcieLinkCount;            // Number of PCIe links
} ExtendedDeviceMetrics;

// ============================================================================
// DeviceInfo APIs
// ============================================================================

/**
 * Get the number of available devices.
 * 
 * @param deviceCount Output parameter for number of devices
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result GetDeviceCount(size_t* deviceCount);

/**
 * Get all available devices information.
 * 
 * @param devices Output buffer for device information (allocated by caller)
 * @param maxCount Maximum number of devices that can fit in the buffer
 * @param deviceCount Output parameter for number of devices actually returned
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result GetAllDevices(ExtendedDeviceInfo* devices, size_t maxCount, size_t* deviceCount);

/**
 * Get device topology including NVLink, IB NIC, and other interconnects.
 * 
 * @param deviceIndexArray Array of device indices to query
 * @param deviceCount Number of devices in array
 * @param topology Output parameter for extended topology (allocated by caller)
 * @param maxConnectionsPerDevice Maximum number of connections per device in topology buffer
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result GetDeviceTopology(int32_t* deviceIndexArray, size_t deviceCount, ExtendedDeviceTopology* topology, size_t maxConnectionsPerDevice);

// ============================================================================
// Virtualization APIs - Partitioned Isolation
// ============================================================================

/**
 * Assign a partition to a device using a template (e.g., create MIG instance).
 * 
 * @param assignment Partition assignment request (templateId, deviceUUID)
 *                   Output: partitionUUID and partitionOverheadBytes
 * @return true on success, false otherwise
 */
bool AssignPartition(PartitionAssignment* assignment);

/**
 * Remove a partition from a device.
 * 
 * @param templateId Template ID used to create the partition
 * @param deviceUUID Device UUID
 * @return true on success, false otherwise
 */
bool RemovePartition(const char* templateId, const char* deviceUUID);

// ============================================================================
// Virtualization APIs - Hard Isolation
// ============================================================================

/**
 * Set hard memory limit for a worker (one-time, called at worker start by limiter.so).
 * 
 * @param workerId Worker identifier
 * @param deviceUUID Device UUID
 * @param memoryLimitBytes Memory limit in bytes
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result SetMemHardLimit(const char* workerId, const char* deviceUUID, uint64_t memoryLimitBytes);

/**
 * Set hard compute unit limit for a worker (one-time, called at worker start).
 * 
 * @param workerId Worker identifier
 * @param deviceUUID Device UUID
 * @param computeUnitLimit Compute unit limit (e.g., percentage 0-100)
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result SetComputeUnitHardLimit(const char* workerId, const char* deviceUUID, uint32_t computeUnitLimit);

// ============================================================================
// Virtualization APIs - Device Snapshot/Migration
// ============================================================================

/**
 * Snapshot device state for processes (lock processes, checkpoint state).
 * Called from hypervisor for migration.
 * 
 * @param processes Array of processes to snapshot
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result Snapshot(ProcessArray* processes);

/**
 * Resume device state for processes (unlock processes, restore state).
 * Called from hypervisor after migration.
 * 
 * @param processes Array of processes to resume
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result Resume(ProcessArray* processes);

// ============================================================================
// Metrics APIs
// ============================================================================

/**
 * Get compute utilization for all processes on all devices.
 * 
 * @param utilizations Output buffer for compute utilizations (allocated by caller)
 * @param maxCount Maximum number of utilizations that can fit in the buffer
 * @param utilizationCount Output parameter for number of utilizations actually returned
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result GetProcessComputeUtilization(
    ComputeUtilization* utilizations,
    size_t maxCount,
    size_t* utilizationCount
);

/**
 * Get memory utilization for all processes on all devices.
 * 
 * @param utilizations Output buffer for memory utilizations (allocated by caller)
 * @param maxCount Maximum number of utilizations that can fit in the buffer
 * @param utilizationCount Output parameter for number of utilizations actually returned
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result GetProcessMemoryUtilization(
    MemoryUtilization* utilizations,
    size_t maxCount,
    size_t* utilizationCount
);

/**
 * Get basic device metrics (power, PCIe, SM active, TC usage, etc.).
 * 
 * @param deviceUUIDArray Array of device UUIDs
 * @param deviceCount Number of devices
 * @param metrics Output buffer for device metrics (allocated by caller, size >= deviceCount)
 * @param maxExtraMetricsPerDevice Maximum number of extra metrics per device
 * @return RESULT_SUCCESS on success, error code otherwise
 * 
 * Note: Caller must allocate extraMetrics arrays for each device metric.
 *       Each metrics[i].extraMetrics should point to an array of size maxExtraMetricsPerDevice.
 *       The function will fill in the metrics and set extraMetricsCount for each device.
 */
Result GetDeviceMetrics(
    const char** deviceUUIDArray,
    size_t deviceCount,
    DeviceMetrics* metrics,
    size_t maxExtraMetricsPerDevice
);

/**
 * Get extended device metrics (NVLink bandwidth, etc.).
 * 
 * @param deviceUUIDArray Array of device UUIDs
 * @param deviceCount Number of devices
 * @param metrics Output buffer for extended device metrics (allocated by caller, size >= deviceCount)
 * @param maxNvlinkPerDevice Maximum number of NVLink connections per device
 * @param maxIbNicPerDevice Maximum number of IB NICs per device
 * @param maxPciePerDevice Maximum number of PCIe links per device
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result GetExtendedDeviceMetrics(
    const char** deviceUUIDArray,
    size_t deviceCount,
    ExtendedDeviceMetrics* metrics,
    size_t maxNvlinkPerDevice,
    size_t maxIbNicPerDevice,
    size_t maxPciePerDevice
);


typedef struct {
    char* hostPath;              // Host path
    char* guestPath;             // Guest path
} Mount;
/**
 * Get vendor mount libs.
 * 
 * @param mounts Output buffer for vendor mount libs (allocated by caller)
 * @param maxCount Maximum number of mounts that can fit in the buffer
 * @param mountCount Output parameter for number of mounts actually returned
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result GetVendorMountLibs(Mount* mounts, size_t maxCount, size_t* mountCount);

// ============================================================================
// Utility APIs
// ============================================================================

/**
 * Log a message (for debugging and diagnostics).
 * 
 * @param level Log level (e.g., "DEBUG", "INFO", "WARN", "ERROR")
 * @param message Log message
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result Log(const char* level, const char* message);

#ifdef __cplusplus
}
#endif

// Include limiter.h after defining Result enum
#include "limiter.h"

#endif // ACCELERATOR_H

