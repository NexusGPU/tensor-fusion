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


// ============================================================================
// Callback Types
// ============================================================================

// Log callback function type
// Called by the library to log messages
// Parameters: level (log level string), message (log message string)
typedef void (*LogCallbackFunc)(const char* level, const char* message);

// ============================================================================
// DeviceInfo Types
// ============================================================================

// Virtualization capabilities
typedef struct {
    bool supportsPartitioning;      // e.g., MIG support
    bool supportsSoftIsolation;     // Hook-based isolation support
    bool supportsHardIsolation;     // One-time limit support
    bool supportsSnapshot;          // Process snapshot/resume support
    bool supportsMetrics;           // Metrics collection support
    bool supportsRemoting;          // Remote device access support
    uint32_t maxPartitions;         // Maximum number of partitions
    uint32_t maxWorkersPerDevice;   // Maximum workers per device
} VirtualizationCapabilities;

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

// Device property key-value pair
typedef struct {
    char key[64];                   // Property key
    char value[256];                // Property value
} DevicePropertyKV;

// Maximum number of device properties
#define MAX_DEVICE_PROPERTIES 64

// Device properties (key-value pairs)
typedef struct {
    DevicePropertyKV properties[MAX_DEVICE_PROPERTIES];  // Array of property key-value pairs
    size_t count;                   // Number of properties (0 to MAX_DEVICE_PROPERTIES)
} DeviceProperties;

// Extended device information
typedef struct {
    DeviceBasicInfo basic;
    DeviceProperties props;
    VirtualizationCapabilities virtualizationCapabilities;
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
} DeviceTopology;

// Extended topology
#define MAX_TOPOLOGY_DEVICES 64

typedef struct {
    DeviceTopology devices[MAX_TOPOLOGY_DEVICES];  // Array of device topologies
    size_t deviceCount;              // Number of devices (0 to MAX_TOPOLOGY_DEVICES)
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
} WorkerInfo;

// Process array for snapshot/resume
#define MAX_PROCESSES 1024

typedef struct {
    pid_t processIds[MAX_PROCESSES];  // Array of process IDs
    size_t processCount;             // Number of processes (0 to MAX_PROCESSES)
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

// Maximum number of extra metrics per device
#define MAX_EXTRA_METRICS 64

// Compute utilization
typedef struct {
    char processId[32];              // Process ID as string
    char deviceUUID[64];             // Device UUID
    double utilizationPercent;      // Utilization percentage (0-100)
    uint64_t activeSMs;              // Active SMs/Compute Units
    uint64_t totalSMs;               // Total SMs/Compute Units
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
    uint32_t utilizationPercent;      // Utilization percentage (0-100)
    uint64_t memoryUsedBytes;         // Memory used
    ExtraMetric extraMetrics[MAX_EXTRA_METRICS];  // Array of extra metrics
    size_t extraMetricsCount;         // Number of extra metrics (0 to MAX_EXTRA_METRICS)
} DeviceMetrics;


// Device UUID array entry (for passing multiple UUIDs)
// DEPRECATED: This struct is no longer used. GetDeviceMetrics now uses const char** instead.
#define MAX_DEVICE_UUIDS 64
#define UUID_STRING_LENGTH 64

typedef struct {
    char uuid[UUID_STRING_LENGTH];   // Device UUID string
} DeviceUUIDEntry;

// Mount structure for vendor libraries
#define MAX_MOUNT_PATH 512

typedef struct {
    char hostPath[MAX_MOUNT_PATH];   // Host path
    char guestPath[MAX_MOUNT_PATH];  // Guest path
} Mount;

Result VirtualGPUInit(void);

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
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result GetDeviceTopology(int32_t* deviceIndexArray, size_t deviceCount, ExtendedDeviceTopology* topology);

// ============================================================================
// Virtualization APIs - Partitioned Isolation
// ============================================================================

/**
 * Assign a partition to a device using a template (e.g., create MIG instance).
 * 
 * @param assignment Partition assignment request (templateId, deviceUUID)
 *                   Output: partitionUUID
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
 * @param deviceUUIDs Array of device UUID strings (null-terminated C strings)
 * @param deviceCount Number of devices
 * @param metrics Output buffer for device metrics (allocated by caller, size >= deviceCount)
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result GetDeviceMetrics(
    const char** deviceUUIDs,
    size_t deviceCount,
    DeviceMetrics* metrics
);


/**
 * Get vendor mount libs returns the mount paths for additional device driver or runtime libraries.
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
 * Register a log callback function.
 * The callback will be called by the library to log messages.
 * 
 * @param callback Log callback function pointer (can be NULL to unregister)
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result RegisterLogCallback(LogCallbackFunc callback);

#ifdef __cplusplus
}
#endif

// Include limiter.h after defining Result enum
#include "limiter.h"

#endif // ACCELERATOR_H
