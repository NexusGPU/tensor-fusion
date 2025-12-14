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

// Interconnect taxonomy to unify PCIe/NVLink/XGMI/CXL/etc.
typedef enum {
    INTERCONNECT_PCIE = 0,
    INTERCONNECT_NVLINK = 1,
    INTERCONNECT_XGMI = 2,
    INTERCONNECT_CXL = 3,
    INTERCONNECT_ETHERNET = 4,
    INTERCONNECT_INFINIBAND = 5,
    INTERCONNECT_HCCL = 6,           // Vendor collective fabric (e.g., Ascend)
    INTERCONNECT_CUSTOM = 7
} InterconnectType;

// Topology link information (peer device connectivity)
typedef struct {
    char deviceUUID[64];             // Peer device UUID
    InterconnectType linkType;       // Connection type (PCIe/NVLink/XGMI/CXL/etc.)
    char linkName[32];               // Vendor label (e.g., "nvlink0", "pcie0")
    uint32_t version;                // Protocol version (PCIe Gen/CXL rev/etc.)
    uint32_t widthLanes;             // Lane width or link width
    uint64_t bandwidthMBps;          // Bandwidth in MB/s (per link or logical channel)
    uint32_t latencyNs;              // Latency in nanoseconds
} TopologyLink;

// Extended device information
typedef struct {
    DeviceBasicInfo basic;
    DeviceProperties props;
    TopologyLink* links;             // Array of topology links to peers
    size_t linkCount;                // Number of links
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
    TopologyLink* links;             // Array of interconnect links
    size_t linkCount;                // Number of links
} DeviceTopology;

// Extended topology (includes NVLink, IB NIC, etc.)
typedef struct {
    DeviceTopology* devices;         // Array of device topologies
    size_t deviceCount;              // Number of devices
    InterconnectType primaryInterconnect; // Dominant fabric type for the group
    char fabricLabel[32];            // Fabric label (e.g., "NVLink", "XGMI", "CXL", "PCIe")
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

// Compute engine taxonomy to cover GPUs, NPUs, TPUs, etc.
typedef enum {
    COMPUTE_ENGINE_GENERAL = 0,      // General-purpose compute units (SM, CU, EU)
    COMPUTE_ENGINE_MATRIX = 1,       // Matrix/tensor cores (e.g., Tensor Core, XMX)
    COMPUTE_ENGINE_VECTOR = 2,       // Vector/SIMD engines
    COMPUTE_ENGINE_AI_CORE = 3,      // Dedicated AI/NPU cores (Ascend AI Core, TPU core)
    COMPUTE_ENGINE_AICPU = 4,        // Ascend AICpu (ARM-based control/scalar processor)
    COMPUTE_ENGINE_DSP = 5,          // DSP or fixed-function signal processors
    COMPUTE_ENGINE_COPY = 6,         // DMA/Copy engines
} ComputeEngineType;

// Precision used for throughput reporting
typedef enum {
    COMPUTE_PRECISION_FP64 = 0,
    COMPUTE_PRECISION_FP32 = 1,
    COMPUTE_PRECISION_TF32 = 2,
    COMPUTE_PRECISION_FP16 = 3,
    COMPUTE_PRECISION_BF16 = 4,
    COMPUTE_PRECISION_INT8 = 5,
    COMPUTE_PRECISION_INT4 = 6,
    COMPUTE_PRECISION_MIXED = 7,
    COMPUTE_PRECISION_UNKNOWN = 8
} ComputePrecision;

// Vendor-specific engine labels (orthogonal to engine type)
typedef enum {
    ENGINE_NAME_UNKNOWN = 0,
    ENGINE_NAME_SM = 1,          // NVIDIA SM
    ENGINE_NAME_TENSOR_CORE = 2, // NVIDIA Tensor Core
    ENGINE_NAME_CU = 3,          // AMD Compute Unit
    ENGINE_NAME_EU = 4,          // Intel Execution Unit
    ENGINE_NAME_XMX = 5,         // Intel XMX/Matrix Engine
    ENGINE_NAME_MATRIX_CORE = 6, // AMD Matrix Core
    ENGINE_NAME_AI_CORE = 7,     // Ascend/TPU AI Core
    ENGINE_NAME_VECTOR_CORE = 8, // Ascend Vector Core or similar
    ENGINE_NAME_AICPU = 9,       // Ascend AICpu (ARM-based control CPU)
    ENGINE_NAME_DSP = 10,        // DSP/ISP/SPS
    ENGINE_NAME_DMA = 11,        // DMA/Copy Engine
    ENGINE_NAME_ENCODER = 12,    // Media encoder
    ENGINE_NAME_DECODER = 13,    // Media decoder
    ENGINE_NAME_OTHER = 14       // Any other vendor block
} ComputeEngineName;

// Per-engine utilization and throughput
typedef struct {
    ComputeEngineType engineType;    // Engine category
    ComputeEngineName engineName;    // Vendor label (use OTHER if not listed)
    ComputePrecision precision;      // Precision of the measured workload
    double utilizationPercent;       // Utilization percentage (0-100) for this engine
    uint64_t activeUnits;            // Active units in the engine category
    uint64_t totalUnits;             // Total units in the engine category
    double throughputOpsPerSec;      // Sustained ops/sec for the precision above
} ComputeEngineUtilization;

// Compute utilization aggregated per process
typedef struct {
    char processId[32];                   // Process ID as string
    char deviceUUID[64];                  // Device UUID
    double overallUtilizationPercent;     // Overall utilization across engines (0-100)
    ComputeEngineUtilization* engines;    // Array of per-engine utilizations
    size_t engineCount;                   // Number of entries in engines
} ComputeUtilization;

// Memory utilization
typedef struct {
    char processId[32];              // Process ID as string
    char deviceUUID[64];             // Device UUID
    uint64_t usedBytes;              // Memory used in bytes
    uint64_t reservedBytes;          // Memory reserved in bytes
    double utilizationPercent;      // Utilization percentage (0-100)
} MemoryUtilization;

// Memory pool taxonomy to represent HBM/GDDR/SRAM/shared system memory
typedef enum {
    MEMORY_POOL_DEVICE_HBM = 0,
    MEMORY_POOL_DEVICE_GDDR = 1,
    MEMORY_POOL_DEVICE_SRAM = 2,
    MEMORY_POOL_LLC = 3,            // Last-level/on-die cache used as capacity
    MEMORY_POOL_SHARED_SYSTEM = 4,  // Shared or unified memory with host/CPU
    MEMORY_POOL_OTHER = 5
} MemoryPoolType;

// Per-memory-pool utilization/bandwidth
typedef struct {
    MemoryPoolType poolType;         // Pool category
    char poolName[32];               // Vendor label (e.g., "HBM0", "SLC", "UMA")
    uint64_t totalBytes;             // Total capacity in bytes
    uint64_t usedBytes;              // Bytes actively used by workloads
    uint64_t reservedBytes;          // Driver-reserved bytes not available to workloads
    uint64_t readBytesPerSec;        // Read bandwidth in bytes/sec
    uint64_t writeBytesPerSec;       // Write bandwidth in bytes/sec
    double utilizationPercent;       // Capacity utilization percentage (0-100)
} MemoryPoolMetrics;

// Per-link throughput/usage metrics
typedef struct {
    InterconnectType linkType;       // Link category
    char linkName[32];               // Vendor label (e.g., "pcie0", "nvlink1")
    uint32_t version;                // Protocol version (PCIe Gen/CXL rev/etc.)
    uint32_t widthLanes;             // Lane width or link width
    uint64_t rxBytes;                // Bytes received over the link
    uint64_t txBytes;                // Bytes transmitted over the link
    double utilizationPercent;       // Link utilization percentage (0-100)
} InterconnectMetrics;

// Basic device metrics (vendor-neutral)
typedef struct {
    char deviceUUID[64];                  // Device UUID
    double powerUsageWatts;               // Current power usage (W)
    double temperatureCelsius;            // Temperature (C)
    ComputeEngineUtilization* compute;    // Array of per-engine utilization
    size_t computeCount;                  // Number of entries in compute
    MemoryPoolMetrics* memoryPools;       // Array of per-pool memory metrics
    size_t memoryPoolCount;               // Number of entries in memoryPools
    ExtraMetric* extraMetrics;            // Array of extra metrics (key-value pairs)
    size_t extraMetricsCount;             // Number of extra metrics
} DeviceMetrics;

// Extended device metrics for interconnect fabrics
typedef struct {
    char deviceUUID[64];                  // Device UUID
    InterconnectMetrics* interconnects;   // Array of per-link metrics
    size_t interconnectCount;             // Number of interconnect entries
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
 * @param maxEnginesPerProcess Maximum number of engine entries per process (size of each engines array)
 * @param utilizationCount Output parameter for number of utilizations actually returned
 * @return RESULT_SUCCESS on success, error code otherwise
 *
 * Note: Caller must allocate engines arrays for each utilization entry.
 *       Each utilizations[i].engines should point to an array of size maxEnginesPerProcess.
 *       The function will fill engineCount for each process.
 */
Result GetProcessComputeUtilization(
    ComputeUtilization* utilizations,
    size_t maxCount,
    size_t maxEnginesPerProcess,
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
 * Get basic device metrics (power/temp + compute engines + memory pools + extra metrics).
 * 
 * @param deviceUUIDArray Array of device UUIDs
 * @param deviceCount Number of devices
 * @param metrics Output buffer for device metrics (allocated by caller, size >= deviceCount)
 * @param maxComputeEnginesPerDevice Maximum number of compute engines per device
 * @param maxMemoryPoolsPerDevice Maximum number of memory pools per device
 * @param maxExtraMetricsPerDevice Maximum number of extra metrics per device
 * @return RESULT_SUCCESS on success, error code otherwise
 * 
 * Note: Caller must allocate compute, memoryPools, and extraMetrics arrays for each device metric.
 *       Each metrics[i].compute should point to an array of size maxComputeEnginesPerDevice.
 *       Each metrics[i].memoryPools should point to an array of size maxMemoryPoolsPerDevice.
 *       Each metrics[i].extraMetrics should point to an array of size maxExtraMetricsPerDevice.
 *       The function will fill counts for each device.
 */
Result GetDeviceMetrics(
    const char** deviceUUIDArray,
    size_t deviceCount,
    DeviceMetrics* metrics,
    size_t maxComputeEnginesPerDevice,
    size_t maxMemoryPoolsPerDevice,
    size_t maxExtraMetricsPerDevice
);

/**
 * Get extended device metrics (interconnect/fabric links).
 * 
 * @param deviceUUIDArray Array of device UUIDs
 * @param deviceCount Number of devices
 * @param metrics Output buffer for extended device metrics (allocated by caller, size >= deviceCount)
 * @param maxInterconnectPerDevice Maximum number of interconnect links per device
 * @return RESULT_SUCCESS on success, error code otherwise
 *
 * Note: Caller must allocate interconnects arrays for each device metric.
 *       Each metrics[i].interconnects should point to an array of size maxInterconnectPerDevice.
 *       The function will fill interconnectCount for each device.
 */
Result GetExtendedDeviceMetrics(
    const char** deviceUUIDArray,
    size_t deviceCount,
    ExtendedDeviceMetrics* metrics,
    size_t maxInterconnectPerDevice
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
