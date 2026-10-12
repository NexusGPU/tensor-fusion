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
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__MINGW64__) && !defined(__CYGWIN__)
typedef int accel_pid_t;
#else
typedef pid_t accel_pid_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(ACCELERATOR_EXPORTS)
#define ACCELERATOR_API __declspec(dllexport)
#else
#define ACCELERATOR_API __declspec(dllimport)
#endif
#else
#define ACCELERATOR_API __attribute__((visibility("default")))
#endif

// ============================================================================
// Common Types
// ============================================================================

typedef enum {
  ACCEL_SUCCESS = 0,
  ACCEL_ERROR_INVALID_PARAM = 1,
  ACCEL_ERROR_NOT_FOUND = 2,
  ACCEL_ERROR_NOT_SUPPORTED = 3,
  ACCEL_ERROR_RESOURCE_EXHAUSTED = 4,
  ACCEL_ERROR_OPERATION_FAILED = 5,
  ACCEL_ERROR_INTERNAL = 6
} AccelResult;

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
  bool supportsPartitioning;     // e.g., MIG support
  bool supportsSoftIsolation;    // Hook-based isolation support
  bool supportsHardIsolation;    // One-time limit support
  bool supportsSnapshot;         // Process snapshot/resume support
  bool supportsMetrics;          // Metrics collection support
  bool supportsRemoting;         // Remote device access support
  uint32_t maxPartitions;        // Maximum number of partitions
  uint32_t maxWorkersPerDevice;  // Maximum workers per device
} VirtualizationCapabilities;

// Basic device information
//
// maxTflops is a theoretical peak estimated by the provider, not a
// vendor-published or benchmarked figure. NVIDIA: FP32 TFLOPS = CUDA cores x
// max SM clock x 2 (one FMA per core per cycle); 0 when the inputs are
// unavailable. TensorFusion schedules and limits in vendor-published FP16
// TFLOPS, which differ widely (RTX 3090: 44.08 here vs 71 published, A100:
// ~19.5 vs 312), so maxTflops must not be used as scheduling capacity when a
// configured FP16 figure exists for the model.
typedef struct {
  char uuid[64];               // Device UUID
  char vendor[32];             // Vendor name (e.g., "NVIDIA", "AMD")
  char model[128];             // Model name (e.g., "A100", "H100")
  char driverVersion[80];      // Driver version
  char firmwareVersion[64];    // Firmware version
  char deviceNode[64];         // Device node path (e.g., "/dev/nvidia0")
  int32_t index;               // Device index
  int32_t numaNode;            // NUMA node ID (-1 if not assigned)
  uint64_t totalMemoryBytes;   // Total memory in bytes
  uint64_t totalComputeUnits;  // Total compute units (e.g., SMs for NVIDIA)
  double maxTflops;            // Estimated theoretical FP32 peak in TFLOPS, not schedulable capacity (see above)
  uint32_t pcieGen;            // PCIe generation
  uint32_t pcieWidth;          // PCIe width (lanes)
} DeviceBasicInfo;

// Device property key-value pair
typedef struct {
  char key[64];     // Property key
  char value[256];  // Property value
} DevicePropertyKV;

// Maximum number of device properties
#define MAX_DEVICE_PROPERTIES 64

// Device properties (key-value pairs)
typedef struct {
  DevicePropertyKV properties[MAX_DEVICE_PROPERTIES];  // Array of property key-value pairs
  size_t count;                                        // Number of properties (0 to MAX_DEVICE_PROPERTIES)
} DeviceProperties;

// Extended device information
typedef struct {
  DeviceBasicInfo basic;
  DeviceProperties props;
  VirtualizationCapabilities virtualizationCapabilities;
} ExtendedDeviceInfo;

// Partition template for hardware partitioning (e.g., MIG)
typedef struct {
  char templateId[64];    // Template identifier
  char name[128];         // Human-readable name
  uint64_t memoryBytes;   // Memory allocated to partition
  uint64_t computeUnits;  // Compute units allocated
  double tflops;          // TFLOPS for this partition
  uint32_t sliceCount;    // Number of slices (for MIG)
  bool isDefault;         // Is this a default template
  char description[256];  // Description
} PartitionTemplate;

// Topology level type (GPU-to-GPU connection type)
typedef enum {
  TOPO_LEVEL_INTERNAL = 0,       // e.g. Tesla K80 (same board)
  TOPO_LEVEL_SINGLE_SWITCH = 1,  // single PCIe switch
  TOPO_LEVEL_MULTI_SWITCH = 2,   // multiple PCIe switches (no host bridge traversal)
  TOPO_LEVEL_HOST_BRIDGE = 3,    // same host bridge
  TOPO_LEVEL_NUMA_NODE = 4,      // same NUMA node
  TOPO_LEVEL_SYSTEM = 5,         // cross NUMA (system level)
  TOPO_LEVEL_SELF = 6,           // same device
  TOPO_LEVEL_UNKNOWN = 7         // unknown or error
} TopoLevelType;

// Estimated bandwidth constants (bytes/sec) for common interconnect types.
// Used as fallback when runtime bandwidth query is unavailable.
#define BANDWIDTH_NVLINK_DEFAULT (600LL * 1024 * 1024 * 1024)  // ~600 GB/s (NVLink 3.0+ bidirectional)

// Topology node: represents connection to another device
typedef struct {
  char peerUUID[64];             // Peer device UUID
  int32_t peerIndex;             // Peer device index
  TopoLevelType topoLevel;       // Topology level to this peer
  int64_t bandwidthBytesPerSec;  // Estimated link bandwidth in bytes/sec, 0 if unknown
} DeviceTopoNode;

// Maximum number of devices in topology matrix
#define MAX_TOPOLOGY_DEVICES 64

// Device topology row: a device and its topology to all other devices
typedef struct {
  char deviceUUID[64];                         // This device's UUID
  int32_t deviceIndex;                         // This device's index
  int32_t numaNode;                            // This device's NUMA node
  DeviceTopoNode peers[MAX_TOPOLOGY_DEVICES];  // Topology to all other devices
  size_t peerCount;                            // Number of peers
} DeviceTopologyInfo;

// Extended topology
typedef struct {
  DeviceTopologyInfo devices[MAX_TOPOLOGY_DEVICES];  // Array of device topology rows
  size_t deviceCount;                                // Number of devices
} ExtendedDeviceTopology;

// ============================================================================
// Virtualization Types
// ============================================================================

// Snapshot context for snapshot/resume operations
// Supports both process-level (CUDA) and device-level (other vendors) snapshots
typedef struct {
  accel_pid_t* processIds;  // Array of process IDs (for process-level snapshot, NULL for device-level)
  size_t processCount;      // Number of processes (0 for device-level snapshot)
  const char* deviceUUID;   // Device UUID (for device-level snapshot, NULL for process-level)
} SnapshotContext;

// Maximum environment variable entries per partition
#define MAX_PARTITION_ENVS 16
#define MAX_ENV_KEY_LENGTH 64
#define MAX_ENV_VALUE_LENGTH 256

// Process array for snapshot/resume
#define MAX_PROCESSES 1024

// ============================================================================
// Metrics Types
// ============================================================================

// Extra metric key-value pair
typedef struct {
  char key[64];  // Metric key name
  double value;  // Metric value
} ExtraMetric;

// Maximum number of extra metrics per device
#define MAX_EXTRA_METRICS 64

// ---------------------------------------------------------------------------
// Optional features (AccelSetFeatures)
//
// A caller written before a feature existed would read its new values as plain
// measurements, so every feature is off until the caller asks for it.
//
// ACCEL_FEATURE_TELEMETRY_MARKERS
//   off (default): ProcessInformation.computeUtilizationPercent is always 0
//     (not measured), and a device whose process list, utilization or memory
//     cannot be read fails the whole AccelGetProcessInformation or
//     AccelGetDeviceMetrics call. No marker below is ever returned.
//   on: per-process utilization is measured, and what cannot be read is
//     marked in-band instead of failing the call:
//     - ProcessInformation.computeUtilizationPercent is the process's share of
//       the device's compute time over the last second (0-100; the processes
//       of a device add up to its utilization; 0.0 means "measured, idle"),
//       or ACCEL_COMPUTE_UTILIZATION_UNAVAILABLE for every process of a device
//       that cannot supply it (unsupported driver, MIG or virtualized device,
//       failed or inconsistent query).
//     - A device whose process list cannot be read contributes exactly one
//       marker row: processId is the empty string, deviceUUID names the
//       device, computeUtilizationPercent is
//       ACCEL_COMPUTE_UTILIZATION_UNAVAILABLE. Nothing is known about that
//       device's processes in this call: it is not "no processes".
//     - DeviceMetrics.utilizationPercent is ACCEL_DEVICE_UTILIZATION_UNAVAILABLE
//       and DeviceMetrics.memoryUsedBytes is ACCEL_DEVICE_MEMORY_UNAVAILABLE
//       when that value could not be read. Each field is marked on its own;
//       the others still hold what was read. Both are marked when the device
//       cannot be opened at all.
//     A call still fails when no requested device could be read.
// ---------------------------------------------------------------------------
#define ACCEL_FEATURE_TELEMETRY_MARKERS (1ULL << 0)

// Markers returned only with ACCEL_FEATURE_TELEMETRY_MARKERS. They lie outside
// the range of the field they mark and need no struct layout change.
#define ACCEL_COMPUTE_UTILIZATION_UNAVAILABLE (-1.0)
#define ACCEL_DEVICE_UTILIZATION_UNAVAILABLE UINT32_MAX
#define ACCEL_DEVICE_MEMORY_UNAVAILABLE UINT64_MAX

// Process information (combines compute and memory utilization)
// Based on AMD SMI amdsmi_proc_info_t structure
typedef struct {
  char processId[32];   // Process ID as string (host PID); empty in a device marker row, see ACCEL_FEATURE_TELEMETRY_MARKERS
  char deviceUUID[64];  // Device UUID
  // Compute utilization
  double computeUtilizationPercent;  // 0 unless ACCEL_FEATURE_TELEMETRY_MARKERS: then 0-100 or ACCEL_COMPUTE_UTILIZATION_UNAVAILABLE
  uint64_t activeSMs;                // Active SMs/Compute Units
  uint64_t totalSMs;                 // Total SMs/Compute Units
  // Memory utilization
  uint64_t memoryUsedBytes;         // Memory used in bytes (0 when the driver cannot report it)
  uint64_t memoryReservedBytes;     // Memory reserved in bytes
  double memoryUtilizationPercent;  // Memory utilization percentage (0-100)
} ProcessInformation;

// Basic device metrics
typedef struct {
  char deviceUUID[64];                          // Device UUID
  double powerUsageWatts;                       // Current power usage (W)
  double temperatureCelsius;                    // Temperature (C)
  uint64_t pcieRxBytes;                         // PCIe RX throughput, last sample (bytes/s); sampled at most every 10 s
  uint64_t pcieTxBytes;                         // PCIe TX throughput, last sample (bytes/s); sampled at most every 10 s
  uint32_t utilizationPercent;                  // Utilization percentage (0-100); with markers: or ACCEL_DEVICE_UTILIZATION_UNAVAILABLE
  uint64_t memoryUsedBytes;                     // Memory used; with markers: or ACCEL_DEVICE_MEMORY_UNAVAILABLE
  ExtraMetric extraMetrics[MAX_EXTRA_METRICS];  // Array of extra metrics
  size_t extraMetricsCount;                     // Number of extra metrics (0 to MAX_EXTRA_METRICS)
} DeviceMetrics;

// Device UUID array entry (for passing multiple UUIDs)
// DEPRECATED: This struct is no longer used. AccelGetDeviceMetrics now uses const char** instead.
#define MAX_DEVICE_UUIDS 64
#define UUID_STRING_LENGTH 64

// Mount structure for vendor libraries
#define MAX_MOUNT_PATH 512

typedef struct {
  char hostPath[MAX_MOUNT_PATH];   // Host path
  char guestPath[MAX_MOUNT_PATH];  // Guest path
} MountPath;

typedef enum {
  PARTITION_TYPE_ENVIRONMENT_VARIABLE = 0,
  PARTITION_TYPE_DEVICE_NODE = 1,
} PartitionResultType;

// Maximum environment variable entries per partition
#define MAX_PARTITION_ENVS 16
#define MAX_ENV_KEY_LENGTH 64
#define MAX_ENV_VALUE_LENGTH 256

// Maximum device-node entries returned by partition APIs.
#define MAX_PARTITION_DEVICE_NODES 16
// Device-node entry format is "hostPath=guestPath".
#define MAX_PARTITION_DEVICE_NODE_LENGTH (MAX_MOUNT_PATH * 2 + 2)

typedef struct {
  PartitionResultType type;
  char deviceUUID[64];  // Device UUID
  // Array of environment variable key-value pairs, A=B, C=D, etc.
  char envVars[MAX_PARTITION_ENVS][MAX_ENV_VALUE_LENGTH];
  // Array of device node mapping key-value pairs, hostPath=guestPath.
  char deviceNodes[MAX_PARTITION_DEVICE_NODES][MAX_PARTITION_DEVICE_NODE_LENGTH];
} PartitionResult;

// ============================================================================
// Initialization APIs
// ============================================================================

/**
 * Initialize the accelerator library.
 *
 * This must be called before device operations. Calls to device operations
 * without a successful AccelInit return ACCEL_ERROR_OPERATION_FAILED.
 * Initialization failures return an error and discard partial provider state.
 * Initialization, shutdown, feature selection and partition changes are
 * serialized against every other call. The read-only queries (device count,
 * devices, topology, process information, device metrics) may run concurrently
 * with each other, so a slow query does not delay another caller's.
 *
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelInit(void);

/**
 * Shutdown the accelerator library.
 * Waits for the current API call to finish. Callers must stop issuing new device
 * operations before shutdown and initialize again before resuming them.
 *
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelShutdown(void);

/**
 * Select the optional features (ACCEL_FEATURE_* bits) the caller understands.
 *
 * Call it after a successful AccelInit and before starting to poll; the
 * selection replaces the previous one and is reset by AccelShutdown. Bits the
 * library does not know are ignored, so check *enabled, not the return value,
 * to learn what is active.
 *
 * This function was added after the first release of this ABI. A caller must
 * resolve it as an optional symbol: a library without it has none of the
 * features and behaves as with all of them off.
 *
 * @param requested Bitwise OR of the ACCEL_FEATURE_* bits to enable (0 disables all)
 * @param enabled Output parameter for the bits now active (may be NULL)
 * @return ACCEL_SUCCESS, or ACCEL_ERROR_OPERATION_FAILED before AccelInit
 */
ACCELERATOR_API AccelResult AccelSetFeatures(uint64_t requested, uint64_t* enabled);

// ============================================================================
// DeviceInfo APIs
// ============================================================================

// NOTE: The DeviceInfo, Virtualization, and Metrics APIs require a successful
// AccelInit. If called before initialization, they return ACCEL_ERROR_OPERATION_FAILED.

/**
 * Get the number of available devices.
 *
 * @param deviceCount Output parameter for number of devices
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelGetDeviceCount(size_t* deviceCount);

/**
 * Get all available devices information.
 *
 * @param devices Output buffer for device information (allocated by caller)
 * @param maxCount Maximum number of devices that can fit in the buffer
 * @param deviceCount Output parameter for number of devices actually returned
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelGetAllDevices(ExtendedDeviceInfo* devices, size_t maxCount, size_t* deviceCount);

/**
 * Get device topology for all devices, including NVLink and PCIe interconnects.
 * Similar to nvidia-smi topo -m output.
 *
 * @param topology Output parameter for extended topology (allocated by caller)
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelGetAllDevicesTopology(ExtendedDeviceTopology* topology);

// ============================================================================
// Virtualization APIs - Partitioned Isolation
// ============================================================================

/**
 * Assign a partition to a device using a template (e.g., create MIG instance).
 *
 * @param templateId Template ID to use for creating the partition
 * @param deviceUUID Target device UUID
 * @param partitionResult Output buffer for assigned partition result (callee allocates)
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelAssignPartition(const char* templateId, const char* deviceUUID, PartitionResult* partitionResult);

/**
 * Remove a partition from a device.
 *
 * @param templateId Template ID used to create the partition
 * @param deviceUUID Device UUID
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelRemovePartition(const char* templateId, const char* deviceUUID);

// ============================================================================
// Virtualization APIs - Hard Isolation
// ============================================================================

/**
 * Set hard memory limit for a worker (one-time, called at worker start by limiter.so).
 *
 * @param deviceUUID Device UUID
 * @param memoryLimitBytes Memory limit in bytes
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelSetMemHardLimit(const char* deviceUUID, uint64_t memoryLimitBytes);

/**
 * Set hard compute unit limit for a worker (one-time, called at worker start).
 *
 * @param deviceUUID Device UUID
 * @param computeUnitLimit Compute unit limit (e.g., percentage 0-100)
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelSetComputeUnitHardLimit(const char* deviceUUID, uint32_t computeUnitLimit);

// ============================================================================
// Virtualization APIs - Device Snapshot/Migration
// ============================================================================

/**
 * Snapshot device/process state (lock and checkpoint state).
 * Called from hypervisor for migration.
 *
 * For process-level snapshot (e.g., CUDA):
 *   - Set processIds and processCount, deviceUUID can be NULL
 * For device-level snapshot (e.g., other vendors):
 *   - Set deviceUUID, processIds can be NULL and processCount = 0
 *
 * @param context Snapshot context containing process IDs and/or device UUID
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelSnapshot(SnapshotContext* context);

/**
 * Resume device/process state (unlock and restore state).
 * Called from hypervisor after migration.
 *
 * For process-level resume (e.g., CUDA):
 *   - Set processIds and processCount, deviceUUID can be NULL
 * For device-level resume (e.g., other vendors):
 *   - Set deviceUUID, processIds can be NULL and processCount = 0
 *
 * @param context Snapshot context containing process IDs and/or device UUID
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelResume(SnapshotContext* context);

// ============================================================================
// Metrics APIs
// ============================================================================

/**
 * Get process information (compute and memory utilization) for all processes on all devices.
 * This combines the functionality of GetProcessComputeUtilization and GetProcessMemoryUtilization
 * into a single call, following AMD SMI style API design.
 *
 * A process is listed once per device it uses, compute and graphics contexts
 * alike. Utilization and failure reporting depend on
 * ACCEL_FEATURE_TELEMETRY_MARKERS; memory fields are valid in every row except
 * a device marker row.
 *
 * If the rows do not fit in maxCount the call returns
 * ACCEL_ERROR_RESOURCE_EXHAUSTED and no rows (a truncated list would
 * under-account memory): call again with a larger buffer.
 *
 * @param processInfos Output buffer for process information (allocated by caller)
 * @param maxCount Maximum number of process infos that can fit in the buffer
 * @param processInfoCount Output parameter for number of process infos actually returned
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelGetProcessInformation(ProcessInformation* processInfos, size_t maxCount, size_t* processInfoCount);

/**
 * Get basic device metrics (power, PCIe, SM active, TC usage, etc.).
 *
 * Every value is read on each call except PCIe throughput, which NVML measures
 * over a blocking 20 ms window per counter: it is sampled at most every 10 s per
 * device and the last sample is returned in between (0 until the first one).
 * How an unreadable device is reported depends on
 * ACCEL_FEATURE_TELEMETRY_MARKERS.
 *
 * @param deviceUUIDs Array of device UUID strings (null-terminated C strings)
 * @param deviceCount Number of devices
 * @param metrics Output buffer for device metrics (allocated by caller, size >= deviceCount)
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelGetDeviceMetrics(const char** deviceUUIDs, size_t deviceCount, DeviceMetrics* metrics);

/**
 * Get vendor mount libs returns the mount paths for additional device driver or runtime libraries.
 *
 * @param mounts Output buffer for vendor mount libs (allocated by caller)
 * @param maxCount Maximum number of mounts that can fit in the buffer
 * @param mountCount Output parameter for number of mounts actually returned
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelGetVendorMountLibs(MountPath* mounts, size_t maxCount, size_t* mountCount);

// ============================================================================
// Utility APIs
// ============================================================================

/**
 * Register a log callback function.
 * The callback will be called by the library to log messages.
 *
 * @param callback Log callback function pointer (can be NULL to unregister)
 * @return ACCEL_SUCCESS on success, error code otherwise
 */
ACCELERATOR_API AccelResult AccelRegisterLogCallback(LogCallbackFunc callback);

#ifdef __cplusplus
}
#endif

#endif  // ACCELERATOR_H
