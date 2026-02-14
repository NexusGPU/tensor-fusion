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
 typedef int pid_t;
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
 typedef struct {
   char uuid[64];               // Device UUID
   char vendor[32];             // Vendor name (e.g., "NVIDIA", "AMD")
   char model[128];             // Model name (e.g., "A100", "H100")
   char driverVersion[80];      // Driver version
   char firmwareVersion[64];    // Firmware version
   int32_t index;               // Device index
   int32_t numaNode;            // NUMA node ID (-1 if not assigned)
   uint64_t totalMemoryBytes;   // Total memory in bytes
   uint64_t totalComputeUnits;  // Total compute units (e.g., SMs for NVIDIA)
   double maxTflops;            // Maximum TFLOPS
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
 
 // Device node host/guest path pair
 #define MAX_DEVICE_NODES 32
 #define MAX_DEVICE_PATH 512
 
 typedef struct {
   char hostPath[MAX_DEVICE_PATH];   // Host device path (e.g., /dev/davinci0)
   char guestPath[MAX_DEVICE_PATH];  // Guest device path (e.g., /dev/davinci0)
 } DeviceNodeKV;
 
 // Device node list
 // Used for exposing device node mappings from discovery to the Go layer.
 typedef struct {
   DeviceNodeKV nodes[MAX_DEVICE_NODES];  // Array of host/guest device node mappings
   size_t count;                          // Number of mappings (0 to MAX_DEVICE_NODES)
 } DeviceNodes;
 
 // Extended device information
 typedef struct {
   DeviceBasicInfo basic;
   DeviceProperties props;
   VirtualizationCapabilities virtualizationCapabilities;
   DeviceNodes deviceNodes;
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
 
 // Topology node: represents connection to another device
 typedef struct {
   char peerUUID[64];        // Peer device UUID
   int32_t peerIndex;        // Peer device index
   TopoLevelType topoLevel;  // Topology level to this peer
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
   pid_t* processIds;       // Array of process IDs (for process-level snapshot, NULL for device-level)
   size_t processCount;     // Number of processes (0 for device-level snapshot)
   const char* deviceUUID;  // Device UUID (for device-level snapshot, NULL for process-level)
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
 
 // Process information (combines compute and memory utilization)
 // Based on AMD SMI amdsmi_proc_info_t structure
 typedef struct {
   char processId[32];   // Process ID as string
   char deviceUUID[64];  // Device UUID
   // Compute utilization
   double computeUtilizationPercent;  // Compute utilization percentage (0-100)
   uint64_t activeSMs;                // Active SMs/Compute Units
   uint64_t totalSMs;                 // Total SMs/Compute Units
   // Memory utilization
   uint64_t memoryUsedBytes;         // Memory used in bytes
   uint64_t memoryReservedBytes;     // Memory reserved in bytes
   double memoryUtilizationPercent;  // Memory utilization percentage (0-100)
 } ProcessInformation;
 
 // Basic device metrics
 typedef struct {
   char deviceUUID[64];                          // Device UUID
   double powerUsageWatts;                       // Current power usage (W)
   double temperatureCelsius;                    // Temperature (C)
   uint64_t pcieRxBytes;                         // PCIe RX bytes
   uint64_t pcieTxBytes;                         // PCIe TX bytes
   uint32_t utilizationPercent;                  // Utilization percentage (0-100)
   uint64_t memoryUsedBytes;                     // Memory used
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
 
typedef struct {
  PartitionResultType type;
  char deviceUUID[64];    // Device UUID
  char envVars[10][256];  // Array of environment variable key-value pairs, A=B, C=D, etc.
  // Optional device node mappings for PARTITION_TYPE_DEVICE_NODE mode.
  // Host path is mounted from node, guest path is exposed inside container.
  DeviceNodes deviceNodes;
} PartitionResult;
 
 // ============================================================================
 // Initialization APIs
 // ============================================================================
 
 /**
  * Initialize the accelerator library.
  *
  * This must be called before any other accelerator API. Calls to other APIs
  * without a successful AccelInit will trigger a TF_PANIC.
  *
  * @return ACCEL_SUCCESS on success, error code otherwise
  */
 ACCELERATOR_API AccelResult AccelInit(void);
 
 /**
  * Shutdown the accelerator library.
  *
  * @return ACCEL_SUCCESS on success, error code otherwise
  */
 ACCELERATOR_API AccelResult AccelShutdown(void);
 
 // ============================================================================
 // DeviceInfo APIs
 // ============================================================================
 
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
  * @param processInfos Output buffer for process information (allocated by caller)
  * @param maxCount Maximum number of process infos that can fit in the buffer
  * @param processInfoCount Output parameter for number of process infos actually returned
  * @return ACCEL_SUCCESS on success, error code otherwise
  */
 ACCELERATOR_API AccelResult AccelGetProcessInformation(ProcessInformation* processInfos, size_t maxCount, size_t* processInfoCount);
 
 /**
  * Get basic device metrics (power, PCIe, SM active, TC usage, etc.).
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
 
