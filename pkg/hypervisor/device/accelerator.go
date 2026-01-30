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

package device

import (
	"fmt"
	"sync"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"k8s.io/klog/v2"
)

// C structure definitions matching accelerator.h
// These must match the C struct definitions exactly

type Result int32

const (
	ResultSuccess                Result = 0
	ResultErrorInvalidParam      Result = 1
	ResultErrorNotFound          Result = 2
	ResultErrorNotSupported      Result = 3
	ResultErrorResourceExhausted Result = 4
	ResultErrorOperationFailed   Result = 5
	ResultErrorInternal          Result = 6
)

type VirtualizationCapabilities struct {
	SupportsPartitioning  bool
	SupportsSoftIsolation bool
	SupportsHardIsolation bool
	SupportsSnapshot      bool
	SupportsMetrics       bool
	SupportsRemoting      bool
	MaxPartitions         uint32
	MaxWorkersPerDevice   uint32
}

// DeviceBasicInfo matches the C struct DeviceBasicInfo in provider/accelerator.h
// Field names in Go are capitalized for export, but memory layout must match C struct exactly
// C struct fields: uuid, vendor, model, driverVersion, firmwareVersion, index, numaNode,
//
//	totalMemoryBytes, totalComputeUnits, maxTflops, pcieGen, pcieWidth
type DeviceBasicInfo struct {
	UUID              [64]byte  // C: char uuid[64]
	Vendor            [32]byte  // C: char vendor[32]
	Model             [128]byte // C: char model[128]
	DriverVersion     [80]byte  // C: char driverVersion[80]
	FirmwareVersion   [64]byte  // C: char firmwareVersion[64]
	Index             int32     // C: int32_t index
	NUMANode          int32     // C: int32_t numaNode
	TotalMemoryBytes  uint64    // C: uint64_t totalMemoryBytes
	TotalComputeUnits uint64    // C: uint64_t totalComputeUnits
	MaxTflops         float64   // C: double maxTflops
	PCIeGen           uint32    // C: uint32_t pcieGen
	PCIeWidth         uint32    // C: uint32_t pcieWidth
}

type DevicePropertyKV struct {
	Key   [64]byte
	Value [256]byte
}

const MaxDeviceProperties = 64

type DeviceProperties struct {
	Properties [MaxDeviceProperties]DevicePropertyKV
	Count      uintptr
}

type ExtendedDeviceInfo struct {
	Basic        DeviceBasicInfo
	Props        DeviceProperties
	Capabilities VirtualizationCapabilities
}

type ExtraMetric struct {
	Key   [64]byte
	Value float64
}

const MaxExtraMetrics = 64

// ProcessInformation combines compute and memory utilization (AMD SMI style)
type ProcessInformation struct {
	ProcessID                 [32]byte
	DeviceUUID                [64]byte
	ComputeUtilizationPercent float64
	ActiveSMs                 uint64
	TotalSMs                  uint64
	MemoryUsedBytes           uint64
	MemoryReservedBytes       uint64
	MemoryUtilizationPercent  float64
}

type DeviceMetrics struct {
	DeviceUUID         [64]byte
	PowerUsageWatts    float64
	TemperatureCelsius float64
	PCIeRxBytes        uint64
	PCIeTxBytes        uint64
	UtilizationPercent uint32
	MemoryUsedBytes    uint64
	ExtraMetrics       [MaxExtraMetrics]ExtraMetric
	ExtraMetricsCount  uintptr
}

const (
	MaxTopologyDevices = 64
)

// TopoLevelType represents GPU-to-GPU connection type
type TopoLevelType int32

const (
	TopoLevelInternal     TopoLevelType = 0 // e.g. Tesla K80 (same board)
	TopoLevelSingleSwitch TopoLevelType = 1 // single PCIe switch
	TopoLevelMultiSwitch  TopoLevelType = 2 // multiple PCIe switches (no host bridge traversal)
	TopoLevelHostBridge   TopoLevelType = 3 // same host bridge
	TopoLevelNUMANode     TopoLevelType = 4 // same NUMA node
	TopoLevelSystem       TopoLevelType = 5 // cross NUMA (system level)
	TopoLevelSelf         TopoLevelType = 6 // same device
	TopoLevelUnknown      TopoLevelType = 7 // unknown or error
)

// DeviceTopoNode represents connection to another device
type DeviceTopoNode struct {
	PeerUUID  [64]byte      // Peer device UUID
	PeerIndex int32         // Peer device index
	TopoLevel TopoLevelType // Topology level to this peer
}

// DeviceTopologyInfo represents a device and its topology to all other devices
type DeviceTopologyInfo struct {
	DeviceUUID  [64]byte                           // This device's UUID
	DeviceIndex int32                              // This device's index
	NUMANode    int32                              // This device's NUMA node
	Peers       [MaxTopologyDevices]DeviceTopoNode // Topology to all other devices
	PeerCount   uintptr                            // Number of peers
}

// ExtendedDeviceTopology contains topology for all devices
type ExtendedDeviceTopology struct {
	Devices     [MaxTopologyDevices]DeviceTopologyInfo // Array of device topology rows
	DeviceCount uintptr                                // Number of devices
}

const MaxMountPath = 512

type Mount struct {
	HostPath  [MaxMountPath]byte
	GuestPath [MaxMountPath]byte
}

const MaxProcesses = 1024

// PartitionResultType represents the type of partition result
type PartitionResultType int32

const (
	PartitionTypeEnvironmentVariable PartitionResultType = 0
	PartitionTypeDeviceNode          PartitionResultType = 1
)

// PartitionResult matches the C struct PartitionResult in provider/accelerator.h
// Field names in Go are capitalized for export, but memory layout must match C struct exactly
// C struct fields: type, deviceUUID, envVars
type PartitionResult struct {
	Type       PartitionResultType // C: PartitionResultType type
	DeviceUUID [64]byte            // C: char deviceUUID[64]
	EnvVars    [10][256]byte       // C: char envVars[10][256], key-value pairs like "A=B"
}

// SnapshotContext for snapshot/resume operations
// Supports both process-level (CUDA) and device-level (other vendors) snapshots
type SnapshotContext struct {
	ProcessIDs   *int32  // Pointer to array of process IDs (for process-level snapshot, NULL for device-level)
	ProcessCount uintptr // Number of processes (0 for device-level snapshot)
	DeviceUUID   *byte   // Device UUID (for device-level snapshot, NULL for process-level)
}

// Function pointer types for purego
var (
	libHandle uintptr
	// DeviceInfo APIs
	vgpuInit              func() Result
	vgpuShutdown          func() Result
	getDeviceCount        func(*uintptr) Result
	getAllDevices         func(*ExtendedDeviceInfo, uintptr, *uintptr) Result
	getAllDevicesTopology func(*ExtendedDeviceTopology) Result
	// Virtualization APIs - signatures match C header exactly:
	// AccelResult AssignPartition(const char* templateId, const char* deviceUUID, PartitionResult* partitionResult);
	assignPartition func(*byte, *byte, *PartitionResult) Result
	// AccelResult RemovePartition(const char* templateId, const char* deviceUUID);
	removePartition func(*byte, *byte) Result
	// AccelResult SetMemHardLimit(const char* deviceUUID, uint64_t memoryLimitBytes);
	setMemHardLimit func(*byte, uint64) Result
	// AccelResult SetComputeUnitHardLimit(const char* deviceUUID, uint32_t computeUnitLimit);
	setComputeUnitHardLimit func(*byte, uint32) Result
	// AccelResult Snapshot(SnapshotContext* context);
	snapshot func(*SnapshotContext) Result
	// AccelResult Resume(SnapshotContext* context);
	resume func(*SnapshotContext) Result
	// Metrics APIs
	getProcessInformation func(*ProcessInformation, uintptr, *uintptr) Result
	getDeviceMetrics      func(**byte, uintptr, *DeviceMetrics) Result
	getVendorMountLibs    func(*Mount, uintptr, *uintptr) Result
	// Utility APIs (Unix only, set by accelerator_unix.go)
	registerLogCallback func(uintptr) Result
)

// AcceleratorInterface provides Go bindings for the C accelerator library using purego
type AcceleratorInterface struct {
	libPath         string
	deviceProcesses map[string][]string
	mu              sync.RWMutex
	loaded          bool
}

// NewAcceleratorInterface creates a new accelerator interface and loads the library
func NewAcceleratorInterface(libPath string) (*AcceleratorInterface, error) {
	accel := &AcceleratorInterface{
		libPath:         libPath,
		deviceProcesses: make(map[string][]string),
		loaded:          false,
	}

	// Load the library
	if err := accel.Load(); err != nil {
		return nil, fmt.Errorf("failed to load accelerator library from %s: %w", libPath, err)
	}

	return accel, nil
}

// Close unloads the accelerator library
func (a *AcceleratorInterface) Close() error {
	if a.loaded && libHandle != 0 {
		// Unregister log callback if it was registered (Unix only)
		func() {
			defer func() {
				if r := recover(); r != nil {
					// registerLogCallback not available or already unregistered
					_ = r // ignore recovery value
				}
			}()
			vgpuShutdown()
			if registerLogCallback != nil {
				registerLogCallback(0)
			}
		}()
		// Note: purego doesn't provide DlClose, but the library will be unloaded when process exits
		a.loaded = false
	}
	return nil
}

// byteArrayToString converts a fixed-size byte array to Go string
func byteArrayToString(arr []byte) string {
	// Find null terminator
	for i, b := range arr {
		if b == 0 {
			return string(arr[:i])
		}
	}
	return string(arr)
}

// GetTotalProcessCount returns the total number of processes across all devices
func (a *AcceleratorInterface) GetTotalProcessCount() int {
	a.mu.RLock()
	defer a.mu.RUnlock()

	total := 0
	for _, processes := range a.deviceProcesses {
		total += len(processes)
	}
	return total
}

// GetDeviceMetrics retrieves device metrics for the specified device UUIDs
func (a *AcceleratorInterface) GetDeviceMetrics(deviceUUIDs []string) ([]*api.GPUUsageMetrics, error) {
	if len(deviceUUIDs) == 0 {
		return []*api.GPUUsageMetrics{}, nil
	}

	const maxStackDevices = 64
	deviceCount := min(len(deviceUUIDs), maxStackDevices)

	// Convert Go strings to C string pointers array
	// Allocate C strings with null terminators
	cStrings := make([]*byte, deviceCount)
	cStringData := make([][]byte, deviceCount)
	for i := range deviceCount {
		// Convert Go string to null-terminated C string
		cStringData[i] = []byte(deviceUUIDs[i])
		cStringData[i] = append(cStringData[i], 0) // null terminator
		cStrings[i] = &cStringData[i][0]
	}

	// Allocate stack buffer for metrics
	var cMetrics [maxStackDevices]DeviceMetrics

	result := getDeviceMetrics(&cStrings[0], uintptr(deviceCount), &cMetrics[0])
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to get device metrics: %d", result)
	}

	// Convert C metrics to Go metrics
	metrics := make([]*api.GPUUsageMetrics, deviceCount)
	for i := range deviceCount {
		cm := &cMetrics[i]
		memoryUsed := cm.MemoryUsedBytes
		var memoryPercentage float64 = 0

		// Convert extra metrics from C array to Go map
		extraMetrics := make(map[string]float64, int(cm.ExtraMetricsCount))
		// Add extra metrics from C array
		for j := 0; j < int(cm.ExtraMetricsCount); j++ {
			em := &cm.ExtraMetrics[j]
			key := byteArrayToString(em.Key[:])
			if key != "" {
				extraMetrics[key] = em.Value
			}
		}

		metrics[i] = &api.GPUUsageMetrics{
			DeviceUUID:        byteArrayToString(cm.DeviceUUID[:]),
			MemoryBytes:       memoryUsed,
			MemoryPercentage:  memoryPercentage,
			ComputePercentage: float64(cm.UtilizationPercent),
			ComputeTflops:     0,
			Rx:                float64(cm.PCIeRxBytes),
			Tx:                float64(cm.PCIeTxBytes),
			Temperature:       float64(cm.TemperatureCelsius),
			PowerUsage:        int64(cm.PowerUsageWatts),
			ExtraMetrics:      extraMetrics,
		}
	}

	return metrics, nil
}

// GetAllDevices retrieves all available devices from the accelerator library
func (a *AcceleratorInterface) GetAllDevices() ([]*api.DeviceInfo, error) {
	// First, get the device count
	var cDeviceCount uintptr
	result := getDeviceCount(&cDeviceCount)
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to get device count: %d", result)
	}

	if cDeviceCount == 0 {
		return []*api.DeviceInfo{}, nil
	}

	// Allocate stack buffer (max 64 devices to avoid stack overflow)
	const maxStackDevices = 64
	var stackDevices [maxStackDevices]ExtendedDeviceInfo
	maxDevices := min(int(cDeviceCount), maxStackDevices)

	var cCount uintptr
	klog.Infof("Getting all devices, max devices count: %d", maxDevices)
	result = getAllDevices(&stackDevices[0], uintptr(maxDevices), &cCount)
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to get all devices: %d", result)
	}

	if cCount == 0 {
		return []*api.DeviceInfo{}, nil
	}

	devices := make([]*api.DeviceInfo, int(cCount))

	for i := 0; i < int(cCount); i++ {
		cInfo := &stackDevices[i]

		// Convert DeviceProperties KV array to map
		properties := make(map[string]string, int(cInfo.Props.Count))
		for j := 0; j < int(cInfo.Props.Count) && j < MaxDeviceProperties; j++ {
			key := byteArrayToString(cInfo.Props.Properties[j].Key[:])
			value := byteArrayToString(cInfo.Props.Properties[j].Value[:])
			if key != "" {
				properties[key] = value
			}
		}

		devices[i] = &api.DeviceInfo{
			UUID:             byteArrayToString(cInfo.Basic.UUID[:]),
			Vendor:           byteArrayToString(cInfo.Basic.Vendor[:]),
			Model:            byteArrayToString(cInfo.Basic.Model[:]),
			Index:            cInfo.Basic.Index,
			NUMANode:         cInfo.Basic.NUMANode,
			TotalMemoryBytes: cInfo.Basic.TotalMemoryBytes,
			MaxTflops:        float64(cInfo.Basic.MaxTflops),
			VirtualizationCapabilities: api.VirtualizationCapabilities{
				SupportsPartitioning:  cInfo.Capabilities.SupportsPartitioning,
				SupportsSoftIsolation: cInfo.Capabilities.SupportsSoftIsolation,
				SupportsHardIsolation: cInfo.Capabilities.SupportsHardIsolation,
				SupportsSnapshot:      cInfo.Capabilities.SupportsSnapshot,
				SupportsMetrics:       cInfo.Capabilities.SupportsMetrics,
				MaxPartitions:         cInfo.Capabilities.MaxPartitions,
				MaxWorkersPerDevice:   cInfo.Capabilities.MaxWorkersPerDevice,
			},
			Properties: properties,
		}
	}
	return devices, nil
}

// AssignPartitionResult represents the result of assigning a partition
type AssignPartitionResult struct {
	PartitionUUID string
	EnvVars       map[string]string
	Type          PartitionResultType
}

// AssignPartition assigns a partition to a device using a template (e.g., create MIG instance)
// Returns the assigned partition result including UUID and environment variables
func (a *AcceleratorInterface) AssignPartition(templateID, deviceUUID string) (*AssignPartitionResult, error) {
	// Validate input lengths
	const maxIDLength = 64
	if len(templateID) >= maxIDLength {
		return nil, fmt.Errorf("template ID is too long (max %d bytes)", maxIDLength-1)
	}
	if len(deviceUUID) >= maxIDLength {
		return nil, fmt.Errorf("device UUID is too long (max %d bytes)", maxIDLength-1)
	}

	// Create null-terminated C strings
	var templateArr [64]byte
	var deviceArr [64]byte

	copy(templateArr[:], templateID)
	copy(deviceArr[:], deviceUUID)
	if len(templateID) < len(templateArr) {
		templateArr[len(templateID)] = 0
	}
	if len(deviceUUID) < len(deviceArr) {
		deviceArr[len(deviceUUID)] = 0
	}

	// Allocate PartitionResult struct for C function to fill
	var cResult PartitionResult

	result := assignPartition(&templateArr[0], &deviceArr[0], &cResult)
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to assign partition: %d", result)
	}

	// Extract partition UUID from deviceUUID field
	partitionUUID := byteArrayToString(cResult.DeviceUUID[:])

	// Parse optional env vars returned by vendor
	envVars := make(map[string]string)
	if cResult.Type == PartitionTypeEnvironmentVariable {
		for i := range 10 {
			envVarStr := byteArrayToString(cResult.EnvVars[i][:])
			if envVarStr == "" {
				continue
			}
			// Parse key-value pair in format "A=B"
			for j := 0; j < len(envVarStr); j++ {
				if envVarStr[j] == '=' {
					key := envVarStr[:j]
					value := envVarStr[j+1:]
					if key != "" {
						envVars[key] = value
					}
					break
				}
			}
		}
	}

	return &AssignPartitionResult{
		PartitionUUID: partitionUUID,
		EnvVars:       envVars,
		Type:          cResult.Type,
	}, nil
}

// RemovePartition removes a partition from a device
// templateID is the template ID used to create the partition
func (a *AcceleratorInterface) RemovePartition(templateID, deviceUUID string) error {
	// Create null-terminated C strings
	var templateArr [64]byte
	var deviceArr [64]byte

	copy(templateArr[:], templateID)
	copy(deviceArr[:], deviceUUID)
	if len(templateID) < len(templateArr) {
		templateArr[len(templateID)] = 0
	}
	if len(deviceUUID) < len(deviceArr) {
		deviceArr[len(deviceUUID)] = 0
	}

	result := removePartition(&templateArr[0], &deviceArr[0])
	if result != ResultSuccess {
		return fmt.Errorf("failed to remove partition: %d", result)
	}

	return nil
}

// SetMemHardLimit sets hard memory limit for a device (one-time, called at worker start by limiter.so)
func (a *AcceleratorInterface) SetMemHardLimit(deviceUUID string, memoryLimitBytes uint64) error {
	var deviceArr [64]byte
	copy(deviceArr[:], deviceUUID)
	if len(deviceUUID) < len(deviceArr) {
		deviceArr[len(deviceUUID)] = 0
	}

	result := setMemHardLimit(&deviceArr[0], memoryLimitBytes)
	if result != ResultSuccess {
		return fmt.Errorf("failed to set memory hard limit: %d", result)
	}

	return nil
}

// SetComputeUnitHardLimit sets hard compute unit limit for a device (one-time, called at worker start)
func (a *AcceleratorInterface) SetComputeUnitHardLimit(deviceUUID string, computeUnitLimit uint32) error {
	var deviceArr [64]byte
	copy(deviceArr[:], deviceUUID)
	if len(deviceUUID) < len(deviceArr) {
		deviceArr[len(deviceUUID)] = 0
	}

	result := setComputeUnitHardLimit(&deviceArr[0], computeUnitLimit)
	if result != ResultSuccess {
		return fmt.Errorf("failed to set compute unit hard limit: %d", result)
	}

	return nil
}

// GetProcessInformation retrieves process information (compute and memory utilization) for all processes
// on all devices. This combines the functionality of GetProcessComputeUtilization and GetProcessMemoryUtilization
// following AMD SMI style API design.
// Note: This directly calls the C API which returns all GPU processes, regardless of what Go tracks internally.
func (a *AcceleratorInterface) GetProcessInformation() ([]api.ProcessInformation, error) {
	// Allocate stack buffer (max 1024 to avoid stack overflow)
	// The C API GetProcessInformation returns all processes on all devices
	const maxStackProcessInfos = 1024
	var stackProcessInfos [maxStackProcessInfos]ProcessInformation

	var cCount uintptr
	result := getProcessInformation(&stackProcessInfos[0], uintptr(maxStackProcessInfos), &cCount)
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to get process information: %d", result)
	}

	if cCount == 0 {
		return []api.ProcessInformation{}, nil
	}

	processInfos := make([]api.ProcessInformation, int(cCount))
	for i := 0; i < int(cCount); i++ {
		pi := &stackProcessInfos[i]
		processInfos[i] = api.ProcessInformation{
			ProcessID:                 byteArrayToString(pi.ProcessID[:]),
			DeviceUUID:                byteArrayToString(pi.DeviceUUID[:]),
			ComputeUtilizationPercent: float64(pi.ComputeUtilizationPercent),
			ActiveSMs:                 pi.ActiveSMs,
			TotalSMs:                  pi.TotalSMs,
			MemoryUsedBytes:           pi.MemoryUsedBytes,
			MemoryReservedBytes:       pi.MemoryReservedBytes,
			MemoryUtilizationPercent:  pi.MemoryUtilizationPercent,
		}
	}

	return processInfos, nil
}

// GetVendorMountLibs retrieves vendor mount libs
func (a *AcceleratorInterface) GetVendorMountLibs() ([]*api.Mount, error) {
	const maxStackMounts = 64
	var stackMounts [maxStackMounts]Mount
	var cCount uintptr

	result := getVendorMountLibs(&stackMounts[0], uintptr(maxStackMounts), &cCount)
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to get vendor mount libs: %d", result)
	}

	if cCount == 0 {
		return []*api.Mount{}, nil
	}

	mounts := make([]*api.Mount, int(cCount))
	for i := 0; i < int(cCount); i++ {
		cm := &stackMounts[i]
		mounts[i] = &api.Mount{
			HostPath:  byteArrayToString(cm.HostPath[:]),
			GuestPath: byteArrayToString(cm.GuestPath[:]),
		}
	}

	return mounts, nil
}
