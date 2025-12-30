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
	"unsafe"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/ebitengine/purego"
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

type IsolationMode int32

const (
	IsolationModeShared      IsolationMode = 0
	IsolationModeSoft        IsolationMode = 1
	IsolationModeHard        IsolationMode = 2
	IsolationModePartitioned IsolationMode = 3
)

type DeviceCapabilities struct {
	SupportsPartitioning  bool
	SupportsSoftIsolation bool
	SupportsHardIsolation bool
	SupportsSnapshot      bool
	SupportsMetrics       bool
	MaxPartitions         uint32
	MaxWorkersPerDevice   uint32
}

type DeviceBasicInfo struct {
	UUID              [64]byte
	Vendor            [32]byte
	Model             [128]byte
	DriverVersion     [64]byte
	FirmwareVersion   [64]byte
	Index             int32
	NUMANode          int32
	TotalMemoryBytes  uint64
	TotalComputeUnits uint64
	MaxTflops         float64
	PCIeGen           uint32
	PCIeWidth         uint32
}

type DeviceProperties struct {
	ClockGraphics          uint32
	ClockSM                uint32
	ClockMem               uint32
	ClockAI                uint32
	PowerLimit             uint32
	TemperatureThreshold   uint32
	ECCEnabled             bool
	PersistenceModeEnabled bool
	ComputeCapability      [16]byte
	ChipType               [32]byte
}

type RelatedDevice struct {
	DeviceUUID     [64]byte
	ConnectionType [32]byte
	BandwidthMBps  uint32
	LatencyNs      uint32
}

const MaxRelatedDevices = 32

type ExtendedDeviceInfo struct {
	Basic              DeviceBasicInfo
	Props              DeviceProperties
	RelatedDevices     [MaxRelatedDevices]RelatedDevice
	RelatedDeviceCount uintptr
	Capabilities       DeviceCapabilities
}

type PartitionAssignment struct {
	TemplateID    [64]byte
	DeviceUUID    [64]byte
	PartitionUUID [64]byte
}

type ExtraMetric struct {
	Key   [64]byte
	Value float64
}

const MaxExtraMetrics = 64

type ComputeUtilization struct {
	ProcessID          [32]byte
	DeviceUUID         [64]byte
	UtilizationPercent float64
	ActiveSMs          uint64
	TotalSMs           uint64
	TflopsUsed         float64
}

type MemoryUtilization struct {
	ProcessID          [32]byte
	DeviceUUID         [64]byte
	UsedBytes          uint64
	ReservedBytes      uint64
	UtilizationPercent float64
}

type DeviceMetrics struct {
	DeviceUUID             [64]byte
	PowerUsageWatts        float64
	TemperatureCelsius     float64
	PCIeRxBytes            uint64
	PCIeTxBytes            uint64
	SMActivePercent        uint32
	TensorCoreUsagePercent uint32
	MemoryUsedBytes        uint64
	MemoryTotalBytes       uint64
	ExtraMetrics           [MaxExtraMetrics]ExtraMetric
	ExtraMetricsCount      uintptr
}

const (
	MaxNvlinkPerDevice      = 18
	MaxIbNicPerDevice       = 8
	MaxPciePerDevice        = 4
	MaxConnectionsPerDevice = 32
	MaxTopologyDevices      = 64
)

type DeviceTopology struct {
	DeviceUUID      [64]byte
	NUMANode        int32
	Connections     [MaxConnectionsPerDevice]RelatedDevice
	ConnectionCount uintptr
}

type ExtendedDeviceTopology struct {
	Devices             [MaxTopologyDevices]DeviceTopology
	DeviceCount         uintptr
	NvlinkBandwidthMBps uint32
	IbNicCount          uint32
	TopologyType        [32]byte
}

type ExtendedDeviceMetrics struct {
	DeviceUUID          [64]byte
	NvlinkBandwidthMBps [MaxNvlinkPerDevice]uint32
	NvlinkCount         uintptr
	IbNicBandwidthMBps  [MaxIbNicPerDevice]uint64
	IbNicCount          uintptr
	PCIeBandwidthMBps   [MaxPciePerDevice]uint32
	PCIeLinkCount       uintptr
}

type DeviceUUIDEntry struct {
	UUID [64]byte
}

const MaxMountPath = 512

type Mount struct {
	HostPath  [MaxMountPath]byte
	GuestPath [MaxMountPath]byte
}

const MaxProcesses = 1024

type ProcessArray struct {
	ProcessIDs   [MaxProcesses]int32
	ProcessCount uintptr
	DeviceUUID   [64]byte
}

// Function pointer types for purego
var (
	libHandle uintptr
	// DeviceInfo APIs
	getDeviceCount    func(*uintptr) Result
	getAllDevices     func(*ExtendedDeviceInfo, uintptr, *uintptr) Result
	getDeviceTopology func(*int32, uintptr, *ExtendedDeviceTopology) Result
	// Virtualization APIs
	assignPartition         func(*PartitionAssignment) bool
	removePartition         func(*byte, *byte) bool
	setMemHardLimit         func(*byte, *byte, uint64) Result
	setComputeUnitHardLimit func(*byte, *byte, uint32) Result
	snapshot                func(*ProcessArray) Result
	resume                  func(*ProcessArray) Result
	// Metrics APIs
	getProcessComputeUtilization func(*ComputeUtilization, uintptr, *uintptr) Result
	getProcessMemoryUtilization  func(*MemoryUtilization, uintptr, *uintptr) Result
	getDeviceMetrics             func(*DeviceUUIDEntry, uintptr, *DeviceMetrics) Result
	getExtendedDeviceMetrics     func(*DeviceUUIDEntry, uintptr, *ExtendedDeviceMetrics) Result
	getVendorMountLibs           func(*Mount, uintptr, *uintptr) Result
	// Utility APIs
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

// Load loads the accelerator library dynamically using purego
func (a *AcceleratorInterface) Load() error {
	if a.libPath == "" {
		return fmt.Errorf("library path is empty")
	}

	handle, err := purego.Dlopen(a.libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("failed to open library: %w", err)
	}
	libHandle = handle

	// Register all required functions
	purego.RegisterLibFunc(&getDeviceCount, handle, "GetDeviceCount")
	purego.RegisterLibFunc(&getAllDevices, handle, "GetAllDevices")
	purego.RegisterLibFunc(&getDeviceTopology, handle, "GetDeviceTopology")
	purego.RegisterLibFunc(&assignPartition, handle, "AssignPartition")
	purego.RegisterLibFunc(&removePartition, handle, "RemovePartition")
	purego.RegisterLibFunc(&setMemHardLimit, handle, "SetMemHardLimit")
	purego.RegisterLibFunc(&setComputeUnitHardLimit, handle, "SetComputeUnitHardLimit")
	purego.RegisterLibFunc(&snapshot, handle, "Snapshot")
	purego.RegisterLibFunc(&resume, handle, "Resume")
	purego.RegisterLibFunc(&getProcessComputeUtilization, handle, "GetProcessComputeUtilization")
	purego.RegisterLibFunc(&getProcessMemoryUtilization, handle, "GetProcessMemoryUtilization")
	purego.RegisterLibFunc(&getDeviceMetrics, handle, "GetDeviceMetrics")
	purego.RegisterLibFunc(&getExtendedDeviceMetrics, handle, "GetExtendedDeviceMetrics")
	purego.RegisterLibFunc(&getVendorMountLibs, handle, "GetVendorMountLibs")

	// Register log callback (optional - may not exist in stub libraries)
	func() {
		defer func() {
			if r := recover(); r != nil {
				// RegisterLogCallback not available in this library, skip callback registration
				klog.V(4).Info("RegisterLogCallback not available in library, skipping log callback registration")
			}
		}()
		purego.RegisterLibFunc(&registerLogCallback, handle, "RegisterLogCallback")

		// If registration succeeded, register the callback
		logCallbackPtr := purego.NewCallback(goLogCallback)
		if registerLogCallback(logCallbackPtr) != ResultSuccess {
			klog.Warning("Failed to register log callback")
		}
	}()

	a.loaded = true
	return nil
}

// Close unloads the accelerator library
func (a *AcceleratorInterface) Close() error {
	if a.loaded && libHandle != 0 {
		// Unregister log callback if it was registered
		func() {
			defer func() {
				if r := recover(); r != nil {
					// registerLogCallback not available or already unregistered
				}
			}()
			if registerLogCallback != nil {
				registerLogCallback(0)
			}
		}()
		// Note: purego doesn't provide Dlclose, but the library will be unloaded when process exits
		a.loaded = false
	}
	return nil
}

// goLogCallback is the Go callback function called by C library for logging
func goLogCallback(level *byte, message *byte) {
	var levelStr, messageStr string
	if level != nil {
		levelStr = cStringToGoString(level)
	}
	if message != nil {
		messageStr = cStringToGoString(message)
	}

	// Map C log levels to klog levels
	switch levelStr {
	case "DEBUG", "debug":
		klog.V(4).Info(messageStr)
	case "INFO", "info":
		klog.Info(messageStr)
	case "WARN", "warn", "WARNING", "warning":
		klog.Warning(messageStr)
	case "ERROR", "error":
		klog.Error(messageStr)
	case "FATAL", "fatal":
		klog.Fatal(messageStr)
	default:
		klog.Info(messageStr)
	}
}

// cStringToGoString converts a C string (null-terminated byte array) to Go string
func cStringToGoString(cstr *byte) string {
	if cstr == nil {
		return ""
	}
	ptr := unsafe.Pointer(cstr)
	length := 0
	for *(*byte)(unsafe.Add(ptr, uintptr(length))) != 0 {
		length++
	}
	return string(unsafe.Slice(cstr, length))
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
	deviceCount := len(deviceUUIDs)
	if deviceCount > maxStackDevices {
		deviceCount = maxStackDevices
	}

	// Convert Go strings to DeviceUUIDEntry array
	uuidEntries := make([]DeviceUUIDEntry, deviceCount)
	for i := 0; i < deviceCount; i++ {
		uuidBytes := []byte(deviceUUIDs[i])
		copy(uuidEntries[i].UUID[:], uuidBytes)
		if len(uuidBytes) < len(uuidEntries[i].UUID) {
			uuidEntries[i].UUID[len(uuidBytes)] = 0 // null terminator
		}
	}

	// Allocate stack buffer for metrics
	var cMetrics [maxStackDevices]DeviceMetrics

	result := getDeviceMetrics(&uuidEntries[0], uintptr(deviceCount), &cMetrics[0])
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to get device metrics: %d", result)
	}

	// Convert C metrics to Go metrics
	metrics := make([]*api.GPUUsageMetrics, deviceCount)
	for i := 0; i < deviceCount; i++ {
		cm := &cMetrics[i]
		memoryTotal := uint64(cm.MemoryTotalBytes)
		memoryUsed := uint64(cm.MemoryUsedBytes)
		var memoryPercentage float64
		if memoryTotal > 0 {
			memoryPercentage = float64(memoryUsed) / float64(memoryTotal) * 100.0
		}

		// Convert extra metrics from C array to Go map
		extraMetrics := make(map[string]float64, int(cm.ExtraMetricsCount)+1)
		// Always include tensorCoreUsagePercent as it's a standard field
		extraMetrics["tensorCoreUsagePercent"] = float64(cm.TensorCoreUsagePercent)

		// Add other extra metrics from C array
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
			ComputePercentage: float64(cm.SMActivePercent),
			ComputeTflops:     0,                                // Not available in DeviceMetrics
			Rx:                float64(cm.PCIeRxBytes) / 1024.0, // Convert bytes to KB
			Tx:                float64(cm.PCIeTxBytes) / 1024.0, // Convert bytes to KB
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
	maxDevices := int(cDeviceCount)
	if maxDevices > maxStackDevices {
		maxDevices = maxStackDevices
	}

	var cCount uintptr
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
		devices[i] = &api.DeviceInfo{
			UUID:             byteArrayToString(cInfo.Basic.UUID[:]),
			Vendor:           byteArrayToString(cInfo.Basic.Vendor[:]),
			Model:            byteArrayToString(cInfo.Basic.Model[:]),
			Index:            cInfo.Basic.Index,
			NUMANode:         cInfo.Basic.NUMANode,
			TotalMemoryBytes: uint64(cInfo.Basic.TotalMemoryBytes),
			MaxTflops:        float64(cInfo.Basic.MaxTflops),
			Capabilities: api.DeviceCapabilities{
				SupportsPartitioning:  bool(cInfo.Capabilities.SupportsPartitioning),
				SupportsSoftIsolation: bool(cInfo.Capabilities.SupportsSoftIsolation),
				SupportsHardIsolation: bool(cInfo.Capabilities.SupportsHardIsolation),
				SupportsSnapshot:      bool(cInfo.Capabilities.SupportsSnapshot),
				SupportsMetrics:       bool(cInfo.Capabilities.SupportsMetrics),
				MaxPartitions:         uint32(cInfo.Capabilities.MaxPartitions),
				MaxWorkersPerDevice:   uint32(cInfo.Capabilities.MaxWorkersPerDevice),
			},
			Properties: make(map[string]string, 0),
		}
	}

	return devices, nil
}

// AssignPartition assigns a partition to a device
func (a *AcceleratorInterface) AssignPartition(templateID, deviceUUID string) (string, error) {
	// Validate input lengths
	const maxIDLength = 64
	if len(templateID) >= maxIDLength {
		return "", fmt.Errorf("template ID is too long (max %d bytes)", maxIDLength-1)
	}
	if len(deviceUUID) >= maxIDLength {
		return "", fmt.Errorf("device UUID is too long (max %d bytes)", maxIDLength-1)
	}

	var assignment PartitionAssignment
	templateBytes := []byte(templateID)
	deviceBytes := []byte(deviceUUID)
	copy(assignment.TemplateID[:], templateBytes)
	copy(assignment.DeviceUUID[:], deviceBytes)
	if len(templateBytes) < len(assignment.TemplateID) {
		assignment.TemplateID[len(templateBytes)] = 0
	}
	if len(deviceBytes) < len(assignment.DeviceUUID) {
		assignment.DeviceUUID[len(deviceBytes)] = 0
	}

	result := assignPartition(&assignment)
	if !result {
		return "", fmt.Errorf("failed to assign partition")
	}

	partitionUUID := byteArrayToString(assignment.PartitionUUID[:])
	return partitionUUID, nil
}

// RemovePartition removes a partition from a device
func (a *AcceleratorInterface) RemovePartition(partitionUUID, deviceUUID string) error {
	partitionBytes := []byte(partitionUUID)
	deviceBytes := []byte(deviceUUID)

	// Create temporary arrays with null terminators
	var partitionArr [64]byte
	var deviceArr [64]byte
	copy(partitionArr[:], partitionBytes)
	copy(deviceArr[:], deviceBytes)
	if len(partitionBytes) < len(partitionArr) {
		partitionArr[len(partitionBytes)] = 0
	}
	if len(deviceBytes) < len(deviceArr) {
		deviceArr[len(deviceBytes)] = 0
	}

	result := removePartition(&partitionArr[0], &deviceArr[0])
	if !result {
		return fmt.Errorf("failed to remove partition")
	}

	return nil
}

// SetMemHardLimit sets hard memory limit for a worker
func (a *AcceleratorInterface) SetMemHardLimit(workerID, deviceUUID string, memoryLimitBytes uint64) error {
	workerBytes := []byte(workerID)
	deviceBytes := []byte(deviceUUID)

	var workerArr [64]byte
	var deviceArr [64]byte
	copy(workerArr[:], workerBytes)
	copy(deviceArr[:], deviceBytes)
	if len(workerBytes) < len(workerArr) {
		workerArr[len(workerBytes)] = 0
	}
	if len(deviceBytes) < len(deviceArr) {
		deviceArr[len(deviceBytes)] = 0
	}

	result := setMemHardLimit(&workerArr[0], &deviceArr[0], memoryLimitBytes)
	if result != ResultSuccess {
		return fmt.Errorf("failed to set memory hard limit: %d", result)
	}

	return nil
}

// SetComputeUnitHardLimit sets hard compute unit limit for a worker
func (a *AcceleratorInterface) SetComputeUnitHardLimit(workerID, deviceUUID string, computeUnitLimit uint32) error {
	workerBytes := []byte(workerID)
	deviceBytes := []byte(deviceUUID)

	var workerArr [64]byte
	var deviceArr [64]byte
	copy(workerArr[:], workerBytes)
	copy(deviceArr[:], deviceBytes)
	if len(workerBytes) < len(workerArr) {
		workerArr[len(workerBytes)] = 0
	}
	if len(deviceBytes) < len(deviceArr) {
		deviceArr[len(deviceBytes)] = 0
	}

	result := setComputeUnitHardLimit(&workerArr[0], &deviceArr[0], computeUnitLimit)
	if result != ResultSuccess {
		return fmt.Errorf("failed to set compute unit hard limit: %d", result)
	}

	return nil
}

// GetProcessComputeUtilization retrieves compute utilization for all tracked processes
func (a *AcceleratorInterface) GetProcessComputeUtilization() ([]api.ComputeUtilization, error) {
	// Get total process count from the map
	totalCount := a.GetTotalProcessCount()
	if totalCount == 0 {
		return []api.ComputeUtilization{}, nil
	}

	// Allocate stack buffer (max 1024 to avoid stack overflow)
	const maxStackUtilizations = 1024
	var stackUtilizations [maxStackUtilizations]ComputeUtilization
	maxCount := totalCount
	if maxCount > maxStackUtilizations {
		maxCount = maxStackUtilizations
	}

	var cCount uintptr
	result := getProcessComputeUtilization(&stackUtilizations[0], uintptr(maxCount), &cCount)
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to get process compute utilization: %d", result)
	}

	if cCount == 0 {
		return []api.ComputeUtilization{}, nil
	}

	utilizations := make([]api.ComputeUtilization, int(cCount))
	for i := 0; i < int(cCount); i++ {
		cu := &stackUtilizations[i]
		utilizations[i] = api.ComputeUtilization{
			ProcessID:          byteArrayToString(cu.ProcessID[:]),
			DeviceUUID:         byteArrayToString(cu.DeviceUUID[:]),
			UtilizationPercent: float64(cu.UtilizationPercent),
		}
	}

	return utilizations, nil
}

// GetProcessMemoryUtilization retrieves memory utilization for all tracked processes
func (a *AcceleratorInterface) GetProcessMemoryUtilization() ([]api.MemoryUtilization, error) {
	// Get total process count from the map
	totalCount := a.GetTotalProcessCount()
	if totalCount == 0 {
		return []api.MemoryUtilization{}, nil
	}

	// Allocate stack buffer (max 1024 to avoid stack overflow)
	const maxStackUtilizations = 1024
	var stackUtilizations [maxStackUtilizations]MemoryUtilization
	maxCount := totalCount
	if maxCount > maxStackUtilizations {
		maxCount = maxStackUtilizations
	}

	var cCount uintptr
	result := getProcessMemoryUtilization(&stackUtilizations[0], uintptr(maxCount), &cCount)
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to get process memory utilization: %d", result)
	}

	if cCount == 0 {
		return []api.MemoryUtilization{}, nil
	}

	utilizations := make([]api.MemoryUtilization, int(cCount))
	for i := 0; i < int(cCount); i++ {
		mu := &stackUtilizations[i]
		utilizations[i] = api.MemoryUtilization{
			ProcessID:     byteArrayToString(mu.ProcessID[:]),
			DeviceUUID:    byteArrayToString(mu.DeviceUUID[:]),
			UsedBytes:     uint64(mu.UsedBytes),
			ReservedBytes: uint64(mu.ReservedBytes),
		}
	}

	return utilizations, nil
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
