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

// DeviceBasicInfo matches the C struct DeviceBasicInfo in vgpu-provider/accelerator.h
// Field names in Go are capitalized for export, but memory layout must match C struct exactly
// C struct fields: uuid, vendor, model, driverVersion, firmwareVersion, index, numaNode,
//
//	totalMemoryBytes, totalComputeUnits, maxTflops, pcieGen, pcieWidth
type DeviceBasicInfo struct {
	UUID              [64]byte  // C: char uuid[64]
	Vendor            [32]byte  // C: char vendor[32]
	Model             [128]byte // C: char model[128]
	DriverVersion     [64]byte  // C: char driverVersion[64]
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

// MaxPartitionEnvs must match MAX_PARTITION_ENVS in accelerator.h
const MaxPartitionEnvs = 16

// MaxEnvKeyLength must match MAX_ENV_KEY_LENGTH in accelerator.h
const MaxEnvKeyLength = 64

// MaxEnvValueLength must match MAX_ENV_VALUE_LENGTH in accelerator.h
const MaxEnvValueLength = 256

// EnvVar matches the C struct EnvVar in accelerator.h
type EnvVar struct {
	Key   [MaxEnvKeyLength]byte
	Value [MaxEnvValueLength]byte
}

type PartitionAssignment struct {
	TemplateID    [64]byte
	DeviceUUID    [64]byte
	PartitionUUID [64]byte
	// Optional env vars returned by vendor-specific partition assignment
	EnvVars     [MaxPartitionEnvs]EnvVar
	EnvVarCount uintptr
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

type DeviceTopology struct {
	DeviceUUID [64]byte
	NUMANode   int32
}

type ExtendedDeviceTopology struct {
	Devices      [MaxTopologyDevices]DeviceTopology
	DeviceCount  uintptr
	TopologyType [32]byte
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
	virtualGPUInit    func() Result
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
	getProcessInformation func(*ProcessInformation, uintptr, *uintptr) Result
	getDeviceMetrics      func(**byte, uintptr, *DeviceMetrics) Result
	getVendorMountLibs    func(*Mount, uintptr, *uintptr) Result
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
	purego.RegisterLibFunc(&virtualGPUInit, handle, "VirtualGPUInit")
	purego.RegisterLibFunc(&getDeviceCount, handle, "GetDeviceCount")
	purego.RegisterLibFunc(&getAllDevices, handle, "GetAllDevices")
	purego.RegisterLibFunc(&getDeviceTopology, handle, "GetDeviceTopology")
	purego.RegisterLibFunc(&assignPartition, handle, "AssignPartition")
	purego.RegisterLibFunc(&removePartition, handle, "RemovePartition")
	purego.RegisterLibFunc(&setMemHardLimit, handle, "SetMemHardLimit")
	purego.RegisterLibFunc(&setComputeUnitHardLimit, handle, "SetComputeUnitHardLimit")
	purego.RegisterLibFunc(&snapshot, handle, "Snapshot")
	purego.RegisterLibFunc(&resume, handle, "Resume")
	purego.RegisterLibFunc(&getProcessInformation, handle, "GetProcessInformation")
	purego.RegisterLibFunc(&getDeviceMetrics, handle, "GetDeviceMetrics")
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

	result := virtualGPUInit()
	if result != ResultSuccess {
		return fmt.Errorf("failed to initialize virtual GPU: %d", result)
	}

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
					_ = r // ignore recovery value
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

		vendor := byteArrayToString(cInfo.Basic.Vendor[:])

		// Initialize device node mappings based on vendor
		deviceNode := make(map[string]string)

		if vendor == "AMD" {
			// AMD GPUs: Look for renderDevice in properties (set by provider)
			// If found, use specific renderD device; otherwise fall back to whole /dev/dri
			if renderDev, ok := properties["renderDevice"]; ok && renderDev != "" {
				klog.Infof("GPU discovery: Found renderDevice=%s for UUID=%s", renderDev, byteArrayToString(cInfo.Basic.UUID[:]))
				deviceNode[renderDev] = renderDev
			} else {
				klog.Warningf("GPU discovery: No renderDevice property for UUID=%s, falling back to /dev/dri", byteArrayToString(cInfo.Basic.UUID[:]))
				// Fallback: mount all /dev/dri (less isolated but works)
				deviceNode["/dev/dri"] = "/dev/dri"
			}
			// Always add /dev/kfd for ROCm
			deviceNode["/dev/kfd"] = "/dev/kfd"
		}

		devices[i] = &api.DeviceInfo{
			UUID:             byteArrayToString(cInfo.Basic.UUID[:]),
			Vendor:           vendor,
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
			DeviceNode: deviceNode,
		}
	}
	return devices, nil
}

// PartitionResult contains the result of assigning a partition
type PartitionResult struct {
	PartitionUUID string
	// EnvVars contains optional environment variables returned by vendor-specific partition assignment
	// Some vendors (e.g., NVIDIA MIG) may return env vars like CUDA_VISIBLE_DEVICES
	EnvVars map[string]string
}

// AssignPartition assigns a partition to a device
// Returns partition UUID and optional env vars that should be injected to worker Pod/Process
func (a *AcceleratorInterface) AssignPartition(templateID, deviceUUID string) (*PartitionResult, error) {
	// Validate input lengths
	const maxIDLength = 64
	if len(templateID) >= maxIDLength {
		return nil, fmt.Errorf("template ID is too long (max %d bytes)", maxIDLength-1)
	}
	if len(deviceUUID) >= maxIDLength {
		return nil, fmt.Errorf("device UUID is too long (max %d bytes)", maxIDLength-1)
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
		return nil, fmt.Errorf("failed to assign partition")
	}

	partitionUUID := byteArrayToString(assignment.PartitionUUID[:])

	// Parse optional env vars returned by vendor
	envVars := make(map[string]string, int(assignment.EnvVarCount))
	for i := 0; i < int(assignment.EnvVarCount) && i < MaxPartitionEnvs; i++ {
		key := byteArrayToString(assignment.EnvVars[i].Key[:])
		value := byteArrayToString(assignment.EnvVars[i].Value[:])
		if key != "" {
			envVars[key] = value
		}
	}

	return &PartitionResult{
		PartitionUUID: partitionUUID,
		EnvVars:       envVars,
	}, nil
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
