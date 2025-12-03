package device

/*
#cgo CFLAGS: -I../../../provider
#cgo LDFLAGS: -ldl
#include "../../../provider/accelerator.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <sys/types.h>

// Forward declarations from wrapper.c
extern int loadAcceleratorLibrary(const char* libPath);
extern void unloadAcceleratorLibrary(void);
extern Result GetDeviceCountWrapper(size_t* deviceCount);
extern Result GetAllDevicesWrapper(ExtendedDeviceInfo* devices, size_t maxCount, size_t* deviceCount);
extern Result GetPartitionTemplatesWrapper(int32_t deviceIndex, PartitionTemplate* templates, size_t maxCount, size_t* templateCount);
extern bool AssignPartitionWrapper(PartitionAssignment* assignment);
extern bool RemovePartitionWrapper(const char* templateId, const char* deviceUUID);
extern Result SetMemHardLimitWrapper(const char* workerId, const char* deviceUUID, uint64_t memoryLimitBytes);
extern Result SetComputeUnitHardLimitWrapper(const char* workerId, const char* deviceUUID, uint32_t computeUnitLimit);
extern Result GetProcessComputeUtilizationWrapper(ComputeUtilization* utilizations, size_t maxCount, size_t* utilizationCount);
extern Result GetProcessMemoryUtilizationWrapper(MemoryUtilization* utilizations, size_t maxCount, size_t* utilizationCount);
extern Result GetDeviceMetricsWrapper(const char** deviceUUIDArray, size_t deviceCount, DeviceMetrics* metrics, size_t maxExtraMetricsPerDevice);
extern Result GetVendorMountLibsWrapper(Mount* mounts, size_t maxCount, size_t* mountCount);
extern const char* getDlError(void);
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
)

// AcceleratorInterface provides Go bindings for the C accelerator library
type AcceleratorInterface struct {
	libPath string
	// deviceProcesses maps device UUID to list of process IDs
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

// Load loads the accelerator library dynamically
func (a *AcceleratorInterface) Load() error {
	if a.libPath == "" {
		return fmt.Errorf("library path is empty")
	}

	cLibPath := C.CString(a.libPath)
	defer C.free(unsafe.Pointer(cLibPath))

	result := C.loadAcceleratorLibrary(cLibPath)
	if result != 0 {
		var errMsg string
		if dlErr := C.getDlError(); dlErr != nil {
			errMsg = C.GoString(dlErr)
		} else {
			errMsg = "unknown error"
		}

		switch result {
		case -1:
			return fmt.Errorf("failed to load library: %s", errMsg)
		case -2:
			return fmt.Errorf("missing required symbols in library: %s", errMsg)
		}
		return fmt.Errorf("failed to load library (code %d): %s", result, errMsg)
	}

	a.loaded = true
	return nil
}

// Close unloads the accelerator library
func (a *AcceleratorInterface) Close() error {
	if a.loaded {
		C.unloadAcceleratorLibrary()
		a.loaded = false
	}
	return nil
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

	// Allocate C strings for device UUIDs
	cDeviceUUIDs := make([]*C.char, deviceCount)
	for i := 0; i < deviceCount; i++ {
		cDeviceUUIDs[i] = C.CString(deviceUUIDs[i])
	}
	defer func() {
		for _, cDeviceUUID := range cDeviceUUIDs {
			if cDeviceUUID != nil {
				C.free(unsafe.Pointer(cDeviceUUID))
			}
		}
	}()

	// Convert Go slice to C array pointer
	// In CGO, we can directly use the slice's underlying array pointer
	var cUUIDArray **C.char
	if deviceCount > 0 {
		cUUIDArray = (**C.char)(unsafe.Pointer(&cDeviceUUIDs[0]))
	}

	// Allocate stack buffer for metrics
	const maxExtraMetricsPerDevice = 32
	var cMetrics [maxStackDevices]C.DeviceMetrics
	var cExtraMetrics [maxStackDevices][maxExtraMetricsPerDevice]C.ExtraMetric

	// Initialize extraMetrics pointers
	for i := 0; i < deviceCount; i++ {
		cMetrics[i].extraMetrics = &cExtraMetrics[i][0]
		cMetrics[i].extraMetricsCount = 0
	}

	//nolint:staticcheck
	result := C.GetDeviceMetricsWrapper(cUUIDArray, C.size_t(deviceCount), &cMetrics[0], C.size_t(maxExtraMetricsPerDevice))
	if result != C.RESULT_SUCCESS {
		return nil, fmt.Errorf("failed to get device metrics: %d", result)
	}

	// Convert C metrics to Go metrics
	metrics := make([]*api.GPUUsageMetrics, deviceCount)
	for i := 0; i < deviceCount; i++ {
		cm := &cMetrics[i]
		memoryTotal := uint64(cm.memoryTotalBytes)
		memoryUsed := uint64(cm.memoryUsedBytes)
		var memoryPercentage float64
		if memoryTotal > 0 {
			memoryPercentage = float64(memoryUsed) / float64(memoryTotal) * 100.0
		}

		// Convert extra metrics from C to Go map
		extraMetrics := make(map[string]float64, int(cm.extraMetricsCount)+1)
		// Always include tensorCoreUsagePercent as it's a standard field
		extraMetrics["tensorCoreUsagePercent"] = float64(cm.tensorCoreUsagePercent)

		// Add other extra metrics from C array
		if cm.extraMetrics != nil && cm.extraMetricsCount > 0 {
			// Convert C pointer to Go slice for indexing
			extraMetricsSlice := (*[maxExtraMetricsPerDevice]C.ExtraMetric)(unsafe.Pointer(cm.extraMetrics))
			for j := 0; j < int(cm.extraMetricsCount); j++ {
				em := &extraMetricsSlice[j]
				key := C.GoString(&em.key[0])
				if key != "" {
					extraMetrics[key] = float64(em.value)
				}
			}
		}

		metrics[i] = &api.GPUUsageMetrics{
			DeviceUUID:        C.GoString(&cm.deviceUUID[0]),
			MemoryBytes:       memoryUsed,
			MemoryPercentage:  memoryPercentage,
			ComputePercentage: float64(cm.smActivePercent),
			ComputeTflops:     0,                                // Not available in DeviceMetrics
			Rx:                float64(cm.pcieRxBytes) / 1024.0, // Convert bytes to KB
			Tx:                float64(cm.pcieTxBytes) / 1024.0, // Convert bytes to KB
			Temperature:       float64(cm.temperatureCelsius),
			PowerUsage:        int64(cm.powerUsageWatts),
			ExtraMetrics:      extraMetrics,
		}
	}

	return metrics, nil
}

// GetAllDevices retrieves all available devices from the accelerator library
func (a *AcceleratorInterface) GetAllDevices() ([]*api.DeviceInfo, error) {
	// First, get the device count
	var cDeviceCount C.size_t
	//nolint:staticcheck
	result := C.GetDeviceCountWrapper(&cDeviceCount)
	if result != C.RESULT_SUCCESS {
		return nil, fmt.Errorf("failed to get device count: %d", result)
	}

	if cDeviceCount == 0 {
		return []*api.DeviceInfo{}, nil
	}

	// Allocate stack buffer (max 64 devices to avoid stack overflow)
	const maxStackDevices = 64
	var stackDevices [maxStackDevices]C.ExtendedDeviceInfo
	maxDevices := int(cDeviceCount)
	if maxDevices > maxStackDevices {
		maxDevices = maxStackDevices
	}

	var cCount C.size_t
	//nolint:staticcheck
	result = C.GetAllDevicesWrapper(&stackDevices[0], C.size_t(maxDevices), &cCount)
	if result != C.RESULT_SUCCESS {
		return nil, fmt.Errorf("failed to get all devices: %d", result)
	}

	if cCount == 0 {
		return []*api.DeviceInfo{}, nil
	}

	devices := make([]*api.DeviceInfo, int(cCount))

	for i := 0; i < int(cCount); i++ {
		cInfo := &stackDevices[i]
		devices[i] = &api.DeviceInfo{
			UUID:             C.GoString(&cInfo.basic.uuid[0]),
			Vendor:           C.GoString(&cInfo.basic.vendor[0]),
			Model:            C.GoString(&cInfo.basic.model[0]),
			Index:            int32(cInfo.basic.index),
			NUMANode:         int32(cInfo.basic.numaNode),
			TotalMemoryBytes: uint64(cInfo.basic.totalMemoryBytes),
			MaxTflops:        float64(cInfo.basic.maxTflops),
			Capabilities: api.DeviceCapabilities{
				SupportsPartitioning:  bool(cInfo.capabilities.supportsPartitioning),
				SupportsSoftIsolation: bool(cInfo.capabilities.supportsSoftIsolation),
				SupportsHardIsolation: bool(cInfo.capabilities.supportsHardIsolation),
				SupportsSnapshot:      bool(cInfo.capabilities.supportsSnapshot),
				SupportsMetrics:       bool(cInfo.capabilities.supportsMetrics),
				MaxPartitions:         uint32(cInfo.capabilities.maxPartitions),
				MaxWorkersPerDevice:   uint32(cInfo.capabilities.maxWorkersPerDevice),
			},
			Properties: make(map[string]string, 0),
		}
	}

	return devices, nil
}

// AssignPartition assigns a partition to a device
func (a *AcceleratorInterface) AssignPartition(templateID, deviceUUID string) (string, error) {
	cTemplateID := C.CString(templateID)
	defer C.free(unsafe.Pointer(cTemplateID))

	cDeviceUUID := C.CString(deviceUUID)
	defer C.free(unsafe.Pointer(cDeviceUUID))

	var assignment C.PartitionAssignment
	C.strncpy(&assignment.templateId[0], cTemplateID, C.size_t(len(templateID)))
	C.strncpy(&assignment.deviceUUID[0], cDeviceUUID, C.size_t(len(deviceUUID)))

	//nolint:staticcheck
	result := C.AssignPartitionWrapper(&assignment)
	if !result {
		return "", fmt.Errorf("failed to assign partition")
	}

	partitionUUID := C.GoString(&assignment.partitionUUID[0])
	return partitionUUID, nil
}

// RemovePartition removes a partition from a device
func (a *AcceleratorInterface) RemovePartition(partitionUUID, deviceUUID string) error {
	cPartitionUUID := C.CString(partitionUUID)
	defer C.free(unsafe.Pointer(cPartitionUUID))

	cDeviceUUID := C.CString(deviceUUID)
	defer C.free(unsafe.Pointer(cDeviceUUID))

	//nolint:staticcheck
	result := C.RemovePartitionWrapper(cPartitionUUID, cDeviceUUID)
	if !result {
		return fmt.Errorf("failed to remove partition")
	}

	return nil
}

// SetMemHardLimit sets hard memory limit for a worker
func (a *AcceleratorInterface) SetMemHardLimit(workerID, deviceUUID string, memoryLimitBytes uint64) error {
	cWorkerID := C.CString(workerID)
	defer C.free(unsafe.Pointer(cWorkerID))

	cDeviceUUID := C.CString(deviceUUID)
	defer C.free(unsafe.Pointer(cDeviceUUID))

	//nolint:staticcheck
	result := C.SetMemHardLimitWrapper(cWorkerID, cDeviceUUID, C.uint64_t(memoryLimitBytes))
	if result != C.RESULT_SUCCESS {
		return fmt.Errorf("failed to set memory hard limit: %d", result)
	}

	return nil
}

// SetComputeUnitHardLimit sets hard compute unit limit for a worker
func (a *AcceleratorInterface) SetComputeUnitHardLimit(workerID, deviceUUID string, computeUnitLimit uint32) error {
	cWorkerID := C.CString(workerID)
	defer C.free(unsafe.Pointer(cWorkerID))

	cDeviceUUID := C.CString(deviceUUID)
	defer C.free(unsafe.Pointer(cDeviceUUID))

	//nolint:staticcheck
	result := C.SetComputeUnitHardLimitWrapper(cWorkerID, cDeviceUUID, C.uint32_t(computeUnitLimit))
	if result != C.RESULT_SUCCESS {
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
	var stackUtilizations [maxStackUtilizations]C.ComputeUtilization
	maxCount := totalCount
	if maxCount > maxStackUtilizations {
		maxCount = maxStackUtilizations
	}

	var cCount C.size_t
	//nolint:staticcheck
	result := C.GetProcessComputeUtilizationWrapper(&stackUtilizations[0], C.size_t(maxCount), &cCount)
	if result != C.RESULT_SUCCESS {
		return nil, fmt.Errorf("failed to get process compute utilization: %d", result)
	}

	if cCount == 0 {
		return []api.ComputeUtilization{}, nil
	}

	utilizations := make([]api.ComputeUtilization, int(cCount))
	for i := 0; i < int(cCount); i++ {
		cu := &stackUtilizations[i]
		utilizations[i] = api.ComputeUtilization{
			ProcessID:          C.GoString(&cu.processId[0]),
			DeviceUUID:         C.GoString(&cu.deviceUUID[0]),
			UtilizationPercent: float64(cu.utilizationPercent),
			// Note: ActiveSMs, TotalSMs, and TFLOPsUsed will be added to ComputeUtilization if needed
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
	var stackUtilizations [maxStackUtilizations]C.MemoryUtilization
	maxCount := totalCount
	if maxCount > maxStackUtilizations {
		maxCount = maxStackUtilizations
	}

	var cCount C.size_t
	//nolint:staticcheck
	result := C.GetProcessMemoryUtilizationWrapper(&stackUtilizations[0], C.size_t(maxCount), &cCount)
	if result != C.RESULT_SUCCESS {
		return nil, fmt.Errorf("failed to get process memory utilization: %d", result)
	}

	if cCount == 0 {
		return []api.MemoryUtilization{}, nil
	}

	utilizations := make([]api.MemoryUtilization, int(cCount))
	for i := 0; i < int(cCount); i++ {
		mu := &stackUtilizations[i]
		utilizations[i] = api.MemoryUtilization{
			ProcessID:     C.GoString(&mu.processId[0]),
			DeviceUUID:    C.GoString(&mu.deviceUUID[0]),
			UsedBytes:     uint64(mu.usedBytes),
			ReservedBytes: uint64(mu.reservedBytes),
			// Note: UtilizationPercent will be calculated separately if needed
		}
	}

	return utilizations, nil
}

// GetVendorMountLibs retrieves vendor mount libs
func (a *AcceleratorInterface) GetVendorMountLibs() ([]*api.Mount, error) {
	const maxStackMounts = 64
	var stackMounts [maxStackMounts]C.Mount
	var cCount C.size_t

	result := C.GetVendorMountLibsWrapper(&stackMounts[0], C.size_t(maxStackMounts), &cCount)
	if result != C.RESULT_SUCCESS {
		return nil, fmt.Errorf("failed to get vendor mount libs: %d", result)
	}

	if cCount == 0 {
		return []*api.Mount{}, nil
	}

	mounts := make([]*api.Mount, int(cCount))
	for i := 0; i < int(cCount); i++ {
		cm := &stackMounts[i]
		var hostPath, guestPath string
		if cm.hostPath != nil {
			hostPath = C.GoString(cm.hostPath)
		}
		if cm.guestPath != nil {
			guestPath = C.GoString(cm.guestPath)
		}
		mounts[i] = &api.Mount{
			HostPath:  hostPath,
			GuestPath: guestPath,
		}
	}

	return mounts, nil
}
