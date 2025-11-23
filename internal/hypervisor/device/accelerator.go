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

	// Allocate stack buffer (max 256 devices to avoid stack overflow)
	const maxStackDevices = 256
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
			UUID:      C.GoString(&cInfo.basic.uuid[0]),
			Vendor:    C.GoString(&cInfo.basic.vendor[0]),
			Model:     C.GoString(&cInfo.basic.model[0]),
			Index:     int32(cInfo.basic.index),
			NUMANode:  int32(cInfo.basic.numaNode),
			Bytes:     uint64(cInfo.basic.totalMemoryBytes),
			MaxTflops: float64(cInfo.basic.maxTflops),
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
func (a *AcceleratorInterface) AssignPartition(templateID, deviceUUID string) (string, uint64, error) {
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
		return "", 0, fmt.Errorf("failed to assign partition")
	}

	partitionUUID := C.GoString(&assignment.partitionUUID[0])
	overhead := uint64(assignment.partitionOverheadBytes)

	return partitionUUID, overhead, nil
}

// RemovePartition removes a partition from a device
func (a *AcceleratorInterface) RemovePartition(templateID, deviceUUID string) error {
	cTemplateID := C.CString(templateID)
	defer C.free(unsafe.Pointer(cTemplateID))

	cDeviceUUID := C.CString(deviceUUID)
	defer C.free(unsafe.Pointer(cDeviceUUID))

	//nolint:staticcheck
	result := C.RemovePartitionWrapper(cTemplateID, cDeviceUUID)
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
			ActiveSMs:          uint64(cu.activeSMs),
			TotalSMs:           uint64(cu.totalSMs),
			TFLOPsUsed:         float64(cu.tflopsUsed),
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
			ProcessID:          C.GoString(&mu.processId[0]),
			DeviceUUID:         C.GoString(&mu.deviceUUID[0]),
			UsedBytes:          uint64(mu.usedBytes),
			ReservedBytes:      uint64(mu.reservedBytes),
			UtilizationPercent: float64(mu.utilizationPercent),
		}
	}

	return utilizations, nil
}
