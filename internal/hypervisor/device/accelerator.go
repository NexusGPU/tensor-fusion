/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package device

/*
#cgo CFLAGS: -I../../../provider
#cgo LDFLAGS: -L../../../provider/build -laccelerator_stub -Wl,-rpath,../../../provider/build
#include <stdlib.h>
#include <string.h>
#include "../../../provider/accelerator.h"

// Forward declarations to help IDE/linter recognize C functions
extern Result GetDeviceCount(size_t* deviceCount);
extern Result GetAllDevices(ExtendedDeviceInfo* devices, size_t maxCount, size_t* deviceCount);
extern Result GetPartitionTemplates(int32_t deviceIndex, PartitionTemplate* templates, size_t maxCount, size_t* templateCount);
extern bool AssignPartition(PartitionAssignment* assignment);
extern bool RemovePartition(const char* templateId, const char* deviceUUID);
extern Result SetMemHardLimit(const char* workerId, const char* deviceUUID, uint64_t memoryLimitBytes);
extern Result SetComputeUnitHardLimit(const char* workerId, const char* deviceUUID, uint32_t computeUnitLimit);
extern Result GetProcessComputeUtilization(ComputeUtilization* utilizations, size_t maxCount, size_t* utilizationCount);
extern Result GetProcessMemoryUtilization(MemoryUtilization* utilizations, size_t maxCount, size_t* utilizationCount);
extern Result Log(const char* level, const char* message);
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

// AcceleratorInterface provides Go bindings for the C accelerator library
type AcceleratorInterface struct {
	libPath string
	// deviceProcesses maps device UUID to list of process IDs
	deviceProcesses map[string][]string
	mu              sync.RWMutex
}

// NewAcceleratorInterface creates a new accelerator interface
func NewAcceleratorInterface(libPath string) *AcceleratorInterface {
	return &AcceleratorInterface{
		libPath:         libPath,
		deviceProcesses: make(map[string][]string),
	}
}

// AddProcess adds a process to the device tracking
func (a *AcceleratorInterface) AddProcess(deviceUUID, processID string) {
	a.mu.Lock()
	defer a.mu.Unlock()

	processes := a.deviceProcesses[deviceUUID]
	// Check if process already exists
	for _, pid := range processes {
		if pid == processID {
			return
		}
	}
	a.deviceProcesses[deviceUUID] = append(processes, processID)
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
func (a *AcceleratorInterface) GetAllDevices() ([]*DeviceInfo, error) {
	// First, get the device count
	var cDeviceCount C.size_t
	//nolint:staticcheck
	result := C.GetDeviceCount(&cDeviceCount)
	if result != C.RESULT_SUCCESS {
		return nil, fmt.Errorf("failed to get device count: %d", result)
	}

	if cDeviceCount == 0 {
		return []*DeviceInfo{}, nil
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
	result = C.GetAllDevices(&stackDevices[0], C.size_t(maxDevices), &cCount)
	if result != C.RESULT_SUCCESS {
		return nil, fmt.Errorf("failed to get all devices: %d", result)
	}

	if cCount == 0 {
		return []*DeviceInfo{}, nil
	}

	devices := make([]*DeviceInfo, int(cCount))

	for i := 0; i < int(cCount); i++ {
		cInfo := &stackDevices[i]
		devices[i] = &DeviceInfo{
			UUID:            C.GoString(&cInfo.basic.uuid[0]),
			Vendor:          C.GoString(&cInfo.basic.vendor[0]),
			Model:           C.GoString(&cInfo.basic.model[0]),
			Index:           int32(cInfo.basic.index),
			NUMANode:        int32(cInfo.basic.numaNode),
			TotalMemory:     uint64(cInfo.basic.totalMemoryBytes),
			TotalCompute:    uint64(cInfo.basic.totalComputeUnits),
			MaxTflops:       float64(cInfo.basic.maxTflops),
			PCIEGen:         uint32(cInfo.basic.pcieGen),
			PCIEWidth:       uint32(cInfo.basic.pcieWidth),
			DriverVersion:   C.GoString(&cInfo.basic.driverVersion[0]),
			FirmwareVersion: C.GoString(&cInfo.basic.firmwareVersion[0]),
			Capabilities: DeviceCapabilities{
				SupportsPartitioning:  bool(cInfo.capabilities.supportsPartitioning),
				SupportsSoftIsolation: bool(cInfo.capabilities.supportsSoftIsolation),
				SupportsHardIsolation: bool(cInfo.capabilities.supportsHardIsolation),
				SupportsSnapshot:      bool(cInfo.capabilities.supportsSnapshot),
				SupportsMetrics:       bool(cInfo.capabilities.supportsMetrics),
				MaxPartitions:         uint32(cInfo.capabilities.maxPartitions),
				MaxWorkersPerDevice:   uint32(cInfo.capabilities.maxWorkersPerDevice),
			},
			Properties: DeviceProperties{
				ClockGraphics:          uint32(cInfo.props.clockGraphics),
				ClockSM:                uint32(cInfo.props.clockSM),
				ClockMem:               uint32(cInfo.props.clockMem),
				ClockAI:                uint32(cInfo.props.clockAI),
				PowerLimit:             uint32(cInfo.props.powerLimit),
				TemperatureThreshold:   uint32(cInfo.props.temperatureThreshold),
				ECCEnabled:             bool(cInfo.props.eccEnabled),
				PersistenceModeEnabled: bool(cInfo.props.persistenceModeEnabled),
				ComputeCapability:      C.GoString(&cInfo.props.computeCapability[0]),
				ChipType:               C.GoString(&cInfo.props.chipType[0]),
			},
		}
	}

	return devices, nil
}

// GetPartitionTemplates retrieves partition templates from the accelerator library
func (a *AcceleratorInterface) GetPartitionTemplates(deviceIndex int32) ([]PartitionTemplate, error) {
	// Allocate stack buffer for templates (max 64 templates)
	const maxTemplates = 64
	var cTemplates [maxTemplates]C.PartitionTemplate
	var cCount C.size_t

	//nolint:staticcheck
	result := C.GetPartitionTemplates(C.int32_t(deviceIndex), &cTemplates[0], C.size_t(maxTemplates), &cCount)
	if result != C.RESULT_SUCCESS {
		return nil, fmt.Errorf("failed to get partition templates: %d", result)
	}

	if cCount == 0 {
		return []PartitionTemplate{}, nil
	}

	templates := make([]PartitionTemplate, int(cCount))

	for i := 0; i < int(cCount); i++ {
		templates[i] = PartitionTemplate{
			TemplateID:   C.GoString(&cTemplates[i].templateId[0]),
			Name:         C.GoString(&cTemplates[i].name[0]),
			MemoryBytes:  uint64(cTemplates[i].memoryBytes),
			ComputeUnits: uint64(cTemplates[i].computeUnits),
			Tflops:       float64(cTemplates[i].tflops),
			SliceCount:   uint32(cTemplates[i].sliceCount),
			IsDefault:    bool(cTemplates[i].isDefault),
			Description:  C.GoString(&cTemplates[i].description[0]),
		}
	}

	return templates, nil
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
	result := C.AssignPartition(&assignment)
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
	result := C.RemovePartition(cTemplateID, cDeviceUUID)
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
	result := C.SetMemHardLimit(cWorkerID, cDeviceUUID, C.uint64_t(memoryLimitBytes))
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
	result := C.SetComputeUnitHardLimit(cWorkerID, cDeviceUUID, C.uint32_t(computeUnitLimit))
	if result != C.RESULT_SUCCESS {
		return fmt.Errorf("failed to set compute unit hard limit: %d", result)
	}

	return nil
}

// GetProcessComputeUtilization retrieves compute utilization for all tracked processes
func (a *AcceleratorInterface) GetProcessComputeUtilization() ([]ComputeUtilization, error) {
	// Get total process count from the map
	totalCount := a.GetTotalProcessCount()
	if totalCount == 0 {
		return []ComputeUtilization{}, nil
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
	result := C.GetProcessComputeUtilization(&stackUtilizations[0], C.size_t(maxCount), &cCount)
	if result != C.RESULT_SUCCESS {
		return nil, fmt.Errorf("failed to get process compute utilization: %d", result)
	}

	if cCount == 0 {
		return []ComputeUtilization{}, nil
	}

	utilizations := make([]ComputeUtilization, int(cCount))
	for i := 0; i < int(cCount); i++ {
		cu := &stackUtilizations[i]
		utilizations[i] = ComputeUtilization{
			ProcessID:          C.GoString(&cu.processId[0]),
			DeviceUUID:         C.GoString(&cu.deviceUUID[0]),
			UtilizationPercent: float64(cu.utilizationPercent),
			ActiveSMs:          uint64(cu.activeSMs),
			TotalSMs:           uint64(cu.totalSMs),
			TflopsUsed:         float64(cu.tflopsUsed),
		}
	}

	return utilizations, nil
}

// GetProcessMemoryUtilization retrieves memory utilization for all tracked processes
func (a *AcceleratorInterface) GetProcessMemoryUtilization() ([]MemoryUtilization, error) {
	// Get total process count from the map
	totalCount := a.GetTotalProcessCount()
	if totalCount == 0 {
		return []MemoryUtilization{}, nil
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
	result := C.GetProcessMemoryUtilization(&stackUtilizations[0], C.size_t(maxCount), &cCount)
	if result != C.RESULT_SUCCESS {
		return nil, fmt.Errorf("failed to get process memory utilization: %d", result)
	}

	if cCount == 0 {
		return []MemoryUtilization{}, nil
	}

	utilizations := make([]MemoryUtilization, int(cCount))
	for i := 0; i < int(cCount); i++ {
		mu := &stackUtilizations[i]
		utilizations[i] = MemoryUtilization{
			ProcessID:          C.GoString(&mu.processId[0]),
			DeviceUUID:         C.GoString(&mu.deviceUUID[0]),
			UsedBytes:          uint64(mu.usedBytes),
			ReservedBytes:      uint64(mu.reservedBytes),
			UtilizationPercent: float64(mu.utilizationPercent),
		}
	}

	return utilizations, nil
}

// Log logs a message using the accelerator library
func (a *AcceleratorInterface) Log(level, message string) error {
	cLevel := C.CString(level)
	defer C.free(unsafe.Pointer(cLevel))

	cMessage := C.CString(message)
	defer C.free(unsafe.Pointer(cMessage))

	//nolint:staticcheck
	result := C.Log(cLevel, cMessage)
	if result != C.RESULT_SUCCESS {
		return fmt.Errorf("failed to log message: %d", result)
	}

	return nil
}
