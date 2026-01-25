package worker

import (
	"maps"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/samber/lo"
	"k8s.io/klog/v2"
)

// AllocationController manages worker device allocations
// This is a shared dependency between DeviceController, WorkerController, and Backend
type AllocationController struct {
	deviceController framework.DeviceController

	mu                sync.RWMutex
	workerAllocations map[string]*api.WorkerAllocation
	deviceAllocations map[string][]*api.WorkerAllocation
}

var _ framework.WorkerAllocationController = &AllocationController{}

// NewAllocationController creates a new AllocationController
func NewAllocationController(deviceController framework.DeviceController) *AllocationController {
	return &AllocationController{
		deviceController:  deviceController,
		workerAllocations: make(map[string]*api.WorkerAllocation, 32),
		deviceAllocations: make(map[string][]*api.WorkerAllocation, 32),
	}
}

// AllocateWorkerDevices allocates devices for a worker request
func (a *AllocationController) AllocateWorkerDevices(request *api.WorkerInfo) (*api.WorkerAllocation, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	deviceIDs := request.AllocatedDeviceUUIDs
	// Backward compatibility: older callers may only populate AllocatedDevices (provider UUIDs).
	if len(deviceIDs) == 0 {
		deviceIDs = request.AllocatedDevices
	}

	deviceInfos := make([]*api.DeviceInfo, 0, len(deviceIDs))

	// partitioned mode, call split device
	isPartitioned := request.IsolationMode == tfv1.IsolationModePartitioned && request.PartitionTemplateID != ""

	for _, deviceUUID := range deviceIDs {
		if device, exists := a.deviceController.GetDevice(deviceUUID); exists {
			if isPartitioned {
				deviceInfo, err := a.deviceController.SplitDevice(deviceUUID, request.PartitionTemplateID)
				if err != nil {
					return nil, err
				}
				deviceInfos = append(deviceInfos, deviceInfo)
			} else {
				deviceInfos = append(deviceInfos, device)
			}
			continue
		}
		klog.Warningf("AllocateWorkerDevices: allocated device %q not found in device controller (after translation attempts)", deviceUUID)
	}

	mounts, err := a.deviceController.GetVendorMountLibs()
	if err != nil {
		klog.Errorf("failed to get vendor mount libs for worker allocation of %s: %v,", request.WorkerUID, err)
		return nil, err
	}

	envs := make(map[string]string, 8)
	devices := make(map[string]*api.DeviceSpec, 8)
	for _, deviceInfo := range deviceInfos {
		maps.Copy(envs, deviceInfo.DeviceEnv)
		for devNode, guestPath := range deviceInfo.DeviceNode {
			if _, exists := devices[devNode]; exists {
				continue
			}
			devices[devNode] = &api.DeviceSpec{
				HostPath:    devNode,
				GuestPath:   guestPath,
				Permissions: "rwm",
			}
		}
	}

	allocation := &api.WorkerAllocation{
		WorkerInfo:  request,
		DeviceInfos: deviceInfos,
		Envs:        envs,
		Mounts:      mounts,
		Devices:     lo.Values(devices),
	}

	a.workerAllocations[request.WorkerUID] = allocation
	for _, deviceUUID := range deviceIDs {
		a.addDeviceAllocation(deviceUUID, allocation)
	}
	return allocation, nil
}

// DeallocateWorker deallocates devices for a worker
// For partitioned devices, this also calls RemovePartitionedDevice to release the partition
func (a *AllocationController) DeallocateWorker(workerUID string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	allocation, exists := a.workerAllocations[workerUID]
	if !exists {
		klog.V(4).Infof("worker allocation not found for worker %s, may have already been deallocated", workerUID)
		return nil
	}
	delete(a.workerAllocations, workerUID)
	deviceIDs := allocation.WorkerInfo.AllocatedDeviceUUIDs
	if len(deviceIDs) == 0 {
		deviceIDs = allocation.WorkerInfo.AllocatedDevices
	}
	for _, deviceUUID := range deviceIDs {
		a.removeDeviceAllocation(deviceUUID, allocation)
	}

	// For partitioned devices, release the partition via device controller
	for _, deviceInfo := range allocation.DeviceInfos {
		if deviceInfo.ParentUUID != "" {
			// This is a partitioned device, release the partition
			if err := a.deviceController.RemovePartitionedDevice(deviceInfo.UUID, deviceInfo.ParentUUID); err != nil {
				klog.Errorf("failed to remove partition %s from device %s for worker %s: %v",
					deviceInfo.UUID, deviceInfo.ParentUUID, workerUID, err)
				// Continue deallocating other resources even if partition removal fails
			}
		}
	}

	klog.Infof("worker %s deallocated", workerUID)
	return nil
}

// GetWorkerAllocation returns the allocation for a specific worker
func (a *AllocationController) GetWorkerAllocation(workerUID string) (*api.WorkerAllocation, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	allocation, exists := a.workerAllocations[workerUID]
	return allocation, exists
}

// GetDeviceAllocations returns all device allocations keyed by device UUID
func (a *AllocationController) GetDeviceAllocations() map[string][]*api.WorkerAllocation {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return maps.Clone(a.deviceAllocations)
}

// addDeviceAllocation adds an allocation to a device (internal, must be called with lock held)
func (a *AllocationController) addDeviceAllocation(deviceUUID string, allocation *api.WorkerAllocation) {
	if _, exists := a.deviceAllocations[deviceUUID]; !exists {
		a.deviceAllocations[deviceUUID] = make([]*api.WorkerAllocation, 0, 8)
	}
	a.deviceAllocations[deviceUUID] = append(a.deviceAllocations[deviceUUID], allocation)
}

// removeDeviceAllocation removes an allocation from a device (internal, must be called with lock held)
func (a *AllocationController) removeDeviceAllocation(deviceUUID string, allocation *api.WorkerAllocation) {
	if _, exists := a.deviceAllocations[deviceUUID]; !exists {
		return
	}
	a.deviceAllocations[deviceUUID] = lo.Filter(a.deviceAllocations[deviceUUID], func(wa *api.WorkerAllocation, _ int) bool {
		return wa.WorkerInfo.WorkerUID != allocation.WorkerInfo.WorkerUID
	})
}
