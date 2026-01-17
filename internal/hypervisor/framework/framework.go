package framework

import (
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
)

type DeviceController interface {
	Start() error

	Stop() error

	DiscoverDevices() error

	ListDevices() ([]*api.DeviceInfo, error)

	GetDevice(deviceUUID string) (*api.DeviceInfo, bool)

	SplitDevice(deviceUUID string, partitionID string) (*api.DeviceInfo, error)

	RemovePartitionedDevice(partitionUUID, deviceUUID string) error

	GetDeviceMetrics() (map[string]*api.GPUUsageMetrics, error)

	// GetProcessInformation returns process-level GPU metrics for all processes on all devices
	// Returns ProcessInformation for each process using GPU, including ProcessID, DeviceUUID,
	// compute utilization, and memory usage
	GetProcessInformation() ([]api.ProcessInformation, error)

	GetVendorMountLibs() ([]*api.Mount, error)

	RegisterDeviceUpdateHandler(handler DeviceChangeHandler)

	GetAcceleratorVendor() string
}

// WorkerAllocationController manages worker device allocations
// This is a shared dependency between DeviceController, WorkerController, and Backend
type WorkerAllocationController interface {
	// AllocateWorkerDevices allocates devices for a worker request
	AllocateWorkerDevices(request *api.WorkerInfo) (*api.WorkerAllocation, error)

	// DeallocateWorker deallocates devices for a worker
	DeallocateWorker(workerUID string) error

	// GetWorkerAllocation returns the allocation for a specific worker
	GetWorkerAllocation(workerUID string) (*api.WorkerAllocation, bool)

	// GetDeviceAllocations returns all device allocations keyed by device UUID
	GetDeviceAllocations() map[string][]*api.WorkerAllocation
}

type WorkerController interface {
	Start() error

	Stop() error

	ListWorkers() ([]*api.WorkerInfo, error)

	// GetWorkerMetrics returns current worker metrics for all workers
	// Returns map keyed by device UUID, then by worker UID, then by process ID
	GetWorkerMetrics() (map[string]map[string]map[string]*api.WorkerMetrics, error)
}

type QuotaController interface {
	// SetQuota sets quota for a worker
	SetQuota(workerUID string) error

	StartSoftQuotaLimiter() error

	StopSoftQuotaLimiter() error

	// GetWorkerQuotaStatus gets quota status for a worker
	GetWorkerQuotaStatus(workerUID string) error
}

// The backend interface for the hypervisor to interact with the underlying infrastructure
type Backend interface {
	Start() error

	Stop() error

	// RegisterWorkerUpdateHandler registers a handler for worker updates
	// The handler will be called for all existing workers (OnAdd) and all future worker changes (add, update, remove)
	RegisterWorkerUpdateHandler(handler WorkerChangeHandler) error

	// StartWorker spawns worker process
	StartWorker(worker *api.WorkerInfo) error

	// StopWorker stops worker process
	StopWorker(workerUID string) error

	// GetProcessMappingInfo gets process mapping information from a host process
	GetProcessMappingInfo(hostPID uint32) (*ProcessMappingInfo, error)

	GetDeviceChangeHandler() DeviceChangeHandler

	ListWorkers() []*api.WorkerInfo
}

// ProcessMappingInfo contains worker information extracted from a process
type ProcessMappingInfo struct {
	// Namespace is the Kubernetes namespace of the pod
	Namespace string
	// PodName is the name of the pod
	PodName string
	// ContainerName is the name of the container within the pod
	ContainerName string
	// GuestID is a composite identifier: namespace_podName_containerName
	GuestID string
	// HostPID is the process ID in the host namespace
	HostPID uint32
	// GuestPID is the process ID in the container namespace
	GuestPID uint32
}

type DeviceChangeHandler struct {
	OnAdd               func(device *api.DeviceInfo)
	OnRemove            func(device *api.DeviceInfo)
	OnUpdate            func(oldDevice, newDevice *api.DeviceInfo)
	OnDiscoveryComplete func(nodeInfo *api.NodeInfo)
}

type WorkerChangeHandler struct {
	OnAdd    func(worker *api.WorkerInfo)
	OnRemove func(worker *api.WorkerInfo)
	OnUpdate func(oldWorker, newWorker *api.WorkerInfo)
}
