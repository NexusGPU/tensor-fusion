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

	GetVendorMountLibs() ([]*api.Mount, error)

	RegisterDeviceUpdateHandler(handler DeviceChangeHandler)

	GetAcceleratorVendor() string

	GetDeviceAllocations() map[string][]*api.WorkerAllocation

	AddDeviceAllocation(deviceUUID string, allocation *api.WorkerAllocation)

	RemoveDeviceAllocation(workerUID string, allocation *api.WorkerAllocation)
}

type WorkerController interface {
	Start() error

	Stop() error

	AllocateWorkerDevices(request *api.WorkerInfo) (*api.WorkerAllocation, error)

	DeallocateWorker(workerUID string) error

	ListWorkers() ([]*api.WorkerInfo, error)

	GetWorkerAllocation(workerUID string) (*api.WorkerAllocation, bool)

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

	// GetProcessMappingInfo gets process mapping information for a worker
	GetProcessMappingInfo(workerUID string, hostPID uint32) (*ProcessMappingInfo, error)

	GetDeviceChangeHandler() DeviceChangeHandler

	ListWorkers() []*api.WorkerInfo
}

// ProcessWorkerInfo contains worker information extracted from a process
type ProcessMappingInfo struct {
	GuestID  string
	HostPID  uint32
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
