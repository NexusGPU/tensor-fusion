package framework

import (
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
)

type DeviceController interface {
	Start() error

	DiscoverDevices() error

	AllocateDevice(request *api.DeviceAllocateRequest) (*api.DeviceAllocateResponse, error)

	// ListDevices returns all discovered devices
	ListDevices() ([]*api.DeviceInfo, error)

	// DevicesUpdates returns a channel that receives device list updates
	// The channel should be closed when Stop() is called
	DevicesUpdates() (<-chan []*api.DeviceInfo, error)

	// GetDevice returns device information by UUID
	GetDevice(deviceUUID string) (*api.DeviceInfo, error)

	// GetDeviceAllocations returns device allocations
	// If deviceUUID is empty, returns all allocations
	GetDeviceAllocations(deviceUUID string) ([]*api.DeviceAllocation, error)

	// GetDeviceAllocationUpdates returns a channel that receives allocation updates
	// The channel should be closed when Stop() is called
	GetDeviceAllocationUpdates(deviceUUID string, allocationID string) (<-chan []*api.DeviceAllocation, error)

	// GetGPUMetrics returns current GPU metrics for all devices
	GetGPUMetrics() (map[string]*api.GPUUsageMetrics, error)
}

type DeviceInterface interface {
	SplitDevice(deviceUUID string) error

	GetDeviceMetrics() (*api.MemoryUtilization, error)
}

type WorkerController interface {
	Start() error

	Stop() error

	// GetWorkerAllocation returns allocation information for a worker
	GetWorkerAllocation(workerUID string) (*api.DeviceAllocation, error)

	// GetWorkerMetricsUpdates returns a channel that receives worker metrics updates
	// The channel should be closed when Stop() is called
	GetWorkerMetricsUpdates() (<-chan *api.DeviceAllocation, error)

	// GetWorkerMetrics returns current worker metrics for all workers
	// Returns map keyed by device UUID, then by worker UID, then by process ID
	GetWorkerMetrics() (map[string]map[string]map[string]*api.WorkerMetrics, error)

	// ListWorkers returns list of all worker UIDs
	ListWorkers() ([]string, error)
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

	// ListAndWatchWorkers gets GPU workers from the workload orchestration platform
	// Returns a channel that receives worker UID lists and a stop channel
	// The channel should be closed when Stop() is called
	ListAndWatchWorkers() (<-chan []string, <-chan struct{}, error)

	// GetWorkerToProcessMap links workers to actual running process list on OS
	GetWorkerToProcessMap() (map[string][]string, error)

	// StartWorker spawns worker process
	StartWorker(workerUID string) error

	// StopWorker stops worker process
	StopWorker(workerUID string) error

	// ReconcileDevices reports devices to backend orchestration and O&M platform
	ReconcileDevices(devices []string) error
}
