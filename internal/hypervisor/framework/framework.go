package framework

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
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

	// ListAndWatchWorkers gets GPU workers from the workload orchestration platform
	// Returns initial list of workers and a channel that receives worker UID lists and a stop channel
	// The channel should be closed when Stop() is called
	ListAndWatchWorkers() ([]*api.WorkerInfo, chan *api.WorkerInfo, error)

	// StartWorker spawns worker process
	StartWorker(workerUID string) error

	// StopWorker stops worker process
	StopWorker(workerUID string) error

	// GetProcessMappingInfo gets process mapping information for a worker
	GetProcessMappingInfo(workerUID string, hostPID uint32) (*ProcessMappingInfo, error)

	CreateOrUpdateState(state *tfv1.GPU) error
}

// ProcessWorkerInfo contains worker information extracted from a process
type ProcessMappingInfo struct {
	GuestID  string
	HostPID  uint32
	GuestPID uint32
}
