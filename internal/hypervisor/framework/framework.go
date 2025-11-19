package framework

import (
	"context"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
)

type DeviceController interface {
	Start() error

	DiscoverDevices() error

	AllocateDevice(request *api.DeviceAllocateRequest) (*api.DeviceAllocateResponse, error)

	ListDevices(ctx context.Context) ([]*api.DeviceInfo, error)

	DevicesUpdates(ctx context.Context) (<-chan []*api.DeviceInfo, error)

	GetDevice(ctx context.Context, deviceUUID string) (*api.DeviceInfo, error)

	GetDeviceAllocations(ctx context.Context, deviceUUID string) ([]*api.DeviceAllocation, error)

	GetDeviceAllocationUpdates(ctx context.Context, deviceUUID string, allocationID string) (<-chan []*api.DeviceAllocation, error)

	// GetGPUMetrics returns current GPU metrics for all devices
	GetGPUMetrics(ctx context.Context) (map[string]*api.GPUUsageMetrics, error)
}

type DeviceInterface interface {
	SplitDevice(ctx context.Context, deviceUUID string) error

	GetDeviceMetrics(ctx context.Context) (*api.MemoryUtilization, error)
}

type WorkerController interface {
	Start() error

	Stop() error

	GetWorkerAllocation(ctx context.Context, workerUID string) (*api.DeviceAllocation, error)

	GetWorkerMetricsUpdates(ctx context.Context) (<-chan *api.DeviceAllocation, error)

	// GetWorkerMetrics returns current worker metrics for all workers
	// Returns map keyed by device UUID, then by worker UID, then by process ID
	GetWorkerMetrics(ctx context.Context) (map[string]map[string]map[string]*api.WorkerMetrics, error)

	// ListWorkers returns list of all worker UIDs
	ListWorkers(ctx context.Context) ([]string, error)
}

type QuotaController interface {
	SetQuota(ctx context.Context, workerUID string) error

	StartSoftQuotaLimiter() error

	StopSoftQuotaLimiter() error

	GetWorkerQuotaStatus(ctx context.Context, workerUID string) error
}

// The backend interface for the hypervisor to interact with the underlying infrastructure
type Backend interface {
	Start() error

	Stop() error

	// Get GPU workers from the workload orchestration platform
	ListAndWatchWorkers(ctx context.Context, stopCh <-chan struct{}) ([]string, error)

	// Link workers to actual running process list on OS
	GetWorkerToProcessMap(ctx context.Context) (map[string][]string, error)

	// Spawn worker process
	StartWorker(ctx context.Context, workerUID string) error

	// Stop worker process
	StopWorker(ctx context.Context, workerUID string) error

	// Report devices to backend orchestration and O&M platform
	ReconcileDevices(ctx context.Context, devices []string) error
}
