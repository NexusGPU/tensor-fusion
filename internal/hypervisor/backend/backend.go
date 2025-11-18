package integration

import (
	"context"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/device"
)

type Framework interface {
	AllocateDevice(ctx context.Context, request *device.DeviceAllocateRequest) (*device.DeviceAllocateResponse, error)

	ListDevices(ctx context.Context) ([]*device.DeviceInfo, error)

	DevicesUpdates(ctx context.Context) (<-chan []*device.DeviceInfo, error)

	GetDevice(ctx context.Context, deviceUUID string) (*device.DeviceInfo, error)

	GetDeviceAllocations(ctx context.Context, deviceUUID string) ([]*device.DeviceAllocation, error)

	GetDeviceAllocationUpdates(ctx context.Context, deviceUUID string, allocationID string) (<-chan []*device.DeviceAllocation, error)
}

// The backend interface for the hypervisor to interact with the underlying infrastructure
type Backend interface {
	Start(ctx context.Context, framework Framework, params map[string]string) error

	// Get GPU workers from the workload orchestration platform
	ListAndWatchWorkers(ctx context.Context) ([]string, error)

	// Report devices to backend orchestration and O&M platform
	ReportDevices(ctx context.Context, devices []string) error

	// Link workers to actual running process list on OS
	GetWorkerProcessMap(ctx context.Context) (map[string][]string, error)

	// Spawn worker process on OS
	StartWorker(ctx context.Context, workerUID string) error

	// Stop worker process on OS
	StopWorker(ctx context.Context, workerUID string) error
}
