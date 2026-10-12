package framework

import "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"

// WorkerComputeUsageProvider is an optional capability of a WorkerController:
// the compute utilization of every worker on every device it holds, integrated
// over time rather than sampled at one instant.
type WorkerComputeUsageProvider interface {
	// GetWorkerComputeUsage returns the accumulated usage keyed by device UUID
	// (lower case), then by worker UID. The result is a copy.
	GetWorkerComputeUsage() map[string]map[string]api.WorkerComputeUsage
}
