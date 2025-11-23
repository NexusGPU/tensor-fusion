package api

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// IsolationMode represents the isolation mode for worker processes
type IsolationMode = tfv1.IsolationModeType

type WorkerInfo struct {
	WorkerUID         string
	AllocatedDevices  []string
	Status            string
	PodUID            string
	PodName           string
	Namespace         string
	PartitionUUID     string
	IsolationMode     tfv1.IsolationModeType
	MemoryLimitBytes  uint64
	ComputeLimitUnits uint32
	TemplateID        string
	Annotations       map[string]string
	PodIndex          string
}

type WorkerAllocation struct {
	WorkerInfo *WorkerInfo

	// the complete or partitioned device info
	DeviceInfos []*DeviceInfo
}
