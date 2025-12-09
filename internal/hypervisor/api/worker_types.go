package api

import (
	"time"

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
	IsolationMode     IsolationMode
	MemoryLimitBytes  uint64
	ComputeLimitUnits uint32
	TemplateID        string
	Annotations       map[string]string
	PodIndex          string

	DeletedAt time.Time
}

type WorkerAllocation struct {
	WorkerInfo *WorkerInfo

	// the complete or partitioned device info
	DeviceInfos []*DeviceInfo

	Envs map[string]string

	Mounts []*Mount

	Devices []*DeviceSpec
}

// DeviceSpec specifies a host device to mount into a container.
type DeviceSpec struct {
	GuestPath string `json:"guestPath,omitempty"`

	HostPath string `json:"hostPath,omitempty"`

	Permissions string `json:"permissions,omitempty"`
}

// Mount specifies a host volume to mount into a container.
// where device library or tools are installed on host and container
type Mount struct {
	GuestPath string `json:"guestPath,omitempty"`

	HostPath string `json:"hostPath,omitempty"`
}
