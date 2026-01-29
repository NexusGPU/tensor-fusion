package api

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// IsolationMode represents the isolation mode for worker processes
type IsolationMode = tfv1.IsolationModeType

// +k8s:deepcopy-gen=true
type WorkerInfo struct {
	WorkerUID        string
	Namespace        string
	WorkerName       string
	AllocatedDevices []string
	Status           WorkerStatus

	QoS           tfv1.QoSLevel
	IsolationMode IsolationMode

	Requests tfv1.Resource
	Limits   tfv1.Resource

	WorkloadName      string
	WorkloadNamespace string

	// Only set for partitioned mode
	PartitionTemplateID string

	// Extra information from backend
	Labels      map[string]string
	Annotations map[string]string

	// Worker Running Info (optional, for non kubernetes workers)
	WorkerRunningInfo *WorkerRunningInfo

	DeletedAt int64
}

// WorkerRunningInfo is the information for the worker running on the host
type WorkerRunningInfo struct {
	Type       WorkerRuntimeType `json:"type"`
	Executable string            `json:"executable"`
	Args       []string          `json:"args"`
	Env        map[string]string `json:"env"`
	WorkingDir string            `json:"workingDir"`

	Restarts  int    `json:"restarts"`
	PID       uint32 `json:"pid"`
	IsRunning bool   `json:"isRunning"`
	ExitCode  int    `json:"exitCode"`
}

type WorkerRuntimeType string

const (
	WorkerRuntimeTypeProcess    WorkerRuntimeType = "process"
	WorkerRuntimeTypeSystemd    WorkerRuntimeType = "systemd"
	WorkerRuntimeTypeContainerd WorkerRuntimeType = "containerd"
)

func (w *WorkerInfo) FilterValue() string {
	return w.WorkerUID + " " + w.WorkerName + " " + w.Namespace
}

// Title returns the display title for the worker in list views
func (w *WorkerInfo) Title() string {
	if w.WorkerName != "" {
		return w.WorkerName
	}
	return w.WorkerUID
}

// Description returns the display description for the worker in list views
func (w *WorkerInfo) Description() string {
	return w.Namespace + "/" + w.WorkerUID
}

type WorkerStatus string

const (
	WorkerStatusPending          WorkerStatus = "Pending"
	WorkerStatusDeviceAllocating WorkerStatus = "DeviceAllocating"
	WorkerStatusRunning          WorkerStatus = "Running"
	WorkerStatusTerminated       WorkerStatus = "Terminated"
)

// +k8s:deepcopy-gen=true
type WorkerAllocation struct {
	WorkerInfo *WorkerInfo

	// the complete or partitioned device info
	DeviceInfos []*DeviceInfo

	Envs map[string]string

	Mounts []*Mount

	Devices []*DeviceSpec
}

// DeviceSpec specifies a host device to mount into a container.
// +k8s:deepcopy-gen=true
type DeviceSpec struct {
	GuestPath string `json:"guestPath,omitempty"`

	HostPath string `json:"hostPath,omitempty"`

	Permissions string `json:"permissions,omitempty"`
}

// Mount specifies a host volume to mount into a container.
// where device library or tools are installed on host and container
// +k8s:deepcopy-gen=true
type Mount struct {
	GuestPath string `json:"guestPath,omitempty"`

	HostPath string `json:"hostPath,omitempty"`
}
