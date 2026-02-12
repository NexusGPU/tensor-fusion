/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// GPUStatus defines the observed state of GPU.
// NOTE: When new fields added, remember to update syncGPUMetadataAndStatusFromCluster
type GPUStatus struct {
	// +kubebuilder:default=Pending
	Phase TensorFusionGPUPhase `json:"phase"`

	// +kubebuilder:default="NVIDIA"
	Vendor string `json:"vendor"`

	Capacity  *Resource `json:"capacity"`
	Available *Resource `json:"available"`

	UUID string `json:"uuid"`

	// +optional
	// +kubebuilder:default=soft
	IsolationMode IsolationModeType `json:"isolationMode,omitempty"`

	// +optional
	Index *int32 `json:"index,omitempty"`

	// When it's -1, it means the GPU is not assigned to any NUMA node
	// +optional
	NUMANode *int32 `json:"numaNode,omitempty"`

	// The host match selector to schedule worker pods
	NodeSelector map[string]string `json:"nodeSelector"`
	GPUModel     string            `json:"gpuModel"`

	// GPU is used by tensor-fusion or nvidia-operator
	// This is the key to be compatible with nvidia-device-plugin to avoid resource overlap
	// Hypervisor will watch kubelet device plugin to report all GPUs already used by nvidia-device-plugin
	// GPUs will be grouped by usedBy to be used by different Pods,
	// tensor-fusion annotation or nvidia-device-plugin resource block
	// +optional
	UsedBy UsedBySystem `json:"usedBy,omitempty"`

	Message string `json:"message"`

	// +optional
	RunningApps []*RunningAppDetail `json:"runningApps,omitempty"`

	// +optional
	// AllocatedPartitions tracks allocated partitions on this GPU
	// Key is partitionUUID, value contains template info and allocated resources
	AllocatedPartitions map[string]AllocatedPartition `json:"allocatedPartitions,omitempty"`
}

// +default="tensor-fusion"
type UsedBySystem string

var (
	UsedByTensorFusion UsedBySystem = UsedBySystem(Domain)
)

type RunningAppDetail struct {
	// Workload name namespace
	Name      string `json:"name,omitempty"`
	Namespace string `json:"namespace,omitempty"`

	// Worker count
	Count int `json:"count"`

	// Pod names that are running this workload
	// +optional
	Pods []*PodGPUInfo `json:"pods,omitempty"`
}

type PodGPUInfo struct {
	Name      string   `json:"name,omitempty"`
	Namespace string   `json:"namespace,omitempty"`
	UID       string   `json:"uid,omitempty"`
	Requests  Resource `json:"requests,omitempty"`
	Limits    Resource `json:"limits,omitempty"`
	QoS       QoSLevel `json:"qos,omitempty"`

	// IsExternal indicates if this allocation is from an external device plugin
	// (e.g., nvidia-device-plugin) rather than TensorFusion scheduler
	// +optional
	IsExternal bool `json:"isExternal,omitempty"`
}

// PartitionTemplate represents a hardware partition template (e.g., MIG profile)
// Only stores template ID and name in GPU status. Detailed resource information
// is stored in public GPU info config.
type PartitionTemplate struct {
	// TemplateID is the unique identifier for this partition template (e.g., "1g.24gb", "4g.94gb")
	TemplateID string `json:"templateId"`

	// Name is a human-readable name for this template
	Name string `json:"name"`
}

// AllocatedPartition represents an allocated partition on a GPU
// Key in AllocatedPartitions map is podUID
type AllocatedPartition struct {
	// TemplateID is the template used to create this partition
	TemplateID string `json:"templateId"`

	// PodUID is the UID of the pod using this partition (used as map key)
	PodUID string `json:"podUid"`

	// PodName is the name of the pod using this partition
	PodName string `json:"podName"`

	// Namespace is the namespace of the pod using this partition
	Namespace string `json:"namespace"`

	// AllocatedAt is when this partition was allocated
	AllocatedAt metav1.Time `json:"allocatedAt"`

	// AllocatedSlotStart is the starting slot position where this partition is allocated
	// This is the actual hardware slot position (0-based index)
	// For NVIDIA MIG: physical slot position (0-7)
	AllocatedSlotStart *uint32 `json:"allocatedSlotStart,omitempty"`

	// AllocatedSlotEnd is the ending slot position (exclusive) where this partition is allocated
	// The partition occupies slots [AllocatedSlotStart, AllocatedSlotEnd)
	AllocatedSlotEnd *uint32 `json:"allocatedSlotEnd,omitempty"`

	// IsolationGroupID is the isolation group where this partition is allocated
	// For Ascend NPU: vGroup ID (0-3)
	// For NVIDIA MIG: this is derived from slot position, not used separately
	// +optional
	IsolationGroupID *uint32 `json:"isolationGroupId,omitempty"`
}

// +kubebuilder:validation:Enum=Pending;Provisioning;Running;Unknown;Destroying;Migrating
type TensorFusionGPUPhase string

const (
	TensorFusionGPUPhasePending    TensorFusionGPUPhase = PhasePending
	TensorFusionGPUPhaseUpdating   TensorFusionGPUPhase = PhaseUpdating
	TensorFusionGPUPhaseRunning    TensorFusionGPUPhase = PhaseRunning
	TensorFusionGPUPhaseUnknown    TensorFusionGPUPhase = PhaseUnknown
	TensorFusionGPUPhaseDestroying TensorFusionGPUPhase = PhaseDestroying
	TensorFusionGPUPhaseMigrating  TensorFusionGPUPhase = PhaseMigrating
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster
// +kubebuilder:printcolumn:name="GPU Model",type="string",JSONPath=".status.gpuModel"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Total TFlops",type="string",JSONPath=".status.capacity.tflops"
// +kubebuilder:printcolumn:name="Total VRAM",type="string",JSONPath=".status.capacity.vram"
// +kubebuilder:printcolumn:name="Available TFlops",type="string",JSONPath=".status.available.tflops"
// +kubebuilder:printcolumn:name="Available VRAM",type="string",JSONPath=".status.available.vram"
// +kubebuilder:printcolumn:name="Device UUID",type="string",JSONPath=".status.uuid"
// +kubebuilder:printcolumn:name="Used By",type="string",JSONPath=".status.usedBy"
// +kubebuilder:printcolumn:name="Node",type="string",JSONPath=".status.nodeSelector"
// GPU is the Schema for the gpus API.
type GPU struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Status GPUStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// GPUList contains a list of GPU.
type GPUList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []GPU `json:"items"`
}

func init() {
	SchemeBuilder.Register(&GPU{}, &GPUList{})
}
