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
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// GPUStatus defines the observed state of GPU.
type GPUStatus struct {
	// +kubebuilder:default=Pending
	Phase TensorFusionGPUPhase `json:"phase"`

	Capacity  *Resource `json:"capacity"`
	Available *Resource `json:"available"`

	UUID string `json:"uuid"`

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
}

// +kubebuilder:validation:Enum=tensor-fusion;nvidia-device-plugin
// +default="tensor-fusion"
type UsedBySystem string

const (
	UsedByTensorFusion       UsedBySystem = "tensor-fusion"
	UsedByNvidiaDevicePlugin UsedBySystem = "nvidia-device-plugin"
)

type RunningAppDetail struct {
	// Workload name namespace
	Name      string `json:"name,omitempty"`
	Namespace string `json:"namespace,omitempty"`

	// Worker count
	Count int `json:"count"`
}

// +kubebuilder:validation:Enum=Pending;Provisioning;Running;Unknown;Destroying;Migrating
type TensorFusionGPUPhase string

const (
	TensorFusionGPUPhasePending    TensorFusionGPUPhase = constants.PhasePending
	TensorFusionGPUPhaseUpdating   TensorFusionGPUPhase = constants.PhaseUpdating
	TensorFusionGPUPhaseRunning    TensorFusionGPUPhase = constants.PhaseRunning
	TensorFusionGPUPhaseUnknown    TensorFusionGPUPhase = constants.PhaseUnknown
	TensorFusionGPUPhaseDestroying TensorFusionGPUPhase = constants.PhaseDestroying
	TensorFusionGPUPhaseMigrating  TensorFusionGPUPhase = constants.PhaseMigrating
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
