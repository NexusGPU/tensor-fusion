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
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// GPUStatus defines the observed state of GPU.
type GPUStatus struct {
	// +kubebuilder:default=Pending
	Phase TensorFusionGPUPhase `json:"phase"`

	Capacity  Resource `json:"capacity"`
	Available Resource `json:"available"`

	UUID string `json:"uuid"`

	// The host match selector to schedule worker pods
	NodeSelector map[string]string `json:"nodeSelector"`

	Message string `json:"message"`
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

type GPUSpec struct {
	GPUModel string `json:"gpuModel"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster
// GPU is the Schema for the gpus API.
type GPU struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   GPUSpec   `json:"spec,omitempty"`
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
