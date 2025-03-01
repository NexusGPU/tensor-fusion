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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type Resource struct {
	Tflops resource.Quantity `json:"tflops"`
	Vram   resource.Quantity `json:"vram"`
}

type Resources struct {
	Requests Resource `json:"requests"`
	Limits   Resource `json:"limits"`
}

// TensorFusionConnectionSpec defines the desired state of TensorFusionConnection.
type TensorFusionConnectionSpec struct {
	PoolName  string    `json:"poolName"`
	Resources Resources `json:"resources"`

	// +optional
	// localGpu mode will schedule the GPU in advance
	GPUs []string `json:"gpu"`
}

type TensorFusionConnectionPhase string

// These are the valid phases of a GpuConnection.
const (
	TensorFusionConnectionPending  TensorFusionConnectionPhase = "Pending"
	TensorFusionConnectionStarting TensorFusionConnectionPhase = "Starting"
	TensorFusionConnectionRunning  TensorFusionConnectionPhase = "Running"
)

// TensorFusionConnectionStatus defines the observed state of TensorFusionConnection.
type TensorFusionConnectionStatus struct {
	Phase         TensorFusionConnectionPhase `json:"phase"`
	ConnectionURL string                      `json:"connectionURL"`
	QoS           QoSLevel                    `json:"qos,omitempty"`
	GPU           string                      `json:"gpu,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// TensorFusionConnection is the Schema for the tensorfusionconnections API.
type TensorFusionConnection struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TensorFusionConnectionSpec   `json:"spec,omitempty"`
	Status TensorFusionConnectionStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// TensorFusionConnectionList contains a list of TensorFusionConnection.
type TensorFusionConnectionList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TensorFusionConnection `json:"items"`
}

func init() {
	SchemeBuilder.Register(&TensorFusionConnection{}, &TensorFusionConnectionList{})
}
