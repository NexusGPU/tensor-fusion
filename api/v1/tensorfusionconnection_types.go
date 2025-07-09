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
	WorkloadName string `json:"workloadName"`
	ClientPod    string `json:"clientPod"`
}

// TensorFusionConnectionStatus defines the observed state of TensorFusionConnection.
type TensorFusionConnectionStatus struct {
	Phase         WorkerPhase `json:"phase"`
	ConnectionURL string      `json:"connectionURL"`
	WorkerName    string      `json:"workerName"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Connection URL",type="string",JSONPath=".status.connectionURL"
// +kubebuilder:printcolumn:name="Worker Name",type="string",JSONPath=".status.workerName"
// +kubebuilder:printcolumn:name="Workload Name",type="string",JSONPath=".spec.workloadName"
// +kubebuilder:printcolumn:name="Client Pod",type="string",JSONPath=".spec.clientPod"

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
