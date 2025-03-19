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

// +kubebuilder:validation:Enum=low;medium;high;critical
type QoSLevel string

const (
	QoSLow      QoSLevel = "low"
	QoSMedium   QoSLevel = "medium"
	QoSHigh     QoSLevel = "high"
	QoSCritical QoSLevel = "critical"
)

// WorkloadProfileSpec defines the desired state of WorkloadProfile.
type WorkloadProfileSpec struct {
	// +optional
	PoolName string `json:"poolName,omitempty"`

	// +optional
	Resources Resources `json:"resources,omitempty"`

	// +optional
	// Qos defines the quality of service level for the client.
	Qos QoSLevel `json:"qos,omitempty"`

	IsLocalGPU bool `json:"isLocalGPU"`
}

// WorkloadProfileStatus defines the observed state of WorkloadProfile.
type WorkloadProfileStatus struct {
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// WorkloadProfile is the Schema for the workloadprofiles API.
type WorkloadProfile struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   WorkloadProfileSpec   `json:"spec,omitempty"`
	Status WorkloadProfileStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// WorkloadProfileList contains a list of WorkloadProfile.
type WorkloadProfileList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []WorkloadProfile `json:"items"`
}

func init() {
	SchemeBuilder.Register(&WorkloadProfile{}, &WorkloadProfileList{})
}
