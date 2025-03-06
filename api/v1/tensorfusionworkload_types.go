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

// TensorFusionWorkloadSpec defines the desired state of TensorFusionWorkload.
type TensorFusionWorkloadSpec struct {
	Replicas *int32 `json:"replicas,omitempty"`
	PoolName string `json:"poolName"`
	// +optional
	Resources Resources `json:"resources"`
	// +optional
	Qos QoSLevel `json:"qos,omitempty"`
	// +optional
	IsLocalGPU bool `json:"isLocalGPU,omitempty"`
}

type WorkerPhase string

const (
	WorkerPending WorkerPhase = "Pending"
	WorkerRunning WorkerPhase = "Running"
	WorkerFailed  WorkerPhase = "Failed"
)

type WorkerStatus struct {
	WorkerPhase WorkerPhase `json:"workerPhase"`

	WorkerName   string            `json:"workerName"`
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// +optional
	WorkerIp string `json:"workerIp,omitempty"`
	// +optional
	WorkerPort int `json:"workerPort,omitempty"`
}

// TensorFusionWorkloadStatus defines the observed state of TensorFusionWorkload.
type TensorFusionWorkloadStatus struct {
	// replicas is the number of Pods created by the Workload controller.
	Replicas int32 `json:"replicas"`

	// readyReplicas is the number of pods created for this Workload with a Ready Condition.
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	WorkerStatuses []WorkerStatus `json:"workerStatuses,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// TensorFusionWorkload is the Schema for the tensorfusionworkloads API.
type TensorFusionWorkload struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TensorFusionWorkloadSpec   `json:"spec,omitempty"`
	Status TensorFusionWorkloadStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// TensorFusionWorkloadList contains a list of TensorFusionWorkload.
type TensorFusionWorkloadList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TensorFusionWorkload `json:"items"`
}

func init() {
	SchemeBuilder.Register(&TensorFusionWorkload{}, &TensorFusionWorkloadList{})
}
