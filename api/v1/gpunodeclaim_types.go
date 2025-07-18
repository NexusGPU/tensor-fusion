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

// GPUNodeClaimStatus defines the observed state of GPUNodeClaim.
type GPUNodeClaimStatus struct {

	// +kubebuilder:default=Pending
	Phase GPUNodeClaimPhase `json:"phase"`

	InstanceID string `json:"instanceID,omitempty"`
}

type GPUNodeClaimPhase string

const (
	GPUNodeClaimPending  GPUNodeClaimPhase = "Pending"
	GPUNodeClaimCreating GPUNodeClaimPhase = "Creating"
	GPUNodeClaimBound    GPUNodeClaimPhase = "Bound"
)

const GPUNodeClaimKind = "GPUNodeClaim"

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"

// GPUNodeClaim is the Schema for the gpunodeclaims API.
type GPUNodeClaim struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`

	Spec   GPUNodeClaimSpec   `json:"spec,omitempty"`
	Status GPUNodeClaimStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// GPUNodeClaimList contains a list of GPUNodeClaim.
type GPUNodeClaimList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []GPUNodeClaim `json:"items"`
}

func init() {
	SchemeBuilder.Register(&GPUNodeClaim{}, &GPUNodeClaimList{})
}

type CapacityTypeEnum string

const (
	CapacityTypeOnDemand CapacityTypeEnum = "OnDemand"

	CapacityTypeReserved CapacityTypeEnum = "Reserved"

	// Spot and Preemptive are aliases of each other, used by different providers
	CapacityTypeSpot CapacityTypeEnum = "Spot"
)

// GPUNodeClaimSpec defines the desired state of GPUNodeClaim.
type GPUNodeClaimSpec struct {
	NodeName     string           `json:"nodeName,omitempty"`
	Region       string           `json:"region,omitempty"`
	Zone         string           `json:"zone,omitempty"`
	InstanceType string           `json:"instanceType,omitempty"`
	NodeClassRef GroupKindName    `json:"nodeClassRef,omitempty"`
	CapacityType CapacityTypeEnum `json:"capacityType,omitempty"`

	TFlopsOffered    resource.Quantity `json:"tflopsOffered"`
	VRAMOffered      resource.Quantity `json:"vramOffered"`
	GPUDeviceOffered int32             `json:"gpuDeviceOffered"`

	ExtraParams map[string]string `json:"extraParams,omitempty"`
}

type GroupKindName struct {
	Group   string `json:"group"`
	Kind    string `json:"kind"`
	Version string `json:"version"`
	Name    string `json:"name"`
}
