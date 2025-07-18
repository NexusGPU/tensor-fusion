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

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// GPUNodeClassSpec defines the desired state of GPUNodeClass.
type GPUNodeClassSpec struct {
	// +optional
	// The launch template to use for VM instances, if set, all other fields could be skipped
	LaunchTemplate NodeClassItemSelectorTerms `json:"launchTemplate"`

	// +optional
	// Could be private or public, varies in different cloud vendor, define where to query the OSImageID
	// +kubebuilder:default="Private"
	OSImageType OSImageTypeEnum `json:"osImageType,omitempty"`

	// the OS image identifier string, default to use first one, if not found, fallback to others
	OSImageSelectorTerms []NodeClassItemSelectorTerms `json:"osImageSelectorTerms,omitempty"`

	// +optional
	// The instance profile to use, assign IAM role and permissions for EC2 instances
	InstanceProfile string `json:"instanceProfile,omitempty"`

	// +optional
	// for AWS only, IMDSv2 metadata service options
	MetadataOptions *NodeClassMetadataOptions `json:"metadataOptions,omitempty"`

	// +optional
	SecurityGroupSelectorTerms []NodeClassItemSelectorTerms `json:"securityGroupSelectorTerms,omitempty"`

	// +optional
	SubnetSelectorTerms []NodeClassItemSelectorTerms `json:"subnetSelectorTerms,omitempty"` // Terms to select subnets

	// +optional
	BlockDeviceMappings []NodeClassBlockDeviceMappings `json:"blockDeviceMappings,omitempty"` // Block device mappings for the instance

	// +optional
	Tags map[string]string `json:"tags,omitempty"` // Tags associated with the resource

	// +optional
	UserData string `json:"userData,omitempty"` // User data script for the instance
}

// +kubebuilder:validation:Enum=Private;Public;System
type OSImageTypeEnum string

const (
	OSImageTypePrivate OSImageTypeEnum = "Private"
	OSImageTypePublic  OSImageTypeEnum = "Public"
	OSImageTypeSystem  OSImageTypeEnum = "System"
)

type NodeClassItemSelectorTerms struct {

	// +optional
	// The item ID
	ID string `json:"id,omitempty"`

	// +optional
	// The item name
	Name string `json:"name,omitempty"`

	// +optional
	// Query by tags
	Tags map[string]string `json:"tags,omitempty"`
}

// AWS IMDSv2 metadata service options
type NodeClassMetadataOptions struct {
	// +optional
	// +kubebuilder:default=true
	HttpEndpoint bool `json:"httpEndpoint,omitempty"`

	// +optional
	// +kubebuilder:default=false
	HttpProtocolIPv6 bool `json:"httpProtocolIPv6,omitempty"`

	// +optional
	// +kubebuilder:default=1
	HttpPutResponseHopLimit int `json:"httpPutResponseHopLimit,omitempty"`

	// +optional
	// +kubebuilder:default="required"
	HttpTokens string `json:"httpTokens,omitempty"`
}

type NodeClassBlockDeviceMappings struct {
	// +optional
	DeviceName string `json:"deviceName,omitempty"` // The device name for the block device

	EBS NodeClassBlockDeviceSettings `json:"ebs,omitempty"`
}

type NodeClassBlockDeviceSettings struct {
	VolumeSize string `json:"volumeSize,omitempty"`

	// +optional
	// Default value would varies based on the cloud vendor
	// For AWS it's gp3, for Alicloud it's cloud_essd
	VolumeType string `json:"volumeType,omitempty"`

	// +optional
	// +kubebuilder:default=true
	DeleteOnTermination bool `json:"deleteOnTermination,omitempty"` // Whether to delete the EBS volume on termination

	// +optional
	// +kubebuilder:default=true
	Encrypted bool `json:"encrypted,omitempty"` // Whether the EBS volume is encrypted

}

// GPUNodeClassStatus defines the observed state of GPUNodeClass.
type GPUNodeClassStatus struct {
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster
// GPUNodeClass is the Schema for the gpunodeclasses API.
type GPUNodeClass struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   GPUNodeClassSpec   `json:"spec,omitempty"`
	Status GPUNodeClassStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// GPUNodeClassList contains a list of GPUNodeClass.
type GPUNodeClassList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []GPUNodeClass `json:"items"`
}

const (
	GPUNodeClassKind = "GPUNodeClass"
)

func init() {
	SchemeBuilder.Register(&GPUNodeClass{}, &GPUNodeClassList{})
}
