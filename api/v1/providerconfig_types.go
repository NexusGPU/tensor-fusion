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

// ProviderConfigSpec defines the desired state of ProviderConfig
// ProviderConfig is used to manage hardware vendor specific configurations
// including images, hardware metadata, virtualization templates and resource names
type ProviderConfigSpec struct {
	// Vendor is the hardware vendor name (e.g., NVIDIA, AMD, Ascend, Apple)
	// +kubebuilder:validation:Required
	Vendor string `json:"vendor"`

	// Images contains container images for different components
	Images ProviderImages `json:"images"`

	// Hypervisor contains vendor-specific hypervisor settings
	// +optional
	Hypervisor *ProviderHypervisorConfig `json:"hypervisor,omitempty"`

	// HardwareMetadata contains GPU/accelerator model information
	// +optional
	HardwareMetadata []HardwareModelInfo `json:"hardwareMetadata,omitempty"`

	// VirtualizationTemplates contains partition/slice templates that can be referenced by hardware metadata
	// This avoids duplicating template info across multiple GPU models
	// +optional
	VirtualizationTemplates []VirtualizationTemplate `json:"virtualizationTemplates,omitempty"`

	// InUseResourceNames contains resource names that should be removed from pods
	// when TensorFusion takes over GPU management (e.g., "nvidia.com/gpu", "amd.com/gpu")
	// +optional
	InUseResourceNames []string `json:"inUseResourceNames,omitempty"`

	// DevicePluginDetection contains settings for detecting existing device plugins
	// +optional
	DevicePluginDetection *DevicePluginDetectionConfig `json:"devicePluginDetection,omitempty"`
}

// ProviderHypervisorConfig contains vendor-specific hypervisor configuration
type ProviderHypervisorConfig struct {
	// PrivilegedHypervisor indicates the hypervisor container should run in privileged mode
	// +optional
	PrivilegedHypervisor bool `json:"privilegedHypervisor,omitempty"`

	// LDLibraryPath appends entries to LD_LIBRARY_PATH for the hypervisor container.
	// Vendor remote workers may also reuse this when host driver libraries must be mounted.
	// +optional
	LDLibraryPath string `json:"ldLibraryPath,omitempty"`

	// ExtraEnv appends extra environment variables to the hypervisor container.
	// This is useful for vendor-specific runtime toggles, for example static vNPU
	// scenarios that need partition runtime options.
	// +optional
	ExtraEnv []ProviderHypervisorEnvVar `json:"extraEnv,omitempty"`

	// HostPathMounts adds host path mounts to the hypervisor pod.
	// Vendor remote workers may also reuse these mounts when they depend on host driver libraries.
	// +optional
	HostPathMounts []ProviderHypervisorHostPathMount `json:"hostPathMounts,omitempty"`

	// DeviceMount defines default device mount strategy used by hypervisor for this vendor.
	// Model-level settings in hardwareMetadata.deviceMount take precedence over this default.
	// +optional
	DeviceMount *ProviderDeviceMountConfig `json:"deviceMount,omitempty"`
}

// ProviderHypervisorEnvVar defines a name/value env var for hypervisor container.
type ProviderHypervisorEnvVar struct {
	// Name is the environment variable name.
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// Value is the environment variable value.
	// +optional
	Value string `json:"value,omitempty"`
}

// ProviderHypervisorHostPathMount defines a hostPath mount for hypervisor
type ProviderHypervisorHostPathMount struct {
	// Name is the volume name
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// HostPath is the path on the host to mount
	// +kubebuilder:validation:Required
	HostPath string `json:"hostPath"`

	// MountPath is the path inside the container
	// +kubebuilder:validation:Required
	MountPath string `json:"mountPath"`

	// ReadOnly indicates if the mount should be read-only
	// +optional
	ReadOnly bool `json:"readOnly,omitempty"`
}

// ProviderImages contains container images for TensorFusion components
type ProviderImages struct {
	// Middleware is the hypervisor/middleware image that runs on GPU nodes
	// This image contains vendor-specific GPU drivers and runtime
	// +kubebuilder:validation:Required
	Middleware string `json:"middleware"`

	// RemoteClient is the client library image injected into user pods
	// +optional
	RemoteClient string `json:"remoteClient,omitempty"`

	// RemoteWorker is the worker process image for remote GPU access
	// +optional
	RemoteWorker string `json:"remoteWorker,omitempty"`
}

// HardwareModelInfo contains information about a specific GPU/accelerator model
type HardwareModelInfo struct {
	// Model is the short model identifier (e.g., "A100_SXM_80G", "RTX4090")
	// +kubebuilder:validation:Required
	Model string `json:"model"`

	// FullModelName is the full human-readable name (e.g., "NVIDIA A100-SXM4-80GB")
	// +kubebuilder:validation:Required
	FullModelName string `json:"fullModelName"`

	// CostPerHour is the average cost per hour for this GPU model
	// +optional
	CostPerHour string `json:"costPerHour,omitempty"`

	// Fp16TFlops is the FP16 performance in TFlops
	// +kubebuilder:validation:Required
	Fp16TFlops resource.Quantity `json:"fp16TFlops"`

	// MaxPartitions is the maximum number of partitions this GPU can support
	// +optional
	MaxPartitions uint32 `json:"maxPartitions,omitempty"`

	// MaxPlacementSlots is the maximum number of placement slots (e.g., 8 for NVIDIA MIG)
	// +optional
	MaxPlacementSlots uint32 `json:"maxPlacementSlots,omitempty"`

	// MaxIsolationGroups is the maximum number of isolation groups (e.g., 4 for Ascend vGroups)
	// +optional
	MaxIsolationGroups uint32 `json:"maxIsolationGroups,omitempty"`

	// TotalExtendedResources defines the total capacity of extended resources
	// For Ascend NPU: {"AICORE": 8, "AICPU": 7, "VPC": 12, ...}
	// +optional
	TotalExtendedResources map[string]uint32 `json:"totalExtendedResources,omitempty"`

	// PartitionTemplateRefs contains references to virtualization templates
	// Use template IDs defined in VirtualizationTemplates
	// +optional
	PartitionTemplateRefs []string `json:"partitionTemplateRefs,omitempty"`

	// DeviceMount defines model-specific device mount strategy used by hypervisor.
	// This overrides spec.hypervisor.deviceMount when set.
	// +optional
	DeviceMount *ProviderDeviceMountConfig `json:"deviceMount,omitempty"`
}

// ProviderDeviceMountConfig defines how hypervisor should decide mounted devices.
type ProviderDeviceMountConfig struct {
	// DeviceMountRule is a CEL expression used in non-partitioned scenarios
	// (e.g. shared/soft/whole-card) to filter compute device nodes.
	// +optional
	DeviceMountRule string `json:"deviceMountRule,omitempty"`

	// PartitionedDeviceMountRule is a CEL expression used in partitioned scenarios
	// when provider-driven partitioning is performed.
	// +optional
	PartitionedDeviceMountRule string `json:"partitionedDeviceMountRule,omitempty"`

	// SharedDevices are always-mounted shared device paths or glob patterns.
	// Examples: /dev/nvidiactl, /dev/uburma/*, /dev/ummu/*
	// +optional
	SharedDevices []string `json:"sharedDevices,omitempty"`
}

// VirtualizationTemplate defines a partition/slice template for GPU virtualization
type VirtualizationTemplate struct {
	// ID is the unique identifier for this template (e.g., "mig-1g-10gb", "vir01")
	// +kubebuilder:validation:Required
	ID string `json:"id"`

	// Name is the vendor-specific name (e.g., "1g.10gb", "vir01")
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// MemoryGigabytes is the memory allocated to this partition
	// +kubebuilder:validation:Required
	MemoryGigabytes uint64 `json:"memoryGigabytes"`

	// ComputePercent is the percentage of compute allocated (0-100), serialized as string
	// +kubebuilder:validation:Required
	ComputePercent string `json:"computePercent"`

	// Description provides additional information about this template
	// +optional
	Description string `json:"description,omitempty"`

	// MaxInstances is the maximum number of instances of this template per GPU
	// +optional
	MaxInstances uint32 `json:"maxInstances,omitempty"`

	// PlacementLimit defines valid placement positions using a bitmask
	// For NVIDIA MIG: defines which slots can host this partition
	// +optional
	PlacementLimit []uint32 `json:"placementLimit,omitempty"`

	// PlacementOffset defines the slot offset for this template
	// +optional
	PlacementOffset uint32 `json:"placementOffset,omitempty"`

	// ExtendedResources contains additional resource dimensions
	// For Ascend NPU: {"AICORE": 1, "AICPU": 1, "VPC": 1, ...}
	// +optional
	ExtendedResources map[string]uint32 `json:"extendedResources,omitempty"`

	// IsolationGroupSharing defines how isolation groups are handled
	// "exclusive" - each partition requires its own isolation group
	// "shared" - multiple partitions can share an isolation group (time-sharing)
	// +optional
	// +kubebuilder:default="exclusive"
	// +kubebuilder:validation:Enum=exclusive;shared
	IsolationGroupSharing string `json:"isolationGroupSharing,omitempty"`

	// MaxPartitionsPerIsolationGroup limits partitions sharing one isolation group
	// Only applicable when IsolationGroupSharing is "shared"
	// +optional
	MaxPartitionsPerIsolationGroup uint32 `json:"maxPartitionsPerIsolationGroup,omitempty"`

	// IsolationGroupSlots defines minimum slots required by this template's isolation group
	// +optional
	IsolationGroupSlots uint32 `json:"isolationGroupSlots,omitempty"`
}

// DevicePluginDetectionConfig contains settings for detecting existing device plugins
type DevicePluginDetectionConfig struct {
	// ResourceNamePrefixes are prefixes used to identify this vendor's device plugin resources
	// e.g., ["nvidia.com/gpu", "nvidia.com/mig"] for NVIDIA
	// +optional
	ResourceNamePrefixes []string `json:"resourceNamePrefixes,omitempty"`

	// UsedBySystemName is the identifier for workloads using this vendor's native device plugin
	// +optional
	UsedBySystemName string `json:"usedBySystemName,omitempty"`
}

// ProviderConfigStatus defines the observed state of ProviderConfig
type ProviderConfigStatus struct{}

// +kubebuilder:object:root=true
// +kubebuilder:resource:scope=Cluster
// +kubebuilder:printcolumn:name="Vendor",type="string",JSONPath=".spec.vendor"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// ProviderConfig is the Schema for managing hardware vendor specific configurations
// Each ProviderConfig represents a single vendor (NVIDIA, AMD, Ascend, etc.)
type ProviderConfig struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ProviderConfigSpec   `json:"spec,omitempty"`
	Status ProviderConfigStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ProviderConfigList contains a list of ProviderConfig
type ProviderConfigList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ProviderConfig `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ProviderConfig{}, &ProviderConfigList{})
}
