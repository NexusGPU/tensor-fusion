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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// GPUResourceQuotaSpec defines the desired state of GPUResourceQuota
type GPUResourceQuotaSpec struct {
	// Total namespace limits (similar to ResourceQuotas)
	Total GPUResourceQuotaTotal `json:"total,omitempty"`

	// Per-workload limits (similar to LimitRanges)
	Single GPUResourceQuotaSingle `json:"single,omitempty"`
}

// GPUResourceQuotaTotal defines total namespace limits
type GPUResourceQuotaTotal struct {
	// Total requests limits for the namespace
	// +optional
	Requests *Resource `json:"requests,omitempty"`

	// Total limits for the namespace
	// +optional
	Limits *Resource `json:"limits,omitempty"`

	// Maximum number of workers in the namespace
	// +optional
	// +kubebuilder:default=32768
	MaxWorkers *int32 `json:"maxWorkers,omitempty"`

	// Alert threshold percentage (0-100)
	// When usage exceeds this percentage, an alert event will be triggered
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	// +kubebuilder:default=95
	// +optional
	AlertThresholdPercent *int32 `json:"alertThresholdPercent,omitempty"`
}

// GPUResourceQuotaSingle defines per-workload limits
type GPUResourceQuotaSingle struct {
	// Maximum resources per workload
	// +optional
	MaxRequests *Resource `json:"maxRequests,omitempty"`

	// +optional
	MaxLimits *Resource `json:"maxLimits,omitempty"`

	// +optional
	MaxGPUCount *int32 `json:"maxGPUCount,omitempty"`

	// Default limits applied to workloads without explicit limits
	// +optional
	DefaultRequests *Resource `json:"defaultRequests,omitempty"`

	// Default requests applied to workloads without explicit requests
	// +optional
	DefaultLimits *Resource `json:"defaultLimits,omitempty"`
}

// GPUResourceQuotaStatus defines the observed state of GPUResourceQuota
type GPUResourceQuotaStatus struct {
	// Current resource usage in the namespace
	Used GPUResourceUsage `json:"used,omitempty"`

	// Available percentage for each resource type
	AvailablePercent GPUResourceAvailablePercent `json:"availablePercent,omitempty"`

	// Conditions represent the latest available observations of the quota's state
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// LastUpdateTime is the last time the status was updated
	// +optional
	LastUpdateTime *metav1.Time `json:"lastUpdateTime,omitempty"`
}

// GPUResourceUsage defines current resource usage
type GPUResourceUsage struct {
	// Current requests usage
	// +optional
	Requests Resource `json:"requests,omitempty"`

	// Current limits usage
	// +optional
	Limits Resource `json:"limits,omitempty"`

	// Current number of workers
	// +optional
	Workers int32 `json:"workers,omitempty"`
}

// GPUResourceAvailablePercent defines available percentage for each resource
// Use string for round(2) float to avoid kubernetes resource can not store float issue
type GPUResourceAvailablePercent struct {
	// +optional
	RequestsTFlops string `json:"requests.tflops,omitempty"`

	// +optional
	RequestsVRAM string `json:"requests.vram,omitempty"`

	// +optional
	LimitsTFlops string `json:"limits.tflops,omitempty"`

	// +optional
	LimitsVRAM string `json:"limits.vram,omitempty"`

	// +optional
	Workers string `json:"workers,omitempty"`
}

// GPUResourceQuotaConditionType defines the condition types for GPUResourceQuota
type GPUResourceQuotaConditionType string

const (
	// GPUResourceQuotaConditionReady indicates the quota is ready and functioning
	GPUResourceQuotaConditionReady GPUResourceQuotaConditionType = "Ready"

	// GPUResourceQuotaConditionAlertThresholdReached indicates the alert threshold has been reached
	GPUResourceQuotaConditionAlertThresholdReached GPUResourceQuotaConditionType = "AlertThresholdReached"
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced
// +kubebuilder:printcolumn:name="Requests TFlops Used",type=string,JSONPath=`.status.used["requests.tflops"]`
// +kubebuilder:printcolumn:name="Requests VRAM Used",type=string,JSONPath=`.status.used["requests.vram"]`
// +kubebuilder:printcolumn:name="Workers Used",type=integer,JSONPath=`.status.used.workers`
// +kubebuilder:printcolumn:name="Alert Threshold",type=integer,JSONPath=`.spec.total.alertThresholdPercent`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// GPUResourceQuota is the Schema for the gpuresourcequotas API
type GPUResourceQuota struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   GPUResourceQuotaSpec   `json:"spec,omitempty"`
	Status GPUResourceQuotaStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// GPUResourceQuotaList contains a list of GPUResourceQuota
type GPUResourceQuotaList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []GPUResourceQuota `json:"items"`
}

type AllocRequest struct {
	// Name of the GPU pool to allocate from
	PoolName string
	// Namespace information for the workload
	WorkloadNameNamespace NameNamespace
	// Resource requirements for the allocation
	Request Resource
	Limit   Resource
	// Number of GPUs to allocate
	Count uint
	// Specific GPU model to allocate, empty string means any model
	GPUModel string
	// Node affinity requirements
	NodeAffinity *v1.NodeAffinity

	// final scheduled GPU IDs for this allocation request
	// This fields is set by GPUAllocator, user should not choose specific GPUs
	GPUNames []string

	// record the pod meta for quota check
	PodMeta metav1.ObjectMeta
}

type GPUAllocationInfo struct {
	Request   Resource `json:"request,omitempty"`
	Limit     Resource `json:"limit,omitempty"`
	PodName   string   `json:"podName,omitempty"`
	PodUID    string   `json:"podUID,omitempty"`
	Namespace string   `json:"namespace,omitempty"`
}

type AdjustRequest struct {
	PodUID     string
	IsScaleUp  bool
	NewRequest Resource
	NewLimit   Resource
}

func (ar *AllocRequest) Clone() framework.StateData {
	return ar
}

func init() {
	SchemeBuilder.Register(&GPUResourceQuota{}, &GPUResourceQuotaList{})
}
