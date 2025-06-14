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
	RequestsTFlops *resource.Quantity `json:"requests.tflops,omitempty"`
	// +optional
	RequestsVRAM *resource.Quantity `json:"requests.vram,omitempty"`

	// Total limits for the namespace
	// +optional
	LimitsTFlops *resource.Quantity `json:"limits.tflops,omitempty"`
	// +optional
	LimitsVRAM *resource.Quantity `json:"limits.vram,omitempty"`

	// Maximum number of workers in the namespace
	// +optional
	Workers *int32 `json:"workers,omitempty"`

	// Alert threshold percentage (0-100)
	// When usage exceeds this percentage, an alert event will be triggered
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	// +kubebuilder:default=95
	AlertThresholdPercent *int32 `json:"alertThresholdPercent,omitempty"`
}

// GPUResourceQuotaSingle defines per-workload limits
type GPUResourceQuotaSingle struct {
	// Maximum resources per workload
	// +optional
	Max *GPUResourceLimits `json:"max,omitempty"`

	// Minimum resources per workload
	// +optional
	Min *GPUResourceLimits `json:"min,omitempty"`

	// Default limits applied to workloads without explicit limits
	// +optional
	Default *GPUResourceDefaults `json:"default,omitempty"`

	// Default requests applied to workloads without explicit requests
	// +optional
	DefaultRequest *GPUResourceDefaults `json:"defaultRequest,omitempty"`
}

// GPUResourceLimits defines resource limits
type GPUResourceLimits struct {
	// TFlops limit
	// +optional
	TFlops *resource.Quantity `json:"tflops,omitempty"`

	// VRAM limit
	// +optional
	VRAM *resource.Quantity `json:"vram,omitempty"`

	// Maximum number of workers
	// +optional
	Workers *int32 `json:"workers,omitempty"`
}

// GPUResourceDefaults defines default resource values
type GPUResourceDefaults struct {
	// Default TFlops
	// +optional
	TFlops *resource.Quantity `json:"tflops,omitempty"`

	// Default VRAM
	// +optional
	VRAM *resource.Quantity `json:"vram,omitempty"`
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
	RequestsTFlops *resource.Quantity `json:"requests.tflops,omitempty"`
	// +optional
	RequestsVRAM *resource.Quantity `json:"requests.vram,omitempty"`

	// Current limits usage
	// +optional
	LimitsTFlops *resource.Quantity `json:"limits.tflops,omitempty"`
	// +optional
	LimitsVRAM *resource.Quantity `json:"limits.vram,omitempty"`

	// Current number of workers
	// +optional
	Workers *int32 `json:"workers,omitempty"`
}

// GPUResourceAvailablePercent defines available percentage for each resource
type GPUResourceAvailablePercent struct {
	// Available percentage for requests.tflops (0-100)
	// +optional
	RequestsTFlops *int64 `json:"requests.tflops,omitempty"`

	// Available percentage for requests.vram (0-100)
	// +optional
	RequestsVRAM *int64 `json:"requests.vram,omitempty"`

	// Available percentage for limits.tflops (0-100)
	// +optional
	LimitsTFlops *int64 `json:"limits.tflops,omitempty"`

	// Available percentage for limits.vram (0-100)
	// +optional
	LimitsVRAM *int64 `json:"limits.vram,omitempty"`

	// Available percentage for workers (0-100)
	// +optional
	Workers *int64 `json:"workers,omitempty"`
}

// GPUResourceQuotaConditionType defines the condition types for GPUResourceQuota
type GPUResourceQuotaConditionType string

const (
	// GPUResourceQuotaConditionReady indicates the quota is ready and functioning
	GPUResourceQuotaConditionReady GPUResourceQuotaConditionType = "Ready"
	// GPUResourceQuotaConditionExceeded indicates the quota has been exceeded
	GPUResourceQuotaConditionExceeded GPUResourceQuotaConditionType = "Exceeded"
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

func init() {
	SchemeBuilder.Register(&GPUResourceQuota{}, &GPUResourceQuotaList{})
}
