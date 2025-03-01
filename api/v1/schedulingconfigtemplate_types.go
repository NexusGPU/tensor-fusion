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
	"k8s.io/apimachinery/pkg/runtime"
)

// Place the workload to right nodes and scale smart.
type SchedulingConfigTemplateSpec struct {

	// place the client or worker to best matched nodes
	Placement PlacementConfig `json:"placement"`

	// scale the workload based on the usage and traffic
	// +optional
	AutoScaling *AutoScalingConfig `json:"autoScaling,omitempty"`

	// avoid hot GPU devices and continuously balance the workload
	// implemented by trigger a simulation scheduling and advise better GPU nodes for scheduler
	// +optional
	ReBalancer *ReBalancerConfig `json:"reBalancer,omitempty"`

	// single GPU device multi-process queuing and fair scheduling with QoS constraint
	// +optional
	Hypervisor *HypervisorScheduling `json:"hypervisor,omitempty"`
}

type PlacementConfig struct {
	// +kubebuilder:default=CompactFirst
	Mode PlacementMode `json:"mode"`

	// +kubebuilder:default=true
	// +optional
	AllowUsingLocalGPU *bool `json:"allowUsingLocalGPU,omitempty"` // If false, workloads will not be scheduled directly to GPU nodes with 'localGPU: true'.

	// +optional
	GPUFilters []GPUFilter `json:"gpuFilters,omitempty"`
}

// +kubebuilder:validation:Enum=CompactFirst;LowLoadFirst
type PlacementMode string

const (
	// default to compactFirst for cost saving and energy saving
	PlacementModeCompactFirst PlacementMode = "CompactFirst"

	// in some cases, use lowLoadFirst for balance and fairness
	PlacementModeLowLoadFirst PlacementMode = "LowLoadFirst"
)

// GPUFilter is to select eligible GPUs for scheduling.
//
// example:
// ```yaml
// - type: avoidTooMuchConnectionsOnSameGPU
// params:
//
//	connectionNum: 150
//
// - type: avoidDifferentZone
// params:
//
//	# by default, GPU worker will be scheduled into the same zone as CPU Client Pod to align AZ and improve performance
//	topologyKey: topology.kubernetes.io/zone
//
// ```
type GPUFilter struct {
	Type   string               `json:"type,omitempty"`
	Params runtime.RawExtension `json:"params,omitempty"`
}

type AutoScalingConfig struct {
	// layer 1 vertical auto-scaling, turbo burst to existing GPU cards quickly
	AutoSetLimits AutoSetLimits `json:"autoSetLimits,omitempty"`

	// layer 2 horizontal auto-scaling, scale up to more GPU cards if max limits threshold hit
	AutoSetReplicas AutoSetReplicas `json:"autoSetReplicas,omitempty"`

	// layer 3 adjusting, to match the actual usage in the long run
	AutoSetRequests AutoSetRequests `json:"autoSetRequests,omitempty"`

	// additional layer to save VRAM, auto-freeze memory and cool down to RAM and Disk
	ScaleToZero ScaleToZero `json:"scaleToZero,omitempty"`
}

// A typical autoLimits algorithm could be checking every 5m, look back 1 day data,
// select 99% of actual usage as preferredLimits,
// calculate finalPreferredLimits, which is preferredLimits*(1+extraBufferRatio)
// if they are equal with each other within a range (eg. 5%), do nothing
// if finalPreferredLimits is less than current limits and exceeded error range,
// set current limits to finalPreferredLimits
// if finalPreferredLimits > current limits and exceeded error range,
// set current limits to max(finalPreferredLimits, current limits * scaleUpStep)
// if AI prediction enabled, it helps to detect history pattern, and set more reasonable, explainable limit value
// the final set limits should be max(finalPreferredLimits, last(predict_value * (1 + extraTFlopsBufferRatio)))
type AutoSetLimits struct {
	EvaluationPeriod string `json:"evaluationPeriod,omitempty"`

	ExtraTFlopsBufferRatio string `json:"extraTFlopsBufferRatio,omitempty"`

	IgnoredDeltaRange string `json:"ignoredDeltaRange,omitempty"`

	ScaleUpStep string `json:"scaleUpStep,omitempty"`

	// the multiplier of requests, to avoid limit set too high, like 5.0
	MaxRatioToRequests string `json:"maxRatioToRequests,omitempty"`

	Prediction *SmartSchedulerModelInput `json:"prediction,omitempty"`
}

// To handle burst traffic, scale up in short time (this feature requires GPU context migration & replication, not available yet)
type AutoSetReplicas struct {
	Enable                *bool  `json:"enable,omitempty"`
	TargetTFlopsOfLimits  string `json:"targetTFlopsOfLimits,omitempty"`
	EvaluationPeriod      string `json:"evaluationPeriod,omitempty"`
	ScaleUpStep           string `json:"scaleUpStep,omitempty"`
	ScaleDownStep         string `json:"scaleDownStep,omitempty"`
	ScaleUpCoolDownTime   string `json:"scaleUpCoolDownTime,omitempty"`
	ScaleDownCoolDownTime string `json:"scaleDownCoolDownTime,omitempty"`
}

type AutoSetRequests struct {
	PercentileForAutoRequests string `json:"percentileForAutoRequests,omitempty"`

	// the request buffer ratio, for example actual usage is 1.0, 10% buffer will be 1.1 as final preferred requests
	ExtraBufferRatio string `json:"extraBufferRatio,omitempty"`

	EvaluationPeriod  string                   `json:"evaluationPeriod,omitempty"`
	AggregationPeriod string                   `json:"aggregationPeriod,omitempty"`
	Prediction        SmartSchedulerModelInput `json:"prediction,omitempty"`
}

type ScaleToZero struct {
	AutoFreeze         []AutoFreeze             `json:"autoFreeze,omitempty"`
	IntelligenceWarmup SmartSchedulerModelInput `json:"intelligenceWarmup,omitempty"`
}

type AutoFreeze struct {
	Qos             QoSLevel `json:"qos,omitempty"`
	FreezeToMemTTL  string   `json:"freezeToMemTTL,omitempty"`
	FreezeToDiskTTL string   `json:"freezeToDiskTTL,omitempty"`
	Enable          *bool    `json:"enable,omitempty"`
}

type SmartSchedulerModelInput struct {
	Enable            *bool  `json:"enable,omitempty"`
	Model             string `json:"model,omitempty"`
	HistoryDataPeriod string `json:"historyDataPeriod,omitempty"`
	PredictionPeriod  string `json:"predictionPeriod,omitempty"`
}

type ReBalancerConfig struct {
	Internal              string             `json:"internal,omitempty"`
	ReBalanceCoolDownTime string             `json:"reBalanceCoolDownTime,omitempty"`
	Threshold             ReBalanceThreshold `json:"threshold,omitempty"`
}

type ReBalanceThreshold struct {
	MatchAny runtime.RawExtension `json:"matchAny,omitempty"`
}

type HypervisorScheduling struct {
	MultiProcessQueuing MultiProcessQueuing `json:"multiProcessQueuing,omitempty"`
}

type MultiProcessQueuing struct {
	Enable               *bool    `json:"enable,omitempty"`
	Interval             string   `json:"interval,omitempty"`
	QueueLevelTimeSlices []string `json:"queueLevelTimeSlices,omitempty"`
}

// SchedulingConfigTemplateStatus defines the observed state of SchedulingConfigTemplate.
type SchedulingConfigTemplateStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster
// +kubebuilder:printcolumn:name="Mode",type="string",JSONPath=".spec.placement.mode"
// +kubebuilder:printcolumn:name="Allow Local GPU",type="string",JSONPath=".spec.placement.allowLocalGPU"
// SchedulingConfigTemplate is the Schema for the schedulingconfigtemplates API.
type SchedulingConfigTemplate struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   SchedulingConfigTemplateSpec   `json:"spec,omitempty"`
	Status SchedulingConfigTemplateStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// SchedulingConfigTemplateList contains a list of SchedulingConfigTemplate.
type SchedulingConfigTemplateList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []SchedulingConfigTemplate `json:"items"`
}

func init() {
	SchemeBuilder.Register(&SchedulingConfigTemplate{}, &SchedulingConfigTemplateList{})
}
