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
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// GPUPoolSpec defines the desired state of GPUPool.
type GPUPoolSpec struct {
	CapacityConfig *CapacityConfig `json:"capacityConfig,omitempty"`

	NodeManagerConfig *NodeManagerConfig `json:"nodeManagerConfig,omitempty"`

	// +optional
	QosConfig *QosConfig `json:"qosConfig,omitempty"`

	// +optional
	ComponentConfig *ComponentConfig `json:"componentConfig,omitempty"`

	// +optional
	SchedulingConfigTemplate *string `json:"schedulingConfigTemplate,omitempty"`
}

type CapacityConfig struct {
	// +optional
	MinResources *GPUOrCPUResourceUnit `json:"minResources,omitempty"`

	// +optional
	MaxResources *GPUOrCPUResourceUnit `json:"maxResources,omitempty"`

	// +optional
	WarmResources *GPUOrCPUResourceUnit `json:"warmResources,omitempty"`

	// +optional
	Oversubscription *Oversubscription `json:"oversubscription,omitempty"`
}

type Oversubscription struct {
	// the percentage of Host RAM appending to GPU VRAM, default to 50%
	// +optional
	// +kubebuilder:default=50
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	VRAMExpandToHostMem int32 `json:"vramExpandToHostMem,omitempty"`

	// the percentage of Host Disk appending to GPU VRAM, default to 70%
	// +optional
	// +kubebuilder:default=70
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	VRAMExpandToHostDisk int32 `json:"vramExpandToHostDisk,omitempty"`

	// The multi of TFlops to oversell, default to 500%, indicates 5 times oversell
	// +optional
	// +kubebuilder:default=500
	// +kubebuilder:validation:Minimum=100
	// +kubebuilder:validation:Maximum=100000
	TFlopsOversellRatio int32 `json:"tflopsOversellRatio,omitempty"`
}

type NodeManagerConfig struct {
	// +kubebuilder:default="AutoSelect"
	ProvisioningMode ProvisioningMode `json:"provisioningMode,omitempty"`

	// +optional
	NodeProvisioner *NodeProvisioner `json:"nodeProvisioner,omitempty"`

	// +optional
	NodeSelector *corev1.NodeSelector `json:"nodeSelector,omitempty"`

	// +optional
	NodeCompaction *NodeCompaction `json:"nodeCompaction,omitempty"`

	// +optional
	NodePoolRollingUpdatePolicy *NodeRollingUpdatePolicy `json:"nodePoolRollingUpdatePolicy,omitempty"`
}

// +kubebuilder:validation:Enum=Provisioned;AutoSelect;Karpenter
type ProvisioningMode string

const (
	ProvisioningModeProvisioned ProvisioningMode = "Provisioned"

	ProvisioningModeAutoSelect ProvisioningMode = "AutoSelect"

	ProvisioningModeKarpenter ProvisioningMode = "Karpenter"
)

// NodeProvisioner or NodeSelector, they are exclusive.
// NodeSelector is for existing GPUs, NodeProvisioner is for Karpenter-like auto management.
type NodeProvisioner struct {

	// TensorFusion GPUNodeClass name
	NodeClass string `json:"nodeClass,omitempty"`

	// Karpenter NodeClass name
	// +optional
	KarpenterNodeClassRef *GroupKindName `json:"karpenterNodeClassRef,omitempty"`

	// +optional
	GPURequirements []Requirement `json:"gpuRequirements,omitempty"`
	// +optional
	GPUTaints []Taint `json:"gpuTaints,omitempty"`
	// +optional
	GPULabels map[string]string `json:"gpuNodeLabels,omitempty"`

	// +optional
	CPURequirements []Requirement `json:"cpuRequirements,omitempty"`
	// +optional
	CPUTaints []Taint `json:"cpuTaints,omitempty"`
	// +optional
	CPULabels map[string]string `json:"cpuNodeLabels,omitempty"`
	// +optional
	GPUAnnotation map[string]string `json:"gpuNodeAnnotations,omitempty"`

	// +optional
	// NodeProvisioner will start an virtual billing based on public pricing or customized pricing, if the VM's costs exceeded any budget constraints, the new VM will not be created, and alerts will be generated
	Budget *PeriodicalBudget `json:"budget,omitempty"`
}

// The budget constraints in dollars
type PeriodicalBudget struct {
	// +kubebuilder:default="100"
	BudgetPerDay string `json:"budgetPerDay,omitempty"`

	// +kubebuilder:default="1000"
	BudgetPerMonth string `json:"budgetPerMonth,omitempty"`

	// +kubebuilder:default="3000"
	BudgetPerQuarter string `json:"budgetPerQuarter,omitempty"`

	// +kubebuilder:default=AlertOnly
	BudgetExceedStrategy BudgetExceedStrategy `json:"budgetExceedStrategy,omitempty"`
}

// +kubebuilder:validation:Enum=AlertOnly;AlertAndTerminateVM
type BudgetExceedStrategy string

const (
	BudgetExceedStrategyAlertOnly           BudgetExceedStrategy = "AlertOnly"
	BudgetExceedStrategyAlertAndTerminateVM BudgetExceedStrategy = "AlertAndTerminateVM"
)

type Requirement struct {
	Key NodeRequirementKey `json:"key,omitempty"`

	// +kubebuilder:default="In"
	// +kubebuilder:validation:Enum=In;Exists;DoesNotExist;Gt;Lt
	Operator corev1.NodeSelectorOperator `json:"operator,omitempty"`

	Values []string `json:"values,omitempty"`
}

// +kubebuilder:validation:Enum=node.kubernetes.io/instance-type;kubernetes.io/arch;kubernetes.io/os;topology.kubernetes.io/region;topology.kubernetes.io/zone;karpenter.sh/capacity-type;tensor-fusion.ai/gpu-vendor;tensor-fusion.ai/gpu-instance-family;tensor-fusion.ai/gpu-instance-size
type NodeRequirementKey string

const (
	NodeRequirementKeyInstanceType NodeRequirementKey = "node.kubernetes.io/instance-type"
	NodeRequirementKeyArchitecture NodeRequirementKey = "kubernetes.io/arch"
	NodeRequirementKeyGPUVendor    NodeRequirementKey = "tensor-fusion.ai/gpu-vendor"

	NodeRequirementKeyOS     NodeRequirementKey = "kubernetes.io/os"
	NodeRequirementKeyRegion NodeRequirementKey = "topology.kubernetes.io/region"
	NodeRequirementKeyZone   NodeRequirementKey = "topology.kubernetes.io/zone"

	// capacity-type is charging method, can be spot/preemptive or on-demand
	NodeRequirementKeyCapacityType NodeRequirementKey = "karpenter.sh/capacity-type"

	NodeRequirementKeyInstanceFamily NodeRequirementKey = "tensor-fusion.ai/gpu-instance-family"
	NodeRequirementKeyInstanceSize   NodeRequirementKey = "tensor-fusion.ai/gpu-instance-size"
)

type Taint struct {
	// +kubebuilder:default=NoSchedule
	// +kubebuilder:validation:Enum=NoSchedule;NoExecute;PreferNoSchedule
	Effect corev1.TaintEffect `json:"effect,omitempty"`
	Key    string             `json:"key,omitempty"`
	Value  string             `json:"value,omitempty"`
}

type NodeCompaction struct {
	// +kubebuilder:default="5m"
	Period string `json:"period,omitempty"`
}
type NodeRollingUpdatePolicy struct {
	// If set to false, updates will be pending in status, and user needs to manually approve updates.
	// Updates will occur immediately or during the next maintenance window.

	// +kubebuilder:default=true
	// +optional
	AutoUpdate *bool `json:"autoUpdate,omitempty"`

	// +kubebuilder:default=100
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	BatchPercentage int32 `json:"batchPercentage,omitempty"`

	// +kubebuilder:default="10m"
	BatchInterval string `json:"batchInterval,omitempty"`

	// +optional
	// +kubebuilder:default="10m"
	MaxDuration string `json:"maxDuration,omitempty"`

	// +optional
	MaintenanceWindow MaintenanceWindow `json:"maintenanceWindow,omitempty"`
}

type MaintenanceWindow struct {
	// crontab syntax.
	Includes []string `json:"includes,omitempty"`
}

// Define different QoS and their price.
type QosConfig struct {
	Definitions []QosDefinition `json:"definitions,omitempty"`
	DefaultQoS  QoSLevel        `json:"defaultQoS,omitempty"`
	Pricing     []QosPricing    `json:"pricing,omitempty"`
}

type QosDefinition struct {
	Name        QoSLevel `json:"name,omitempty"`
	Description string   `json:"description,omitempty"`
	Priority    int      `json:"priority,omitempty"` // Range from 1-100, reflects the scheduling priority when GPU is full and tasks are in the queue.
}

type GPUResourceUnit struct {
	// Tera floating point operations per second
	TFlops resource.Quantity `json:"tflops,omitempty"`

	// VRAM is short for Video memory, namely GPU RAM
	VRAM resource.Quantity `json:"vram,omitempty"`
}

type GPUOrCPUResourceUnit struct {
	TFlops resource.Quantity `json:"tflops,omitempty"`

	VRAM resource.Quantity `json:"vram,omitempty"`

	// CPU/Memory is only available when CloudVendor connection is enabled
	// +optional
	CPU resource.Quantity `json:"cpu,omitempty"`

	// +optional
	Memory resource.Quantity `json:"memory,omitempty"`
}

type QosPricing struct {
	Qos QoSLevel `json:"qos,omitempty"`

	Requests GPUResourcePricingUnit `json:"requests,omitempty"`

	// Default requests and limitsOverRequests are same, indicates normal on-demand serverless GPU usage, in hands-on lab low QoS case, limitsOverRequests should be lower, so that user can get burstable GPU resources with very low cost
	// +kubebuilder:default="1"
	LimitsOverRequestsChargingRatio string `json:"limitsOverRequests,omitempty"`
}

// The default pricing based on second level pricing from https://modal.com/pricing
// with Tensor/CUDA Core : HBM = 2:1
type GPUResourcePricingUnit struct {
	// price is per hour, billing period is any time unit

	// +kubebuilder:default="$0.0069228"
	PerFP16TFlopsPerHour string `json:"perFP16TFlopsPerHour,omitempty"`

	// +kubebuilder:default="$0.01548"
	PerGBOfVRAMPerHour string `json:"perGBOfVRAMPerHour,omitempty"`
}

// Customize system components for seamless onboarding.
type ComponentConfig struct {
	// +optional
	Worker *WorkerConfig `json:"worker,omitempty"`

	// +optional
	Hypervisor *HypervisorConfig `json:"hypervisor,omitempty"`

	// +optional
	NodeDiscovery *NodeDiscoveryConfig `json:"nodeDiscovery,omitempty"`

	// +optional
	Client *ClientConfig `json:"client,omitempty"`
}
type NodeDiscoveryConfig struct {
	Image string `json:"image,omitempty"`

	// +optional
	PodTemplate *runtime.RawExtension `json:"podTemplate,omitempty"`
}

type HypervisorConfig struct {
	Image string `json:"image,omitempty"`

	VectorImage string `json:"vectorImage,omitempty"`

	// +kubebuilder:default=8000
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=65535
	// +optional
	PortNumber *int32 `json:"portNumber,omitempty"`

	// +optional
	PodTemplate *runtime.RawExtension `json:"podTemplate,omitempty"`

	// +optional
	EnableVector bool `json:"enableVector,omitempty"`
}

type WorkerConfig struct {
	Image string `json:"image,omitempty"`
	// +optional
	PodTemplate *runtime.RawExtension `json:"podTemplate,omitempty"`
}

type ClientConfig struct {
	RemoteModeImage string `json:"remoteModeImage,omitempty"`

	EmbeddedModeImage string `json:"embeddedModeImage,omitempty"`

	OperatorEndpoint string `json:"operatorEndpoint,omitempty"`

	// +optional
	PatchToPod *runtime.RawExtension `json:"patchToPod,omitempty"`

	// +optional
	PatchToContainer *runtime.RawExtension `json:"patchToContainer,omitempty"`

	// +optional
	PatchToEmbeddedWorkerContainer *runtime.RawExtension `json:"patchToEmbeddedWorkerContainer,omitempty"`

	// +optional
	PatchEmbeddedWorkerToPod *runtime.RawExtension `json:"patchEmbeddedWorkerToPod,omitempty"`
}

// GPUPoolStatus defines the observed state of GPUPool.
type GPUPoolStatus struct {
	Cluster string `json:"cluster,omitempty"`

	// +kubebuilder:default=Pending
	Phase TensorFusionPoolPhase `json:"phase"`

	Conditions []metav1.Condition `json:"conditions,omitempty"`

	TotalNodes    int32 `json:"totalNodes,omitempty"`
	TotalGPUs     int32 `json:"totalGPUs,omitempty"`
	ReadyNodes    int32 `json:"readyNodes"`
	NotReadyNodes int32 `json:"notReadyNodes"`

	TotalTFlops resource.Quantity `json:"totalTFlops"`
	TotalVRAM   resource.Quantity `json:"totalVRAM"`

	VirtualTFlops resource.Quantity `json:"virtualTFlops"`
	VirtualVRAM   resource.Quantity `json:"virtualVRAM"`

	AvailableTFlops resource.Quantity `json:"availableTFlops"`
	AvailableVRAM   resource.Quantity `json:"availableVRAM"`

	RunningAppsCnt int32 `json:"runningAppsCnt,omitempty"`

	// +optional
	VirtualAvailableTFlops *resource.Quantity `json:"virtualAvailableTFlops,omitempty"`
	// +optional
	VirtualAvailableVRAM *resource.Quantity `json:"virtualAvailableVRAM,omitempty"`

	// when updating any component version or config, pool controller will perform rolling update.
	// the status will be updated periodically, default to 5s, progress will be 0-100.
	// when the progress is 100, the component version or config is fully updated.
	ComponentStatus PoolComponentStatus `json:"componentStatus"`

	// TODO: calculated every 1h/1d/1w average
	UtilizedTFlopsPercent string `json:"utilizedTFlopsPercent,omitempty"`
	UtilizedVRAMPercent   string `json:"utilizedVRAMPercent,omitempty"`

	// TODO: updated with interval
	AllocatedTFlopsPercent string `json:"allocatedTFlopsPercent,omitempty"`
	AllocatedVRAMPercent   string `json:"allocatedVRAMPercent,omitempty"`

	// TODO: aggregated with interval
	SavedCostsPerMonth       string `json:"savedCostsPerMonth,omitempty"`
	PotentialSavingsPerMonth string `json:"potentialSavingsPerMonth,omitempty"`

	// +kubebuilder:default=""
	// If the budget is exceeded, the set value in comma separated string to indicate which period caused the exceeding.
	// If this field is not empty, scheduler will not schedule new AI workloads and stop scaling-up check.
	// TODO not implemented yet
	BudgetExceeded string `json:"budgetExceeded,omitempty"`

	// +optional
	// +kubebuilder:default="None"
	ProvisioningPhase ProvisioningPhase `json:"provisioningPhase,omitempty"`

	// +optional
	LastCompactionTime *metav1.Time `json:"lastCompactionTime,omitempty"`
}

// +kubebuilder:validation:Enum=Pending;Running;Updating;Destroying;Unknown
type TensorFusionPoolPhase string

const (
	TensorFusionPoolPhasePending    = TensorFusionPoolPhase(constants.PhasePending)
	TensorFusionPoolPhaseRunning    = TensorFusionPoolPhase(constants.PhaseRunning)
	TensorFusionPoolPhaseUpdating   = TensorFusionPoolPhase(constants.PhaseUpdating)
	TensorFusionPoolPhaseUnknown    = TensorFusionPoolPhase(constants.PhaseUnknown)
	TensorFusionPoolPhaseDestroying = TensorFusionPoolPhase(constants.PhaseDestroying)
)

// +kubebuilder:validation:Enum=None;Initializing;Provisioning;Completed
type ProvisioningPhase string

const (
	// None means not in provisioning mode
	ProvisioningPhaseNone = ProvisioningPhase("None")

	// When NodeClaim created and pending GPUNodeClaim not empty, it's provisioning state,
	// check until all GPUNodeClaims are bound, unless next scale up should not happen
	ProvisioningPhaseProvisioning = ProvisioningPhase("Provisioning")

	// When all GPUNodeClaims are bound, set to Completed
	ProvisioningPhaseCompleted = ProvisioningPhase("Completed")
)

type PoolProvisioningStatus struct {
	InitializingNodes int32 `json:"initializingNodes,omitempty"`
	TerminatingNodes  int32 `json:"terminatingNodes,omitempty"`
	AvailableNodes    int32 `json:"availableNodes,omitempty"`
}

type PoolComponentStatus struct {
	WorkerVersion        string `json:"worker,omitempty"`
	WorkerConfigSynced   bool   `json:"workerConfigSynced,omitempty"`
	WorkerUpdateProgress int32  `json:"workerUpdateProgress,omitempty"`

	HypervisorVersion        string `json:"hypervisor,omitempty"`
	HypervisorConfigSynced   bool   `json:"hypervisorConfigSynced,omitempty"`
	HyperVisorUpdateProgress int32  `json:"hypervisorUpdateProgress,omitempty"`

	ClientVersion        string `json:"client,omitempty"`
	ClientConfigSynced   bool   `json:"clientConfigSynced,omitempty"`
	ClientUpdateProgress int32  `json:"clientUpdateProgress,omitempty"`
}

// GPUPool is the Schema for the gpupools API.
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster

// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="TFlops Oversubscription",type="string",JSONPath=".spec.capacityConfig.oversubscription.tflopsOversellRatio"
// +kubebuilder:printcolumn:name="Mode",type="string",JSONPath=".status.mode"
// +kubebuilder:printcolumn:name="Default Scheduling Strategy",type="string",JSONPath=".spec.schedulingConfigTemplate"
// +kubebuilder:printcolumn:name="Total Nodes",type="string",JSONPath=".status.totalNodes"
// +kubebuilder:printcolumn:name="Total GPU",type="string",JSONPath=".status.totalGPUs"
// +kubebuilder:printcolumn:name="Total Tflops",type="string",JSONPath=".status.totalTFlops"
// +kubebuilder:printcolumn:name="Total VRAM",type="string",JSONPath=".status.totalVRAM"
// +kubebuilder:printcolumn:name="Available Tflops",type="string",JSONPath=".status.availableTFlops"
// +kubebuilder:printcolumn:name="Available VRAM",type="string",JSONPath=".status.availableVRAM"
type GPUPool struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   GPUPoolSpec   `json:"spec,omitempty"`
	Status GPUPoolStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// GPUPoolList contains a list of GPUPool.
type GPUPoolList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []GPUPool `json:"items"`
}

func init() {
	SchemeBuilder.Register(&GPUPool{}, &GPUPoolList{})
}
