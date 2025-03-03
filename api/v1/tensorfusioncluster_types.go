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
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// TensorFusionClusterSpec defines the desired state of TensorFusionCluster.
type TensorFusionClusterSpec struct {
	GPUPools []GPUPoolDefinition `json:"gpuPools,omitempty"`

	// +optional
	ComputingVendor *ComputingVendorConfig `json:"computingVendor,omitempty"`

	// +optional
	StorageVendor *StorageVendorConfig `json:"storageVendor,omitempty"`

	// +optional
	DataPipelines *DataPipelinesConfig `json:"dataPipelines,omitempty"`
}

// TensorFusionClusterStatus defines the observed state of TensorFusionCluster.
type TensorFusionClusterStatus struct {

	// +kubebuilder:default=Pending
	Phase TensorFusionClusterPhase `json:"phase,omitempty"`

	Conditions []metav1.Condition `json:"conditions,omitempty"`

	TotalPools int32 `json:"totalPools"`
	TotalNodes int32 `json:"totalNodes"`
	TotalGPUs  int32 `json:"totalGPUs"`

	TotalTFlops resource.Quantity `json:"totalTFlops"`
	TotalVRAM   resource.Quantity `json:"totalVRAM"`

	VirtualTFlops resource.Quantity `json:"virtualTFlops"`
	VirtualVRAM   resource.Quantity `json:"virtualVRAM"`

	AvailableTFlops resource.Quantity `json:"availableTFlops"`
	AvailableVRAM   resource.Quantity `json:"availableVRAM"`

	// +optional
	VirtualAvailableTFlops *resource.Quantity `json:"virtualAvailableTFlops,omitempty"`
	// +optional
	VirtualAvailableVRAM *resource.Quantity `json:"virtualAvailableVRAM,omitempty"`

	// +optional
	ReadyGPUPools []string `json:"readyGPUPools"`

	// +optional
	NotReadyGPUPools []string `json:"notReadyGPUPools"`

	// +kubebuilder:default=0
	//
	RetryCount int64 `json:"retryCount"`

	// calculated every 5m average
	UtilizedTFlopsPercent string `json:"utilizedTFlopsPercent,omitempty"`
	UtilizedVRAMPercent   string `json:"utilizedVRAMPercent,omitempty"`

	// updated with interval
	AllocatedTFlopsPercent string `json:"allocatedTFlopsPercent,omitempty"`
	AllocatedVRAMPercent   string `json:"allocatedVRAMPercent,omitempty"`

	// aggregated with interval
	SavedCostsPerMonth       string `json:"savedCostsPerMonth,omitempty"`
	PotentialSavingsPerMonth string `json:"potentialSavingsPerMonth,omitempty"`

	CloudVendorConfigHash string `json:"cloudVendorConfigHash,omitempty"`
}

// +kubebuilder:validation:Enum=Pending;Running;Updating;Destroying;Unknown
// TensorFusionClusterPhase represents the phase of the TensorFusionCluster resource.
type TensorFusionClusterPhase string

const (
	TensorFusionClusterPending    = TensorFusionClusterPhase(constants.PhasePending)
	TensorFusionClusterRunning    = TensorFusionClusterPhase(constants.PhaseRunning)
	TensorFusionClusterUpdating   = TensorFusionClusterPhase(constants.PhaseUpdating)
	TensorFusionClusterDestroying = TensorFusionClusterPhase(constants.PhaseDestroying)
	TensorFusionClusterUnknown    = TensorFusionClusterPhase(constants.PhaseUnknown)
)

// GPUPool defines how to create a GPU pool, could be URL or inline
type GPUPoolDefinition struct {
	Name string `json:"name,omitempty"` // Name of the GPU pool.

	SpecTemplate GPUPoolSpec `json:"specTemplate"`
}

// ComputingVendorConfig defines the Cloud vendor connection such as AWS, GCP, Azure etc.
type ComputingVendorConfig struct {
	Name string `json:"name,omitempty"`

	// support popular cloud providers
	Type ComputingVendorName `json:"type,omitempty"`

	AuthType AuthTypeEnum `json:"authType,omitempty"` // Authentication type (e.g., accessKey, serviceAccount).

	// +optional
	// +kubebuilder:default=true
	Enable *bool `json:"enable,omitempty"` // Enable or disable the computing vendor.

	Params ComputingVendorParams `json:"params,omitempty"`
}

// +kubebuilder:validation:Enum=accessKey;serviceAccountRole
type AuthTypeEnum string

const (
	AuthTypeAccessKey          AuthTypeEnum = "accessKey"
	AuthTypeServiceAccountRole AuthTypeEnum = "serviceAccountRole"
)

// +kubebuilder:validation:Enum=aws;lambda-labs;gcp;azure;oracle-oci;ibm;openshift;vultr;together-ai;alibaba;nvidia;tencent;runpod;mock
type ComputingVendorName string

const (
	ComputingVendorAWS        ComputingVendorName = "aws"
	ComputingVendorGCP        ComputingVendorName = "gcp"
	ComputingVendorAzure      ComputingVendorName = "azure"
	ComputingVendorOracle     ComputingVendorName = "oracle-oci"
	ComputingVendorIBM        ComputingVendorName = "ibm"
	ComputingVendorOpenShift  ComputingVendorName = "openshift"
	ComputingVendorVultr      ComputingVendorName = "vultr"
	ComputingVendorTogetherAI ComputingVendorName = "together-ai"
	ComputingVendorLambdaLabs ComputingVendorName = "lambda-labs"
	ComputingVendorAlibaba    ComputingVendorName = "alibaba"
	ComputingVendorNvidia     ComputingVendorName = "nvidia"
	ComputingVendorTencent    ComputingVendorName = "tencent"
	ComputingVendorRunPod     ComputingVendorName = "runpod"

	// This is not unit/integration testing only, no cloud provider is involved
	ComputingVendorMock ComputingVendorName = "mock"
)

type ComputingVendorParams struct {
	// +optional
	DefaultRegion string `json:"defaultRegion,omitempty"` // Region for the computing vendor.

	// the secret of access key and secret key or config file, must be mounted as file path
	// +optional
	AccessKeyPath string `json:"accessKeyPath,omitempty"`
	// +optional
	SecretKeyPath string `json:"secretKeyPath,omitempty"`

	// preferred IAM role since it's more secure
	// +optional
	IAMRole string `json:"iamRole,omitempty"`

	// +optional
	ConfigFile string `json:"configFile,omitempty"`

	// +optional
	// User can set extra cloud vendor params, eg.
	// in ali cloud:" spotPriceLimit, spotDuration, spotInterruptionBehavior, systemDiskCategory, systemDiskSize, dataDiskPerformanceLevel
	// in aws cloud: TODO
	ExtraParams map[string]string `json:"extraParams,omitempty"`
}

// StorageVendorConfig defines Postgres database with extensions for timeseries storage and other resource aggregation results, system events and diagnostics reports etc.
type StorageVendorConfig struct {
	Mode  string `json:"mode,omitempty"`  // Mode of the storage vendor (e.g., cloudnative-pg, timescale-db, RDS for PG).
	Image string `json:"image,omitempty"` // Image for the storage vendor (default to timescale).

	// +optional
	InstallCloudNativePGOperator *bool `json:"installCloudNativePGOperator,omitempty"` // Whether to install CloudNative-PG operator.

	StorageClass      string               `json:"storageClass,omitempty"`      // Storage class for the storage vendor.
	PGExtensions      []string             `json:"pgExtensions,omitempty"`      // List of PostgreSQL extensions to install.
	PGClusterTemplate runtime.RawExtension `json:"pgClusterTemplate,omitempty"` // Extra spec for the PostgreSQL cluster template.
}

// DataPipelinesConfig defines the aggregation jobs that can make statistics on the data and then report to cloud if configured.
type DataPipelinesConfig struct {
	Resources DataPipeline4ResourcesConfig `json:"resources,omitempty"`

	Timeseries DataPipeline4TimeSeriesConfig `json:"timeseries,omitempty"`
}

type DataPipeline4ResourcesConfig struct {
	// +optional
	SyncToCloud *bool `json:"syncToCloud,omitempty"` // Whether to sync resources to the cloud.

	// +optional human readable time like 1h, 1d, default to 1h
	SyncPeriod string `json:"syncPeriod,omitempty"` // Period for syncing resources.
}

type DataPipeline4TimeSeriesConfig struct {
	AggregationPeriods       []string          `json:"aggregationPeriods,omitempty"`       // List of aggregation periods.
	RawDataRetention         string            `json:"rawDataRetention,omitempty"`         // Retention period for raw data.
	AggregationDataRetention string            `json:"aggregationDataRetention,omitempty"` // Retention period for aggregated data.
	RemoteWrite              RemoteWriteConfig `json:"remoteWrite,omitempty"`              // Configuration for remote write.
}

// RemoteWriteConfig represents the configuration for remote write.
type RemoteWriteConfig struct {
	Connection DataPipelineResultRemoteWriteConfig `json:"connection,omitempty"`
	Metrics    []string                            `json:"metrics,omitempty"` // List of metrics to remote write.
}

type DataPipelineResultRemoteWriteConfig struct {
	Type string `json:"type,omitempty"` // Type of the connection (e.g., datadog).
	URL  string `json:"url,omitempty"`  // URL of the connection.
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Total Pools",type="string",JSONPath=".status.totalPools"
// +kubebuilder:printcolumn:name="Total Nodes",type="string",JSONPath=".status.totalNodes"
// +kubebuilder:printcolumn:name="Total GPU",type="string",JSONPath=".status.totalGPUs"

// +kubebuilder:printcolumn:name="Total Tflops",type="string",JSONPath=".status.totalTFlops"
// +kubebuilder:printcolumn:name="Total VRAM",type="string",JSONPath=".status.totalVRAM"
// +kubebuilder:printcolumn:name="Available Tflops",type="string",JSONPath=".status.availableTFlops"
// +kubebuilder:printcolumn:name="Available VRAM",type="string",JSONPath=".status.availableVRAM"
// TensorFusionCluster is the Schema for the tensorfusionclusters API.
type TensorFusionCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TensorFusionClusterSpec   `json:"spec,omitempty"`
	Status TensorFusionClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// TensorFusionClusterList contains a list of TensorFusionCluster.
type TensorFusionClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TensorFusionCluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&TensorFusionCluster{}, &TensorFusionClusterList{})
}
