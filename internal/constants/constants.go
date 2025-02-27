package constants

import "time"

const (
	// Domain is the domain prefix used for all tensor-fusion.ai related annotations and finalizers
	Domain = "tensor-fusion.ai"

	// Finalizer constants
	FinalizerSuffix = "finalizer"
	Finalizer       = Domain + "/" + FinalizerSuffix

	LabelKeyOwner        = Domain + "/managed-by"
	LabelKeyClusterOwner = Domain + "/cluster"
	LabelKeyNodeClass    = Domain + "/node-class"

	GPUNodePoolIdentifierLabelPrefix = Domain + "/pool-"
	GPUNodePoolIdentifierLabelFormat = Domain + "/pool-%s"
	NodeDeletionMark                 = Domain + "/should-delete"

	TensorFusionEnabledLabelKey = Domain + "/enabled"
	InitialGPUNodeSelector      = "nvidia.com/gpu.present=true"

	GPULastReportTimeAnnotationKey = Domain + "/last-sync"

	// Annotation key constants
	GpuPoolAnnotationKey = Domain + "/gpupool"
	// %s -> container_name
	TFLOPSRequestAnnotationFormat = Domain + "/tflops-request-%s"
	VRAMRequestAnnotationFormat   = Domain + "/vram-request-%s"
	TFLOPSLimitAnnotationFormat   = Domain + "/tflops-limit-%s"
	VRAMLimitAnnotationFormat     = Domain + "/vram-limit-%s"

	PendingRequeueDuration = time.Second * 3
	StatusCheckInterval    = time.Second * 6

	GetConnectionURLEnv    = "TENSOR_FUSION_OPERATOR_GET_CONNECTION_URL"
	ConnectionNameEnv      = "TENSOR_FUSION_CONNECTION_NAME"
	ConnectionNamespaceEnv = "TENSOR_FUSION_CONNECTION_NAMESPACE"

	WorkerPortEnv       = "TENSOR_FUSION_WORKER_PORT"
	NamespaceEnv        = "OPERATOR_NAMESPACE"
	NamespaceDefaultVal = "tensor-fusion"
)

const (
	ConditionStatusTypeReady           = "Ready"
	ConditionStatusTypeGPUScheduled    = "GPUScheduled"
	ConditionStatusTypeConnectionReady = "ConnectionReady"
	ConditionStatusTypeNodeProvisioned = "NodeProvisioned"
	ConditionStatusTypePoolReady       = "PoolReady"

	ConditionStatusTypeGPUPool               = "GPUPoolReady"
	ConditionStatusTypeTimeSeriesDatabase    = "TimeSeriesDatabaseReady"
	ConditionStatusTypeCloudVendorConnection = "CloudVendorConnectionReady"
)

const (
	PhaseUnknown    = "Unknown"
	PhasePending    = "Pending"
	PhaseUpdating   = "Updating"
	PhaseScheduling = "Scheduling"
	PhaseMigrating  = "Migrating"
	PhaseDestroying = "Destroying"

	PhaseRunning   = "Running"
	PhaseSucceeded = "Succeeded"
	PhaseFailed    = "Failed"
)

const (
	// No disrupt label, similar to Karpenter, avoid TFConnection/Worker/GPUNode to be moved to another node or destroying node.
	// Refer: https://karpenter.sh/docs/concepts/disruption/
	SchedulingDoNotDisruptLabel = "tensor-fusion.ai/do-not-disrupt"
)

const (
	GPUNodeOSLinux   = "linux"
	GPUNodeOSWindows = "windows"
	GPUNodeOSMacOS   = "macos"
)

// To match GPUNode with K8S node, when creating from cloud vendor, must set a label from cloud-init userdata
const (
	ProvisionerLabelKey        = "tensor-fusion.ai/node-provisioner"
	ProvisionerNamePlaceholder = "__GPU_NODE_RESOURCE_NAME__"
)
const (
	NodeDiscoveryReportGPUNodeEnvName = "NODE_DISCOVERY_REPORT_GPU_NODE"
)

const TFDataPath = "/tmp/tensor-fusion/data"
const DataVolumeName = "tf-data"
