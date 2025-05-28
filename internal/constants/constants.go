package constants

import "time"

const (
	NvidiaGPUKey = "nvidia.com/gpu"
)
const (
	// Domain is the domain prefix used for all tensor-fusion.ai related annotations and finalizers
	Domain = "tensor-fusion.ai"

	// Finalizer constants
	FinalizerSuffix = "finalizer"
	Finalizer       = Domain + "/" + FinalizerSuffix

	LabelKeyOwner           = Domain + "/managed-by"
	LabelKeyUser            = Domain + "/used-by"
	LabelKeyClusterOwner    = Domain + "/cluster"
	LabelKeyNodeClass       = Domain + "/node-class"
	LabelKeyPodTemplateHash = Domain + "/pod-template-hash"
	TrueStringValue         = "true"

	GPUNodePoolIdentifierLabelPrefix = Domain + "/pool-"
	GPUNodePoolIdentifierLabelFormat = Domain + "/pool-%s"
	NodeDeletionMark                 = Domain + "/should-delete"

	TensorFusionEnabledLabelKey = Domain + "/enabled"
	InitialGPUNodeSelector      = "nvidia.com/gpu.present=true"

	GPULastReportTimeAnnotationKey = Domain + "/last-sync"
	WorkloadKey                    = Domain + "/workload"
	GpuKey                         = Domain + "/gpu"
	GpuPoolKey                     = Domain + "/gpupool"

	// Annotation key constants
	GpuCountKey                      = Domain + "/gpu-count"
	TFLOPSRequestAnnotation          = Domain + "/tflops-request"
	VRAMRequestAnnotation            = Domain + "/vram-request"
	TFLOPSLimitAnnotation            = Domain + "/tflops-limit"
	VRAMLimitAnnotation              = Domain + "/vram-limit"
	WorkloadProfileAnnotation        = Domain + "/client-profile"
	InjectContainerAnnotation        = Domain + "/inject-container"
	ReplicasAnnotation               = Domain + "/replicas"
	GenWorkloadAnnotation            = Domain + "/generate-workload"
	IsLocalGPUAnnotation             = Domain + "/is-local-gpu"
	NoStandaloneWorkerModeAnnotation = Domain + "/no-standalone-worker-mode"

	AutoScaleLimitsAnnotation   = Domain + "/auto-limits"
	AutoScaleRequestsAnnotation = Domain + "/auto-requests"
	AutoScaleReplicasAnnotation = Domain + "/auto-replicas"

	// GPUModelAnnotation specifies the required GPU model (e.g., "A100", "H100")
	GPUModelAnnotation = Domain + "/gpu-model"

	GpuReleasedAnnotation = Domain + "/gpu-released"

	TensorFusionPodCounterKeyAnnotation   = Domain + "/pod-counter-key"
	TensorFusionPodCountAnnotation        = Domain + "/tf-pod-count"
	TensorFusionEnabledReplicasAnnotation = Domain + "/enabled-replicas"

	PendingRequeueDuration = time.Second * 3
	StatusCheckInterval    = time.Second * 6

	GetConnectionURLEnv    = "TENSOR_FUSION_OPERATOR_GET_CONNECTION_URL"
	ConnectionNameEnv      = "TENSOR_FUSION_CONNECTION_NAME"
	ConnectionNamespaceEnv = "TENSOR_FUSION_CONNECTION_NAMESPACE"

	WorkerPortEnv              = "TENSOR_FUSION_WORKER_PORT"
	WorkerCudaUpLimitTflopsEnv = "TENSOR_FUSION_CUDA_UP_LIMIT_TFLOPS"
	WorkerCudaUpLimitEnv       = "TENSOR_FUSION_CUDA_UP_LIMIT"
	WorkerCudaMemLimitEnv      = "TENSOR_FUSION_CUDA_MEM_LIMIT"
	WorkerPodNameEnv           = "POD_NAME"
	NamespaceEnv               = "OPERATOR_NAMESPACE"
	NamespaceDefaultVal        = "tensor-fusion-sys"
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
	SchedulingDoNotDisruptLabel = Domain + "/do-not-disrupt"
)

const (
	GPUNodeOSLinux   = "linux"
	GPUNodeOSWindows = "windows"
	GPUNodeOSMacOS   = "macos"
)

// To match GPUNode with K8S node, when creating from cloud vendor, must set a label from cloud-init userdata
const (
	ProvisionerLabelKey        = Domain + "/node-provisioner"
	ProvisionerNamePlaceholder = "__GPU_NODE_RESOURCE_NAME__"
)
const (
	NodeDiscoveryReportGPUNodeEnvName = "NODE_DISCOVERY_REPORT_GPU_NODE"
)

const TFDataPath = "/tmp/tensor-fusion/data"
const DataVolumeName = "tf-data"
const TensorFusionPoolManualCompaction = Domain + "/manual-compaction"
