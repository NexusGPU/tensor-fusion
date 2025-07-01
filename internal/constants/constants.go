package constants

import "time"

const (
	NvidiaGPUKey = "nvidia.com/gpu"
)

var (
	PendingRequeueDuration = time.Second * 3
	StatusCheckInterval    = time.Second * 6
)

const (
	// Domain is the domain prefix used for all tensor-fusion.ai related annotations and finalizers
	Domain = "tensor-fusion.ai"

	// Finalizer constants
	FinalizerSuffix = "finalizer"
	Finalizer       = Domain + "/" + FinalizerSuffix

	SchedulerName = "tensor-fusion-scheduler"

	LabelKeyOwner           = Domain + "/managed-by"
	LabelKeyClusterOwner    = Domain + "/cluster"
	LabelKeyNodeClass       = Domain + "/node-class"
	LabelKeyPodTemplateHash = Domain + "/pod-template-hash"
	LabelComponent          = Domain + "/component"
	// used by TF connection, for matching the related connections when worker Pod state changed
	LabelWorkerName  = Domain + "/worker-name"
	TrueStringValue  = "true"
	FalseStringValue = "false"

	ComponentClient        = "client"
	ComponentWorker        = "worker"
	ComponentHypervisor    = "hypervisor"
	ComponentNodeDiscovery = "node-discovery"
	ComponentOperator      = "operator"

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
	GpuCountAnnotation             = Domain + "/gpu-count"
	TFLOPSRequestAnnotation        = Domain + "/tflops-request"
	VRAMRequestAnnotation          = Domain + "/vram-request"
	TFLOPSLimitAnnotation          = Domain + "/tflops-limit"
	VRAMLimitAnnotation            = Domain + "/vram-limit"
	WorkloadProfileAnnotation      = Domain + "/client-profile"
	InjectContainerAnnotation      = Domain + "/inject-container"
	IsLocalGPUAnnotation           = Domain + "/is-local-gpu"
	EmbeddedWorkerAnnotation       = Domain + "/embedded-worker"
	DedicatedWorkerAnnotation      = Domain + "/dedicated-worker"
	StandaloneWorkerModeAnnotation = Domain + "/no-standalone-worker-mode"
	// GPUModelAnnotation specifies the required GPU model (e.g., "A100", "H100")
	GPUModelAnnotation = Domain + "/gpu-model"
	// GPU ID list is assigned by scheduler, should not specified by user
	GPUDeviceIDsAnnotation            = Domain + "/gpu-ids"
	SetPendingOwnedWorkloadAnnotation = Domain + "/pending-owned-workload"

	GenHostPortLabel             = Domain + "/host-port"
	GenHostPortLabelValue        = "auto"
	GenHostPortNameLabel         = Domain + "/port-name"
	GenPortNumberAnnotation      = Domain + "/port-number"
	TensorFusionWorkerPortNumber = 8000

	AutoScaleLimitsAnnotation   = Domain + "/auto-limits"
	AutoScaleRequestsAnnotation = Domain + "/auto-requests"
	AutoScaleReplicasAnnotation = Domain + "/auto-replicas"

	GpuReleasedAnnotation = Domain + "/gpu-released"

	TensorFusionPodCounterKeyAnnotation = Domain + "/pod-counter-key"
	TensorFusionPodCountAnnotation      = Domain + "/tf-pod-count"
	TensorFusionWorkerSuffix            = "-tf"

	// For grey release
	TensorFusionEnabledReplicasAnnotation = Domain + "/enabled-replicas"
	TensorFusionDefaultPoolKeyAnnotation  = Domain + "/is-default-pool"

	GetConnectionURLEnv    = "TENSOR_FUSION_OPERATOR_GET_CONNECTION_URL"
	ConnectionNameEnv      = "TENSOR_FUSION_CONNECTION_NAME"
	ConnectionNamespaceEnv = "TENSOR_FUSION_CONNECTION_NAMESPACE"

	WorkerCudaUpLimitTflopsEnv = "TENSOR_FUSION_CUDA_UP_LIMIT_TFLOPS"
	WorkerCudaUpLimitEnv       = "TENSOR_FUSION_CUDA_UP_LIMIT"
	WorkerCudaMemLimitEnv      = "TENSOR_FUSION_CUDA_MEM_LIMIT"
	WorkloadNameEnv            = "TENSOR_FUSION_WORKLOAD_NAME"
	PoolNameEnv                = "TENSOR_FUSION_POOL_NAME"
	PodNameEnv                 = "POD_NAME"
	GPUNodeNameEnv             = "GPU_NODE_NAME"
	NamespaceEnv               = "OPERATOR_NAMESPACE"
	NamespaceDefaultVal        = "tensor-fusion-sys"

	KubernetesHostNameLabel      = "kubernetes.io/hostname"
	GiBToBytes                   = 1024 * 1024 * 1024
	HypervisorServiceAccountName = "tensor-fusion-hypervisor-sa"

	TSDBVersionConfigMap = "tensor-fusion-tsdb-version"

	QoSLevelLow      = "low"
	QoSLevelMedium   = "medium"
	QoSLevelHigh     = "high"
	QoSLevelCritical = "critical"

	EnableWebhookEnv   = "ENABLE_WEBHOOKS"
	EnableSchedulerEnv = "ENABLE_SCHEDULER"
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
const AlertJobName = "tensor-fusion"
const HypervisorSchedulingConfigEnv = "TF_HYPERVISOR_SCHEDULING_CONFIG"

const (
	LeaderInfoConfigMapName        = "tensor-fusion-operator-leader-info"
	LeaderInfoConfigMapLeaderIPKey = "leader-ip"
)

const ShortUUIDAlphabet = "123456789abcdefghijkmnopqrstuvwxy"
const NvidiaVisibleAllDeviceEnv = "NVIDIA_VISIBLE_DEVICES"
const NvidiaVisibleAllDeviceValue = "all"

const (
	LowFrequencyObjFailureInitialDelay        = 100 * time.Millisecond
	LowFrequencyObjFailureMaxDelay            = 1000 * time.Second
	LowFrequencyObjFailureMaxRPS              = 1
	LowFrequencyObjFailureMaxBurst            = 1
	LowFrequencyObjFailureConcurrentReconcile = 5
)
