package constants

// Controller itself envs
const NamespaceEnv = "OPERATOR_NAMESPACE"

// System feature toggles
const (
	EnableWebhookEnv                  = "ENABLE_WEBHOOKS"
	EnableSchedulerEnv                = "ENABLE_SCHEDULER"
	EnableCustomResourceControllerEnv = "ENABLE_CR_CONTROLLER"

	// TensorFusion ControllerManager's http endpoint will verify Pod JWT signature
	// if this env var is set, will disable the verification, it's enabled by default
	// should not set to true in production environment
	DisableConnectionAuthEnv = "DISABLE_CONNECTION_AUTH"

	NvidiaOperatorProgressiveMigrationEnv = "NVIDIA_OPERATOR_PROGRESSIVE_MIGRATION"
)

// General envs used in compose components manifest
const (
	NvidiaVisibleAllDeviceEnv   = "NVIDIA_VISIBLE_DEVICES"
	NvidiaVisibleAllDeviceValue = "all"

	TensorFusionGPUInfoConfigName       = "tensor-fusion-sys-public-gpu-info"
	TensorFusionGPUInfoConfigVolumeName = "gpu-info"
	TensorFusionGPUInfoConfigMountPath  = "/etc/tensor-fusion/gpu-info.yaml"
	TensorFusionGPUInfoConfigSubPath    = "gpu-info.yaml"
	TensorFusionGPUInfoEnvVar           = "TENSOR_FUSION_GPU_INFO_PATH"

	KubeletDevicePluginVolumeName = "device-plugin"
	KubeletDevicePluginPath       = "/var/lib/kubelet/device-plugins"

	KubeletPodResourcesVolumeName = "pod-resources"
	KubeletPodResourcesPath       = "/var/lib/kubelet/pod-resources"

	TensorFusionVectorConfigName       = "tensor-fusion-sys-vector-config"
	TensorFusionVectorConfigVolumeName = "vector-config"
	TensorFusionVectorConfigMountPath  = "/etc/vector/vector.yaml"
	TensorFusionVectorConfigSubPath    = "vector-hypervisor.yaml"

	LogsVolumeName           = "logs"
	KubernetesLogsVolumeName = "kubernetes-logs"
	KubernetesLogsPath       = "/var/log/pods"
	TensorFusionLogPath      = "/logs"

	DefaultHttpBindIP = "0.0.0.0"
)

const (
	TFContainerNameClient        = "inject-lib"
	TFContainerNameWorker        = "tensorfusion-worker"
	TFContainerNameHypervisor    = "tensorfusion-hypervisor"
	TFContainerNameNodeDiscovery = "tensorfusion-node-discovery"
	TFContainerVector            = "vector"
)

// TensorFusion client related envs
const (
	GetConnectionURLEnv    = "TENSOR_FUSION_OPERATOR_GET_CONNECTION_URL"
	ConnectionNameEnv      = "TENSOR_FUSION_CONNECTION_NAME"
	ConnectionNamespaceEnv = "TENSOR_FUSION_CONNECTION_NAMESPACE"

	RealNvmlLibPathEnv   = "TF_NVML_LIB_PATH"
	RealCUDALibPathEnv   = "TF_CUDA_LIB_PATH"
	RealNvmlLibPathValue = "/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
	RealCUDALibPathValue = "/lib/x86_64-linux-gnu/libcuda.so"

	PrependPathEnv          = "TF_PREPEND_PATH"
	PrependLDLibraryPathEnv = "TF_PREPEND_LD_LIBRARY_PATH"

	LdPreloadFileName = "ld.so.preload"
	LdPreloadFile     = "/etc/ld.so.preload"

	TFLibsVolumeName       = "tf-libs"
	TFLibsVolumeMountPath  = "/tensor-fusion"
	TFConnectionNamePrefix = "tf-vgpu-"

	HostIPFieldRef       = "status.hostIP"
	NodeNameFieldRef     = "spec.nodeName"
	ResourceNameFieldRef = "metadata.name"
	NamespaceFieldRef    = "metadata.namespace"
)

// TensorFusion worker related envs
const (
	HypervisorIPEnv   = "HYPERVISOR_IP"
	HypervisorPortEnv = "HYPERVISOR_PORT"

	PodNamespaceEnv  = "POD_NAMESPACE"
	ContainerNameEnv = "CONTAINER_NAME"

	// the path of nGPU lib for limiter to load
	NGPUPathEnv   = "TENSOR_FUSION_NGPU_PATH"
	NGPUPathValue = TFLibsVolumeMountPath + "/libcuda.so"

	LdPreloadEnv     = "LD_PRELOAD"
	LdPreloadLimiter = "/home/app/libcuda_limiter.so"

	SharedMemDeviceName   = "/dev/shm"
	SharedMemMountSubPath = "shm"

	// disable GPU limiter, for emergency use
	DisableGpuLimiterEnv = "DISABLE_GPU_LIMITER"
	// directly forward CUDA calls to GPU driver in nGPU mode, for emergency use
	DisableCudaOptimizationEnv = "TF_ENABLE_DISPATCH_FORWARD"
	// disable vram manager, for emergency use
	DisableVRAMManagerEnv      = "TF_DISABLE_MEMORY_MANAGER"
	DisableWorkerFeatureEnvVal = "1"

	TensorFusionRemoteWorkerPortNumber = 8000
	TensorFusionRemoteWorkerPortName   = "remote-vgpu"
)

// TensorFusion hypervisor related envs
const (
	HypervisorPoolNameEnv           = "TENSOR_FUSION_POOL_NAME"
	PodNameEnv                      = "POD_NAME"
	VectorPodNodeNameEnv            = "NODE_NAME"
	HypervisorGPUNodeNameEnv        = "GPU_NODE_NAME"
	HypervisorSchedulingConfigEnv   = "TF_HYPERVISOR_SCHEDULING_CONFIG"
	HypervisorListenAddrEnv         = "API_LISTEN_ADDR"
	HypervisorMetricsFormatEnv      = "TF_HYPERVISOR_METRICS_FORMAT"
	HypervisorMetricsExtraLabelsEnv = "TF_HYPERVISOR_METRICS_EXTRA_LABELS"
	HypervisorDetectUsedGPUEnv      = "DETECT_IN_USED_GPUS"

	// Add ptrace capability to hypervisor container, to trace all host PID using GPU
	SystemPtraceCapability = "SYS_PTRACE"

	HypervisorDefaultPortNumber int32  = 8000
	HypervisorPortName          string = "http"

	// For security enhancement, there are 2 types of endpoints to protect
	// 1. client call operator /connection API, to obtain tensor fusion worker's URL
	// 2. worker call hypervisor API, to obtain current workers GPU quota info
	// if this env var is set on operator and hypervisor, will try to verify JWT signature for each call
	// not implemented yet, iss is public in EKS and most K8S distribution
	// but k3s and some K8S distribution may not support, need to find some way to get SA token JWT pub key
	HypervisorVerifyServiceAccountEnabledEnvVar   = "SA_TOKEN_VERIFY_ENABLED"
	HypervisorVerifyServiceAccountPublicKeyEnvVar = "SA_TOKEN_VERIFY_PUBLIC_KEY"
)

// Node discovery related envs
const (
	NodeDiscoveryReportGPUNodeEnvName = "NODE_DISCOVERY_REPORT_GPU_NODE"
	NodeDiscoveryHostNameEnv          = "HOSTNAME"
)

const (
	KubeApiVersionMajorEnv = "KUBE_API_VERSION_MAJOR"
	KubeApiVersionMinorEnv = "KUBE_API_VERSION_MINOR"
)
