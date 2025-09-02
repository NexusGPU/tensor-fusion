package cel_filter

// CEL variable names available in expressions
const (
	// Root variables
	CELVarGPU          = "gpu"
	CELVarWorkerPodKey = "workerPodKey"
	CELVarRequest      = "request"
)

// GPU object field names
const (
	// Basic GPU metadata
	GPUFieldName      = "name"
	GPUFieldNamespace = "namespace"
	GPUFieldGPUModel  = "gpuModel"
	GPUFieldUUID      = "uuid"
	GPUFieldPhase     = "phase"
	GPUFieldUsedBy    = "usedBy"
	GPUFieldMessage   = "message"

	// Kubernetes metadata
	GPUFieldLabels      = "labels"
	GPUFieldAnnotations = "annotations"

	// Resource information
	GPUFieldAvailable    = "available"
	GPUFieldNodeSelector = "nodeSelector"
	GPUFieldRunningApps  = "runningApps"

	// Resource sub-fields
	ResourceFieldTFlops = "tflops"
	ResourceFieldVRAM   = "vram"

	// Running app sub-fields
	AppFieldName      = "name"
	AppFieldNamespace = "namespace"
	AppFieldCount     = "count"

	// WorkerPodKey fields
	PodKeyFieldName      = "name"
	PodKeyFieldNamespace = "namespace"
)

// Request object field names
const (
	RequestFieldWorkerPodKey = "workerPodKey"
	RequestFieldCount        = "count"
	RequestFieldGPUModel     = "gpuModel"
	RequestFieldRequest      = "request"
	RequestFieldLimit        = "limit"
)
