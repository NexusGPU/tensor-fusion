package device

// PartitionResultHandler encapsulates how each partition result type contributes
// device nodes in SplitDevice flow.
type PartitionResultHandler interface {
	BaseDeviceNodes(partitionResult *AssignPartitionResult) map[string]string
	UseProviderNodesFallback() bool
}

type envPartitionResultHandler struct{}

func (envPartitionResultHandler) BaseDeviceNodes(_ *AssignPartitionResult) map[string]string {
	// Env-managed partitions should not expose compute nodes unless policy adds them.
	return nil
}

func (envPartitionResultHandler) UseProviderNodesFallback() bool {
	return false
}

type deviceNodePartitionResultHandler struct{}

func (deviceNodePartitionResultHandler) BaseDeviceNodes(partitionResult *AssignPartitionResult) map[string]string {
	if partitionResult == nil {
		return nil
	}
	return partitionResult.DeviceNodes
}

func (deviceNodePartitionResultHandler) UseProviderNodesFallback() bool {
	return true
}

var (
	envPartitionHandler        = envPartitionResultHandler{}
	deviceNodePartitionHandler = deviceNodePartitionResultHandler{}
)

func resolvePartitionResultHandler(resultType PartitionResultType) (PartitionResultHandler, bool) {
	switch resultType {
	case PartitionTypeEnvironmentVariable:
		return envPartitionHandler, true
	case PartitionTypeDeviceNode:
		return deviceNodePartitionHandler, true
	default:
		return nil, false
	}
}
