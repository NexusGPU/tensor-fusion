package external_dp

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

const (
	resourceNvidiaGPU = "nvidia.com/gpu"
)

// NvidiaDevicePluginDetector handles NVIDIA-specific device plugin detection
type NvidiaDevicePluginDetector struct{}

// NewNvidiaDevicePluginDetector creates a new NVIDIA device plugin detector
func NewNvidiaDevicePluginDetector() *NvidiaDevicePluginDetector {
	return &NvidiaDevicePluginDetector{}
}

// GetResourceName returns the resource name this detector handles
func (n *NvidiaDevicePluginDetector) GetResourceName() string {
	return resourceNvidiaGPU
}

// GetUsedBySystem returns the UsedBy system name for NVIDIA
func (n *NvidiaDevicePluginDetector) GetUsedBySystem() string {
	return string(tfv1.UsedByNvidiaDevicePlugin)
}
