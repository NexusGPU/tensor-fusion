package external_dp

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

const (
	resourceNvidiaGPU  = "nvidia.com/gpu"
	resourceNvidiaMIG  = "nvidia.com/mig"
	realDeviceIDLength = 40
)

var UsedByNvidiaDevicePlugin = tfv1.UsedBySystem("nvidia-device-plugin")
var UsedBy3rdPartyDevicePlugin = tfv1.UsedBySystem("3rd-party-device-plugin")

// NvidiaDevicePluginDetector handles NVIDIA-specific device plugin detection
type NvidiaDevicePluginDetector struct{}

// NewNvidiaDevicePluginDetector creates a new NVIDIA device plugin detector
func NewNvidiaDevicePluginDetector() *NvidiaDevicePluginDetector {
	return &NvidiaDevicePluginDetector{}
}

// GetResourceName returns the resource name this detector handles
func (n *NvidiaDevicePluginDetector) GetResourceNamePrefixes() []string {
	return []string{resourceNvidiaGPU, resourceNvidiaMIG}
}

// GetUsedBySystem returns the UsedBy system name for NVIDIA
func (n *NvidiaDevicePluginDetector) GetUsedBySystemAndRealDeviceID(deviceID, resourceName string) (system string, realDeviceID string) {
	if resourceName == resourceNvidiaGPU {
		// Some external device plugin's device ID is GPU-(UUID)-0, 1, 2, 3 (e.g. HAMI)
		// Need to recover to real device ID
		if len(deviceID) > realDeviceIDLength {
			return string(UsedBy3rdPartyDevicePlugin), deviceID[:realDeviceIDLength]
		} else {
			return string(UsedByNvidiaDevicePlugin), deviceID
		}
	} else {
		return string(UsedByNvidiaDevicePlugin), deviceID
	}
}
