package filter

import (
	"encoding/json"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

type gpuVirtualizationCapabilities struct {
	SupportsPartitioning  bool   `json:"supportsPartitioning"`
	SupportsSoftIsolation bool   `json:"supportsSoftIsolation"`
	SupportsHardIsolation bool   `json:"supportsHardIsolation"`
	SupportsSnapshot      bool   `json:"supportsSnapshot"`
	SupportsMetrics       bool   `json:"supportsMetrics"`
	SupportsRemoting      bool   `json:"supportsRemoting"`
	MaxPartitions         uint32 `json:"maxPartitions"`
	MaxWorkersPerDevice   uint32 `json:"maxWorkersPerDevice"`

	metadataPresent bool `json:"-"`
}

func extractVirtualizationCapabilities(gpu *tfv1.GPU) gpuVirtualizationCapabilities {
	if gpu == nil || len(gpu.Annotations) == 0 {
		return gpuVirtualizationCapabilities{}
	}
	raw := gpu.Annotations[constants.GPUVirtualizationCapabilitiesAnnotation]
	if raw == "" {
		return gpuVirtualizationCapabilities{}
	}
	caps := gpuVirtualizationCapabilities{}
	if err := json.Unmarshal([]byte(raw), &caps); err != nil {
		return gpuVirtualizationCapabilities{}
	}
	caps.metadataPresent = true
	return caps
}

func hasVirtualizationCapabilityMetadata(caps gpuVirtualizationCapabilities) bool {
	return caps.metadataPresent
}
