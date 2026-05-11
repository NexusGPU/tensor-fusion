package filter

import (
	"encoding/json"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

type gpuVirtualizationCapabilities struct {
	SupportsPartitioning  bool   `json:"SupportsPartitioning"`
	SupportsSoftIsolation bool   `json:"SupportsSoftIsolation"`
	SupportsHardIsolation bool   `json:"SupportsHardIsolation"`
	SupportsSnapshot      bool   `json:"SupportsSnapshot"`
	SupportsMetrics       bool   `json:"SupportsMetrics"`
	SupportsRemoting      bool   `json:"SupportsRemoting"`
	MaxPartitions         uint32 `json:"MaxPartitions"`
	MaxWorkersPerDevice   uint32 `json:"MaxWorkersPerDevice"`

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
