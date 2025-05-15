package filter

import (
	"context"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

// SameNodeFilter ensures that the selected GPUs are from the same node
type SameNodeFilter struct {
	count uint // Number of GPUs required
}

// NewSameNodeFilter creates a new SameNodeFilter with the specified count
func NewSameNodeFilter(count uint) *SameNodeFilter {
	return &SameNodeFilter{
		count: count,
	}
}

// Filter implements GPUFilter.Filter
// It groups GPUs by node and returns only those nodes that have at least 'count' GPUs
func (f *SameNodeFilter) Filter(_ context.Context, gpus []tfv1.GPU) ([]tfv1.GPU, error) {
	// If count is 1 or 0, no need to filter by node
	if f.count <= 1 {
		return gpus, nil
	}

	// Group GPUs by node
	gpusByNode := make(map[string][]tfv1.GPU)
	for _, gpu := range gpus {
		if gpu.Labels == nil {
			continue
		}

		nodeName, exists := gpu.Labels[constants.LabelKeyOwner]
		if !exists {
			continue
		}

		gpusByNode[nodeName] = append(gpusByNode[nodeName], gpu)
	}

	// Filter nodes that have at least 'count' GPUs
	var result []tfv1.GPU
	for _, nodeGPUs := range gpusByNode {
		if uint(len(nodeGPUs)) >= f.count {
			// Add all GPUs from this node to the result
			result = append(result, nodeGPUs...)
		}
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("no node has at least %d available GPUs", f.count)
	}

	return result, nil
}
