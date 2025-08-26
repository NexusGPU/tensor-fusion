package filter

import (
	"context"
	"fmt"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/samber/lo"
	"sigs.k8s.io/controller-runtime/pkg/log"
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
func (f *SameNodeFilter) Filter(ctx context.Context, workerPodKey tfv1.NameNamespace, gpus []*tfv1.GPU) ([]*tfv1.GPU, error) {
	// If count is 1 or 0, no need to filter by node
	if f.count <= 1 {
		return gpus, nil
	}

	gpusByNode := make(map[string][]*tfv1.GPU, len(gpus)/2)
	for _, gpu := range gpus {
		nodeName, exists := gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
		if !exists {
			continue
		}
		gpusByNode[nodeName] = append(gpusByNode[nodeName], gpu)
	}

	// Filter nodes that have at least 'count' GPUs
	result := make([]*tfv1.GPU, 0, len(gpus))
	for _, nodeGPUs := range gpusByNode {
		if uint(len(nodeGPUs)) >= f.count {
			// Add all GPUs from this node to the result
			result = append(result, nodeGPUs...)
		}
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("no node has at least %d available GPUs", f.count)
	}

	if log.FromContext(ctx).V(6).Enabled() {
		log.FromContext(ctx).V(6).Info("Apply SameNodeMultipleGPU Filter",
			"before", strings.Join(lo.Map(gpus, func(gpu *tfv1.GPU, _ int) string {
				return gpu.Name
			}), ","),
			"after", strings.Join(lo.Map(result, func(gpu *tfv1.GPU, _ int) string {
				return gpu.Name
			}), ","), "pod", workerPodKey)
	}

	return result, nil
}

func (f *SameNodeFilter) Name() string {
	return "SameNodeFilter"
}
