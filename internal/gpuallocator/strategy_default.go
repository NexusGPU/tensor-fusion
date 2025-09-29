package gpuallocator

import (
	"fmt"
	"slices"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/samber/lo"
)

// NodeCompactGPULowLoad selects GPU with maximum available resources (least utilized)
// to distribute workloads more evenly across GPUs
// default to this mode since it balance the cost and stability, scatter workload on single node with multiple GPUs
type NodeCompactGPULowLoad struct {
	cfg          *config.GPUFitConfig
	nodeGpuStore map[string]map[string]*tfv1.GPU
}

var _ Strategy = NodeCompactGPULowLoad{}

// GPU selector is not used by Kubernetes scheduler framework,
// just used for allocator testing as of now, framework will compose similar logic
var DefaultGPUSelector = func(
	strategy Strategy,
	nodeGPUMap map[string]map[string]*tfv1.GPU,
	validGPUs []*tfv1.GPU, count uint,
) ([]*tfv1.GPU, error) {
	if len(validGPUs) == 0 {
		return nil, fmt.Errorf("no GPUs available")
	}

	// step 1. group gpus by its node
	gpuMap := lo.GroupBy(validGPUs, func(gpu *tfv1.GPU) string {
		return gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
	})

	// step 2. for each node, get all its gpus from, and calculate the score
	nodeScores := make([]struct {
		node  string
		score int
	}, 0, len(gpuMap))
	for node := range gpuMap {
		score := 0
		allGPUs := nodeGPUMap[node]
		for _, gpu := range allGPUs {
			score += strategy.Score(gpu)
		}
		nodeScores = append(nodeScores, struct {
			node  string
			score int
		}{node, score})
	}

	// step 3. sort node by score
	slices.SortFunc(nodeScores, func(a, b struct {
		node  string
		score int
	}) int {
		return b.score - a.score
	})

	// step 4. find first node that have enough gpus >= count, return these count gpus
	for _, nodeScore := range nodeScores {
		gpus := gpuMap[nodeScore.node]
		if len(gpus) >= int(count) {
			// choose highest score GPUs
			slices.SortFunc(gpus, func(a, b *tfv1.GPU) int {
				return strategy.Score(b) - strategy.Score(a)
			})
			return gpus[:count], nil
		}
	}
	// should not happen
	return nil, fmt.Errorf("not enough gpus in scoring stage")
}

// SelectGPUs selects multiple GPUs from the same node with the most available resources (least loaded)
func (l NodeCompactGPULowLoad) SelectGPUs(gpus []*tfv1.GPU, count uint) ([]*tfv1.GPU, error) {
	return DefaultGPUSelector(l, l.nodeGpuStore, gpus, count)
}

// Score function is using by Kubernetes scheduler framework
func (l NodeCompactGPULowLoad) Score(gpu *tfv1.GPU) int {
	// TODO: should consider node level score, and GPU score is second level
	// probably Add ScoreNode function (with all GPU score context)
	tflopsAvailablePercentage := gpu.Status.Available.Tflops.AsApproximateFloat64() / gpu.Status.Capacity.Tflops.AsApproximateFloat64() * 100
	vramAvailablePercentage := gpu.Status.Available.Vram.AsApproximateFloat64() / gpu.Status.Capacity.Vram.AsApproximateFloat64() * 100
	return normalizeScore(l.cfg, vramAvailablePercentage, tflopsAvailablePercentage)
}
