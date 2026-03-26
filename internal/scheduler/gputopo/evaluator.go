package scheduler

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// Evaluator evaluates GPU topology for a set of candidate GPUs and returns
// the best combination for the requested GPU count.
type Evaluator interface {
	// Name returns the evaluator name for logging/debugging.
	Name() string
	// Evaluate selects the best GPU combination from the given candidates.
	// gpus: candidate GPUs on a single node.
	// count: number of GPUs requested.
	// preferLeastDamage: if true, single-GPU requests prefer GPUs that
	//   cause the least damage to high-quality topology clusters.
	Evaluate(gpus []*tfv1.GPU, count uint, preferLeastDamage bool) (*NodeTopologyPlan, error)
}

// AutoEvaluator selects the appropriate evaluator based on available topology data.
type AutoEvaluator struct {
	peerEvaluator *PeerTopologyEvaluator
	numaEvaluator *NUMAEvaluator
}

// NewAutoEvaluator creates an AutoEvaluator with the given max allowed tier.
func NewAutoEvaluator(maxAllowedTier int) *AutoEvaluator {
	return &AutoEvaluator{
		peerEvaluator: NewPeerTopologyEvaluator(maxAllowedTier),
		numaEvaluator: NewNUMAEvaluator(maxAllowedTier),
	}
}

func (e *AutoEvaluator) Name() string {
	return TopologySourceAuto
}

// Evaluate delegates to the appropriate evaluator based on GPU topology data.
// Priority:
// 1. If any GPU has Topology.Peers with valid tier data → PeerTopologyEvaluator
// 2. If any GPU has NUMANode set → NUMAEvaluator
// 3. Otherwise → returns TierUnknown result
func (e *AutoEvaluator) Evaluate(gpus []*tfv1.GPU, count uint, preferLeastDamage bool) (*NodeTopologyPlan, error) {
	if len(gpus) == 0 {
		return &NodeTopologyPlan{
			CandidateGPUIds: []string{},
			Tier:            TierUnknown,
			Score:           0,
			ModeSatisfied:   false,
			Reason:          "no candidate GPUs",
		}, nil
	}

	// Check if vendor topology (peer link) data is available
	for _, gpu := range gpus {
		if gpu.Status.Topology != nil && len(gpu.Status.Topology.Peers) > 0 {
			return e.peerEvaluator.Evaluate(gpus, count, preferLeastDamage)
		}
	}

	// Check if NUMA data is available
	for _, gpu := range gpus {
		if gpu.Status.NUMANode != nil && *gpu.Status.NUMANode >= 0 {
			return e.numaEvaluator.Evaluate(gpus, count, preferLeastDamage)
		}
	}

	// No topology data available
	gpuNames := make([]string, len(gpus))
	for i, gpu := range gpus {
		gpuNames[i] = gpu.Name
	}
	bestGPUs := gpuNames
	if count > 0 && int(count) < len(gpuNames) {
		bestGPUs = gpuNames[:count]
	}

	return &NodeTopologyPlan{
		CandidateGPUIds: gpuNames,
		BestGPUIds:      bestGPUs,
		Tier:            TierUnknown,
		Score:           0,
		ModeSatisfied:   false,
		Reason:          "no topology data available, using unknown tier",
	}, nil
}
