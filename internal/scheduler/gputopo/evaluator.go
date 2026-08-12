package scheduler

import (
	"sort"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// GPUScorer returns a placement score for one GPU. Higher scores are preferred.
// Topology evaluators use it only after topology quality is equal.
type GPUScorer func(*tfv1.GPU) int

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

	// EvaluateWithScorer keeps topology as the primary ordering and uses the
	// placement score only to break ties between equivalent GPU combinations.
	EvaluateWithScorer(gpus []*tfv1.GPU, count uint, preferLeastDamage bool, scorer GPUScorer) (*NodeTopologyPlan, error)
}

func sortGPUsByPlacement(gpus []*tfv1.GPU, scorer GPUScorer) {
	sort.SliceStable(gpus, func(i, j int) bool {
		if scorer != nil {
			si, sj := scorer(gpus[i]), scorer(gpus[j])
			if si != sj {
				return si > sj
			}
		}
		return gpus[i].Name < gpus[j].Name
	})
}

func placementScore(gpus []*tfv1.GPU, scorer GPUScorer) int {
	if scorer == nil {
		return 0
	}
	total := 0
	for _, gpu := range gpus {
		total += scorer(gpu)
	}
	return total
}

func bestPlacementScore(gpus []*tfv1.GPU, count int, scorer GPUScorer) int {
	ordered := append([]*tfv1.GPU(nil), gpus...)
	sortGPUsByPlacement(ordered, scorer)
	if count < len(ordered) {
		ordered = ordered[:count]
	}
	return placementScore(ordered, scorer)
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
	return e.EvaluateWithScorer(gpus, count, preferLeastDamage, nil)
}

func (e *AutoEvaluator) EvaluateWithScorer(gpus []*tfv1.GPU, count uint, preferLeastDamage bool, scorer GPUScorer) (*NodeTopologyPlan, error) {
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
			return e.peerEvaluator.EvaluateWithScorer(gpus, count, preferLeastDamage, scorer)
		}
	}

	// Check if NUMA data is available
	for _, gpu := range gpus {
		if gpu.Status.NUMANode != nil && *gpu.Status.NUMANode >= 0 {
			return e.numaEvaluator.EvaluateWithScorer(gpus, count, preferLeastDamage, scorer)
		}
	}

	// No topology data available
	gpuNames := make([]string, len(gpus))
	for i, gpu := range gpus {
		gpuNames[i] = gpu.Name
	}
	ordered := append([]*tfv1.GPU(nil), gpus...)
	sortGPUsByPlacement(ordered, scorer)
	bestGPUs := make([]string, len(ordered))
	for i, gpu := range ordered {
		bestGPUs[i] = gpu.Name
	}
	if count > 0 && int(count) < len(gpuNames) {
		bestGPUs = bestGPUs[:count]
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
