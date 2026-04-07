package scheduler

import (
	fwk "k8s.io/kube-scheduler/framework"
)

// GPUAffinityTier represents the topology affinity level between GPUs.
// Lower values indicate closer/better affinity.
type GPUAffinityTier int

const (
	// TierSameInterconnect indicates GPUs connected via high-speed interconnect (NVLink/HCCS/XGMI).
	// Reserved for future vendor topology support.
	TierSameInterconnect GPUAffinityTier = 0
	// TierSameNUMA indicates GPUs on the same NUMA node.
	TierSameNUMA GPUAffinityTier = 1
	// TierCrossNUMA indicates GPUs on different NUMA nodes.
	TierCrossNUMA GPUAffinityTier = 2
	// TierUnknown indicates NUMA/topology information is unavailable.
	TierUnknown GPUAffinityTier = 3
)

const (
	TopologySourceAuto   = "auto"
	TopologySourceNUMA   = "numa"
	TopologySourceVendor = "vendor"

	TopologyModeSoft = "soft"
	TopologyModeHard = "hard"

	AnnotationBoolTrue  = "true"
	AnnotationBoolFalse = "false"
)

// CycleStateGPUTopologyResult is the key used to store topology evaluation results in CycleState.
const CycleStateGPUTopologyResult = "gpuTopologyResult"

// MaxCombinationSearch is the maximum number of GPU combinations to enumerate
// before falling back to heuristic selection.
const MaxCombinationSearch = 256

// Pairwise affinity scores used in combination scoring.
const (
	AffinityScoreSameNUMA  = 10
	AffinityScoreCrossNUMA = 2
	AffinityScoreUnknown   = 0
)

// Same-NUMA excess scoring parameters.
const (
	SameNUMAExcessPenalty = 5  // score deduction per excess GPU in the NUMA domain
	SameNUMAMinIntraScore = 50 // floor for same-NUMA intra-tier score
)

// numTiers is the total number of defined affinity tiers (0..3).
const numTiers = 4

// TierBandedScore computes a final score in [0, 100] that encodes both tier and
// intra-tier quality. Lower tiers (better topology) always produce higher scores
// than higher tiers (worse topology), regardless of intra-tier score.
// intraScore should be in [0, 100].
func TierBandedScore(tier GPUAffinityTier, intraScore int64) int64 {
	bandSize := int64(100 / numTiers) // 25 points per tier band
	tierBase := bandSize * int64(numTiers-1-int(tier))
	scaled := intraScore * (bandSize - 1) / 100
	return tierBase + scaled
}

// NodeTopologyPlan represents the topology evaluation result for a single node.
type NodeTopologyPlan struct {
	// NodeName is the name of the evaluated node.
	NodeName string
	// CandidateGPUIds are the GPU names from GPUResourcesFit's candidate set.
	CandidateGPUIds []string
	// BestGPUIds are the GPU names selected by topology evaluation.
	BestGPUIds []string
	// Tier is the worst-case affinity tier of the selected combination.
	Tier GPUAffinityTier
	// Score is the topology score for this combination (higher is better).
	Score int64
	// ModeSatisfied indicates whether the combination satisfies the hard mode constraint.
	ModeSatisfied bool
	// Reason provides debug information about the evaluation.
	Reason string
}

// GPUTopologyStateData stores topology evaluation results for all candidate nodes.
type GPUTopologyStateData struct {
	// Plans maps node names to their topology evaluation results.
	Plans map[string]*NodeTopologyPlan
}

func (d *GPUTopologyStateData) Clone() fwk.StateData {
	return d
}

// GetBestGPUIds returns the BestGPUIds for the given node, or nil if not found.
// This method satisfies the topologyBestGPUProvider interface used by gpuresources
// to avoid circular imports.
func (d *GPUTopologyStateData) GetBestGPUIds(nodeName string) []string {
	if d == nil || d.Plans == nil {
		return nil
	}
	plan, exists := d.Plans[nodeName]
	if !exists {
		return nil
	}
	return plan.BestGPUIds
}
