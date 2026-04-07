package scheduler

import (
	"fmt"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func int32Ptr(v int32) *int32 { return &v }

func makeGPU(name string, numaNode *int32) *tfv1.GPU {
	return &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status: tfv1.GPUStatus{
			NUMANode: numaNode,
			UUID:     name + "-uuid",
		},
	}
}

// --- NUMAEvaluator tests ---

func TestNUMAEvaluator_MultiGPU_SameNUMA(t *testing.T) {
	eval := NewNUMAEvaluator(1)
	gpus := []*tfv1.GPU{
		makeGPU("gpu-0", int32Ptr(0)),
		makeGPU("gpu-1", int32Ptr(0)),
		makeGPU("gpu-2", int32Ptr(1)),
		makeGPU("gpu-3", int32Ptr(1)),
	}

	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierSameNUMA {
		t.Errorf("expected TierSameNUMA, got %d", plan.Tier)
	}
	if !plan.ModeSatisfied {
		t.Errorf("expected ModeSatisfied=true")
	}
	if len(plan.BestGPUIds) != 2 {
		t.Fatalf("expected 2 BestGPUIds, got %d", len(plan.BestGPUIds))
	}
	// All selected GPUs should be on the same NUMA
	numaIDs := make(map[int32]bool)
	for _, name := range plan.BestGPUIds {
		for _, gpu := range gpus {
			if gpu.Name == name {
				numaIDs[*gpu.Status.NUMANode] = true
			}
		}
	}
	if len(numaIDs) != 1 {
		t.Errorf("expected all GPUs on same NUMA, got NUMA nodes: %v", numaIDs)
	}
}

func TestNUMAEvaluator_MultiGPU_NoSameNUMA_HardMode(t *testing.T) {
	eval := NewNUMAEvaluator(1) // maxAllowedTier=1 means only TierSameNUMA is allowed
	gpus := []*tfv1.GPU{
		makeGPU("gpu-0", int32Ptr(0)),
		makeGPU("gpu-1", int32Ptr(1)),
		makeGPU("gpu-2", int32Ptr(2)),
		makeGPU("gpu-3", int32Ptr(3)),
	}

	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierCrossNUMA {
		t.Errorf("expected TierCrossNUMA, got %d", plan.Tier)
	}
	if plan.ModeSatisfied {
		t.Errorf("expected ModeSatisfied=false for hard mode with cross-NUMA")
	}
}

func TestNUMAEvaluator_MultiGPU_NoSameNUMA_SoftMode(t *testing.T) {
	eval := NewNUMAEvaluator(2) // maxAllowedTier=2 means TierCrossNUMA is allowed
	gpus := []*tfv1.GPU{
		makeGPU("gpu-0", int32Ptr(0)),
		makeGPU("gpu-1", int32Ptr(1)),
		makeGPU("gpu-2", int32Ptr(2)),
		makeGPU("gpu-3", int32Ptr(3)),
	}

	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierCrossNUMA {
		t.Errorf("expected TierCrossNUMA, got %d", plan.Tier)
	}
	if !plan.ModeSatisfied {
		t.Errorf("expected ModeSatisfied=true for soft mode with cross-NUMA and maxTier=2")
	}
	if len(plan.BestGPUIds) != 2 {
		t.Fatalf("expected 2 BestGPUIds, got %d", len(plan.BestGPUIds))
	}
}

func TestNUMAEvaluator_SingleGPU_LeastDamage(t *testing.T) {
	eval := NewNUMAEvaluator(1)
	gpus := []*tfv1.GPU{
		makeGPU("gpu-0", int32Ptr(0)), // NUMA 0: 3 GPUs (large cluster)
		makeGPU("gpu-1", int32Ptr(0)),
		makeGPU("gpu-2", int32Ptr(0)),
		makeGPU("gpu-3", int32Ptr(1)), // NUMA 1: 1 GPU (orphan)
	}

	plan, err := eval.Evaluate(gpus, 1, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(plan.BestGPUIds) != 1 {
		t.Fatalf("expected 1 BestGPUId, got %d", len(plan.BestGPUIds))
	}
	// Should pick the orphan gpu-3 to preserve the NUMA 0 cluster
	if plan.BestGPUIds[0] != "gpu-3" {
		t.Errorf("expected gpu-3 (orphan), got %s", plan.BestGPUIds[0])
	}
}

func TestNUMAEvaluator_SingleGPU_NoLeastDamage(t *testing.T) {
	eval := NewNUMAEvaluator(1)
	gpus := []*tfv1.GPU{
		makeGPU("gpu-0", int32Ptr(0)),
		makeGPU("gpu-1", int32Ptr(0)),
		makeGPU("gpu-2", int32Ptr(1)),
	}

	plan, err := eval.Evaluate(gpus, 1, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(plan.BestGPUIds) != 1 {
		t.Fatalf("expected 1 BestGPUId, got %d", len(plan.BestGPUIds))
	}
	// Without least damage, just picks the first GPU
	if plan.BestGPUIds[0] != "gpu-0" {
		t.Errorf("expected gpu-0 (first), got %s", plan.BestGPUIds[0])
	}
}

func TestNUMAEvaluator_NUMAUnknown(t *testing.T) {
	eval := NewNUMAEvaluator(1)
	gpus := []*tfv1.GPU{
		makeGPU("gpu-0", nil),
		makeGPU("gpu-1", nil),
		makeGPU("gpu-2", nil),
		makeGPU("gpu-3", nil),
	}

	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierUnknown {
		t.Errorf("expected TierUnknown, got %d", plan.Tier)
	}
	if plan.ModeSatisfied {
		t.Errorf("expected ModeSatisfied=false for unknown NUMA with maxTier=1")
	}
}

func TestNUMAEvaluator_NUMAUnknown_TreatAsWorst(t *testing.T) {
	eval := NewNUMAEvaluator(3) // maxAllowedTier=3 includes TierUnknown
	gpus := []*tfv1.GPU{
		makeGPU("gpu-0", nil),
		makeGPU("gpu-1", nil),
	}

	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierUnknown {
		t.Errorf("expected TierUnknown, got %d", plan.Tier)
	}
	if !plan.ModeSatisfied {
		t.Errorf("expected ModeSatisfied=true with maxTier=3")
	}
}

func TestNUMAEvaluator_PreferCompactNUMA(t *testing.T) {
	eval := NewNUMAEvaluator(1)
	gpus := []*tfv1.GPU{
		makeGPU("gpu-0", int32Ptr(0)), // NUMA 0: 2 GPUs (exact fit)
		makeGPU("gpu-1", int32Ptr(0)),
		makeGPU("gpu-2", int32Ptr(1)), // NUMA 1: 4 GPUs (excess)
		makeGPU("gpu-3", int32Ptr(1)),
		makeGPU("gpu-4", int32Ptr(1)),
		makeGPU("gpu-5", int32Ptr(1)),
	}

	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierSameNUMA {
		t.Errorf("expected TierSameNUMA, got %d", plan.Tier)
	}
	// Should prefer NUMA 0 (exact fit, less fragmentation)
	for _, name := range plan.BestGPUIds {
		if name != "gpu-0" && name != "gpu-1" {
			t.Errorf("expected GPUs from NUMA 0, got %s", name)
		}
	}
}

func TestNUMAEvaluator_LargeNodeHeuristic(t *testing.T) {
	// Create 16 GPUs spread across 4 NUMA nodes
	gpus := make([]*tfv1.GPU, 16)
	for i := 0; i < 16; i++ {
		numaID := int32(i / 4)
		gpus[i] = makeGPU(fmt.Sprintf("gpu-%d", i), int32Ptr(numaID))
	}

	eval := NewNUMAEvaluator(2)
	plan, err := eval.Evaluate(gpus, 8, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(plan.BestGPUIds) != 8 {
		t.Fatalf("expected 8 BestGPUIds, got %d", len(plan.BestGPUIds))
	}
	// Should have a valid plan
	if plan.Score <= 0 {
		t.Errorf("expected positive score, got %d", plan.Score)
	}
}

func TestNUMAEvaluator_DomainPruning(t *testing.T) {
	// 16 GPUs across 4 NUMA domains (4 each). Request 4.
	// C(16,4) = 1820 > 256 → triggers domain pruning.
	// After pruning to top 1 domain (4 GPUs): C(4,4) = 1, enumerable.
	// Should find a same-NUMA combination via pruned enumeration.
	gpus := make([]*tfv1.GPU, 16)
	for i := 0; i < 16; i++ {
		numaID := int32(i / 4)
		gpus[i] = makeGPU(fmt.Sprintf("gpu-%d", i), int32Ptr(numaID))
	}

	eval := NewNUMAEvaluator(2)
	plan, err := eval.Evaluate(gpus, 4, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(plan.BestGPUIds) != 4 {
		t.Fatalf("expected 4 BestGPUIds, got %d", len(plan.BestGPUIds))
	}
	// With 4 GPUs per NUMA, same-NUMA should be found via Phase 1
	if plan.Tier != TierSameNUMA {
		t.Errorf("expected TierSameNUMA, got %d (reason: %s)", plan.Tier, plan.Reason)
	}
}

func TestNUMAEvaluator_DomainPruning_CrossNUMA(t *testing.T) {
	// 20 GPUs: NUMA 0 has 8, NUMA 1 has 6, NUMA 2 has 4, NUMA 3 has 2. Request 10.
	// No single domain has 10, so same-NUMA fails (Phase 1 returns nil).
	// C(20,10) is huge → full enumeration skipped.
	// Pruning: top 2 domains (8+6=14 GPUs), C(14,10) = 1001 > 256.
	// Add NUMA 2: (8+6+4=18 GPUs), C(18,10) = 43758 > 256.
	// All domains: same as original → pruning returns nil.
	// Falls through to heuristic.
	var gpus []*tfv1.GPU
	sizes := []int{8, 6, 4, 2}
	idx := 0
	for numaID, size := range sizes {
		for j := 0; j < size; j++ {
			gpus = append(gpus, makeGPU(fmt.Sprintf("gpu-%d", idx), int32Ptr(int32(numaID))))
			idx++
		}
	}

	eval := NewNUMAEvaluator(2)
	plan, err := eval.Evaluate(gpus, 10, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(plan.BestGPUIds) != 10 {
		t.Fatalf("expected 10 BestGPUIds, got %d", len(plan.BestGPUIds))
	}
	// Heuristic fills from largest domain first, so most GPUs should be from NUMA 0
	if plan.Score <= 0 {
		t.Errorf("expected positive score, got %d", plan.Score)
	}
}

func TestNUMAEvaluator_DomainPruning_EnablesEnumeration(t *testing.T) {
	// 16 GPUs: NUMA 0 has 6, NUMA 1 has 6, NUMA 2 has 2, NUMA 3 has 2. Request 6.
	// C(16,6) = 8008 > 256 → full enumeration skipped.
	// Pruning: top 1 domain (6 GPUs), C(6,6) = 1 ≤ 256 → pruned enumeration.
	// Result should be same-NUMA from NUMA 0 or NUMA 1.
	var gpus []*tfv1.GPU
	sizes := []int{6, 6, 2, 2}
	idx := 0
	for numaID, size := range sizes {
		for j := 0; j < size; j++ {
			gpus = append(gpus, makeGPU(fmt.Sprintf("gpu-%d", idx), int32Ptr(int32(numaID))))
			idx++
		}
	}

	eval := NewNUMAEvaluator(2)
	plan, err := eval.Evaluate(gpus, 6, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierSameNUMA {
		t.Errorf("expected TierSameNUMA via domain pruning, got %d (reason: %s)", plan.Tier, plan.Reason)
	}
}

// --- AutoEvaluator tests ---

func TestAutoEvaluator_WithNUMA(t *testing.T) {
	eval := NewAutoEvaluator(1)
	gpus := []*tfv1.GPU{
		makeGPU("gpu-0", int32Ptr(0)),
		makeGPU("gpu-1", int32Ptr(0)),
	}

	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierSameNUMA {
		t.Errorf("expected TierSameNUMA from auto evaluator, got %d", plan.Tier)
	}
}

func TestAutoEvaluator_WithoutNUMA(t *testing.T) {
	eval := NewAutoEvaluator(1)
	gpus := []*tfv1.GPU{
		makeGPU("gpu-0", nil),
		makeGPU("gpu-1", nil),
	}

	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierUnknown {
		t.Errorf("expected TierUnknown from auto evaluator without NUMA, got %d", plan.Tier)
	}
}

func TestAutoEvaluator_EmptyGPUs(t *testing.T) {
	eval := NewAutoEvaluator(1)
	plan, err := eval.Evaluate(nil, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierUnknown {
		t.Errorf("expected TierUnknown for empty GPUs, got %d", plan.Tier)
	}
	if plan.ModeSatisfied {
		t.Errorf("expected ModeSatisfied=false for empty GPUs")
	}
}

// --- GPUTopologyStateData tests ---

func TestGPUTopologyStateData_GetBestGPUIds(t *testing.T) {
	state := &GPUTopologyStateData{
		Plans: map[string]*NodeTopologyPlan{
			"node-1": {
				BestGPUIds: []string{"gpu-0", "gpu-1"},
			},
		},
	}

	if ids := state.GetBestGPUIds("node-1"); len(ids) != 2 {
		t.Errorf("expected 2 GPU IDs for node-1, got %d", len(ids))
	}
	if ids := state.GetBestGPUIds("node-2"); ids != nil {
		t.Errorf("expected nil for non-existent node, got %v", ids)
	}

	var nilState *GPUTopologyStateData
	if ids := nilState.GetBestGPUIds("node-1"); ids != nil {
		t.Errorf("expected nil for nil state, got %v", ids)
	}
}

// --- Tier-banded scoring tests ---

func TestTierBandedScore_Ordering(t *testing.T) {
	// Any same-NUMA score must beat any cross-NUMA score
	sameNUMAWorst := TierBandedScore(TierSameNUMA, 0)
	crossNUMABest := TierBandedScore(TierCrossNUMA, 100)
	if sameNUMAWorst <= crossNUMABest {
		t.Errorf("same-NUMA worst (%d) should beat cross-NUMA best (%d)", sameNUMAWorst, crossNUMABest)
	}

	// Any cross-NUMA score must beat any unknown score
	crossNUMAWorst := TierBandedScore(TierCrossNUMA, 0)
	unknownBest := TierBandedScore(TierUnknown, 100)
	if crossNUMAWorst <= unknownBest {
		t.Errorf("cross-NUMA worst (%d) should beat unknown best (%d)", crossNUMAWorst, unknownBest)
	}

	// Within same tier, higher intra-score should give higher banded score
	high := TierBandedScore(TierSameNUMA, 100)
	low := TierBandedScore(TierSameNUMA, 0)
	if high <= low {
		t.Errorf("higher intra-score (%d) should beat lower (%d) within same tier", high, low)
	}
}

// --- Single GPU with unknown NUMA tests ---

func TestNUMAEvaluator_SingleGPU_UnknownNUMA_NoLeastDamage(t *testing.T) {
	eval := NewNUMAEvaluator(1) // maxTier=1 (SameNUMA only)
	gpus := []*tfv1.GPU{
		makeGPU("gpu-0", nil), // unknown NUMA
	}

	plan, err := eval.Evaluate(gpus, 1, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierUnknown {
		t.Errorf("expected TierUnknown for GPU with nil NUMA, got %d", plan.Tier)
	}
	if plan.ModeSatisfied {
		t.Errorf("expected ModeSatisfied=false for unknown NUMA with maxTier=1")
	}
}

func TestNUMAEvaluator_SingleGPU_UnknownNUMA_LeastDamage(t *testing.T) {
	eval := NewNUMAEvaluator(3) // maxTier=3 (all allowed)
	gpus := []*tfv1.GPU{
		makeGPU("gpu-0", int32Ptr(0)),
		makeGPU("gpu-1", nil), // unknown NUMA
	}

	plan, err := eval.Evaluate(gpus, 1, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Least-damage should prefer unknown GPU (least valuable)
	if plan.BestGPUIds[0] != "gpu-1" {
		t.Errorf("expected gpu-1 (unknown, least damage), got %s", plan.BestGPUIds[0])
	}
	if plan.Tier != TierUnknown {
		t.Errorf("expected TierUnknown when selecting unknown GPU, got %d", plan.Tier)
	}
	if !plan.ModeSatisfied {
		t.Errorf("expected ModeSatisfied=true with maxTier=3")
	}
}

// --- Zero count edge case ---

func TestNUMAEvaluator_ZeroCount(t *testing.T) {
	eval := NewNUMAEvaluator(3)
	gpus := []*tfv1.GPU{makeGPU("gpu-0", int32Ptr(0))}

	plan, err := eval.Evaluate(gpus, 0, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.ModeSatisfied {
		t.Errorf("expected ModeSatisfied=false for zero count")
	}
	if plan.CandidateGPUIds == nil {
		t.Errorf("expected non-nil CandidateGPUIds")
	}
}

// --- Cross-tier score ordering in soft mode ---

func TestNUMAEvaluator_SameNUMA_AlwaysOutscores_CrossNUMA(t *testing.T) {
	eval := NewNUMAEvaluator(3)

	// Node A: 2 GPUs on same NUMA with heavy excess (10 GPUs, request 2)
	gpusA := make([]*tfv1.GPU, 10)
	for i := 0; i < 10; i++ {
		gpusA[i] = makeGPU(fmt.Sprintf("a-gpu-%d", i), int32Ptr(0))
	}
	planA, _ := eval.Evaluate(gpusA, 2, true)

	// Node B: 2 GPUs on different NUMA (cross-NUMA)
	gpusB := []*tfv1.GPU{
		makeGPU("b-gpu-0", int32Ptr(0)),
		makeGPU("b-gpu-1", int32Ptr(1)),
	}
	planB, _ := eval.Evaluate(gpusB, 2, true)

	if planA.Score <= planB.Score {
		t.Errorf("same-NUMA (score=%d) should outscore cross-NUMA (score=%d) in soft mode", planA.Score, planB.Score)
	}
}

// --- Combination utility tests ---

func TestCombinationCount(t *testing.T) {
	tests := []struct {
		n, k int
		want int
	}{
		{8, 2, 28},
		{8, 4, 70},
		{8, 8, 1},
		{4, 0, 1},
		{4, 5, 0},
		{16, 4, 1820},
	}
	for _, tt := range tests {
		got := combinationCount(tt.n, tt.k)
		if got != tt.want {
			t.Errorf("C(%d,%d) = %d, want %d", tt.n, tt.k, got, tt.want)
		}
	}
}

func TestNextCombination(t *testing.T) {
	indices := []int{0, 1}
	n := 4
	count := 1 // already have the first combination
	for nextCombination(indices, n) {
		count++
	}
	if count != 6 { // C(4,2) = 6
		t.Errorf("expected 6 combinations, got %d", count)
	}
}
