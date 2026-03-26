package scheduler

import (
	"fmt"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func makeGPUWithTopology(name, uuid string, numaNode *int32, peers []tfv1.GPUPeerLinkStatus) *tfv1.GPU {
	var topo *tfv1.GPUTopologyStatus
	if peers != nil {
		topo = &tfv1.GPUTopologyStatus{Peers: peers}
	}
	return &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status: tfv1.GPUStatus{
			UUID:     uuid,
			NUMANode: numaNode,
			Topology: topo,
		},
	}
}

func TestPeerEvaluator_SameInterconnect(t *testing.T) {
	// 4 GPUs all NVLink-connected (tier 0)
	gpus := []*tfv1.GPU{
		makeGPUWithTopology("gpu-0", "uuid-0", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-1", Tier: 0, LinkType: "Internal"},
			{PeerGPUUUID: "uuid-2", Tier: 0, LinkType: "Internal"},
			{PeerGPUUUID: "uuid-3", Tier: 0, LinkType: "Internal"},
		}),
		makeGPUWithTopology("gpu-1", "uuid-1", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-0", Tier: 0, LinkType: "Internal"},
			{PeerGPUUUID: "uuid-2", Tier: 0, LinkType: "Internal"},
			{PeerGPUUUID: "uuid-3", Tier: 0, LinkType: "Internal"},
		}),
		makeGPUWithTopology("gpu-2", "uuid-2", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-0", Tier: 0, LinkType: "Internal"},
			{PeerGPUUUID: "uuid-1", Tier: 0, LinkType: "Internal"},
			{PeerGPUUUID: "uuid-3", Tier: 0, LinkType: "Internal"},
		}),
		makeGPUWithTopology("gpu-3", "uuid-3", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-0", Tier: 0, LinkType: "Internal"},
			{PeerGPUUUID: "uuid-1", Tier: 0, LinkType: "Internal"},
			{PeerGPUUUID: "uuid-2", Tier: 0, LinkType: "Internal"},
		}),
	}

	eval := NewPeerTopologyEvaluator(1)
	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierSameInterconnect {
		t.Errorf("expected TierSameInterconnect, got %d (reason: %s)", plan.Tier, plan.Reason)
	}
	if !plan.ModeSatisfied {
		t.Errorf("expected ModeSatisfied=true for TierSameInterconnect with maxTier=1")
	}
	if len(plan.BestGPUIds) != 2 {
		t.Fatalf("expected 2 BestGPUIds, got %d", len(plan.BestGPUIds))
	}
}

func TestPeerEvaluator_TwoClusters_CrossNUMA(t *testing.T) {
	// 4 GPUs: cluster A (gpu-0, gpu-1) with NVLink, cluster B (gpu-2, gpu-3) with NVLink
	// Cross-cluster is tier 2 (system/cross-NUMA)
	gpus := []*tfv1.GPU{
		makeGPUWithTopology("gpu-0", "uuid-0", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-1", Tier: 0},
			{PeerGPUUUID: "uuid-2", Tier: 2},
			{PeerGPUUUID: "uuid-3", Tier: 2},
		}),
		makeGPUWithTopology("gpu-1", "uuid-1", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-0", Tier: 0},
			{PeerGPUUUID: "uuid-2", Tier: 2},
			{PeerGPUUUID: "uuid-3", Tier: 2},
		}),
		makeGPUWithTopology("gpu-2", "uuid-2", int32Ptr(1), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-0", Tier: 2},
			{PeerGPUUUID: "uuid-1", Tier: 2},
			{PeerGPUUUID: "uuid-3", Tier: 0},
		}),
		makeGPUWithTopology("gpu-3", "uuid-3", int32Ptr(1), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-0", Tier: 2},
			{PeerGPUUUID: "uuid-1", Tier: 2},
			{PeerGPUUUID: "uuid-2", Tier: 0},
		}),
	}

	eval := NewPeerTopologyEvaluator(2)

	// Request 2: should find same-interconnect cluster
	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierSameInterconnect {
		t.Errorf("request 2: expected TierSameInterconnect, got %d", plan.Tier)
	}

	// Request 4: must go cross-cluster → TierCrossNUMA
	plan, err = eval.Evaluate(gpus, 4, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierCrossNUMA {
		t.Errorf("request 4: expected TierCrossNUMA, got %d (reason: %s)", plan.Tier, plan.Reason)
	}
}

func TestPeerEvaluator_InterconnectBeatsNUMA(t *testing.T) {
	// Peer evaluator with TierSameInterconnect should always outscore NUMA evaluator's TierSameNUMA
	peerScore := TierBandedScore(TierSameInterconnect, 50) // worst same-interconnect
	numaScore := TierBandedScore(TierSameNUMA, 100)        // best same-NUMA

	if peerScore <= numaScore {
		t.Errorf("same-interconnect (%d) should outscore same-NUMA (%d)", peerScore, numaScore)
	}
}

func TestPeerEvaluator_SingleGPU_LeastDamage(t *testing.T) {
	// 3 NVLink-connected GPUs + 1 orphan with no peer data
	gpus := []*tfv1.GPU{
		makeGPUWithTopology("gpu-0", "uuid-0", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-1", Tier: 0},
			{PeerGPUUUID: "uuid-2", Tier: 0},
		}),
		makeGPUWithTopology("gpu-1", "uuid-1", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-0", Tier: 0},
			{PeerGPUUUID: "uuid-2", Tier: 0},
		}),
		makeGPUWithTopology("gpu-2", "uuid-2", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-0", Tier: 0},
			{PeerGPUUUID: "uuid-1", Tier: 0},
		}),
		makeGPUWithTopology("gpu-3", "uuid-3", int32Ptr(1), nil), // orphan, no peers
	}

	eval := NewPeerTopologyEvaluator(3)
	plan, err := eval.Evaluate(gpus, 1, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Should pick orphan gpu-3 to preserve the NVLink cluster
	if plan.BestGPUIds[0] != "gpu-3" {
		t.Errorf("expected gpu-3 (orphan), got %s", plan.BestGPUIds[0])
	}
}

func TestPeerEvaluator_FallbackToNUMA(t *testing.T) {
	// GPUs with partial peer data: gpu-0 has peers, gpu-1 and gpu-2 don't
	// But gpu-1 and gpu-2 are on same NUMA → should fall back to NUMA tier
	gpus := []*tfv1.GPU{
		makeGPUWithTopology("gpu-0", "uuid-0", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-1", Tier: 0},
		}),
		makeGPUWithTopology("gpu-1", "uuid-1", int32Ptr(0), nil), // no peers but has NUMA
		makeGPUWithTopology("gpu-2", "uuid-2", int32Ptr(0), nil), // no peers but has NUMA
	}

	eval := NewPeerTopologyEvaluator(3)
	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// gpu-0 and gpu-1 have tier 0 (from peer data), so should be SameInterconnect
	if plan.Tier != TierSameInterconnect {
		t.Logf("BestGPUIds: %v, Reason: %s", plan.BestGPUIds, plan.Reason)
		t.Errorf("expected TierSameInterconnect (gpu-0+gpu-1 have tier 0), got %d", plan.Tier)
	}
}

func TestPeerEvaluator_AllUnknown(t *testing.T) {
	// GPUs with no topology data at all
	gpus := []*tfv1.GPU{
		makeGPUWithTopology("gpu-0", "uuid-0", nil, nil),
		makeGPUWithTopology("gpu-1", "uuid-1", nil, nil),
	}

	eval := NewPeerTopologyEvaluator(1)
	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.Tier != TierUnknown {
		t.Errorf("expected TierUnknown, got %d", plan.Tier)
	}
	if plan.ModeSatisfied {
		t.Errorf("expected ModeSatisfied=false for TierUnknown with maxTier=1")
	}
}

func TestPeerEvaluator_HardMode_RejectsCrossCluster(t *testing.T) {
	// maxAllowedTier=0: only TierSameInterconnect allowed
	// 2 separate clusters, request spans both
	gpus := []*tfv1.GPU{
		makeGPUWithTopology("gpu-0", "uuid-0", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-1", Tier: 2},
		}),
		makeGPUWithTopology("gpu-1", "uuid-1", int32Ptr(1), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-0", Tier: 2},
		}),
	}

	eval := NewPeerTopologyEvaluator(0)
	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.ModeSatisfied {
		t.Errorf("expected ModeSatisfied=false for cross-cluster with maxTier=0")
	}
}

func TestPeerEvaluator_LargeNode_Heuristic(t *testing.T) {
	// 16 GPUs: 4 NVLink clusters of 4. Request 8 (triggers heuristic).
	gpus := make([]*tfv1.GPU, 16)
	for i := 0; i < 16; i++ {
		clusterBase := (i / 4) * 4
		peers := make([]tfv1.GPUPeerLinkStatus, 0)
		for j := 0; j < 16; j++ {
			if j == i {
				continue
			}
			tier := int32(2) // cross-cluster
			if j/4 == i/4 {
				tier = 0 // same cluster
			}
			peers = append(peers, tfv1.GPUPeerLinkStatus{
				PeerGPUUUID: fmt.Sprintf("uuid-%d", j),
				Tier:        tier,
			})
		}
		numaID := int32(clusterBase / 4)
		gpus[i] = makeGPUWithTopology(
			fmt.Sprintf("gpu-%d", i),
			fmt.Sprintf("uuid-%d", i),
			&numaID,
			peers,
		)
	}

	eval := NewPeerTopologyEvaluator(2)
	plan, err := eval.Evaluate(gpus, 8, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(plan.BestGPUIds) != 8 {
		t.Fatalf("expected 8 BestGPUIds, got %d", len(plan.BestGPUIds))
	}
	if plan.Score <= 0 {
		t.Errorf("expected positive score, got %d", plan.Score)
	}
}

func TestAutoEvaluator_PrefersVendorOverNUMA(t *testing.T) {
	// GPU with both NUMA and vendor topology: auto should pick vendor evaluator
	gpus := []*tfv1.GPU{
		makeGPUWithTopology("gpu-0", "uuid-0", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-1", Tier: 0},
		}),
		makeGPUWithTopology("gpu-1", "uuid-1", int32Ptr(0), []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-0", Tier: 0},
		}),
	}

	eval := NewAutoEvaluator(1)
	plan, err := eval.Evaluate(gpus, 2, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// With vendor data, should get TierSameInterconnect (0), not TierSameNUMA (1)
	if plan.Tier != TierSameInterconnect {
		t.Errorf("auto evaluator should prefer vendor: expected TierSameInterconnect, got %d", plan.Tier)
	}
}

func TestBuildPeerTierMatrix_Symmetrize(t *testing.T) {
	// Asymmetric peer data: gpu-0 sees gpu-1 at tier 0, but gpu-1 has no data
	gpus := []*tfv1.GPU{
		makeGPUWithTopology("gpu-0", "uuid-0", nil, []tfv1.GPUPeerLinkStatus{
			{PeerGPUUUID: "uuid-1", Tier: 0},
		}),
		makeGPUWithTopology("gpu-1", "uuid-1", nil, nil),
	}

	matrix := buildPeerTierMatrix(gpus)
	if matrix.tiers[0][1] != 0 {
		t.Errorf("expected tier 0 from gpu-0 to gpu-1, got %d", matrix.tiers[0][1])
	}
	if matrix.tiers[1][0] != 0 {
		t.Errorf("expected symmetrized tier 0 from gpu-1 to gpu-0, got %d", matrix.tiers[1][0])
	}
}
