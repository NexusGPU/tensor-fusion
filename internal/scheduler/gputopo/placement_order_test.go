package scheduler

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	framework "k8s.io/kubernetes/pkg/scheduler/framework"
)

func placementTestGPU(name, available string, numa int32) *tfv1.GPU {
	gpu := makeGPU(name, int32Ptr(numa))
	gpu.Status.Capacity = &tfv1.Resource{
		Tflops: resource.MustParse("100"),
		Vram:   resource.MustParse("100Mi"),
	}
	gpu.Status.Available = &tfv1.Resource{
		Tflops: resource.MustParse(available),
		Vram:   resource.MustParse(available + "Mi"),
	}
	return gpu
}

func scorerForMode(mode tfv1.PlacementMode) GPUScorer {
	strategy := gpuallocator.NewStrategy(mode, &config.GPUFitConfig{
		TflopsWeight: 0.5,
		VramWeight:   0.5,
	}, nil)
	return func(gpu *tfv1.GPU) int { return strategy.Score(gpu, false) }
}

func TestNUMATopologyUsesPlacementToBreakEquivalentCombinationTies(t *testing.T) {
	gpus := []*tfv1.GPU{
		placementTestGPU("gpu-most-used", "10", 0),
		placementTestGPU("gpu-used", "30", 0),
		placementTestGPU("gpu-idle", "100", 0),
		placementTestGPU("gpu-less-used", "80", 0),
	}

	tests := []struct {
		name string
		mode tfv1.PlacementMode
		want []string
	}{
		{name: "compact first chooses the most used GPUs", mode: tfv1.PlacementModeCompactFirst, want: []string{"gpu-most-used", "gpu-used"}},
		{name: "node compact GPU low load chooses the least used GPUs", mode: tfv1.PlacementModeNodeCompactGPULowLoad, want: []string{"gpu-idle", "gpu-less-used"}},
		{name: "low load first chooses the least used GPUs", mode: tfv1.PlacementModeLowLoadFirst, want: []string{"gpu-idle", "gpu-less-used"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan, err := NewNUMAEvaluator(1).EvaluateWithScorer(gpus, 2, true, scorerForMode(tt.mode))
			if err != nil {
				t.Fatalf("EvaluateWithScorer() error = %v", err)
			}
			if len(plan.BestGPUIds) != len(tt.want) {
				t.Fatalf("BestGPUIds = %v, want %v", plan.BestGPUIds, tt.want)
			}
			for i := range tt.want {
				if plan.BestGPUIds[i] != tt.want[i] {
					t.Fatalf("BestGPUIds = %v, want %v", plan.BestGPUIds, tt.want)
				}
			}
		})
	}
}

func TestPeerTopologyUsesPlacementInsideEquivalentNVLinkGroup(t *testing.T) {
	gpus := []*tfv1.GPU{
		placementTestGPU("gpu-most-used", "10", 0),
		placementTestGPU("gpu-used", "30", 0),
		placementTestGPU("gpu-idle", "100", 0),
		placementTestGPU("gpu-less-used", "80", 0),
	}
	for _, gpu := range gpus {
		peers := make([]tfv1.GPUPeerLinkStatus, 0, len(gpus)-1)
		for _, peer := range gpus {
			if peer.Name != gpu.Name {
				peers = append(peers, tfv1.GPUPeerLinkStatus{PeerGPUUUID: peer.Status.UUID, Tier: int32(TierSameInterconnect)})
			}
		}
		gpu.Status.Topology = &tfv1.GPUTopologyStatus{Peers: peers}
	}

	compact, err := NewPeerTopologyEvaluator(1).EvaluateWithScorer(gpus, 2, true, scorerForMode(tfv1.PlacementModeCompactFirst))
	if err != nil {
		t.Fatalf("compact EvaluateWithScorer() error = %v", err)
	}
	if compact.BestGPUIds[0] != "gpu-most-used" || compact.BestGPUIds[1] != "gpu-used" {
		t.Fatalf("compact BestGPUIds = %v, want most-used pair", compact.BestGPUIds)
	}

	lowLoad, err := NewPeerTopologyEvaluator(1).EvaluateWithScorer(gpus, 2, true, scorerForMode(tfv1.PlacementModeNodeCompactGPULowLoad))
	if err != nil {
		t.Fatalf("low-load EvaluateWithScorer() error = %v", err)
	}
	if lowLoad.BestGPUIds[0] != "gpu-idle" || lowLoad.BestGPUIds[1] != "gpu-less-used" {
		t.Fatalf("low-load BestGPUIds = %v, want least-used pair", lowLoad.BestGPUIds)
	}
}

func TestBetterTopologyWinsBeforePlacementScore(t *testing.T) {
	// CompactFirst strongly prefers the used GPUs, but they span NUMA domains.
	// The idle pair is same-NUMA, so topology must win first.
	gpus := []*tfv1.GPU{
		placementTestGPU("used-numa-0", "10", 0),
		placementTestGPU("used-numa-1", "10", 1),
		placementTestGPU("idle-numa-2-a", "100", 2),
		placementTestGPU("idle-numa-2-b", "100", 2),
	}

	plan, err := NewNUMAEvaluator(2).EvaluateWithScorer(gpus, 2, true, scorerForMode(tfv1.PlacementModeCompactFirst))
	if err != nil {
		t.Fatalf("EvaluateWithScorer() error = %v", err)
	}
	if plan.Tier != TierSameNUMA || plan.BestGPUIds[0] != "idle-numa-2-a" || plan.BestGPUIds[1] != "idle-numa-2-b" {
		t.Fatalf("plan = tier %d GPUs %v, want same-NUMA idle pair", plan.Tier, plan.BestGPUIds)
	}
}

func TestTopologyScoreDoesNotChangeNodePlacement(t *testing.T) {
	plugin := &GPUNetworkTopologyAware{}
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Labels: map[string]string{constants.LabelComponent: constants.ComponentWorker},
	}}

	score, status := plugin.Score(context.Background(), framework.NewCycleState(), pod, nil)
	if !status.IsSuccess() {
		t.Fatalf("Score() status = %v", status)
	}
	if score != 0 {
		t.Fatalf("Score() = %d, want 0 so placement owns node ordering", score)
	}
}
