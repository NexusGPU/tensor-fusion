package scheduler

import (
	"context"
	"strings"
	"testing"

	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
	framework "k8s.io/kubernetes/pkg/scheduler/framework"
)

func TestFilterUsesPerPodMaxTierInDiagnostic(t *testing.T) {
	plugin := &GPUNetworkTopologyAware{
		cfg: &config.GPUNetworkTopologyAwareConfig{
			Mode:           TopologyModeHard,
			TopologySource: TopologySourceAuto,
			MaxAllowedTier: 1,
		},
	}
	state := framework.NewCycleState()
	state.Write(CycleStateGPUTopologyResult, &GPUTopologyStateData{
		Plans: map[string]*NodeTopologyPlan{
			"node-a": {
				Tier:          TierSameNUMA,
				ModeSatisfied: false,
				Reason:        "test topology",
			},
		},
	})
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{constants.LabelComponent: constants.ComponentWorker},
			Annotations: map[string]string{
				AnnotationRequireTopology: AnnotationBoolTrue,
				AnnotationTopologyMaxTier: "0",
			},
		},
	}
	nodeInfo := framework.NewNodeInfo()
	nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}})

	status := plugin.Filter(context.Background(), state, pod, nodeInfo)
	if status.Code() != fwk.Unschedulable {
		t.Fatalf("Filter() code = %v, want Unschedulable", status.Code())
	}
	if want := "maxAllowed=0"; !strings.Contains(status.Message(), want) {
		t.Fatalf("Filter() message = %q, want %q", status.Message(), want)
	}
}

func TestResolvePerPodConfig_DefaultsToSoftWithoutAnnotation(t *testing.T) {
	plugin := &GPUNetworkTopologyAware{
		cfg: &config.GPUNetworkTopologyAwareConfig{
			Mode:           "soft",
			TopologySource: "auto",
			MaxAllowedTier: 1,
		},
	}

	mode, maxTier, source := plugin.resolvePerPodConfig(&v1.Pod{})
	if mode != "soft" {
		t.Fatalf("expected default mode soft, got %q", mode)
	}
	if maxTier != 1 {
		t.Fatalf("expected default maxTier 1, got %d", maxTier)
	}
	if source != "auto" {
		t.Fatalf("expected default source auto, got %q", source)
	}
}

func TestResolvePerPodConfig_RequireTopologyOverridesMode(t *testing.T) {
	plugin := &GPUNetworkTopologyAware{
		cfg: &config.GPUNetworkTopologyAwareConfig{
			Mode:           "soft",
			TopologySource: "auto",
			MaxAllowedTier: 1,
		},
	}

	pod := &v1.Pod{}
	pod.Annotations = map[string]string{
		AnnotationRequireTopology: "true",
	}

	mode, _, _ := plugin.resolvePerPodConfig(pod)
	if mode != "hard" {
		t.Fatalf("expected mode hard when require topology is true, got %q", mode)
	}

	pod.Annotations[AnnotationRequireTopology] = "false"
	mode, _, _ = plugin.resolvePerPodConfig(pod)
	if mode != "soft" {
		t.Fatalf("expected mode soft when require topology is false, got %q", mode)
	}
}

func TestResolvePerPodConfig_StillHonorsTierAndSourceOverrides(t *testing.T) {
	plugin := &GPUNetworkTopologyAware{
		cfg: &config.GPUNetworkTopologyAwareConfig{
			Mode:           "soft",
			TopologySource: "auto",
			MaxAllowedTier: 1,
		},
	}

	pod := &v1.Pod{}
	pod.Annotations = map[string]string{
		AnnotationRequireTopology: "true",
		AnnotationTopologyMaxTier: "2",
		AnnotationTopologySource:  "vendor",
	}

	mode, maxTier, source := plugin.resolvePerPodConfig(pod)
	if mode != "hard" {
		t.Fatalf("expected mode hard, got %q", mode)
	}
	if maxTier != 2 {
		t.Fatalf("expected maxTier 2, got %d", maxTier)
	}
	if source != "vendor" {
		t.Fatalf("expected source vendor, got %q", source)
	}
}
