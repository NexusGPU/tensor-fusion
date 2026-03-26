package scheduler

import (
	"testing"

	"github.com/NexusGPU/tensor-fusion/internal/config"
	v1 "k8s.io/api/core/v1"
)

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
