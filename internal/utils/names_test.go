package utils_test

import (
	"strings"
	"testing"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

func TestBuildHypervisorPodNameKeepsShortNames(t *testing.T) {
	t.Parallel()

	nodeName := "ip-10-0-0-1.ap-southeast-2.compute.internal"
	got := utils.BuildHypervisorPodName(nodeName)
	want := "hypervisor-" + nodeName

	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestBuildHypervisorPodNameTruncatesLongNames(t *testing.T) {
	t.Parallel()

	nodeName := "ip-10-0-123-456.ap-southeast-2.compute.internal-cluster-very-long-node-name-for-hypervisor"
	got := utils.BuildHypervisorPodName(nodeName)

	if len(got) > 63 {
		t.Fatalf("expected name length <= 63, got %d (%q)", len(got), got)
	}
	if !strings.HasPrefix(got, "hypervisor-") {
		t.Fatalf("expected hypervisor prefix, got %q", got)
	}
	if got == "hypervisor-"+nodeName {
		t.Fatalf("expected long name to be truncated, got %q", got)
	}
}

func TestBuildHypervisorPodNameAddsHashForUniqueness(t *testing.T) {
	t.Parallel()

	common := strings.Repeat("a", 80)
	nameA := common + "node-a"
	nameB := common + "node-b"

	gotA := utils.BuildHypervisorPodName(nameA)
	gotB := utils.BuildHypervisorPodName(nameB)

	if gotA == gotB {
		t.Fatalf("expected distinct names for long node names, got %q", gotA)
	}
	if len(gotA) > 63 || len(gotB) > 63 {
		t.Fatalf("expected both names to be <= 63, got %d and %d", len(gotA), len(gotB))
	}
}
