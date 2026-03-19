package controller

import (
	"strings"
	"testing"
)

func TestBuildNodeJobNameKeepsShortNames(t *testing.T) {
	t.Parallel()

	nodeName := "ip-10-0-0-1.ap-southeast-2.compute.internal"
	got := getDiscoveryJobName(nodeName)
	want := "node-discovery-" + nodeName

	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestBuildNodeJobNameTruncatesLongNames(t *testing.T) {
	t.Parallel()

	nodeName := "ip-10-0-123-456.ap-southeast-2.compute.internal-cluster-very-long-node-name-for-job"
	got := getDiscoveryJobName(nodeName)

	if len(got) > maxNodeJobNameLength {
		t.Fatalf("expected name length <= %d, got %d (%q)", maxNodeJobNameLength, len(got), got)
	}
	if !strings.HasPrefix(got, "node-discovery-") {
		t.Fatalf("expected discovery prefix, got %q", got)
	}
	if got == "node-discovery-"+nodeName {
		t.Fatalf("expected long name to be truncated, got %q", got)
	}
}

func TestBuildNodeJobNameAddsHashForUniqueness(t *testing.T) {
	t.Parallel()

	prefix := "driver-probe"
	common := strings.Repeat("a", 80)
	nameA := common + "node-a"
	nameB := common + "node-b"

	gotA := buildNodeJobName(prefix, nameA)
	gotB := buildNodeJobName(prefix, nameB)

	if gotA == gotB {
		t.Fatalf("expected distinct job names for long node names, got %q", gotA)
	}
	if len(gotA) > maxNodeJobNameLength || len(gotB) > maxNodeJobNameLength {
		t.Fatalf("expected both names to be <= %d, got %d and %d", maxNodeJobNameLength, len(gotA), len(gotB))
	}
}
