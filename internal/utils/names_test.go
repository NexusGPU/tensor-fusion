package utils

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBuildHypervisorPodName(t *testing.T) {
	t.Run("short name preserved", func(t *testing.T) {
		name := BuildHypervisorPodName("gpu-node-01")
		assert.Equal(t, "hypervisor-gpu-node-01", name)
	})

	t.Run("long name truncated within limit", func(t *testing.T) {
		longNode := strings.Repeat("a", 200)
		name := BuildHypervisorPodName(longNode)
		assert.LessOrEqual(t, len(name), 63)
		assert.True(t, strings.HasPrefix(name, "hypervisor-"))
	})

	t.Run("different long names produce different results", func(t *testing.T) {
		node1 := strings.Repeat("a", 200)
		node2 := strings.Repeat("a", 199) + "b"
		name1 := BuildHypervisorPodName(node1)
		name2 := BuildHypervisorPodName(node2)
		assert.NotEqual(t, name1, name2)
	})

	t.Run("boundary length name", func(t *testing.T) {
		// "hypervisor-" is 11 chars, so 52 char node name = 63 total
		node := strings.Repeat("x", 52)
		name := BuildHypervisorPodName(node)
		assert.Equal(t, "hypervisor-"+node, name)
		assert.Equal(t, 63, len(name))
	})

	t.Run("one over boundary triggers truncation", func(t *testing.T) {
		node := strings.Repeat("x", 53)
		name := BuildHypervisorPodName(node)
		assert.LessOrEqual(t, len(name), 63)
		assert.NotEqual(t, "hypervisor-"+node, name)
	})

	t.Run("v1 compatibility trims hyphen at truncation boundary", func(t *testing.T) {
		// v1 leaves 43 characters for the truncated node segment. Make the
		// boundary character a hyphen to cover its TrimRight behavior.
		node := strings.Repeat("x", 42) + "-" + strings.Repeat("y", 30)
		name := BuildHypervisorPodName(node)
		expected := fmt.Sprintf("hypervisor-%s-%s", strings.Repeat("x", 42), shortHash(node, 8))
		assert.Equal(t, expected, name)
		assert.NotContains(t, name, "--")
	})
}

func TestBuildNodeScopedName(t *testing.T) {
	t.Run("custom prefix and max length", func(t *testing.T) {
		name := BuildNodeScopedName("driver-probe", "my-node", 63, 8)
		assert.Equal(t, "driver-probe-my-node", name)
	})

	t.Run("truncation with custom params", func(t *testing.T) {
		longNode := strings.Repeat("n", 100)
		name := BuildNodeScopedName("dp", longNode, 30, 6)
		assert.LessOrEqual(t, len(name), 30)
		assert.True(t, strings.HasPrefix(name, "dp-"))
	})
}
