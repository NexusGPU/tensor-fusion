package utils

import (
	"crypto/sha256"
	"fmt"
	"strings"
)

const (
	maxHypervisorPodNameLength = 63
	nodeScopedNameHashLength   = 8
)

// BuildHypervisorPodName returns a stable pod name for a node-scoped hypervisor.
// Short names are preserved as-is, while long names are truncated and suffixed
// with a short hash to remain unique and DNS-label safe.
func BuildHypervisorPodName(nodeName string) string {
	return BuildNodeScopedName("hypervisor", nodeName, maxHypervisorPodNameLength, nodeScopedNameHashLength)
}

// BuildNodeScopedName returns a stable DNS-label-safe name for resources that
// are keyed by a node name and prefixed by a component identifier.
func BuildNodeScopedName(prefix, nodeName string, maxLength, hashLength int) string {
	name := fmt.Sprintf("%s-%s", prefix, nodeName)
	if len(name) <= maxLength {
		return name
	}

	hash := buildNodeScopedNameHash(nodeName)
	if len(hash) > hashLength {
		hash = hash[:hashLength]
	}

	availableNodeNameLen := maxLength - len(prefix) - len(hash) - 2
	if availableNodeNameLen <= 0 {
		return fmt.Sprintf("%s-%s", prefix, hash)
	}

	truncatedNodeName := nodeName
	if len(truncatedNodeName) > availableNodeNameLen {
		truncatedNodeName = truncatedNodeName[:availableNodeNameLen]
	}
	truncatedNodeName = strings.TrimRight(truncatedNodeName, "-")
	if truncatedNodeName == "" {
		return fmt.Sprintf("%s-%s", prefix, hash)
	}

	return fmt.Sprintf("%s-%s-%s", prefix, truncatedNodeName, hash)
}

func buildNodeScopedNameHash(name string) string {
	sum := sha256.Sum256([]byte(name))
	return fmt.Sprintf("%x", sum[:])
}
