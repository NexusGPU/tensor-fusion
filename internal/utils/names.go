package utils

import (
	"crypto/sha256"
	"fmt"
)

const (
	// maxHypervisorPodNameLength is the DNS-label safe limit for pod names.
	maxHypervisorPodNameLength = 63
	// defaultHashLength is the number of hex chars appended when truncating.
	defaultHashLength = 8
)

// BuildHypervisorPodName returns a stable pod name for a node-scoped hypervisor.
// Short names are preserved as-is, while long names are truncated and suffixed
// with a short hash to remain unique and DNS-label safe.
func BuildHypervisorPodName(nodeName string) string {
	return BuildNodeScopedName("tf-hypervisor", nodeName, maxHypervisorPodNameLength, defaultHashLength)
}

// BuildNodeScopedName constructs a name in the form "prefix-nodeName", ensuring
// the result does not exceed maxLen characters. When truncation is needed, a
// hash suffix derived from the full nodeName is appended to preserve uniqueness.
func BuildNodeScopedName(prefix, nodeName string, maxLen, hashLen int) string {
	name := prefix + "-" + nodeName
	if len(name) <= maxLen {
		return name
	}

	hash := shortHash(nodeName, hashLen)
	// prefix + "-" + truncatedNode + "-" + hash
	maxNodeLen := maxLen - len(prefix) - 1 - 1 - hashLen
	if maxNodeLen < 0 {
		maxNodeLen = 0
	}
	truncated := nodeName
	if len(truncated) > maxNodeLen {
		truncated = truncated[:maxNodeLen]
	}
	return fmt.Sprintf("%s-%s-%s", prefix, truncated, hash)
}

func shortHash(s string, length int) string {
	h := sha256.Sum256([]byte(s))
	hex := fmt.Sprintf("%x", h)
	if len(hex) > length {
		return hex[:length]
	}
	return hex
}
