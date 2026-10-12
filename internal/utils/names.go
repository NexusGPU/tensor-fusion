package utils

import (
	"crypto/sha256"
	"fmt"
	"strings"
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
	// This name is shared with v1 and is part of the in-place upgrade contract:
	// GPUPool rolling updates replace the existing pod instead of running two
	// node-level hypervisors concurrently.
	return buildV1CompatibleHypervisorPodName(nodeName)
}

func buildV1CompatibleHypervisorPodName(nodeName string) string {
	const prefix = "hypervisor"
	name := prefix + "-" + nodeName
	if len(name) <= maxHypervisorPodNameLength {
		return name
	}

	hash := shortHash(nodeName, defaultHashLength)
	maxNodeLen := maxHypervisorPodNameLength - len(prefix) - len(hash) - 2
	truncated := nodeName
	if len(truncated) > maxNodeLen {
		truncated = truncated[:maxNodeLen]
	}
	truncated = strings.TrimRight(truncated, "-")
	if truncated == "" {
		return prefix + "-" + hash
	}
	return fmt.Sprintf("%s-%s-%s", prefix, truncated, hash)
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

const (
	// MaxLabelValueLength is the API server's limit for a label value, in bytes.
	MaxLabelValueLength = 63
	// labelValueHashLength is the number of hex chars that keep a truncated
	// label value unique.
	labelValueHashLength = 10
)

// LabelSafeValue maps an object name to a value the API server accepts as a
// label value. Object names (nodes in particular) may be up to 253 characters,
// a label value at most 63.
//
// A name that fits is returned unchanged, so every object written before this
// function existed keeps its label and stays selectable. A longer name becomes
// "<first characters>-<10 hex of its SHA-256>", exactly as long as allowed at
// most. Writers and selectors must both go through this function; code that
// needs the name itself has to read it from somewhere else (for a GPU: its
// GPUNode owner reference, see GPUNodeNameOfGPU), never from the label.
func LabelSafeValue(name string) string {
	if len(name) <= MaxLabelValueLength {
		return name
	}
	hash := shortHash(name, labelValueHashLength)
	// A label value starts and ends with an alphanumeric character; the hash
	// takes care of the end, the name's own first character of the start.
	prefix := strings.TrimRight(name[:MaxLabelValueLength-labelValueHashLength-1], "-_.")
	if prefix == "" {
		return hash
	}
	return prefix + "-" + hash
}
