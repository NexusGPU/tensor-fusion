/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package utils

import (
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
)

// ResolveNodeIsolationMode returns the effective runtime isolation mode for a
// Kubernetes node. The first matching pool rule wins, followed by the pool
// default.
func ResolveNodeIsolationMode(node *corev1.Node, cfg *tfv1.NodeManagerConfig) (tfv1.IsolationModeType, error) {
	if cfg != nil {
		for i := range cfg.IsolationModeRules {
			rule := &cfg.IsolationModeRules[i]
			selector, err := metav1.LabelSelectorAsSelector(&rule.Selector)
			if err != nil {
				return "", fmt.Errorf("invalid isolation mode rule %d selector: %w", i, err)
			}
			if node != nil && selector.Matches(labels.Set(node.Labels)) {
				return rule.Mode, nil
			}
		}
		if cfg.DefaultIsolationMode != "" {
			return cfg.DefaultIsolationMode, nil
		}
	}

	return tfv1.IsolationModeSoft, nil
}

// NodeIsolationPolicyHash identifies pool-wide policy changes. The resolved
// mode is hashed separately per node so unchanged nodes are not restarted.
func NodeIsolationPolicyHash(cfg *tfv1.NodeManagerConfig) string {
	if cfg == nil {
		return GetObjectHash(tfv1.IsolationModeSoft)
	}
	return GetObjectHash(struct {
		DefaultMode tfv1.IsolationModeType
		Rules       []tfv1.NodeIsolationModeRule
	}{
		DefaultMode: effectiveDefaultIsolationMode(cfg),
		Rules:       cfg.IsolationModeRules,
	})
}

func effectiveDefaultIsolationMode(cfg *tfv1.NodeManagerConfig) tfv1.IsolationModeType {
	if cfg != nil && cfg.DefaultIsolationMode != "" {
		return cfg.DefaultIsolationMode
	}
	return tfv1.IsolationModeSoft
}
