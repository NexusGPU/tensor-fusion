/*
Copyright 2024.

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

package v1

import (
	"context"
	"fmt"
	"slices"

	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// ShouldAutoMigrateGPUPod determines if a Pod should be automatically migrated to TensorFusion
// based on the auto migration configuration.
func ShouldAutoMigrateGPUPod(ctx context.Context, c client.Client, pod *corev1.Pod) (bool, error) {

	// Skip migration if pod explicitly disables TensorFusion via tensor-fusion.ai/enabled=false label
	if utils.IsTensorFusionPodDisabled(pod) {
		return false, nil
	}

	globalConfig := config.GetGlobalConfig()
	if globalConfig == nil || globalConfig.AutoMigration == nil {
		return false, nil
	}

	autoMigration := globalConfig.AutoMigration
	if !autoMigration.Enable {
		// When enable=false, auto migration is disabled
		// Fall back to previous logic (IsProgressiveMigration check)
		return false, nil
	}

	// When enable=true, check scope rules
	scope := autoMigration.Scope
	if scope == nil {
		// No scope means migrate all GPU Pods
		return true, nil
	}

	// Check if Pod matches exclude rules first
	if scope.Excludes != nil {
		excluded, err := matchesRules(ctx, c, pod, scope.Excludes)
		if err != nil {
			return false, fmt.Errorf("failed to check exclude rules: %w", err)
		}
		if excluded {
			return false, nil
		}
	}

	// Check include rules
	if scope.Includes != nil {
		included, err := matchesRules(ctx, c, pod, scope.Includes)
		if err != nil {
			return false, fmt.Errorf("failed to check include rules: %w", err)
		}
		if !included {
			return false, nil
		}
	}
	return true, nil
}

// matchesRules checks if a Pod matches the given auto migration rules
func matchesRules(ctx context.Context, c client.Client, pod *corev1.Pod, rules *config.AutoMigrationRules) (bool, error) {
	// Check namespace names
	if len(rules.NamespaceNames) > 0 {
		if slices.Contains(rules.NamespaceNames, pod.Namespace) {
			return true, nil
		}
	}

	// Check namespace selector
	if rules.NamespaceSelector != nil {
		namespace := &corev1.Namespace{}
		if err := c.Get(ctx, client.ObjectKey{Name: pod.Namespace}, namespace); err != nil {
			return false, fmt.Errorf("failed to get namespace %s: %w", pod.Namespace, err)
		}
		selector, err := metav1.LabelSelectorAsSelector(rules.NamespaceSelector)
		if err != nil {
			return false, fmt.Errorf("invalid namespace selector: %w", err)
		}
		if selector.Matches(labels.Set(namespace.Labels)) {
			return true, nil
		}
	}

	// Check pod selector
	if rules.PodSelector != nil {
		selector, err := metav1.LabelSelectorAsSelector(rules.PodSelector)
		if err != nil {
			return false, fmt.Errorf("invalid pod selector: %w", err)
		}
		if selector.Matches(labels.Set(pod.Labels)) {
			return true, nil
		}
	}
	return false, nil
}
