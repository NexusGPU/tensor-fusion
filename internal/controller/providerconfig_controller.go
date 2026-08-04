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

package controller

import (
	"context"
	"fmt"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/util/retry"
	schedulingcorev1 "k8s.io/component-helpers/scheduling/corev1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// ProviderConfigReconciler reconciles a ProviderConfig object
type ProviderConfigReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	ProviderManager *provider.Manager
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=providerconfigs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpunodes,verbs=get;list;watch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpupools,verbs=get;list;watch;patch;update
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch

// Reconcile handles ProviderConfig create/update/delete events
func (r *ProviderConfigReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var providerConfig tfv1.ProviderConfig
	if err := r.Get(ctx, req.NamespacedName, &providerConfig); err != nil {
		if errors.IsNotFound(err) {
			vendor := r.ProviderManager.DeleteProviderByName(req.Name)
			gpuInfos := r.ProviderManager.GetAllGpuInfos()
			gpuallocator.LoadPartitionTemplatesFromConfig(gpuInfos)
			if vendor != "" {
				if err := r.publishProviderRevision(ctx, vendor, ""); err != nil {
					return ctrl.Result{}, err
				}
			}
			logger.Info("ProviderConfig deleted, caches cleaned", "name", req.Name)
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Always refresh the in-memory caches: after an operator restart the
	// ProviderManager is empty and must be repopulated from every ProviderConfig.
	r.ProviderManager.UpdateProvider(&providerConfig)
	gpuInfos := r.ProviderManager.GetAllGpuInfos()
	gpuallocator.LoadPartitionTemplatesFromConfig(gpuInfos)
	logger.Info("partition templates refreshed", "gpuInfoCount", len(gpuInfos))
	logger.Info("ProviderConfig synced", "vendor", providerConfig.Spec.Vendor)

	// Publish the revision instead of deleting pods directly. GPUPool owns the
	// rollout state machine (batch size, interval and progress), while per-node
	// hashes limit recreation to nodes that use this vendor.
	newHash := utils.GetObjectHash(providerConfig.Spec)
	oldHash := providerConfig.Annotations[constants.ProviderConfigSpecHashAnnotation]

	if oldHash != newHash {
		if err := r.publishProviderRevision(ctx, providerConfig.Spec.Vendor, newHash); err != nil {
			logger.Error(err, "failed to publish ProviderConfig revision", "vendor", providerConfig.Spec.Vendor)
			return ctrl.Result{}, err
		}
		patch := client.MergeFrom(providerConfig.DeepCopy())
		if providerConfig.Annotations == nil {
			providerConfig.Annotations = map[string]string{}
		}
		providerConfig.Annotations[constants.ProviderConfigSpecHashAnnotation] = newHash
		if err := r.Patch(ctx, &providerConfig, patch); err != nil {
			return ctrl.Result{}, fmt.Errorf("persist provider-config spec hash: %w", err)
		}
	}

	return ctrl.Result{}, nil
}

func (r *ProviderConfigReconciler) publishProviderRevision(ctx context.Context, vendor, revision string) error {
	logger := log.FromContext(ctx)
	if vendor == "" {
		return nil
	}

	poolNames := map[string]struct{}{}
	var poolList tfv1.GPUPoolList
	if err := r.List(ctx, &poolList); err != nil {
		return fmt.Errorf("list GPU pools for ProviderConfig rollout: %w", err)
	}
	for i := range poolList.Items {
		pool := &poolList.Items[i]
		if poolMayUseVendor(pool, vendor) {
			poolNames[pool.Name] = struct{}{}
		}
	}

	// GPUNode labels are the observed vendor source of truth and also cover
	// manually-created or temporarily inconsistent pool configurations.
	var nodeList tfv1.GPUNodeList
	if err := r.List(ctx, &nodeList); err != nil {
		return fmt.Errorf("list GPU nodes for ProviderConfig rollout: %w", err)
	}
	for i := range nodeList.Items {
		node := &nodeList.Items[i]
		nodeVendor, err := r.resolveNodeVendor(ctx, node)
		if err != nil {
			logger.Error(err, "failed to resolve node vendor while publishing revision", "node", node.Name)
			continue
		}
		if strings.EqualFold(nodeVendor, vendor) {
			if poolName := utils.ExtractPoolNameFromNodeLabel(node); poolName != "" {
				poolNames[poolName] = struct{}{}
			}
		}
	}

	for poolName := range poolNames {
		if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
			pool := &tfv1.GPUPool{}
			if err := r.Get(ctx, client.ObjectKey{Name: poolName}, pool); err != nil {
				return err
			}
			before := client.MergeFrom(pool.DeepCopy())
			if err := utils.SetProviderConfigRevision(pool, vendor, revision); err != nil {
				return err
			}
			return r.Patch(ctx, pool, before)
		}); err != nil {
			return fmt.Errorf("publish ProviderConfig revision to pool %s: %w", poolName, err)
		}
		logger.Info("published ProviderConfig revision for rolling update",
			"pool", poolName, "vendor", vendor, "revision", revision)
	}

	return nil
}

func poolMayUseVendor(pool *tfv1.GPUPool, vendor string) bool {
	if pool == nil || pool.Spec.NodeManagerConfig == nil {
		return strings.EqualFold(vendor, constants.AcceleratorVendorNvidia)
	}
	cfg := pool.Spec.NodeManagerConfig
	for configuredVendor := range cfg.MultiVendorNodeSelector {
		if strings.EqualFold(configuredVendor, vendor) {
			return true
		}
	}
	if len(cfg.MultiVendorNodeSelector) > 0 {
		return false
	}
	if cfg.DefaultVendor != "" {
		return strings.EqualFold(cfg.DefaultVendor, vendor)
	}
	return strings.EqualFold(vendor, constants.AcceleratorVendorNvidia)
}

func (r *ProviderConfigReconciler) resolveNodeVendor(ctx context.Context, node *tfv1.GPUNode) (string, error) {
	if node.Labels != nil {
		if vendor := node.Labels[constants.AcceleratorLabelVendor]; vendor != "" {
			return vendor, nil
		}
	}

	poolName := utils.ExtractPoolNameFromNodeLabel(node)
	if poolName == "" {
		return "", fmt.Errorf("missing pool label for node %s", node.Name)
	}

	pool := &tfv1.GPUPool{}
	if err := r.Get(ctx, client.ObjectKey{Name: poolName}, pool); err != nil {
		return "", fmt.Errorf("failed to get pool %s: %w", poolName, err)
	}

	cfg := pool.Spec.NodeManagerConfig
	if cfg == nil {
		return constants.AcceleratorVendorNvidia, nil
	}

	if len(cfg.MultiVendorNodeSelector) == 0 && cfg.NodeSelector == nil {
		if cfg.DefaultVendor != "" {
			return cfg.DefaultVendor, nil
		}
		return constants.AcceleratorVendorNvidia, nil
	}

	k8sNode := &corev1.Node{}
	if err := r.Get(ctx, client.ObjectKey{Name: node.Name}, k8sNode); err != nil {
		return "", fmt.Errorf("failed to get k8s node %s: %w", node.Name, err)
	}

	return matchVendorFromNode(k8sNode, cfg)
}

func matchVendorFromNode(node *corev1.Node, nodeManagerConfig *tfv1.NodeManagerConfig) (string, error) {
	if nodeManagerConfig == nil {
		return constants.AcceleratorVendorNvidia, nil
	}

	if len(nodeManagerConfig.MultiVendorNodeSelector) > 0 {
		for vendor, nodeSelector := range nodeManagerConfig.MultiVendorNodeSelector {
			if nodeSelector == nil {
				continue
			}
			matches, err := schedulingcorev1.MatchNodeSelectorTerms(node, nodeSelector)
			if err != nil {
				return "", err
			}
			if matches {
				return vendor, nil
			}
		}
		return "", fmt.Errorf("no vendor matched in MultiVendorNodeSelector")
	}

	if nodeManagerConfig.DefaultVendor != "" {
		return nodeManagerConfig.DefaultVendor, nil
	}

	return constants.AcceleratorVendorNvidia, nil
}

// SetupWithManager sets up the controller with the Manager
func (r *ProviderConfigReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.ProviderConfig{}).
		Named("provider-config").
		Complete(r)
}
