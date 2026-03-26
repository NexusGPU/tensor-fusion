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
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpupools,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;delete

// Reconcile handles ProviderConfig create/update/delete events
func (r *ProviderConfigReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var providerConfig tfv1.ProviderConfig
	if err := r.Get(ctx, req.NamespacedName, &providerConfig); err != nil {
		if errors.IsNotFound(err) {
			r.ProviderManager.DeleteProviderByName(req.Name)
			gpuInfos := r.ProviderManager.GetAllGpuInfos()
			gpuallocator.LoadPartitionTemplatesFromConfig(gpuInfos)
			logger.Info("ProviderConfig deleted, caches cleaned", "name", req.Name)
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Update the provider manager cache
	r.ProviderManager.UpdateProvider(&providerConfig)
	gpuInfos := r.ProviderManager.GetAllGpuInfos()
	gpuallocator.LoadPartitionTemplatesFromConfig(gpuInfos)
	logger.Info("partition templates refreshed", "gpuInfoCount", len(gpuInfos))
	logger.Info("ProviderConfig synced", "vendor", providerConfig.Spec.Vendor)

	if err := r.restartHypervisorPodsForVendor(ctx, providerConfig.Spec.Vendor); err != nil {
		logger.Error(err, "failed to restart hypervisor pods", "vendor", providerConfig.Spec.Vendor)
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

func (r *ProviderConfigReconciler) restartHypervisorPodsForVendor(ctx context.Context, vendor string) error {
	logger := log.FromContext(ctx)
	if vendor == "" {
		return nil
	}

	var nodeList tfv1.GPUNodeList
	if err := r.List(ctx, &nodeList); err != nil {
		return fmt.Errorf("failed to list GPU nodes: %w", err)
	}

	for i := range nodeList.Items {
		node := &nodeList.Items[i]
		nodeVendor, err := r.resolveNodeVendor(ctx, node)
		if err != nil {
			logger.Error(err, "failed to resolve node vendor, skipping hypervisor restart", "node", node.Name)
			continue
		}
		if !strings.EqualFold(nodeVendor, vendor) {
			continue
		}

		key := client.ObjectKey{
			Namespace: utils.CurrentNamespace(),
			Name:      utils.BuildHypervisorPodName(node.Name),
		}
		pod := &corev1.Pod{}
		if err := r.Get(ctx, key, pod); err != nil {
			if errors.IsNotFound(err) {
				continue
			}
			logger.Error(err, "failed to get hypervisor pod", "pod", key.Name, "node", node.Name)
			continue
		}
		if err := r.Delete(ctx, pod); err != nil {
			if errors.IsNotFound(err) {
				continue
			}
			logger.Error(err, "failed to delete hypervisor pod", "pod", key.Name, "node", node.Name)
			continue
		}
		logger.Info("deleted hypervisor pod due to ProviderConfig update", "pod", key.Name, "node", node.Name, "vendor", vendor)
	}

	return nil
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
