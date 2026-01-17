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

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/pkg/util/taints"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	schedulingcorev1 "k8s.io/component-helpers/scheduling/corev1"
)

// PodReconciler reconciles a Pod object
type NodeReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=core,resources=nodes,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=nodes/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=core,resources=nodes/finalizers,verbs=create;get;patch;update

// Reconcile k8s nodes to create and update GPUNode
//
//nolint:gocyclo
func (r *NodeReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)
	node := &corev1.Node{}
	if err := r.Get(ctx, req.NamespacedName, node); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		log.Error(err, "Failed to get Node")
		return ctrl.Result{}, err
	}

	if node.Spec.Unschedulable {
		log.Info("Node is unschedulable, skip reconciling", "node", node.Name)
		return ctrl.Result{}, nil
	}

	// Remove TensorFusion taint if exists (TODO: Remove after version 1.50)
	// Skip taint removal if node is being deleted or evicted
	if node.DeletionTimestamp.IsZero() {
		taintRemoved, err := r.removeTensorFusionTaint(ctx, node)
		if err != nil {
			log.Error(err, "Failed to remove TensorFusion taint from node", "node", node.Name)
			// Continue processing even if taint removal fails
		} else if taintRemoved {
			// Re-fetch node to get latest resourceVersion after taint removal
			if err := r.Get(ctx, req.NamespacedName, node); err != nil {
				log.Error(err, "Failed to re-fetch node after taint removal", "node", node.Name)
				return ctrl.Result{}, err
			}
		}
	}

	if node.Labels == nil {
		node.Labels = map[string]string{}
	}
	provisioner := node.Labels[constants.ProvisionerLabelKey]

	// Select mode, GPU node is controlled by K8S node
	var poolList tfv1.GPUPoolList
	if err := r.List(ctx, &poolList); err != nil {
		return ctrl.Result{}, fmt.Errorf("can not list gpuPool : %w", err)
	}
	pool, matched, err := getMatchedPoolName(node, poolList.Items)
	if err != nil {
		return ctrl.Result{}, err
	}
	if !matched {
		existingGPUNode := &tfv1.GPUNode{}
		if err := r.Get(ctx, client.ObjectKey{Name: node.Name}, existingGPUNode); err != nil {
			if errors.IsNotFound(err) {
				return ctrl.Result{}, nil
			}
			return ctrl.Result{}, fmt.Errorf("can not get gpuNode(%s) : %w", node.Name, err)
		}
		// delete existing gpunode if no matched pool
		if err := r.Delete(ctx, existingGPUNode); err != nil {
			// requeue if the gpunode is not generated
			if errors.IsNotFound(err) {
				return ctrl.Result{}, nil
			}
			return ctrl.Result{}, fmt.Errorf("can not delete gpuNode(%s) : %w", node.Name, err)
		}
		return ctrl.Result{}, nil
	}

	// Skip creation if the GPUNode already exists
	gpuNode := &tfv1.GPUNode{}
	if err := r.Get(ctx, client.ObjectKey{Name: node.Name}, gpuNode); err != nil {
		if errors.IsNotFound(err) {
			gpuNode = r.generateGPUNode(node, pool, provisioner)
			if e := r.Create(ctx, gpuNode); e != nil {
				return ctrl.Result{}, fmt.Errorf("failed to create GPUNode: %w", e)
			}
		}
	}

	// If node changed to other AI accelerator hardware vendor, update gpuNode label vendor and trigger hypervisor update
	if gpuNode.Labels[constants.AcceleratorLabelVendor] != node.Labels[constants.AcceleratorLabelVendor] {
		gpuNode.Labels[constants.AcceleratorLabelVendor] = node.Labels[constants.AcceleratorLabelVendor]
		if err := r.Update(ctx, gpuNode); err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to update GPU node vendor: %w", err)
		}
	}

	if !node.DeletionTimestamp.IsZero() {
		log.Info("GPU node is being deleted, mark related GPUNode resource as destroying", "node", node.Name)
		gpuNode.Status.Phase = tfv1.TensorFusionGPUNodePhaseDestroying
		if err := r.Status().Update(ctx, gpuNode); err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to update GPU node status: %w", err)
		}
		return ctrl.Result{}, nil
	}

	// update k8s node hash
	hash := ""
	if len(pool.Spec.NodeManagerConfig.MultiVendorNodeSelector) > 0 {
		hash = utils.GetObjectHash(pool.Spec.NodeManagerConfig.MultiVendorNodeSelector)
	} else {
		hash = utils.GetObjectHash(pool.Spec.NodeManagerConfig.NodeSelector)
	}
	if node.Labels[constants.LabelNodeSelectorHash] != hash {
		if err := UpdateK8SNodeSelectorHashAndVendor(ctx, r.Client, node, hash, node.Labels[constants.AcceleratorLabelVendor]); err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to update k8s node hash: %w", err)
		}
	}

	provisioningMode := pool.Spec.NodeManagerConfig.ProvisioningMode
	isDirectManagedMode := provisioningMode == tfv1.ProvisioningModeProvisioned
	isManagedNode := isDirectManagedMode || provisioningMode == tfv1.ProvisioningModeKarpenter
	// No owner for auto select mode, and node's owner will be Karpenter NodeClaim and then GPUNodeClaim in Karpenter mode
	if isManagedNode {
		if gpuNode.Labels != nil && provisioner != "" &&
			gpuNode.Labels[constants.ProvisionerLabelKey] != provisioner {
			gpuNode.Labels[constants.ProvisionerLabelKey] = provisioner
			if err := r.Update(ctx, gpuNode); err != nil {
				return ctrl.Result{}, fmt.Errorf("failed to update owner of node: %w", err)
			}
		}

		gpuNodeClaim := &tfv1.GPUNodeClaim{}
		provisionerName := node.Labels[constants.ProvisionerLabelKey]
		if provisionerName == "" {
			// Indicates the node can not be linked to GPUNodeClaim, should be marked as existing instance
			// when GPUNodeClaim keeps Creating state, should measure and trigger warning.
			// when GPUNodeClaim not exists, it's caused by migrating existing GPUNodes
			log.Info("GPU node found but no linked to GPUNodeClaim since node-provisioner label missing", "node", node.Name)
			return ctrl.Result{}, nil
		}

		if err := r.Get(ctx, client.ObjectKey{Name: provisionerName}, gpuNodeClaim); err != nil {
			if errors.IsNotFound(err) {
				if node.Labels[constants.ProvisionerMissingLabel] == constants.TrueStringValue {
					return ctrl.Result{}, nil
				} else {
					log.Info("GPUNodeClaim is missing, Node is still there, should mark it as Orphan", "node", node.Name)
					node.Labels[constants.ProvisionerMissingLabel] = constants.TrueStringValue
					if err := r.Update(ctx, node); err != nil {
						return ctrl.Result{}, fmt.Errorf("failed to update owner of node: %w", err)
					}
				}
				return ctrl.Result{}, nil
			}
			return ctrl.Result{}, fmt.Errorf("failed to get GPUNodeClaim provisioner(%s): %w", provisionerName, err)
		}

		if isDirectManagedMode {
			// direct provisioned nodes, set owner ref to GPUNodeClaim
			// while in Karpenter mode, owner auto set by Karpenter controller, and NodeClaim is owned by GPUNodeClaim
			_ = controllerutil.SetControllerReference(gpuNodeClaim, node, r.Scheme)
			if err := r.Update(ctx, node); err != nil {
				return ctrl.Result{}, fmt.Errorf("failed to update owner of node: %w", err)
			}
		}
	}
	return ctrl.Result{}, nil
}

func (r *NodeReconciler) generateGPUNode(node *corev1.Node, pool *tfv1.GPUPool, provisioner string) *tfv1.GPUNode {
	mode := tfv1.GPUNodeManageModeAutoSelect
	if provisioner != "" {
		mode = tfv1.GPUNodeManageModeProvisioned
	}
	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: node.Name,
			Labels: map[string]string{
				constants.LabelKeyOwner: pool.Name,
				fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name): "true",
			},
		},
		Spec: tfv1.GPUNodeSpec{
			ManageMode: mode,
		},
	}
	if provisioner != "" {
		gpuNode.Labels[constants.ProvisionerLabelKey] = provisioner
	}
	// Copy vendor label from k8s node to GPUNode
	if node.Labels != nil && node.Labels[constants.AcceleratorLabelVendor] != "" {
		gpuNode.Labels[constants.AcceleratorLabelVendor] = node.Labels[constants.AcceleratorLabelVendor]
	}
	_ = controllerutil.SetControllerReference(pool, gpuNode, r.Scheme)
	return gpuNode
}

// SetupWithManager sets up the controller with the Manager.
func (r *NodeReconciler) SetupWithManager(mgr ctrl.Manager) error {
	ctr := ctrl.NewControllerManagedBy(mgr)
	// Prefer to choose an initial label selector to avoid performance impact in large Kubernetes clusters that has lots of CPU nodes
	selectors := utils.GetInitialGPUNodeSelector()
	if len(selectors) == 2 {
		p, err := predicate.LabelSelectorPredicate(metav1.LabelSelector{
			MatchLabels: map[string]string{
				selectors[0]: selectors[1],
			},
		})
		if err != nil {
			return fmt.Errorf("unable to create predicate: %w", err)
		}
		ctr.For(&corev1.Node{}, builder.WithPredicates(p))
	} else {
		ctr.For(&corev1.Node{})
	}

	return ctr.
		Named("node").
		Complete(r)
}

// Remove TensorFusion taint if exists (TODO: Remove after version 1.50)
func (r *NodeReconciler) removeTensorFusionTaint(ctx context.Context, node *corev1.Node) (bool, error) {
	taintKey := constants.NodeUsedByTaintKey
	taintValue := constants.TensorFusionSystemName

	taint := &corev1.Taint{
		Key:    taintKey,
		Effect: corev1.TaintEffectNoSchedule,
		Value:  taintValue,
	}

	// Check if taint exists first
	if !taints.TaintExists(node.Spec.Taints, taint) {
		return false, nil // No update needed, taint doesn't exist
	}

	// Use retry mechanism to handle concurrent updates
	var updated bool
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		latest := &corev1.Node{}
		if err := r.Get(ctx, client.ObjectKey{Name: node.Name}, latest); err != nil {
			return err
		}

		// Check if taint still exists
		if !taints.TaintExists(latest.Spec.Taints, taint) {
			updated = false
			return nil // Taint already removed, no update needed
		}

		var changed bool
		latest, changed, _ = taints.RemoveTaint(latest, taint)
		if !changed {
			updated = false
			return nil // No update needed, taint removal didn't change anything
		}

		if err := r.Update(ctx, latest); err != nil {
			return err
		}
		updated = true
		return nil
	}); err != nil {
		return false, err
	}

	if updated {
		log.FromContext(ctx).Info("removed TensorFusion taint from node", "node", node.Name)
	}
	return updated, nil
}

func getMatchedPoolName(node *corev1.Node, poolList []tfv1.GPUPool) (*tfv1.GPUPool, bool, error) {
	for _, pool := range poolList {
		nodeManagerConfig := pool.Spec.NodeManagerConfig
		if nodeManagerConfig == nil {
			continue
		}

		// Prioritize MultiVendorNodeSelector if it has entries
		// TODO(Finn): Regarding the sorting problem of selector, if there are more types of selector in the later stage
		// the selector selection in all code logic can be merged into one function call.
		if len(nodeManagerConfig.MultiVendorNodeSelector) > 0 {
			for _, nodeSelector := range nodeManagerConfig.MultiVendorNodeSelector {
				if nodeSelector == nil {
					continue
				}
				matches, err := schedulingcorev1.MatchNodeSelectorTerms(node, nodeSelector)
				if err != nil {
					return nil, false, err
				}
				if matches {
					return &pool, true, nil
				}
			}
			continue
		}

		// Fall back to NodeSelector
		if nodeManagerConfig.NodeSelector != nil {
			matches, err := schedulingcorev1.MatchNodeSelectorTerms(node, nodeManagerConfig.NodeSelector)
			if err != nil {
				return nil, false, err
			}
			if matches {
				return &pool, true, nil
			}
		}
	}
	return nil, false, nil
}
