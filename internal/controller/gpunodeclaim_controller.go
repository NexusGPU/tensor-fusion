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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	log "sigs.k8s.io/controller-runtime/pkg/log"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

// GPUNodeClaimReconciler reconciles a GPUNodeClaim object
type GPUNodeClaimReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpunodeclaims,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpunodeclaims/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpunodeclaims/finalizers,verbs=update

// GPUNodeClaim is responsible for creating cloud vendor GPU nodes
func (r *GPUNodeClaimReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Reconciling GPUNodeClaim", "name", req.Name)

	claim := &tfv1.GPUNodeClaim{}
	if err := r.Get(ctx, req.NamespacedName, claim); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	clusterName := claim.GetLabels()[constants.LabelKeyClusterOwner]
	cluster := &tfv1.TensorFusionCluster{}
	if err := r.Get(ctx, client.ObjectKey{Name: clusterName}, cluster); err != nil {
		if errors.IsNotFound(err) {
			r.Recorder.Eventf(claim, corev1.EventTypeWarning, "OrphanedNode", "provisioned node not found, this could result in orphaned nodes, please check manually: %s", claim.Name)
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	vendorCfg := cluster.Spec.ComputingVendor
	if vendorCfg == nil {
		return ctrl.Result{}, fmt.Errorf("failed to get computing vendor config for cluster %s", clusterName)
	}

	poolName := claim.GetLabels()[constants.LabelKeyOwner]
	pool := &tfv1.GPUPool{}
	if err := r.Get(ctx, client.ObjectKey{Name: poolName}, pool); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	shouldReturn, err := utils.HandleFinalizer(ctx, claim, r.Client, func(ctx context.Context, claim *tfv1.GPUNodeClaim) (bool, error) {
		nodeList := &corev1.NodeList{}
		if err := r.List(ctx, nodeList, client.MatchingLabels{constants.ProvisionerLabelKey: claim.Name}); err != nil {
			if errors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		}
		if len(nodeList.Items) > 0 {
			for _, node := range nodeList.Items {
				if !node.DeletionTimestamp.IsZero() {
					continue
				}
				// TODO: in karpenter mode, delete the NodeClaim
				// in direct provisioning mode, call terminate instance and then delete k8s node
				err := r.Delete(ctx, &node)
				if err != nil {
					return false, err
				}
			}
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return ctrl.Result{}, err
	}
	if shouldReturn {
		return ctrl.Result{}, nil
	}

	// create cloud vendor node
	if err := r.reconcileCloudVendorNode(ctx, claim, pool); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *GPUNodeClaimReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.GPUNodeClaim{}).
		Named("gpunodeclaim").
		Complete(r)
}

func (r *GPUNodeClaimReconciler) reconcileCloudVendorNode(ctx context.Context, claim *tfv1.GPUNodeClaim, pool *tfv1.GPUPool) error {
	// No NodeInfo, should create new one
	provider, _, err := createProvisionerAndQueryCluster(ctx, pool, r.Client)
	if err != nil {
		return err
	}

	// TODO: query cloud vendor by node name
	status, err := provider.CreateNode(ctx, &claim.Spec)
	if err != nil {
		return err
	}

	// TODO: fix me, GPUNode is not created yet, node info should be stored in GPUNodeClaim status and sync in gpunode controller later !

	// Update GPUNode status about the cloud vendor info
	// To match GPUNode - K8S node, the --node-label in Kubelet is MUST-have, like Karpenter, it force set userdata to add a provisionerId label, k8s node controller then can set its ownerReference to the GPUNode
	gpuNode := &tfv1.GPUNode{}
	err = r.Get(ctx, client.ObjectKey{Name: claim.Spec.NodeName}, gpuNode)
	if err != nil {
		return err
	}
	gpuNode.Status.Phase = tfv1.TensorFusionGPUNodePhasePending
	gpuNode.Status.NodeInfo.IP = status.PrivateIP
	gpuNode.Status.NodeInfo.InstanceID = status.InstanceID
	gpuNode.Status.NodeInfo.Region = claim.Spec.Region

	// Retry status update until success to handle version conflicts
	err = retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		// Get the latest version before attempting an update
		latest := &tfv1.GPUNode{}
		if err := r.Get(ctx, client.ObjectKey{Name: gpuNode.Name}, latest); err != nil {
			return err
		}

		// Apply our status updates to the latest version
		latest.Status.Phase = tfv1.TensorFusionGPUNodePhasePending
		latest.Status.NodeInfo.IP = status.PrivateIP
		latest.Status.NodeInfo.InstanceID = status.InstanceID
		latest.Status.NodeInfo.Region = claim.Spec.Region

		// Attempt to update with the latest version
		return r.Client.Status().Update(ctx, latest)
	})

	if err != nil {
		log.FromContext(ctx).Error(err, "Failed to update GPUNode status after retries, must terminate node to keep operation atomic", "name", claim.Spec.NodeName)
		errTerminate := provider.TerminateNode(ctx, &types.NodeIdentityParam{
			InstanceID: status.InstanceID,
			Region:     claim.Spec.Region,
		})
		if errTerminate != nil {
			log.FromContext(ctx).Error(errTerminate, "Failed to terminate cloud vendor node when GPUNode status failed to update")
			panic(errTerminate)
		}
		return nil
	}

	r.Recorder.Eventf(pool, corev1.EventTypeNormal, "ManagedNodeCreated", "Created node: %s, IP: %s", status.InstanceID, status.PrivateIP)
	return nil
}
