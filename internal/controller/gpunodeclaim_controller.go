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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	karpv1 "sigs.k8s.io/karpenter/pkg/apis/v1"
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
	poolName := claim.GetLabels()[constants.LabelKeyOwner]
	pool := &tfv1.GPUPool{}
	if err := r.Get(ctx, client.ObjectKey{Name: poolName}, pool); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	provider, cluster, err := createProvisionerAndQueryCluster(ctx, pool, r.Client)
	vendorCfg := cluster.Spec.ComputingVendor
	if vendorCfg == nil {
		return ctrl.Result{}, fmt.Errorf("failed to get computing vendor config for cluster %s", cluster.Name)
	}

	provisioningMode := pool.Spec.NodeManagerConfig.ProvisioningMode
	if provisioningMode == tfv1.ProvisioningModeAutoSelect {
		log.Info("AutoSelect mode, skip provision node", "name", claim.Name)
		return ctrl.Result{}, nil
	}
	isKarpenterMode := provisioningMode == tfv1.ProvisioningModeKarpenter

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
				if isKarpenterMode {
					// karpenter mode, delete the NodeClaim
					kClaim := &karpv1.NodeClaim{
						ObjectMeta: metav1.ObjectMeta{
							Name: claim.Spec.NodeName,
						},
					}
					log.Info("Deleting karpenter node claim", "name", claim.Spec.NodeName)
					err := r.Delete(ctx, kClaim)
					if err != nil {
						return false, err
					}
				} else {
					// TODO: should be async, return deletion request ID and check deletion status by request ID
					log.Info("Deleting cloud vendor node", "instanceID", claim.Status.InstanceID, "region", claim.Spec.Region)
					err = provider.TerminateNode(ctx, &types.NodeIdentityParam{
						InstanceID: claim.Status.InstanceID,
						Region:     claim.Spec.Region,
					})
					if err != nil {
						return false, err
					}
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

	// create or check cloud vendor node
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

	if claim.Status.InstanceID == "" {
		// TODO: should be async with request ID
		status, err := provider.CreateNode(ctx, &claim.Spec)
		if err != nil {
			return err
		}
		r.Recorder.Eventf(pool, corev1.EventTypeNormal, "ManagedNodeCreated", "Created node: %s, IP: %s", status.InstanceID, status.PrivateIP)
		// Retry status update until success to handle version conflicts
		if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
			// Get the latest version before attempting an update
			latest := &tfv1.GPUNodeClaim{}
			if err := r.Get(ctx, client.ObjectKey{Name: claim.Name}, latest); err != nil {
				return err
			}
			// Apply our status updates to the latest version
			latest.Status.InstanceID = status.InstanceID
			latest.Status.Phase = tfv1.GPUNodeClaimBound

			// Attempt to update with the latest version
			return r.Status().Update(ctx, latest)
		}); err != nil {
			return err
		}
	}
	return nil
}
