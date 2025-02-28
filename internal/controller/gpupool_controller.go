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
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
	utils "github.com/NexusGPU/tensor-fusion-operator/internal/utils"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// GPUPoolReconciler reconciles a GPUPool object
type GPUPoolReconciler struct {
	client.Client

	LastProcessedItems sync.Map

	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpupools,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpupools/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpupools/finalizers,verbs=update

// Reconcile GPU pools
func (r *GPUPoolReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	runNow, alreadyQueued, waitTime := utils.DebouncedReconcileCheck(ctx, &r.LastProcessedItems, req.NamespacedName)
	if alreadyQueued {
		return ctrl.Result{}, nil
	}
	if !runNow {
		return ctrl.Result{RequeueAfter: waitTime}, nil
	}

	log.Info("Reconciling GPUPool", "name", req.NamespacedName.Name)
	defer func() {
		log.Info("Finished reconciling GPUPool", "name", req.NamespacedName.Name)
	}()

	pool := &tfv1.GPUPool{}
	if err := r.Get(ctx, req.NamespacedName, pool); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// TODO: if phase is destroying, stop all existing workers and hypervisors, stop time series flow aggregations
	deleted, err := utils.HandleFinalizer(ctx, pool, r.Client, func(ctx context.Context, pool *tfv1.GPUPool) (bool, error) {
		if pool.Status.Phase != tfv1.TensorFusionPoolPhaseDestroying {
			pool.Status.Phase = tfv1.TensorFusionPoolPhaseDestroying
			if err := r.Status().Update(ctx, pool); err != nil {
				return false, err
			}
		}
		// check if all nodes has been deleted
		nodes := &tfv1.GPUNodeList{}
		if err := r.Client.List(ctx, nodes, client.MatchingLabels{constants.LabelKeyOwner: pool.Name}); err != nil {
			return false, err
		}
		return len(nodes.Items) == 0, nil
	})
	if err != nil {
		return ctrl.Result{}, err
	}
	if deleted {
		return ctrl.Result{RequeueAfter: constants.PendingRequeueDuration}, nil
	}

	if err := r.reconcilePoolCurrentCapacityAndReadiness(ctx, pool); err != nil {
		return ctrl.Result{}, err
	}

	// For provisioning mode, check if need to scale up GPUNodes upon AvailableCapacity changed
	isProvisioningMode := pool.Spec.NodeManagerConfig.ProvisioningMode == tfv1.ProvisioningModeProvisioned

	// Provisioning mode, check capacity and scale up if needed
	if isProvisioningMode {
		newNodeCreated, err := r.reconcilePoolCapacityWithProvisioner(ctx, pool)
		if err != nil {
			return ctrl.Result{}, err
		}
		// Set phase to updating and let GPUNode event trigger the check and update capacity loop, util all nodes are ready
		if newNodeCreated {
			// Refresh the capacity again since new node has been created
			pool.Status.Phase = tfv1.TensorFusionPoolPhaseUpdating
			if err := r.Status().Patch(ctx, pool, client.Merge); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{RequeueAfter: constants.StatusCheckInterval}, nil
		}
	}

	// TODO, when componentConfig changed, it should notify corresponding resource to upgrade
	// eg. when hypervisor changed, should change all owned GPUNode's status.phase to Updating

	return ctrl.Result{}, nil
}

func (r *GPUPoolReconciler) reconcilePoolCurrentCapacityAndReadiness(ctx context.Context, pool *tfv1.GPUPool) error {
	log := log.FromContext(ctx)

	nodes := &tfv1.GPUNodeList{}
	if err := r.Client.List(ctx, nodes, client.MatchingLabels{constants.LabelKeyOwner: pool.Name}); err != nil {
		return fmt.Errorf("list nodes of Pool %s failed: %v", pool.Name, err)
	}

	log.Info("Calculate current capacity and readiness for pool", "name", pool.Name)

	totalGPUs := int32(0)
	readyNodes := 0
	totalVRAM := resource.Quantity{}
	virtualVRAM := resource.Quantity{}
	totalTFlops := resource.Quantity{}
	virtualTFlops := resource.Quantity{}

	availableVRAM := resource.Quantity{}
	availableTFlops := resource.Quantity{}

	for _, node := range nodes.Items {
		totalGPUs = totalGPUs + node.Status.TotalGPUs
		totalVRAM.Add(node.Status.TotalVRAM)
		totalTFlops.Add(node.Status.TotalTFlops)
		if node.Status.Phase == tfv1.TensorFusionGPUNodePhaseRunning {
			readyNodes++
		}
		virtualVRAM.Add(node.Status.VirtualVRAM)
		virtualTFlops.Add(node.Status.VirtualTFlops)

		availableVRAM.Add(node.Status.AvailableVRAM)
		availableTFlops.Add(node.Status.AvailableTFlops)
	}

	pool.Status.TotalGPUs = totalGPUs
	pool.Status.TotalNodes = int32(len(nodes.Items))
	pool.Status.TotalVRAM = totalVRAM
	pool.Status.TotalTFlops = totalTFlops
	pool.Status.AvailableTFlops = availableTFlops
	pool.Status.AvailableVRAM = availableVRAM

	pool.Status.ReadyNodes = int32(readyNodes)
	pool.Status.NotReadyNodes = int32(len(nodes.Items)) - pool.Status.ReadyNodes

	pool.Status.VirtualTFlops = virtualTFlops
	pool.Status.VirtualVRAM = virtualVRAM

	allowScaleToZero := true
	if pool.Spec.CapacityConfig != nil && pool.Spec.CapacityConfig.MinResources != nil {
		minTFlops, _ := pool.Spec.CapacityConfig.MinResources.TFlops.AsInt64()
		minVRAM, _ := pool.Spec.CapacityConfig.MinResources.VRAM.AsInt64()

		allowScaleToZero = minTFlops == 0 && minVRAM == 0
	}

	allNodesReady := readyNodes == len(nodes.Items)
	if allNodesReady && readyNodes == 0 {
		if !allowScaleToZero {
			allNodesReady = false
		}
	}

	if allNodesReady {
		pool.Status.Phase = tfv1.TensorFusionPoolPhaseRunning
		log.Info("Pool is running, all nodes are ready", "name", pool.Name, "nodes", len(nodes.Items))
	} else {
		// set back to updating, wait GPUNode change triggering the pool change
		pool.Status.Phase = tfv1.TensorFusionPoolPhaseUpdating
	}

	if err := r.Client.Status().Update(ctx, pool); err != nil {
		return fmt.Errorf("update pool status: %w", err)
	}
	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *GPUPoolReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.GPUPool{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Named("gpupool").
		Owns(&tfv1.GPUNode{}).
		Complete(r)
}
