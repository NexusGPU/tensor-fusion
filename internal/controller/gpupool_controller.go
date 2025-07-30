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
	"sort"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/component"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	utils "github.com/NexusGPU/tensor-fusion/internal/utils"
	"golang.org/x/time/rate"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

var (
	// Killer switch to avoid creating too much cloud vendor nodes
	// Controlled by /api/provision?enable=true/false
	ProvisioningToggle = true

	// creating nodes, next round capacity check should consider the assumed resources
	// map key is pool name, second level is GPUClaim name
	PendingGPUNodeClaim map[string]map[string]tfv1.Resource

	// deleting nodes, must be serialized, delete one round by one round
	// map key is pool name, value is GPUNode name list
	PendingDeletionGPUNodes map[string][]string

	// TODO add metrics for node provisioning and compaction

	// When add/remove pending provisioning or deleting nodes, lock the memory
	pendingGPUNodeStateLock sync.Mutex
)

func init() {
	PendingGPUNodeClaim = make(map[string]map[string]tfv1.Resource)
	PendingDeletionGPUNodes = make(map[string][]string)
}

// GPUPoolReconciler reconciles a GPUPool object
type GPUPoolReconciler struct {
	client.Client

	LastProcessedItems sync.Map

	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

// First round reconcile should build correct capacity, and then check provisioning
// for each pool first reconcile, record the time + StatusCheckInterval * 2
// and requeue until current time after that, start provisioning loop
var provisioningInitializationMinTime = map[string]time.Time{}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpupools,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpupools/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpupools/finalizers,verbs=update

// Reconcile GPU pools
func (r *GPUPoolReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	log.Info("Reconciling GPUPool", "name", req.Name)
	defer func() {
		log.Info("Finished reconciling GPUPool", "name", req.Name)
	}()

	pool := &tfv1.GPUPool{}
	if err := r.Get(ctx, req.NamespacedName, pool); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// TODO: if phase is destroying, stop all existing workers and hypervisors, stop time series flow aggregations
	shouldReturn, err := utils.HandleFinalizer(ctx, pool, r.Client, func(ctx context.Context, pool *tfv1.GPUPool) (bool, error) {
		return r.cleanUpPool(ctx, pool)
	})
	if err != nil {
		return ctrl.Result{}, err
	}
	if shouldReturn || !pool.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	if err := r.reconcilePoolCurrentCapacityAndReadiness(ctx, pool); err != nil {
		return ctrl.Result{}, err
	}

	if _, ok := provisioningInitializationMinTime[pool.Name]; !ok {
		provisioningInitializationMinTime[pool.Name] = time.Now().Add(constants.StatusCheckInterval * 3)
	}

	if ctrlResult, err := r.reconcilePoolComponents(ctx, pool); err != nil {
		return ctrl.Result{}, err
	} else if ctrlResult != nil {
		return *ctrlResult, nil
	}

	// For provisioning mode, check if need to scale up GPUNodes upon AvailableCapacity changed
	isProvisioningMode := pool.Spec.NodeManagerConfig.ProvisioningMode == tfv1.ProvisioningModeProvisioned ||
		pool.Spec.NodeManagerConfig.ProvisioningMode == tfv1.ProvisioningModeKarpenter

	// Provisioning mode, check capacity and scale up if needed
	if isProvisioningMode {
		// avoid provision before first round reconcile
		if time.Now().Before(provisioningInitializationMinTime[pool.Name]) {
			return ctrl.Result{RequeueAfter: constants.StatusCheckInterval}, nil
		}
		// avoid concurrent provisioning, must wait pending nodes bound, then start next round capacity check
		newCreatedNodes, err := r.reconcilePoolCapacityWithProvisioner(ctx, pool)
		if err != nil {
			return ctrl.Result{}, err
		}
		// Set phase to updating and let GPUNode event trigger the check and update capacity loop, util all nodes are ready
		if len(newCreatedNodes) > 0 {
			pendingGPUNodeStateLock.Lock()
			for claimName := range newCreatedNodes {
				if PendingGPUNodeClaim[pool.Name] == nil {
					PendingGPUNodeClaim[pool.Name] = make(map[string]tfv1.Resource, len(newCreatedNodes)*2)
				}
				PendingGPUNodeClaim[pool.Name][claimName] = newCreatedNodes[claimName]
			}
			pendingGPUNodeStateLock.Unlock()
			// Refresh the capacity again since new node has been created
			pool.Status.ProvisioningPhase = tfv1.ProvisioningPhaseProvisioning
			if err := r.Status().Patch(ctx, pool, client.Merge); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{RequeueAfter: constants.StatusCheckInterval}, nil
		}

		if len(PendingDeletionGPUNodes[pool.Name]) > 0 {
			if err := r.reconcilePendingDeletingNodes(ctx, pool); err != nil {
				return ctrl.Result{}, err
			}
		}

		if len(PendingGPUNodeClaim[pool.Name]) > 0 {
			if err := r.reconcilePendingCreatingNodes(ctx, pool); err != nil {
				return ctrl.Result{}, err
			}
		}
	}
	return ctrl.Result{}, nil
}

func (r *GPUPoolReconciler) reconcilePendingDeletingNodes(ctx context.Context, pool *tfv1.GPUPool) error {
	log := log.FromContext(ctx)
	remainDeletingNodes := []string{}
	deletingNodes := PendingDeletionGPUNodes[pool.Name]
	for _, gpuNodeName := range deletingNodes {
		gpuNodeList := &tfv1.GPUNodeList{}
		if err := r.List(ctx, gpuNodeList, client.MatchingLabels{constants.ProvisionerLabelKey: gpuNodeName}); err != nil {
			if errors.IsNotFound(err) {
				// Already deleted, skip adding to deletingNodes, trigger pool status update
				continue
			} else {
				return err
			}
		}
		if len(gpuNodeList.Items) == 0 {
			log.Info("GPU node already deleted, skip deleting", "gpuNodeName", gpuNodeName)
			continue
		}

		gpuNodeClaim := &tfv1.GPUNodeClaim{}
		provisionedClaimName := gpuNodeList.Items[0].Labels[constants.ProvisionerLabelKey]
		if provisionedClaimName == "" {
			log.Error(nil, "invalid GPU provisioner name label for GPU node", "gpuNodeName", gpuNodeName)
			continue
		}
		if err := r.Get(ctx, client.ObjectKey{Name: provisionedClaimName}, gpuNodeClaim); err != nil {
			log.Error(err, "failed to get GPU node claim when GPUNode exists, should not happen",
				"gpuNodeName", gpuNodeName, "claimName", provisionedClaimName)
			return err
		}
		// Delete the root provisioning object GPU NodeClaim, to trigger the deletion of the GPU Node ultimately
		if gpuNodeClaim.DeletionTimestamp.IsZero() {
			if err := r.Delete(ctx, gpuNodeClaim); err != nil {
				return err
			}
			log.Info("deleted GPU node claim", "gpuNodeName", gpuNodeName, "claimName", provisionedClaimName)
		}
		remainDeletingNodes = append(remainDeletingNodes, gpuNodeName)
	}

	if len(remainDeletingNodes) != len(deletingNodes) {
		pendingGPUNodeStateLock.Lock()
		PendingDeletionGPUNodes[pool.Name] = remainDeletingNodes
		pendingGPUNodeStateLock.Unlock()
		return nil
	}
	return nil
}

func (r *GPUPoolReconciler) reconcilePendingCreatingNodes(ctx context.Context, pool *tfv1.GPUPool) error {
	latestPendingClaim := make(map[string]tfv1.Resource, len(PendingGPUNodeClaim[pool.Name]))
	completedBoundClaims := []string{}
	for claimName := range PendingGPUNodeClaim[pool.Name] {
		gpuNodeClaim := &tfv1.GPUNodeClaim{}
		if err := r.Get(ctx, client.ObjectKey{Name: claimName}, gpuNodeClaim); err != nil {
			return err
		}
		if gpuNodeClaim.Status.Phase == tfv1.GPUNodeClaimBound {
			completedBoundClaims = append(completedBoundClaims, claimName)
		} else {
			latestPendingClaim[claimName] = tfv1.Resource{
				Tflops: gpuNodeClaim.Spec.TFlopsOffered,
				Vram:   gpuNodeClaim.Spec.VRAMOffered,
			}
		}
	}
	if len(completedBoundClaims) > 0 {
		pendingGPUNodeStateLock.Lock()
		PendingGPUNodeClaim[pool.Name] = latestPendingClaim
		pendingGPUNodeStateLock.Unlock()
		log.FromContext(ctx).Info("GPU node provisioned and bound", "pool", pool.Name,
			"bound node claims", strings.Join(completedBoundClaims, ","))
	}
	return nil
}

func (r *GPUPoolReconciler) reconcilePoolCurrentCapacityAndReadiness(
	ctx context.Context, pool *tfv1.GPUPool) error {
	log := log.FromContext(ctx)
	staleStatus := pool.Status.DeepCopy()

	nodes := &tfv1.GPUNodeList{}
	if err := r.List(ctx, nodes, client.MatchingLabels{constants.LabelKeyOwner: pool.Name}); err != nil {
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
	virtualAvailableVRAM := resource.Quantity{}
	virtualAvailableTFlops := resource.Quantity{}

	runningAppsCnt := int32(0)
	deduplicationMap := make(map[string]struct{})

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

		if node.Status.VirtualAvailableVRAM != nil {
			virtualAvailableVRAM.Add(*node.Status.VirtualAvailableVRAM)
		}
		if node.Status.VirtualAvailableTFlops != nil {
			virtualAvailableTFlops.Add(*node.Status.VirtualAvailableTFlops)
		}

		for _, runningApp := range node.Status.AllocationInfo {
			workloadIdentifier := runningApp.Name + "_" + runningApp.Namespace
			if _, ok := deduplicationMap[workloadIdentifier]; !ok {
				runningAppsCnt++
				deduplicationMap[workloadIdentifier] = struct{}{}
			}
		}
	}

	pool.Status.TotalGPUs = totalGPUs
	pool.Status.TotalNodes = int32(len(nodes.Items))
	pool.Status.TotalVRAM = totalVRAM
	pool.Status.TotalTFlops = totalTFlops
	pool.Status.AvailableTFlops = availableTFlops
	pool.Status.AvailableVRAM = availableVRAM

	pool.Status.VirtualAvailableTFlops = &virtualAvailableTFlops
	pool.Status.VirtualAvailableVRAM = &virtualAvailableVRAM

	pool.Status.ReadyNodes = int32(readyNodes)
	pool.Status.NotReadyNodes = int32(len(nodes.Items)) - pool.Status.ReadyNodes

	pool.Status.VirtualTFlops = virtualTFlops
	pool.Status.VirtualVRAM = virtualVRAM

	pool.Status.RunningAppsCnt = runningAppsCnt

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

	if !equality.Semantic.DeepEqual(&pool.Status, staleStatus) {
		if err := r.Client.Status().Update(ctx, pool); err != nil {
			return fmt.Errorf("failed to update pool status: %w", err)
		}
		log.Info("updated pool status because of capacity or node status changed", "pool", pool.Name)
	}
	return nil
}

func (r *GPUPoolReconciler) reconcilePoolComponents(ctx context.Context, pool *tfv1.GPUPool) (*ctrl.Result, error) {
	if pool.Spec.ComponentConfig == nil {
		return nil, fmt.Errorf(`missing componentconfig in pool spec`)
	}

	log := log.FromContext(ctx)
	startTime := time.Now()
	log.V(6).Info("Started reconciling components", "startTime", startTime)
	defer func() {
		log.V(6).Info("Finished reconciling components", "duration", time.Since(startTime))
	}()

	components := []component.Interface{
		&component.Hypervisor{},
		&component.Worker{},
		&component.Client{},
	}

	errs := []error{}
	ctrlResults := []*ctrl.Result{}
	for _, c := range components {
		ctrlResult, err := component.ManageUpdate(r.Client, ctx, pool, c)
		if err != nil {
			errs = append(errs, err)
		}
		if ctrlResult != nil {
			ctrlResults = append(ctrlResults, ctrlResult)
		}
	}

	var ctrlResult *ctrl.Result
	if len(ctrlResults) > 0 {
		sort.Slice(ctrlResults, func(i, j int) bool {
			return ctrlResults[i].RequeueAfter < ctrlResults[j].RequeueAfter
		})
		ctrlResult = ctrlResults[0]
	}

	return ctrlResult, utilerrors.NewAggregate(errs)
}

func (r *GPUPoolReconciler) cleanUpPool(ctx context.Context, pool *tfv1.GPUPool) (bool, error) {
	log := log.FromContext(ctx)
	log.Info("TensorFusionGPUPool is being deleted", "name", pool.Name)
	if pool.Status.Phase != tfv1.TensorFusionPoolPhaseDestroying {
		pool.Status.Phase = tfv1.TensorFusionPoolPhaseDestroying
		if err := r.Status().Update(ctx, pool); err != nil {
			return false, err
		}
	}

	claims := &tfv1.GPUNodeClaimList{}
	if err := r.List(ctx, claims, client.MatchingLabels{constants.LabelKeyOwner: pool.Name}); err != nil {
		return false, err
	}
	if len(claims.Items) > 0 {
		for _, claim := range claims.Items {
			if claim.DeletionTimestamp.IsZero() {
				log.Info("Pool deleting, cleanup GPUNodeClaim", "name", claim.Name)
				if err := r.Delete(ctx, &claim); err != nil {
					return false, err
				}
			}
		}
	}

	nodes := &tfv1.GPUNodeList{}
	if err := r.List(ctx, nodes, client.MatchingLabels{constants.LabelKeyOwner: pool.Name}); err != nil {
		return false, err
	}
	return len(nodes.Items) == 0, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *GPUPoolReconciler) SetupWithManager(mgr ctrl.Manager, addLimiter bool) error {
	rateLimiterOption := controller.Options{
		RateLimiter: workqueue.NewTypedMaxOfRateLimiter(
			workqueue.NewTypedItemExponentialFailureRateLimiter[reconcile.Request](
				constants.LowFrequencyObjFailureInitialDelay,
				constants.LowFrequencyObjFailureMaxDelay,
			),
			&workqueue.TypedBucketRateLimiter[reconcile.Request]{
				Limiter: rate.NewLimiter(rate.Limit(
					constants.LowFrequencyObjFailureMaxRPS),
					constants.LowFrequencyObjFailureMaxBurst),
			},
		),
		MaxConcurrentReconciles: constants.LowFrequencyObjFailureConcurrentReconcile,
	}
	ctr := ctrl.NewControllerManagedBy(mgr)
	if addLimiter {
		ctr = ctr.WithOptions(rateLimiterOption)
	}
	return ctr.For(&tfv1.GPUPool{}).
		Named("gpupool").
		Owns(&tfv1.GPUNode{}).
		Owns(&tfv1.GPUNodeClaim{}).
		Complete(r)
}
