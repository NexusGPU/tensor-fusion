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

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	controllerutil "sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	utils "github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apimachinery/pkg/api/equality"
)

// TensorFusionClusterReconciler reconciles a TensorFusionCluster object
type TensorFusionClusterReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder

	LastProcessedItems sync.Map
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionclusters/finalizers,verbs=update
// +kubebuilder:rbac:groups=core,resources=events,verbs=create;patch
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=batch,resources=cronjobs,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=namespaces;configmaps,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups=apps,resources=daemonsets;statefulsets;replicasets,verbs=get;list;watch;create;update;patch;delete

// Reconcile a TensorFusionCluster object, create and monitor GPU Pool, managing cluster level component versions
func (r *TensorFusionClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	runNow, _, waitTime := utils.DebouncedReconcileCheck(ctx, &r.LastProcessedItems, req.NamespacedName)
	// If delete event happen during wait time and no other event trigger reconciliation, it will not be deleted
	// if alreadyQueued {
	// 	return ctrl.Result{}, nil
	// }
	if !runNow {
		return ctrl.Result{RequeueAfter: waitTime}, nil
	}

	log.Info("Reconciling TensorFusionCluster", "name", req.NamespacedName.Name)
	defer func() {
		log.Info("Finished reconciling TensorFusionCluster", "name", req.NamespacedName.Name)
	}()

	// Get the TensorFusionConnection object
	tfc := &tfv1.TensorFusionCluster{}
	if err := r.Get(ctx, req.NamespacedName, tfc); err != nil {
		if errors.IsNotFound(err) {
			// Object not found, could have been deleted after reconcile request, return without error
			return ctrl.Result{}, nil
		}
		log.Error(err, "Failed to get TensorFusionCluster")
		return ctrl.Result{}, err
	}
	originalStatus := tfc.Status.DeepCopy()

	shouldReturn, err := utils.HandleFinalizer(ctx, tfc, r.Client, func(context context.Context, tfc *tfv1.TensorFusionCluster) (bool, error) {
		log.Info("TensorFusionCluster is being deleted", "name", tfc.Name)
		if tfc.Status.Phase != tfv1.TensorFusionClusterDestroying {
			tfc.Status.Phase = tfv1.TensorFusionClusterDestroying
			if err := r.Status().Update(ctx, tfc); err != nil {
				return false, err
			}
		}
		var poolList tfv1.GPUPoolList
		if err := r.List(ctx, &poolList, client.MatchingLabels{constants.LabelKeyOwner: tfc.Name}); err != nil {
			return false, err
		}
		return len(poolList.Items) == 0, nil
	})
	if err != nil {
		return ctrl.Result{}, err
	}
	if shouldReturn {
		// requeue for next loop
		// we need manually requeue cause GenerationChangedPredicate
		return ctrl.Result{Requeue: true}, nil
	}

	if tfc.Status.Phase == "" || tfc.Status.Phase == constants.PhaseUnknown {
		tfc.SetAsPending()
		if err := r.updateTFClusterStatus(ctx, tfc, originalStatus); err != nil {
			return ctrl.Result{}, err
		}
		// Next loop to make sure the custom resources are created
		return ctrl.Result{Requeue: true}, nil
	}

	// reconcile GPUPool
	poolChanged, err := r.reconcileGPUPool(ctx, tfc)
	if err != nil {
		return ctrl.Result{}, err
	}
	if poolChanged {
		tfc.SetAsUpdating()
	}

	// reconcile TSDB
	tsdbChanged, err := r.reconcileTimeSeriesDatabase(ctx, tfc)
	if err != nil {
		return ctrl.Result{}, err
	}
	if tsdbChanged {
		tfc.SetAsUpdating()
	}

	// reconcile CloudVendorConnection
	cloudConnectionChanged, err := r.reconcileCloudVendorConnection(ctx, tfc)
	if err != nil {
		return ctrl.Result{}, err
	}
	if cloudConnectionChanged {
		tfc.SetAsUpdating()
	}

	if poolChanged || tsdbChanged || cloudConnectionChanged {
		if err := r.updateTFClusterStatus(ctx, tfc, originalStatus); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{RequeueAfter: constants.PendingRequeueDuration}, nil
	}

	// when updating, check util they are ready
	// check status, if not ready, requeue after backoff delay, if all components are ready, set as ready
	if ready, conditions, err := r.checkTFClusterComponentsReady(ctx, tfc); err != nil {
		return ctrl.Result{}, err
	} else if !ready {
		// update retry count
		tfc.Status.RetryCount = tfc.Status.RetryCount + 1
		// store additional check result for each component
		tfc.SetAsUpdating(conditions...)

		if err := r.updateTFClusterStatus(ctx, tfc, originalStatus); err != nil {
			return ctrl.Result{}, err
		}
		delay := utils.CalculateExponentialBackoffWithJitter(tfc.Status.RetryCount)
		return ctrl.Result{RequeueAfter: delay}, nil
	} else {
		// all components are ready, set cluster as ready
		tfc.Status.RetryCount = 0
		tfc.SetAsReady(conditions...)
		gpupools, err := r.listOwnedGPUPools(ctx, tfc)
		if err != nil {
			return ctrl.Result{}, err
		}
		tfc.RefreshStatus(gpupools)
		if err := r.updateTFClusterStatus(ctx, tfc, originalStatus); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}
}

func (r *TensorFusionClusterReconciler) listOwnedGPUPools(ctx context.Context, tfc *tfv1.TensorFusionCluster) ([]tfv1.GPUPool, error) {
	var gpupoolsList tfv1.GPUPoolList
	err := r.List(ctx, &gpupoolsList, client.MatchingLabels(map[string]string{
		constants.LabelKeyOwner: tfc.GetName(),
	}))
	if err != nil {
		return nil, err
	}
	return gpupoolsList.Items, nil
}

func (r *TensorFusionClusterReconciler) reconcileTimeSeriesDatabase(ctx context.Context, tfc *tfv1.TensorFusionCluster) (bool, error) {
	// TODO: Not implemented yet
	return false, nil
}

func (r *TensorFusionClusterReconciler) reconcileCloudVendorConnection(ctx context.Context, tfc *tfv1.TensorFusionCluster) (bool, error) {
	if (tfc.Spec.ComputingVendor == nil) || (tfc.Spec.ComputingVendor.Type == "") {
		return false, nil
	}

	cfgHash := utils.GetObjectHash(tfc.Spec.ComputingVendor)
	if tfc.Status.CloudVendorConfigHash == "" || tfc.Status.CloudVendorConfigHash != cfgHash {
		// test the cloud vendor connection only when config changed
		provider, err := cloudprovider.GetProvider(*tfc.Spec.ComputingVendor)
		if err != nil {
			return false, err
		}
		err = (*provider).TestConnection()
		if err != nil {
			tfc.SetAsUpdating(metav1.Condition{
				Type:    constants.ConditionStatusTypeCloudVendorConnection,
				Status:  metav1.ConditionFalse,
				Message: err.Error(),
				Reason:  "CloudVendorConnectionFailed",
			})
			if errUpdateStatus := r.updateTFClusterStatus(ctx, tfc, nil); errUpdateStatus != nil {
				return true, errUpdateStatus
			}
			return true, err
		}
		tfc.Status.CloudVendorConfigHash = cfgHash
		meta.SetStatusCondition(&tfc.Status.Conditions, metav1.Condition{
			Type:   constants.ConditionStatusTypeCloudVendorConnection,
			Status: metav1.ConditionTrue,
			Reason: "CloudVendorConnectionOK",
		})
		return true, nil
	}

	return false, nil
}

func (r *TensorFusionClusterReconciler) reconcileGPUPool(ctx context.Context, tfc *tfv1.TensorFusionCluster) (bool, error) {
	// Fetch existing GPUPools that belong to this cluster
	var gpupoolsList tfv1.GPUPoolList
	err := r.List(ctx, &gpupoolsList, client.MatchingLabels(map[string]string{
		constants.LabelKeyOwner: tfc.GetName(),
	}))
	if err != nil {
		return false, fmt.Errorf("failed to list GPUPools: %w", err)
	}

	// Map existing GPUPools by their unique identifier (e.g., name)
	existingGPUPools := make(map[string]*tfv1.GPUPool)
	for _, pool := range gpupoolsList.Items {
		if pool.OwnerReferences != nil {
			owner := &metav1.OwnerReference{}
			for i := range pool.OwnerReferences {
				if controllerRef := pool.OwnerReferences[i]; controllerRef.Name == tfc.GetName() {
					owner = &controllerRef
					break
				}
			}
			if owner != nil && owner.Name == tfc.GetName() {
				existingGPUPools[pool.Name] = &pool
			}
		}
	}

	errors := []error{}
	anyPoolChanged := false

	// Process each intended GPUPool in the cluster spec
	for _, poolSpec := range tfc.Spec.GPUPools {
		key := tfc.Name + "-" + poolSpec.Name

		// Check if the GPUPool already exists
		existingPool := existingGPUPools[key]
		if existingPool == nil {
			poolLabels := map[string]string{
				constants.LabelKeyOwner: tfc.GetName(),
			}
			for k, v := range tfc.Labels {
				poolLabels[k] = v
			}

			// Create new GPUPool
			gpupool := &tfv1.GPUPool{
				ObjectMeta: metav1.ObjectMeta{
					Name:   key,
					Labels: poolLabels,
				},
				Spec: poolSpec.SpecTemplate,
			}
			e := controllerutil.SetControllerReference(tfc, gpupool, r.Scheme)
			if e != nil {
				errors = append(errors, fmt.Errorf("failed to set controller reference: %w", e))
				continue
			}
			err = r.Create(ctx, gpupool)
			anyPoolChanged = true
			if err != nil {
				errors = append(errors, fmt.Errorf("failed to create GPUPool %s: %w", key, err))
				continue
			}
		} else {
			// Update existing GPUPool if spec changed
			if !equality.Semantic.DeepEqual(&existingPool.Spec, &poolSpec.SpecTemplate) {
				existingPool.Spec = poolSpec.SpecTemplate
				err = r.Update(ctx, existingPool)
				if err != nil {
					errors = append(errors, fmt.Errorf("failed to update GPUPool %s: %w", key, err))
				}
				anyPoolChanged = true
			}
		}
	}

	// Delete any GPUPools that are no longer in the spec
	for poolName := range existingGPUPools {
		found := false
		for _, poolSpec := range tfc.Spec.GPUPools {
			key := tfc.Name + "-" + poolSpec.Name
			if key == poolName {
				found = true
				break
			}
		}
		if !found {
			existingPool := existingGPUPools[poolName]
			err = r.Delete(ctx, existingPool)
			anyPoolChanged = true
			if err != nil {
				errors = append(errors, fmt.Errorf("failed to delete GPUPool %s: %w", poolName, err))
			}
		}
	}

	// If there were any errors, return them; otherwise, set status to Running
	if len(errors) > 0 {
		for _, err := range errors {
			return false, fmt.Errorf("reconcile pool in tensor-fusion cluster failed: %w", err)
		}
	}
	return anyPoolChanged, nil
}

func (r *TensorFusionClusterReconciler) checkTFClusterComponentsReady(ctx context.Context, tfc *tfv1.TensorFusionCluster) (bool, []metav1.Condition, error) {
	allPass := true
	conditions := []metav1.Condition{
		{
			Type:   constants.ConditionStatusTypeGPUPool,
			Status: metav1.ConditionTrue,
		},
		{
			Type:   constants.ConditionStatusTypeTimeSeriesDatabase,
			Status: metav1.ConditionTrue,
		},
	}

	// check if all conditions are true, any not ready component will make allPass false

	// Step 1. check GPUPools
	var pools tfv1.GPUPoolList
	err := r.List(ctx, &pools, client.MatchingLabels(map[string]string{
		constants.LabelKeyOwner: tfc.GetName(),
	}))
	if err != nil {
		r.Recorder.Eventf(tfc, corev1.EventTypeWarning, "CheckComponentStatusError", err.Error())
		return false, nil, fmt.Errorf("failed to list GPUPools: %w", err)
	}
	if len(pools.Items) != len(tfc.Spec.GPUPools) {
		allPass = false
		conditions[0].Status = metav1.ConditionFalse
	} else {
		for i := range pools.Items {
			if pools.Items[i].Status.Phase != constants.PhaseRunning {
				allPass = false
				conditions[0].Status = metav1.ConditionFalse
				break
			}
		}
	}

	// Step 2. check TimeSeriesDatabase connection Model/Snapshot Distributor etc. TODO

	return allPass, conditions, nil
}

func (r *TensorFusionClusterReconciler) updateTFClusterStatus(ctx context.Context, tfc *tfv1.TensorFusionCluster, prevStatus *tfv1.TensorFusionClusterStatus) error {
	// diff the status to ensure only changed status updated, ignore retryCount field since it's updated in the controller
	if prevStatus != nil {
		if equality.Semantic.DeepEqual(tfc.Status, *prevStatus) {
			return nil
		}
	}
	if err := r.Status().Update(ctx, tfc); err != nil {
		r.Recorder.Eventf(tfc, corev1.EventTypeWarning, "UpdateClusterStatusError", err.Error())
		return err
	}
	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *TensorFusionClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.TensorFusionCluster{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Named("tensorfusioncluster").
		Owns(&tfv1.GPUPool{}).
		Complete(r)
}
