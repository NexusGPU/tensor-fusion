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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/internal/worker"
	"github.com/samber/lo"
)

// TensorFusionWorkloadReconciler reconciles a TensorFusionWorkload object
type TensorFusionWorkloadReconciler struct {
	client.Client
	Scheme        *runtime.Scheme
	Recorder      record.EventRecorder
	PortAllocator *portallocator.PortAllocator
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionworkloads,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionworkloads/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionworkloads/finalizers,verbs=update

// TensorFusionWorkload Reconciler
//
//nolint:gocyclo
func (r *TensorFusionWorkloadReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Reconciling TensorFusionWorkload", "request", req)

	// Fetch the TensorFusionWorkload instance
	workload := &tfv1.TensorFusionWorkload{}
	if err := r.Get(ctx, req.NamespacedName, workload); err != nil {
		if errors.IsNotFound(err) {
			// Object not found, return without error
			return ctrl.Result{}, nil
		}
		// Error reading the object
		return ctrl.Result{}, err
	}

	podList := &corev1.PodList{}
	if err := r.List(ctx, podList,
		client.InNamespace(req.Namespace),
		client.MatchingLabels{constants.WorkloadKey: workload.Name}); err != nil {
		return ctrl.Result{}, fmt.Errorf("list pods: %w", err)
	}
	// only calculate state based on not deleted pods, otherwise will cause wrong total replica count
	podList.Items = lo.Filter(podList.Items, func(pod corev1.Pod, _ int) bool {
		return pod.DeletionTimestamp.IsZero()
	})

	// handle finalizer
	shouldReturn, err := utils.HandleFinalizer(ctx, workload, r.Client, func(ctx context.Context, workload *tfv1.TensorFusionWorkload) (bool, error) {
		if workload.Spec.IsDynamicReplica() {
			// dynamic replica mode, should be deleted by owner, not itself
			return true, nil
		}

		// fixed replica mode which created by user, should trigger pod deletion and stop scale up
		// when all pods are deleted, finalizer will be removed
		if len(podList.Items) == 0 {
			return true, nil
		}
		if err := r.scaleDownWorkers(ctx, workload, podList.Items); err != nil {
			return false, err
		}
		return false, nil
	})
	if err != nil {
		return ctrl.Result{}, err
	}
	if shouldReturn {
		return ctrl.Result{}, nil
	}

	if !workload.DeletionTimestamp.IsZero() {
		log.Info("Workload is being deleted, skipping pod creation", "name", workload.Name, "namespace", workload.Namespace)
		return ctrl.Result{}, nil
	}

	// Fetch the GPUPool
	pool := &tfv1.GPUPool{}
	if err := r.Get(ctx, client.ObjectKey{Name: workload.Spec.PoolName}, pool); err != nil {
		return ctrl.Result{}, fmt.Errorf("gpu pool(%s) does not exist", workload.Spec.PoolName)
	}

	// Create worker generator
	workerGenerator := &worker.WorkerGenerator{
		WorkerConfig:     pool.Spec.ComponentConfig.Worker,
		HypervisorConfig: pool.Spec.ComponentConfig.Hypervisor,
	}

	podTemplateHash, err := workerGenerator.PodTemplateHash(workload.Spec)
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("get pod template hash: %w", err)
	}

	if workload.Status.PodTemplateHash != podTemplateHash {
		workload.Status.PodTemplateHash = podTemplateHash
		if err := r.Status().Update(ctx, workload); err != nil {
			return ctrl.Result{}, fmt.Errorf("update status: %w", err)
		}
		return ctrl.Result{}, nil
	}

	// When it is not dynamic replica, workload maintains worker replicas by itself,
	// In this mode, allow any Pod select connection to connect to any worker,
	// to achieve a sub-pool for lower costs when CPU side scaling frequency is high
	if !workload.Spec.IsDynamicReplica() {
		err := r.reconcileScaling(ctx, workload, podList, workerGenerator, podTemplateHash)
		if err != nil {
			return ctrl.Result{}, err
		}
	}

	if err := r.updateStatus(ctx, workload, podList.Items); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// reconcileScaling handles scaling up and down of worker pods and updates replica status
func (r *TensorFusionWorkloadReconciler) reconcileScaling(
	ctx context.Context,
	workload *tfv1.TensorFusionWorkload,
	podList *corev1.PodList,
	workerGenerator *worker.WorkerGenerator,
	podTemplateHash string,
) error {
	log := log.FromContext(ctx)
	// Check if there are any Pods using the old podTemplateHash and delete them if any
	if len(podList.Items) > 0 {
		// make oldest pod first, to delete from oldest to latest outdated pod
		sort.Slice(podList.Items, func(i, j int) bool {
			return podList.Items[i].CreationTimestamp.Before(&podList.Items[j].CreationTimestamp)
		})

		var outdatedPods []corev1.Pod
		for i := range podList.Items {
			pod := &podList.Items[i]
			if pod.Labels[constants.LabelKeyPodTemplateHash] != podTemplateHash {
				outdatedPods = append(outdatedPods, *pod)
			}
		}

		if len(outdatedPods) > 0 {
			log.Info("Found outdated pods with different template hash", "count", len(outdatedPods))
			if err := r.scaleDownWorkers(ctx, workload, outdatedPods); err != nil {
				return err
			}
			// After deletion, requeue will be triggered by deleted Pod
			return nil
		}
	}

	// Determine the number of replicas
	desiredReplicas := int32(1)
	if workload.Spec.Replicas != nil {
		desiredReplicas = *workload.Spec.Replicas
	}

	// Count current replicas
	currentReplicas := int32(len(podList.Items))
	log.Info("Current replicas", "count", currentReplicas, "desired", desiredReplicas)

	// Update workload status
	if workload.Status.WorkerCount != currentReplicas {
		workload.Status.WorkerCount = currentReplicas
		if err := r.Status().Update(ctx, workload); err != nil {
			return fmt.Errorf("update status: %w", err)
		}
	}

	// Scale up if needed
	if currentReplicas < desiredReplicas {
		log.Info("Scaling up workers", "from", currentReplicas, "to", desiredReplicas)

		// Calculate how many pods need to be added
		podsToAdd := int(desiredReplicas - currentReplicas)
		if err := r.scaleUpWorkers(ctx, workerGenerator, workload, podsToAdd, podTemplateHash); err != nil {
			return fmt.Errorf("scale up workers: %w", err)
		}
	} else if currentReplicas > desiredReplicas {
		log.Info("Scaling down workers", "from", currentReplicas, "to", desiredReplicas)

		// Sort pods by creation time (oldest first)
		sort.Slice(podList.Items, func(i, j int) bool {
			return podList.Items[i].CreationTimestamp.Before(&podList.Items[j].CreationTimestamp)
		})

		// Calculate how many pods need to be removed
		podsToRemove := int(currentReplicas - desiredReplicas)
		if err := r.scaleDownWorkers(ctx, workload, podList.Items[:podsToRemove]); err != nil {
			return err
		}
	}

	return nil
}

func (r *TensorFusionWorkloadReconciler) tryStartWorker(
	ctx context.Context,
	workerGenerator *worker.WorkerGenerator,
	workload *tfv1.TensorFusionWorkload,
	hash string,
) (*corev1.Pod, error) {
	pod, err := workerGenerator.GenerateWorkerPod(ctx, workload)
	if err != nil {
		return nil, fmt.Errorf("generate worker pod %w", err)
	}

	pod.Labels[constants.WorkloadKey] = workload.Name
	pod.Labels[constants.LabelKeyPodTemplateHash] = hash

	// Add finalizer for GPU resource cleanup
	pod.Finalizers = append(pod.Finalizers, constants.Finalizer)

	if err := ctrl.SetControllerReference(workload, pod, r.Scheme); err != nil {
		return nil, fmt.Errorf("set owner reference %w", err)
	}
	if err := r.Create(ctx, pod); err != nil {
		return nil, fmt.Errorf("create pod %w", err)
	}
	return pod, nil
}

// scaleDownWorkers handles the scaling down of worker pods
func (r *TensorFusionWorkloadReconciler) scaleDownWorkers(ctx context.Context, workload *tfv1.TensorFusionWorkload, pods []corev1.Pod) error {
	log := log.FromContext(ctx)
	for i := range pods {
		podToDelete := &pods[i]
		log.Info("Scaling down worker pod", "name", podToDelete.Name, "workload", workload.Name)

		// If it's already being deleting, should avoid call delete multiple times
		if !podToDelete.DeletionTimestamp.IsZero() {

			// handle a corner case when pod don't have worker label and finalizer exists
			if podToDelete.Labels[constants.LabelComponent] != constants.ComponentWorker &&
				controllerutil.RemoveFinalizer(podToDelete, constants.Finalizer) {
				if err := r.Update(ctx, podToDelete); err != nil {
					return fmt.Errorf("can not remove wrong added finalizer: %w", err)
				}
			}
			continue
		}

		// Delete the pod with foreground deletion policy
		// The finalizer will handle GPU resource cleanup
		if err := r.deletePod(ctx, podToDelete); err != nil {
			return err
		}
	}
	return nil
}

// deletePod deletes a pod
func (r *TensorFusionWorkloadReconciler) deletePod(ctx context.Context, pod *corev1.Pod) error {
	log := log.FromContext(ctx)

	if err := r.Delete(ctx, pod); err != nil {
		log.Error(err, "Failed to delete worker pod", "name", pod.Name)
		return fmt.Errorf("delete worker pod: %w", err)
	}

	log.Info("Deleted worker pod", "name", pod.Name)
	return nil
}

// scaleUpWorkers handles the scaling up of worker pods
func (r *TensorFusionWorkloadReconciler) scaleUpWorkers(ctx context.Context, workerGenerator *worker.WorkerGenerator, workload *tfv1.TensorFusionWorkload, count int, hash string) error {
	// Create worker pods
	for range count {
		_, err := r.tryStartWorker(ctx, workerGenerator, workload, hash)
		if err != nil {
			return fmt.Errorf("create worker pod: %w", err)
		}
	}
	return nil
}

// updateStatus updates the status of a TensorFusionWorkload
func (r *TensorFusionWorkloadReconciler) updateStatus(
	ctx context.Context,
	workload *tfv1.TensorFusionWorkload,
	pods []corev1.Pod,
) error {
	log := log.FromContext(ctx)
	readyReplicas := int32(0)
	failedWorkers := 0

	for _, pod := range pods {
		// Skip pods that are being deleted
		if pod.DeletionTimestamp != nil {
			continue
		}
		switch pod.Status.Phase {
		case corev1.PodRunning:
			if utils.IsPodConditionTrue(pod.Status.Conditions, corev1.PodReady) {
				readyReplicas++
			}
		case corev1.PodFailed:
			failedWorkers++
		}

	}

	// Determine workload phase
	var phase tfv1.TensorFusionWorkloadPhase
	var conditions []metav1.Condition

	// Update Ready condition based on readyReplicas and desired replicas
	readyCondition := metav1.Condition{
		Type:               constants.ConditionStatusTypeReady,
		LastTransitionTime: metav1.Now(),
	}

	if failedWorkers > 0 {
		// when any worker failed, workload status should be false
		phase = tfv1.TensorFusionWorkloadPhaseFailed
		readyCondition.Status = metav1.ConditionFalse
		readyCondition.Reason = "WorkerFailed"
		readyCondition.Message = fmt.Sprintf("Failed workers num: %d", failedWorkers)
		r.Recorder.Eventf(workload, corev1.EventTypeWarning, "WorkerFailed", "Failed workers num: %d", failedWorkers)
	} else if workload.Spec.IsDynamicReplica() {
		// for dynamic replicas, if no worker failed, indicate workload is running
		phase = tfv1.TensorFusionWorkloadPhaseRunning
		readyCondition.Status = metav1.ConditionTrue
		readyCondition.Reason = "WorkloadReady"
		readyCondition.Message = "All dynamic worker replicas are running"
	} else if workload.Spec.Replicas != nil && readyReplicas == *workload.Spec.Replicas {
		phase = tfv1.TensorFusionWorkloadPhaseRunning
		readyCondition.Status = metav1.ConditionTrue
		readyCondition.Reason = "WorkloadReady"
		readyCondition.Message = "All workers are running"
	} else {
		phase = tfv1.TensorFusionWorkloadPhasePending
		readyCondition.Status = metav1.ConditionFalse
		readyCondition.Reason = "WaitingForWorkers"
		readyCondition.Message = fmt.Sprintf("Ready replicas: %d/%d", readyReplicas, *workload.Spec.Replicas)
	}
	conditions = append(conditions, readyCondition)

	// Check if we need to update status
	totalReplicasChangedInDynamicReplicaMode :=
		workload.Status.WorkerCount != int32(len(pods)) && workload.Spec.IsDynamicReplica()
	if totalReplicasChangedInDynamicReplicaMode {
		workload.Status.WorkerCount = int32(len(pods))
	}
	statusChanged := totalReplicasChangedInDynamicReplicaMode || workload.Status.ReadyWorkers != readyReplicas ||
		workload.Status.Phase != phase ||
		!utils.EqualConditionsDisregardTransitionTime(workload.Status.Conditions, conditions)

	if statusChanged {
		log.Info("Updating workload status", "phase", phase, "readyReplicas", readyReplicas)
		workload.Status.Phase = phase
		workload.Status.Conditions = conditions
		workload.Status.ReadyWorkers = readyReplicas
		if err := r.Status().Update(ctx, workload); err != nil {
			return fmt.Errorf("update workload status: %w", err)
		}
	}
	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *TensorFusionWorkloadReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.TensorFusionWorkload{}).
		Named("tensorfusionworkload").
		Watches(&corev1.Pod{}, handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []reconcile.Request {
			pod := obj.(*corev1.Pod)
			if pod.Labels == nil || pod.Labels[constants.WorkloadKey] == "" {
				return nil
			}
			return []reconcile.Request{
				{NamespacedName: client.ObjectKey{
					Name:      pod.Labels[constants.WorkloadKey],
					Namespace: pod.Namespace,
				}},
			}
		})).
		Complete(r)
}
