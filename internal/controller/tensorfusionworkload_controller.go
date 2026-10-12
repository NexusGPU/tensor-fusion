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
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/events"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/pkg/controller"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	gangscheduler "github.com/NexusGPU/tensor-fusion/internal/gang"
	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/internal/worker"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

// TensorFusionWorkloadReconciler reconciles a TensorFusionWorkload object
type TensorFusionWorkloadReconciler struct {
	client.Client
	Scheme        *runtime.Scheme
	Recorder      events.EventRecorder
	PortAllocator *portallocator.PortAllocator

	// apiReader reads from the API server instead of the informer cache. It is
	// set by SetupWithManager; without it the cached client is used.
	apiReader client.Reader
	// orphanGracePeriod replaces WorkloadOrphanGracePeriod when set (tests).
	orphanGracePeriod time.Duration
}

// WorkloadOrphanGracePeriod is how long a workload the pod webhook created may
// be without an owner reference and without a pod before it is removed (see
// collectOrphanWorkload). It is counted from the later of the workload's
// creation and constants.WorkloadLastPodSeenAnnotation.
//
// The webhook creates the workload, or records an admission on an existing
// one, while the pod's create request is still being handled. After the
// webhook has answered (it has timeoutSeconds, at most 30s, to do so) the API
// server runs the remaining admission steps and stores the pod, all inside its
// request timeout (--request-timeout, 60s by default). So about a minute later
// the pod is either stored or never will be. That the workload has no pod is
// confirmed with a list from the API server before it is deleted, so the lag
// of the informer cache is not part of the budget. The remaining minutes are
// margin: for clock differences between the API server, the operator replica
// that served the admission and the one that runs this check, for the time
// recorded on the workload being up to workloadPodSeenRefreshInterval old, and
// for an API server with a longer request timeout. The margin is generous on
// purpose: unlike an admission slot nobody holds (PodCounterLeakGracePeriod),
// a workload without a pod holds no GPU, quota or replica slot, so keeping it
// a few minutes longer costs nothing, while removing the workload of a pod
// that is about to appear leaves that pod without one.
//
// A workload whose owner has scaled it to zero is not subject to this period
// at all: it has an owner reference and is never considered.
const WorkloadOrphanGracePeriod = 5 * time.Minute

// workloadPodSeenRefreshInterval bounds how often the controller writes
// constants.WorkloadLastPodSeenAnnotation while a workload without an owner
// reference has pods, and is how soon such a workload is looked at again.
const workloadPodSeenRefreshInterval = time.Minute

// workloadOrphanRecheckSlack is added to the delay until the grace period has
// passed, so the next reconcile does not arrive a moment too early.
const workloadOrphanRecheckSlack = time.Second

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionworkloads,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionworkloads/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionworkloads/finalizers,verbs=update

// Reconcile first removes the workload if the pod webhook created it for a
// pod that was never stored (see collectOrphanWorkload) and otherwise handles
// it as usual (see reconcileWorkload). A failure of the first step is logged
// and tried again later instead of failing the reconcile of a workload that
// may be perfectly in use.
func (r *TensorFusionWorkloadReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	deleted, recheck, err := r.collectOrphanWorkload(ctx, req.NamespacedName)
	if err != nil {
		log.FromContext(ctx).Error(err, "failed to check workload for a missing owner and pod", "workload", req.NamespacedName)
		recheck = workloadPodSeenRefreshInterval
	}
	if deleted {
		// The deletion arrives as an event of its own; that reconcile takes
		// the finalizer off.
		return ctrl.Result{}, nil
	}
	res, err := r.reconcileWorkload(ctx, req)
	if err != nil {
		return res, err
	}
	// An immediate requeue comes back here and asks again; a delay must not
	// replace it.
	//nolint:staticcheck // Requeue is still what reconcileWorkload returns on a conflict
	if recheck > 0 && !res.Requeue && (res.RequeueAfter == 0 || recheck < res.RequeueAfter) {
		res.RequeueAfter = recheck
	}
	return res, nil
}

// isOrphanCandidate reports whether the workload is one collectOrphanWorkload
// may remove once it has had no pod for the grace period:
//   - the pod webhook created it (it marks its workloads with the dynamic
//     workload mode); a workload a user created is never touched;
//   - it has no owner reference. The webhook gives the workload the pod's
//     controller (Deployment, StatefulSet, Job, ...) as owner, and for a pod
//     without a controller the pod controller makes the stored pod the owner.
//     Either reference hands the workload's lifetime to the Kubernetes garbage
//     collector: while the owner exists the workload stays, with or without
//     pods (an owner scaled to zero, or one whose pods are all being refused),
//     and when the owner is gone the collector deletes the workload. What no
//     reference covers is the workload of a pod without a controller that was
//     refused after admission, or deleted before the pod controller got to it.
func isOrphanCandidate(workload *tfv1.TensorFusionWorkload) bool {
	return workload.DeletionTimestamp.IsZero() &&
		workload.Annotations[constants.WorkloadModeAnnotation] == constants.WorkloadModeDynamic &&
		len(workload.OwnerReferences) == 0
}

// podRefersToWorkload reports whether the pod belongs to the workload with
// that name in its namespace: a worker or a local-GPU client (label), a client
// of a remote worker (selected-workload annotation), or a pod that is about to
// become the workload's owner.
func podRefersToWorkload(pod *corev1.Pod, workloadName string) bool {
	return pod.Labels[constants.WorkloadKey] == workloadName ||
		pod.Annotations[constants.SelectedWorkloadAnnotation] == workloadName ||
		pod.Annotations[constants.SetPendingOwnedWorkloadAnnotation] == workloadName
}

// workloadHasPod reports whether any pod, terminating ones included, refers to
// the workload. The annotations cannot be selected on, so all pods of the
// namespace are looked at; they are only read.
func workloadHasPod(ctx context.Context, reader client.Reader, workload *tfv1.TensorFusionWorkload) (bool, error) {
	podList := &corev1.PodList{}
	if err := reader.List(ctx, podList, client.InNamespace(workload.Namespace), client.UnsafeDisableDeepCopy); err != nil {
		return false, fmt.Errorf("list pods: %w", err)
	}
	for i := range podList.Items {
		if podRefersToWorkload(&podList.Items[i], workload.Name) {
			return true, nil
		}
	}
	return false, nil
}

// orphanClockStart returns the time from which the workload counts as being
// without a pod: the later of its creation and the recorded last time a pod of
// it was admitted or seen. Both are stored on the workload, so a restart of
// the operator neither starts the period again nor cuts it short. readable is
// false when the recorded time cannot be parsed.
func orphanClockStart(workload *tfv1.TensorFusionWorkload) (start time.Time, readable bool) {
	start = workload.CreationTimestamp.Time
	value, ok := workload.Annotations[constants.WorkloadLastPodSeenAnnotation]
	if !ok {
		return start, true
	}
	seen, err := time.Parse(time.RFC3339, value)
	if err != nil {
		return start, false
	}
	if seen.After(start) {
		start = seen
	}
	return start, true
}

// collectOrphanWorkload deletes a workload that the pod webhook created for a
// pod that does not exist. The webhook creates the workload during admission,
// before the pod is stored; when a later admission step, a quota or the
// validation refuses the pod, or the request times out, nothing else would
// ever remove a workload that has no owner reference (see isOrphanCandidate).
//
// The workload is deleted when no pod has referred to it for the grace period
// (see WorkloadOrphanGracePeriod and orphanClockStart). Until then the
// returned delay asks for another look when the period has passed. While pods
// exist the time is recorded on the workload, at most once per
// workloadPodSeenRefreshInterval; normally that never happens because the
// workload gets its owner reference within seconds.
//
// Two things keep a pod that is being admitted right now from losing its
// workload: the webhook records the admission on a workload it reuses, which
// starts the period again, and the delete carries the resource version this
// decision was made on, so it is refused when the workload has changed since
// (the webhook's record, a new owner reference).
//
// Deleting the workload releases nothing: GPU allocations, quota and the
// enabled-replicas counter are all held per pod, and this workload has none.
// Its finalizer is taken off by the next reconcile (reconcileWorkload).
func (r *TensorFusionWorkloadReconciler) collectOrphanWorkload(
	ctx context.Context,
	key client.ObjectKey,
) (deleted bool, recheck time.Duration, err error) {
	workload := &tfv1.TensorFusionWorkload{}
	if err := r.Get(ctx, key, workload); err != nil {
		return false, 0, client.IgnoreNotFound(err)
	}
	if !isOrphanCandidate(workload) {
		return false, 0, nil
	}

	grace := r.orphanGracePeriod
	if grace <= 0 {
		grace = WorkloadOrphanGracePeriod
	}
	refresh := min(workloadPodSeenRefreshInterval, grace/2)
	now := time.Now()
	start, readable := orphanClockStart(workload)

	hasPod, err := workloadHasPod(ctx, r.Client, workload)
	if err != nil {
		return false, 0, err
	}
	if hasPod {
		if !readable || now.Sub(start) >= refresh {
			if err := r.recordPodSeen(ctx, workload, now); err != nil {
				return false, 0, err
			}
		}
		// A client of a remote worker does not trigger this reconcile when it
		// goes away, so look again without waiting for an event.
		return false, refresh, nil
	}

	if !readable {
		// Without a usable time there is no telling how long the workload has
		// been without a pod: start the period now instead of guessing.
		if err := r.recordPodSeen(ctx, workload, now); err != nil {
			return false, 0, err
		}
		return false, grace + workloadOrphanRecheckSlack, nil
	}
	if wait := start.Add(grace).Sub(now); wait > 0 {
		return false, wait + workloadOrphanRecheckSlack, nil
	}

	// The cache may be behind: a pod stored a moment ago must be seen.
	reader := r.apiReader
	if reader == nil {
		reader = r.Client
	}
	if hasPod, err = workloadHasPod(ctx, reader, workload); err != nil {
		return false, 0, err
	}
	if hasPod {
		return false, refresh, nil
	}

	if err := r.Delete(ctx, workload, client.Preconditions{
		UID:             &workload.UID,
		ResourceVersion: &workload.ResourceVersion,
	}); err != nil {
		if errors.IsNotFound(err) {
			return true, 0, nil
		}
		if errors.IsConflict(err) {
			// Changed since it was read; decide again on what it is now.
			return false, workloadOrphanRecheckSlack, nil
		}
		return false, 0, fmt.Errorf("delete workload without owner and pod: %w", err)
	}
	log.FromContext(ctx).Info("Deleted workload created by the pod webhook: it has no owner and has had no pod",
		"workload", key, "withoutPodSince", start, "gracePeriod", grace)
	if r.Recorder != nil {
		r.Recorder.Eventf(workload, nil, corev1.EventTypeNormal, "OrphanWorkloadDeleted", "Delete",
			"Workload has no owner and has had no pod since %s", start.Format(time.RFC3339))
	}
	return true, 0, nil
}

// recordPodSeen stores the time as constants.WorkloadLastPodSeenAnnotation.
func (r *TensorFusionWorkloadReconciler) recordPodSeen(
	ctx context.Context,
	workload *tfv1.TensorFusionWorkload,
	seen time.Time,
) error {
	patch := client.MergeFrom(workload.DeepCopy())
	if workload.Annotations == nil {
		workload.Annotations = map[string]string{}
	}
	workload.Annotations[constants.WorkloadLastPodSeenAnnotation] = seen.UTC().Format(time.RFC3339)
	if err := r.Patch(ctx, workload, patch); err != nil {
		return client.IgnoreNotFound(fmt.Errorf("record last pod seen: %w", err))
	}
	return nil
}

// reconcileWorkload handles one TensorFusionWorkload.
//
//nolint:gocyclo
func (r *TensorFusionWorkloadReconciler) reconcileWorkload(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
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
	activePods := filterActivePods(podList)

	// handle finalizer
	shouldReturn, err := utils.HandleFinalizer(ctx, workload, r.Client, func(ctx context.Context, workload *tfv1.TensorFusionWorkload) (bool, error) {
		if workload.Spec.IsDynamicReplica() {
			// dynamic replica mode, should be deleted by owner, not itself
			return true, nil
		}

		// fixed replica mode which created by user, should trigger pod deletion and stop scale up
		// when all pods are deleted, finalizer will be removed
		if len(activePods) == 0 {
			return true, nil
		}
		if err := r.scaleDownWorkers(ctx, workload, activePods); err != nil {
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
	var workerConfig *tfv1.WorkerConfig
	var hypervisorConfig *tfv1.HypervisorConfig
	if pool.Spec.ComponentConfig != nil {
		workerConfig = pool.Spec.ComponentConfig.Worker
		hypervisorConfig = pool.Spec.ComponentConfig.Hypervisor
	}
	workerGenerator := &worker.WorkerGenerator{
		WorkerConfig:     workerConfig,
		HypervisorConfig: hypervisorConfig,
	}

	podTemplateHash, err := workerGenerator.PodTemplateHash(workload.Spec)
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("get pod template hash: %w", err)
	}

	if workload.Status.PodTemplateHash != podTemplateHash {
		if err := r.patchWorkloadStatus(ctx, client.ObjectKeyFromObject(workload), func(latest *tfv1.TensorFusionWorkload) bool {
			if latest.Status.PodTemplateHash == podTemplateHash {
				return false
			}
			latest.Status.PodTemplateHash = podTemplateHash
			return true
		}); err != nil {
			if errors.IsConflict(err) {
				log.V(1).Info("status patch conflict, requeue", "name", workload.Name)
				return ctrl.Result{Requeue: true}, nil
			}
			return ctrl.Result{}, fmt.Errorf("update status: %w", err)
		}
		return ctrl.Result{}, nil
	}

	// When it is not dynamic replica, workload maintains worker replicas by itself,
	// In this mode, allow any Pod select connection to connect to any worker,
	// to achieve a sub-pool for lower costs when CPU side scaling frequency is high
	if !workload.Spec.IsDynamicReplica() {
		err := r.reconcileScaling(ctx, workload, activePods, workerGenerator, podTemplateHash)
		if err != nil {
			if errors.IsConflict(err) {
				log.V(1).Info("scaling status conflict, requeue", "name", workload.Name)
				return ctrl.Result{Requeue: true}, nil
			}
			return ctrl.Result{}, err
		}
	}

	if err := r.updateStatus(ctx, workload, activePods); err != nil {
		if errors.IsConflict(err) {
			log.V(1).Info("workload status conflict, requeue", "name", workload.Name)
			return ctrl.Result{Requeue: true}, nil
		}
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// reconcileScaling handles scaling up and down of worker pods and updates replica status
func (r *TensorFusionWorkloadReconciler) reconcileScaling(
	ctx context.Context,
	workload *tfv1.TensorFusionWorkload,
	activePods []*corev1.Pod,
	workerGenerator *worker.WorkerGenerator,
	podTemplateHash string,
) error {
	log := log.FromContext(ctx)
	// Check if there are any Pods using the old podTemplateHash and delete them if any
	if len(activePods) > 0 {
		// make oldest pod first, to delete from oldest to latest outdated pod
		sort.Slice(activePods, func(i, j int) bool {
			return activePods[i].CreationTimestamp.Before(&activePods[j].CreationTimestamp)
		})

		var outdatedPods []*corev1.Pod
		for _, pod := range activePods {
			if pod.Labels[constants.LabelKeyPodTemplateHash] != podTemplateHash {
				outdatedPods = append(outdatedPods, pod)
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
	currentReplicas := int32(len(activePods))
	log.Info("Current replicas", "count", currentReplicas, "desired", desiredReplicas)

	// Update workload status
	if workload.Status.WorkerCount != currentReplicas {
		if err := r.patchWorkloadStatus(ctx, client.ObjectKeyFromObject(workload), func(latest *tfv1.TensorFusionWorkload) bool {
			if latest.Status.WorkerCount == currentReplicas {
				return false
			}
			latest.Status.WorkerCount = currentReplicas
			return true
		}); err != nil {
			return fmt.Errorf("update status: %w", err)
		}
	}

	diff := currentReplicas - desiredReplicas
	// Scale up if needed
	if diff < 0 {
		log.Info("Scaling up workers", "from", currentReplicas, "to", desiredReplicas)

		if err := r.scaleUpWorkers(ctx, workerGenerator, workload, int(-diff), podTemplateHash); err != nil {
			return fmt.Errorf("scale up workers: %w", err)
		}
	} else if diff > 0 {
		log.Info("Scaling down workers", "from", currentReplicas, "to", desiredReplicas)

		// No need to sort if we are about to delete all pods
		if diff < int32(len(activePods)) {
			sort.Sort(controller.ActivePods(activePods))
		}

		if err := r.scaleDownWorkers(ctx, workload, activePods[:diff]); err != nil {
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
	desiredMembers := gangscheduler.ResolveDesiredMembers(ctx, r.Client, workload)
	pod, err := workerGenerator.GenerateWorkerPod(ctx, workload, desiredMembers)
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
func (r *TensorFusionWorkloadReconciler) scaleDownWorkers(ctx context.Context, workload *tfv1.TensorFusionWorkload, pods []*corev1.Pod) error {
	log := log.FromContext(ctx)
	for _, podToDelete := range pods {
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
	pods []*corev1.Pod,
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
	desiredReplicas := gangscheduler.ResolveDesiredMembers(ctx, r.Client, workload)
	phase, readyCondition := deriveWorkloadReadiness(workload, desiredReplicas, readyReplicas, failedWorkers)

	log.Info("Updating workload status", "phase", phase, "readyReplicas", readyReplicas)
	if err := r.patchWorkloadStatus(ctx, client.ObjectKeyFromObject(workload), func(latest *tfv1.TensorFusionWorkload) bool {
		changed := false

		if latest.Spec.IsDynamicReplica() && latest.Status.WorkerCount != int32(len(pods)) {
			latest.Status.WorkerCount = int32(len(pods))
			changed = true
		}
		if latest.Status.ReadyWorkers != readyReplicas {
			latest.Status.ReadyWorkers = readyReplicas
			changed = true
		}
		if reconcileGangStatus(ctx, r.Client, latest, readyReplicas, failedWorkers) {
			changed = true
		}

		effectivePhase, effectiveCondition := reconcileWorkloadReadinessWithGang(latest, phase, readyCondition)
		if latest.Status.Phase != effectivePhase {
			latest.Status.Phase = effectivePhase
			changed = true
		}
		if meta.SetStatusCondition(&latest.Status.Conditions, effectiveCondition) {
			changed = true
		}

		return changed
	}); err != nil {
		return fmt.Errorf("update workload status: %w", err)
	}
	return nil
}

func deriveWorkloadReadiness(
	workload *tfv1.TensorFusionWorkload,
	desiredReplicas, readyReplicas int32,
	failedWorkers int,
) (tfv1.TensorFusionWorkloadPhase, metav1.Condition) {
	readyCondition := metav1.Condition{
		Type:               constants.ConditionStatusTypeReady,
		LastTransitionTime: metav1.Now(),
	}

	switch {
	case failedWorkers > 0:
		readyCondition.Status = metav1.ConditionFalse
		readyCondition.Reason = "WorkerFailed"
		readyCondition.Message = fmt.Sprintf("Failed workers num: %d", failedWorkers)
		return tfv1.TensorFusionWorkloadPhaseFailed, readyCondition
	case desiredReplicas == 0:
		readyCondition.Status = metav1.ConditionFalse
		readyCondition.Reason = "ScaledToZero"
		readyCondition.Message = "Workload is scaled to zero"
		return tfv1.TensorFusionWorkloadPhasePending, readyCondition
	case workload != nil && workload.Spec.IsDynamicReplica():
		readyCondition.Status = metav1.ConditionTrue
		readyCondition.Reason = "WorkloadReady"
		readyCondition.Message = "All dynamic worker replicas are running"
		return tfv1.TensorFusionWorkloadPhaseRunning, readyCondition
	case readyReplicas == desiredReplicas:
		readyCondition.Status = metav1.ConditionTrue
		readyCondition.Reason = "WorkloadReady"
		readyCondition.Message = "All workers are running"
		return tfv1.TensorFusionWorkloadPhaseRunning, readyCondition
	default:
		readyCondition.Status = metav1.ConditionFalse
		readyCondition.Reason = "WaitingForWorkers"
		readyCondition.Message = fmt.Sprintf("Ready replicas: %d/%d", readyReplicas, desiredReplicas)
		return tfv1.TensorFusionWorkloadPhasePending, readyCondition
	}
}

func (r *TensorFusionWorkloadReconciler) patchWorkloadStatus(
	ctx context.Context,
	key client.ObjectKey,
	mutate func(*tfv1.TensorFusionWorkload) bool,
) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		latest := &tfv1.TensorFusionWorkload{}
		if err := r.Get(ctx, key, latest); err != nil {
			if errors.IsNotFound(err) {
				return nil
			}
			return err
		}

		if !mutate(latest) {
			return nil
		}

		// Use full status update so required status fields remain present.
		return r.Status().Update(ctx, latest)
	})
}

// reconcileGangStatus updates gang status based on actual pod states.
// The scheduler owns scheduling-phase transitions (Waiting/Scheduling/TimedOut),
// while the controller owns runtime-phase transitions (Running/Failed).
// The controller only overwrites the scheduler's phase when pods have actually
// reached a terminal runtime state, to avoid a write-conflict loop.
func reconcileGangStatus(
	ctx context.Context,
	c client.Client,
	workload *tfv1.TensorFusionWorkload,
	readyReplicas int32,
	failedWorkers int,
) bool {
	desiredMembers := gangscheduler.ResolveDesiredMembers(ctx, c, workload)
	if !gangscheduler.GangEnabledForWorkload(workload, desiredMembers) {
		// If the workload has a GangScheduling spec but desiredMembers is
		// temporarily unresolvable (e.g., owner not yet visible), do NOT
		// clear existing gang status — just skip this reconcile cycle.
		if workload.Spec.GangScheduling != nil && workload.Status.Gang != nil {
			return false
		}
		if workload.Status.Gang == nil {
			return false
		}
		workload.Status.Gang = nil
		return true
	}

	// Start from existing status written by the scheduler, or create a new one.
	var updated *tfv1.GangSchedulingStatus
	if workload.Status.Gang != nil {
		updated = workload.Status.Gang.DeepCopy()
	} else {
		updated = &tfv1.GangSchedulingStatus{
			Phase: tfv1.GangSchedulingPhasePending,
		}
		setGangPhase(updated, tfv1.GangSchedulingPhasePending, "GangPending",
			fmt.Sprintf("Gang waiting for scheduling: %d/%d ready", readyReplicas, desiredMembers))
	}

	// Always update counters that the controller can observe.
	updated.GroupKey = fmt.Sprintf("%s/%s", workload.Namespace, workload.Name)
	updated.DesiredMembers = desiredMembers
	updated.ReadyMembers = readyReplicas
	if updated.ScheduledMembers < readyReplicas {
		updated.ScheduledMembers = readyReplicas
	}

	// Only transition phase for states the controller owns (runtime observation).
	// Scheduler-owned phases (Waiting, Scheduling, TimedOut) are preserved.
	switch {
	case failedWorkers > 0:
		setGangPhase(updated, tfv1.GangSchedulingPhaseFailed, "GangWorkerFailed",
			fmt.Sprintf("Failed workers num: %d", failedWorkers))
		updated.WaitingMembers = 0
	case readyReplicas >= desiredMembers:
		setGangPhase(updated, tfv1.GangSchedulingPhaseRunning, "GangReady",
			fmt.Sprintf("Gang members running: %d/%d ready", readyReplicas, desiredMembers))
		updated.WaitingMembers = 0
		updated.BackoffUntil = ""
	default:
		// Only reset to Pending if the current phase is a terminal/runtime phase
		// that is no longer valid. Never overwrite scheduler's in-progress phases.
		if isSchedulerOwnedPhase(updated.Phase) {
			// Scheduler is actively managing this phase — don't touch it.
		} else if updated.Phase == "" || updated.Phase == tfv1.GangSchedulingPhaseRunning {
			setGangPhase(updated, tfv1.GangSchedulingPhasePending, "GangPending",
				fmt.Sprintf("Gang waiting for scheduling: %d/%d ready", readyReplicas, desiredMembers))
		}
	}

	if gangscheduler.SameGangStatus(workload.Status.Gang, updated) {
		return false
	}
	workload.Status.Gang = updated
	return true
}

// isSchedulerOwnedPhase returns true for phases that the scheduler is responsible for.
// The controller should not overwrite these phases unless pods reach a runtime state.
func isSchedulerOwnedPhase(phase tfv1.GangSchedulingPhase) bool {
	switch phase {
	case tfv1.GangSchedulingPhaseWaiting, tfv1.GangSchedulingPhaseScheduling, tfv1.GangSchedulingPhaseTimedOut:
		return true
	default:
		return false
	}
}

func reconcileWorkloadReadinessWithGang(
	workload *tfv1.TensorFusionWorkload,
	phase tfv1.TensorFusionWorkloadPhase,
	readyCondition metav1.Condition,
) (tfv1.TensorFusionWorkloadPhase, metav1.Condition) {
	if workload == nil || workload.Status.Gang == nil {
		return phase, readyCondition
	}

	switch workload.Status.Gang.Phase {
	case tfv1.GangSchedulingPhaseRunning:
		return phase, readyCondition
	case tfv1.GangSchedulingPhaseFailed:
		readyCondition.Status = metav1.ConditionFalse
		readyCondition.Reason = workload.Status.Gang.Reason
		readyCondition.Message = workload.Status.Gang.Message
		return tfv1.TensorFusionWorkloadPhaseFailed, readyCondition
	case tfv1.GangSchedulingPhaseTimedOut, tfv1.GangSchedulingPhasePending, tfv1.GangSchedulingPhaseWaiting, tfv1.GangSchedulingPhaseScheduling:
		readyCondition.Status = metav1.ConditionFalse
		readyCondition.Reason = workload.Status.Gang.Reason
		readyCondition.Message = workload.Status.Gang.Message
		return tfv1.TensorFusionWorkloadPhasePending, readyCondition
	default:
		return phase, readyCondition
	}
}

func setGangPhase(status *tfv1.GangSchedulingStatus, phase tfv1.GangSchedulingPhase, reason, message string) {
	if status == nil {
		return
	}
	if status.Phase != phase || status.Reason != reason || status.Message != message {
		status.LastTransitionTime = metav1.Now()
	}
	status.Phase = phase
	status.Reason = reason
	status.Message = message
}

func filterActivePods(podList *corev1.PodList) []*corev1.Pod {
	var activePods []*corev1.Pod
	for _, pod := range podList.Items {
		if pod.DeletionTimestamp.IsZero() {
			activePods = append(activePods, &pod)
		}
	}
	return activePods
}

// SetupWithManager sets up the controller with the Manager.
func (r *TensorFusionWorkloadReconciler) SetupWithManager(mgr ctrl.Manager) error {
	r.apiReader = mgr.GetAPIReader()
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
