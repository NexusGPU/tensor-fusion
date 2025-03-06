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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"slices"

	"reflect"

	tensorfusionaiv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
	scheduler "github.com/NexusGPU/tensor-fusion-operator/internal/scheduler"
	"github.com/NexusGPU/tensor-fusion-operator/internal/utils"
	"github.com/NexusGPU/tensor-fusion-operator/internal/worker"
)

// TensorFusionWorkloadReconciler reconciles a TensorFusionWorkload object
type TensorFusionWorkloadReconciler struct {
	client.Client
	Scheme    *runtime.Scheme
	Scheduler scheduler.Scheduler
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionworkloads,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionworkloads/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionworkloads/finalizers,verbs=update
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

	// First, handle pods with finalizers that need GPU resource cleanup
	podList := &corev1.PodList{}
	if err := r.List(ctx, podList,
		client.InNamespace(req.Namespace),
		client.MatchingLabels{constants.WorkloadKey: workload.Name}); err != nil {
		return ctrl.Result{}, fmt.Errorf("list pods: %w", err)
	}

	hasdeletion := false
	// Process pods with our finalizer
	for i := range podList.Items {
		pod := &podList.Items[i]

		// Handle our GPU resource cleanup finalizer
		deleted, err := utils.HandleFinalizer(ctx, pod, r.Client, func(ctx context.Context, obj *corev1.Pod) (bool, error) {
			return r.handlePodGPUCleanup(ctx, pod, workload)
		})

		if err != nil {
			return ctrl.Result{}, err
		}
		hasdeletion = hasdeletion || deleted
	}

	if hasdeletion {
		return ctrl.Result{Requeue: true, RequeueAfter: constants.PendingRequeueDuration}, nil
	}

	// Fetch the GPUPool
	pool := &tfv1.GPUPool{}
	if err := r.Get(ctx, client.ObjectKey{Name: workload.Spec.PoolName}, pool); err != nil {
		return ctrl.Result{}, fmt.Errorf("gpu pool(%s) does not exist", workload.Spec.PoolName)
	}

	// Create worker generator
	workerGenerator := &worker.WorkerGenerator{WorkerConfig: pool.Spec.ComponentConfig.Worker}

	// Determine the number of replicas
	desiredReplicas := int32(1)
	if workload.Spec.Replicas != nil {
		desiredReplicas = *workload.Spec.Replicas
	}

	// Count current replicas
	currentReplicas := int32(len(podList.Items))
	log.Info("Current replicas", "count", currentReplicas, "desired", desiredReplicas)

	// Update workload status
	if workload.Status.Replicas != currentReplicas {
		workload.Status.Replicas = currentReplicas
		if err := r.Status().Update(ctx, workload); err != nil {
			return ctrl.Result{}, fmt.Errorf("update status: %w", err)
		}
	}

	// Scale up if needed
	if currentReplicas < desiredReplicas {
		log.Info("Scaling up workers", "from", currentReplicas, "to", desiredReplicas)

		// Calculate how many pods need to be added
		podsToAdd := int(desiredReplicas - currentReplicas)
		if err := r.scaleUpWorkers(ctx, workerGenerator, workload, podsToAdd, req.Namespace); err != nil {
			return ctrl.Result{}, err
		}
	} else if currentReplicas > desiredReplicas {
		log.Info("Scaling down workers", "from", currentReplicas, "to", desiredReplicas)

		// Sort pods by creation time (oldest first)
		sort.Slice(podList.Items, func(i, j int) bool {
			return podList.Items[i].CreationTimestamp.Before(&podList.Items[j].CreationTimestamp)
		})

		// Calculate how many pods need to be removed
		podsToRemove := int(currentReplicas - desiredReplicas)
		if err := r.scaleDownWorkers(ctx, podList.Items[:podsToRemove]); err != nil {
			return ctrl.Result{}, err
		}
	}

	if err := r.updateStatus(ctx, workload, podList.Items, workerGenerator); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

func (r *TensorFusionWorkloadReconciler) tryStartWorker(
	ctx context.Context,
	workerGenerator *worker.WorkerGenerator,
	gpu *tfv1.GPU,
	workload *tfv1.TensorFusionWorkload,
	namespacedName types.NamespacedName,
) (*corev1.Pod, error) {
	// Try to get the Pod
	pod := &corev1.Pod{}
	if err := r.Get(ctx, namespacedName, pod); err != nil {
		if errors.IsNotFound(err) {
			// Pod doesn't exist, create a new one
			port := workerGenerator.AllocPort()
			pod, err = workerGenerator.GenerateWorkerPod(gpu, namespacedName, port)
			if err != nil {
				return nil, fmt.Errorf("generate worker pod %w", err)
			}

			// Add labels to identify this pod as part of the workload
			if pod.Labels == nil {
				pod.Labels = make(map[string]string)
			}
			pod.Labels[constants.WorkloadKey] = workload.Name
			pod.Labels[constants.GpuKey] = gpu.Name

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
		return nil, err
	}
	return pod, nil
}

// scaleDownWorkers handles the scaling down of worker pods
func (r *TensorFusionWorkloadReconciler) scaleDownWorkers(ctx context.Context, pods []corev1.Pod) error {
	log := log.FromContext(ctx)

	for i := range pods {
		podToDelete := &pods[i]
		log.Info("Scaling down worker pod", "name", podToDelete.Name)
		// Delete the pod with foreground deletion policy
		// The finalizer will handle GPU resource cleanup
		if err := r.deletePod(ctx, podToDelete); err != nil {
			return err
		}
	}
	return nil
}

// handlePodGPUCleanup handles the cleanup of GPU resources when a pod is being deleted
func (r *TensorFusionWorkloadReconciler) handlePodGPUCleanup(ctx context.Context, pod *corev1.Pod, workload *tfv1.TensorFusionWorkload) (bool, error) {
	log := log.FromContext(ctx)

	// Check if this is our finalizer
	if !containsFinalizer(pod, constants.Finalizer) {
		// Not our finalizer, skip processing
		return true, nil
	}
	log.Info("Processing pod with GPU resource cleanup finalizer", "pod", pod.Name)
	// Get GPU name from pod label
	gpuName, ok := pod.Labels[constants.GpuKey]
	if !ok {
		log.Info("Pod has finalizer but no GPU label", "pod", pod.Name)
		return true, nil
	}

	// Get the GPU
	gpu := &tfv1.GPU{}
	if err := r.Get(ctx, client.ObjectKey{Name: gpuName}, gpu); err != nil {
		if errors.IsNotFound(err) {
			// GPU not found, just continue
			log.Info("GPU not found", "gpu", gpuName, "pod", pod.Name)
			return true, nil
		}
		// Error getting GPU, retry later
		log.Error(err, "Failed to get GPU", "gpu", gpuName, "pod", pod.Name)
		return false, err
	}

	// Release GPU resources
	if err := r.Scheduler.Release(ctx, workload.Spec.Resources.Requests, gpu); err != nil {
		log.Error(err, "Failed to release GPU resources, will retry", "gpu", gpuName, "pod", pod.Name)
		return false, err
	}

	log.Info("Released GPU resources via finalizer", "gpu", gpuName, "pod", pod.Name)
	return true, nil
}

// Helper function to check if a pod has a specific finalizer
func containsFinalizer(pod *corev1.Pod, finalizer string) bool {
	return slices.Contains(pod.Finalizers, finalizer)
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
func (r *TensorFusionWorkloadReconciler) scaleUpWorkers(ctx context.Context, workerGenerator *worker.WorkerGenerator, workload *tfv1.TensorFusionWorkload, count int, namespace string) error {
	log := log.FromContext(ctx)

	// Create worker pods
	currentCount := int(workload.Status.Replicas)
	for i := range count {
		// Schedule GPU for the worker
		gpu, err := r.Scheduler.Schedule(ctx, workload.Spec.PoolName, workload.Spec.Resources.Requests)
		if err != nil {
			return fmt.Errorf("schedule GPU: %w", err)
		}

		// Create worker pod
		workerName := fmt.Sprintf("%s-worker-%d", workload.Name, currentCount+i)
		namespacedName := types.NamespacedName{
			Namespace: namespace,
			Name:      workerName,
		}

		_, err = r.tryStartWorker(ctx, workerGenerator, gpu, workload, namespacedName)
		if err != nil {
			// Try to release the GPU resource if pod creation fails
			releaseErr := r.Scheduler.Release(ctx, workload.Spec.Resources.Requests, gpu)
			if releaseErr != nil {
				log.Error(releaseErr, "Failed to release GPU after pod creation failure")
			}
			return fmt.Errorf("create worker pod: %w", err)
		}

		log.Info("Created worker pod", "name", workerName)
	}

	return nil
}

// updateStatus updates the WorkerStatuses and readyReplicas field in the workload status
func (r *TensorFusionWorkloadReconciler) updateStatus(
	ctx context.Context,
	workload *tfv1.TensorFusionWorkload,
	pods []corev1.Pod,
	workerGenerator *worker.WorkerGenerator,
) error {
	log := log.FromContext(ctx)
	readyReplicas := int32(0)

	// Create a worker statuses slice to hold all worker status information
	workerStatuses := []tfv1.WorkerStatus{}

	for _, pod := range pods {
		// Skip pods that are being deleted
		if pod.DeletionTimestamp != nil {
			continue
		}

		readyReplicas++

		// Get worker IP and port information from pod
		ip := pod.Status.PodIP
		port, err := workerGenerator.WorkerPort(&pod)
		if err != nil {
			log.Error(err, "can not get worker port", "pod", pod.Name, "error", err)
			continue
		}

		var workerPhase tfv1.WorkerPhase
		switch pod.Status.Phase {
		case corev1.PodPending:
			workerPhase = tfv1.WorkerPending
		case corev1.PodRunning:
			workerPhase = tfv1.WorkerRunning
		case corev1.PodFailed:
			workerPhase = tfv1.WorkerFailed
		default:
			workerPhase = tfv1.WorkerPending
		}

		// Create and append worker status
		workerStatus := tfv1.WorkerStatus{
			WorkerPhase:  workerPhase,
			WorkerName:   pod.Name,
			WorkerIp:     ip,
			WorkerPort:   port,
			NodeSelector: pod.Spec.NodeSelector,
		}

		workerStatuses = append(workerStatuses, workerStatus)
	}

	// Check if we need to update status
	statusChanged := workload.Status.ReadyReplicas != readyReplicas ||
		!reflect.DeepEqual(workload.Status.WorkerStatuses, workerStatuses)

	if statusChanged {
		log.Info("Updating workload status", "readyReplicas", readyReplicas, "workerCount", len(workerStatuses))
		workload.Status.ReadyReplicas = readyReplicas
		workload.Status.WorkerStatuses = workerStatuses
		if err := r.Status().Update(ctx, workload); err != nil {
			return fmt.Errorf("update workload status: %w", err)
		}
	}
	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *TensorFusionWorkloadReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tensorfusionaiv1.TensorFusionWorkload{}).
		Named("tensorfusionworkload").
		Owns(&corev1.Pod{}).
		Complete(r)
}
