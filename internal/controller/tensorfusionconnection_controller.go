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

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
	scheduler "github.com/NexusGPU/tensor-fusion-operator/internal/scheduler"
	"github.com/NexusGPU/tensor-fusion-operator/internal/utils"
	"github.com/NexusGPU/tensor-fusion-operator/internal/worker"
	corev1 "k8s.io/api/core/v1"
)

// TensorFusionConnectionReconciler reconciles a TensorFusionConnection object
type TensorFusionConnectionReconciler struct {
	client.Client
	Scheme    *runtime.Scheme
	Scheduler scheduler.Scheduler
}

// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionconnections,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionconnections/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionconnections/finalizers,verbs=update

// Add and monitor GPU worker Pod for a TensorFusionConnection
func (r *TensorFusionConnectionReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	log.Info("Reconciling TensorFusionConnection", "name", req.NamespacedName.Name)
	defer func() {
		log.Info("Finished reconciling TensorFusionConnection", "name", req.NamespacedName.Name)
	}()

	// Get the TensorFusionConnection object
	connection := &tfv1.TensorFusionConnection{}
	if err := r.Get(ctx, req.NamespacedName, connection); err != nil {
		if errors.IsNotFound(err) {
			// Object not found, could have been deleted after reconcile request, return without error
			return ctrl.Result{}, nil
		}
		log.Error(err, "Failed to get TensorFusionConnection")
		return ctrl.Result{}, err
	}

	deleted, err := utils.HandleFinalizer(ctx, connection, r.Client, r.handleDeletion)
	if err != nil {
		return ctrl.Result{}, err
	}
	if deleted {
		return ctrl.Result{RequeueAfter: constants.PendingRequeueDuration}, nil
	}

	var gpu *tfv1.GPU
	// If status is not set or pending, try to schedule
	if connection.Status.Phase == "" || connection.Status.Phase == tfv1.TensorFusionConnectionPending {
		// Try to get an available gpu from scheduler
		var err error
		gpu, err = r.Scheduler.Schedule(connection.Spec.Resources.Requests)
		if err != nil {
			log.Error(err, "Failed to schedule gpu instance")
			connection.Status.Phase = tfv1.TensorFusionConnectionPending
		} else if gpu != nil {
			connection.Status.Phase = tfv1.TensorFusionConnectionStarting
			// Store the gpu name for cleanup
			connection.Status.GPU = gpu.Name
		} else {
			// Init status
			connection.Status.Phase = tfv1.TensorFusionConnectionPending
		}
	}

	// check schedule result
	if gpu == nil && connection.Status.GPU != "" {
		gpu = &tfv1.GPU{}
		if err := r.Get(ctx, client.ObjectKey{Name: connection.Status.GPU}, gpu); err != nil {
			log.Error(err, "Failed to get GPU.", "gpu", connection.Status.GPU)
			gpu = nil
		}
	}

	// Start worker Pod
	if connection.Status.Phase != tfv1.TensorFusionConnectionPending && gpu != nil {
		pool := &tfv1.GPUPool{}
		if err := r.Get(ctx, client.ObjectKey{Name: connection.Spec.PoolName}, pool); err != nil {
			return ctrl.Result{}, fmt.Errorf("gpu pool(%s) does not exist", connection.Spec.PoolName)
		}
		workerGenerator := &worker.WorkerGenerator{WorkerConfig: pool.Spec.ComponentConfig.Worker}
		// Start worker job
		workerPod, err := r.tryStartWorker(ctx, workerGenerator, gpu, connection, client.ObjectKeyFromObject(connection))
		if err != nil {
			log.Error(err, "Failed to start worker pod")
			return ctrl.Result{}, err
		}

		if workerPod.Status.Phase == corev1.PodRunning {
			connection.Status.Phase = tfv1.TensorFusionConnectionRunning
			connection.Status.ConnectionURL, err = workerGenerator.GenerateConnectionURL(connection, workerPod)
			if err != nil {
				return ctrl.Result{}, err
			}
		}
		// TODO: Handle PodFailure
	}

	if err := r.mustUpdateTFConnectionStatus(ctx, connection, gpu); err != nil {
		return ctrl.Result{}, err
	}

	if connection.Status.Phase == tfv1.TensorFusionConnectionPending {
		// requeue
		return ctrl.Result{RequeueAfter: constants.PendingRequeueDuration}, nil
	}

	return ctrl.Result{}, nil
}

func (r *TensorFusionConnectionReconciler) tryStartWorker(
	ctx context.Context,
	workerGenerator *worker.WorkerGenerator,
	gpu *tfv1.GPU,
	connection *tfv1.TensorFusionConnection,
	namespacedName types.NamespacedName,
) (*corev1.Pod, error) {
	// Try to get the Pod
	pod := &corev1.Pod{}
	if err := r.Get(ctx, namespacedName, pod); err != nil {
		if errors.IsNotFound(err) {
			// Pod doesn't exist, create a new one
			port := workerGenerator.AllocPort()
			pod, err = workerGenerator.GenerateWorkerPod(gpu, connection, namespacedName, port)
			if err != nil {
				return nil, fmt.Errorf("generate worker pod %w", err)
			}
			if err := ctrl.SetControllerReference(connection, pod, r.Scheme); err != nil {
				return nil, fmt.Errorf("set owner reference %w", err)
			}
			if err := r.Create(ctx, pod); err != nil {
				return nil, fmt.Errorf("create pod %w", err)
			}
			return pod, nil
		}
	}
	return pod, nil
}

// handleDeletion handles cleanup of external dependencies
func (r *TensorFusionConnectionReconciler) handleDeletion(ctx context.Context, connection *tfv1.TensorFusionConnection) (bool, error) {
	if connection.Status.GPU == "" {
		return true, nil // No gpu was allocated, nothing to clean up
	}

	// Get the gpu
	gpu := &tfv1.GPU{}
	if err := r.Get(ctx, client.ObjectKey{Name: connection.Status.GPU}, gpu); err != nil {
		if errors.IsNotFound(err) {
			// gpu is already gone, nothing to do
			return true, nil
		}
		return false, err
	}

	// Release the resources
	if err := r.Scheduler.Release(connection.Spec.Resources.Requests, gpu); err != nil {
		return false, err
	}

	return true, r.mustUpdateTFConnectionStatus(ctx, connection, gpu)
}

func (r *TensorFusionConnectionReconciler) mustUpdateTFConnectionStatus(ctx context.Context, connection *tfv1.TensorFusionConnection, gpu *tfv1.GPU) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		// Get the latest version of the connection
		latestConnection := &tfv1.TensorFusionConnection{}
		if err := r.Get(ctx, client.ObjectKey{
			Name:      connection.Name,
			Namespace: connection.Namespace,
		}, latestConnection); err != nil {
			return err
		}

		// Update the status fields we care about
		latestConnection.Status = connection.Status

		// Update the connection status
		if err := r.Status().Update(ctx, latestConnection); err != nil {
			return err
		}

		if gpu != nil {
			// Get the latest version of the gpu
			latestgpu := &tfv1.GPU{}

			if err := r.Get(ctx, client.ObjectKey{
				Name: gpu.Name,
			}, latestgpu); err != nil {
				return err
			}

			// Update the status fields we care about
			latestgpu.Status.Available = gpu.Status.Available
			if err := r.Status().Update(ctx, latestgpu); err != nil {
				return err
			}
		}
		return nil
	})
}

// SetupWithManager sets up the controller with the Manager.
func (r *TensorFusionConnectionReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.TensorFusionConnection{}).
		Owns(&corev1.Pod{}).
		Named("tensorfusionconnection").
		Complete(r)
}
