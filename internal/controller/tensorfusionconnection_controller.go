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
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/worker"
)

// TensorFusionConnectionReconciler reconciles a TensorFusionConnection object
type TensorFusionConnectionReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionconnections,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionconnections/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionconnections/finalizers,verbs=update

// Add and monitor GPU worker Pod for a TensorFusionConnection
func (r *TensorFusionConnectionReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	log.V(6).Info("Reconciling TensorFusionConnection", "name", req.Name)
	defer func() {
		log.V(6).Info("Finished reconciling TensorFusionConnection", "name", req.Name)
	}()

	connection := &tfv1.TensorFusionConnection{}
	if err := r.Get(ctx, req.NamespacedName, connection); err != nil {
		if errors.IsNotFound(err) {
			// Object not found, could have been deleted after reconcile request, return without error
			return ctrl.Result{}, nil
		}
		log.Error(err, "Failed to get TensorFusionConnection")
		return ctrl.Result{}, err
	}
	needSelectWorker := false
	if connection.Status.WorkerName != "" {
		// check if worker pod is still running
		pod := &v1.Pod{}
		if err := r.Get(ctx, client.ObjectKey{Name: connection.Status.WorkerName, Namespace: connection.Namespace}, pod); err != nil {
			if errors.IsNotFound(err) {
				needSelectWorker = true
			} else {
				log.Error(err, "Failed to get worker pod")
				return ctrl.Result{}, err
			}
		}
		if pod.Status.Phase != v1.PodRunning {
			connection.Status.WorkerName = ""
			connection.Status.Phase = tfv1.WorkerFailed
			connection.Status.ConnectionURL = ""
			// set worker name to empty to trigger select worker again
			if updateErr := r.Status().Update(ctx, connection); updateErr != nil {
				return ctrl.Result{}, fmt.Errorf("failed to update connection status: %w", updateErr)
			}
			return ctrl.Result{}, nil
		}
	} else {
		needSelectWorker = true
	}

	if needSelectWorker {
		workloadName, ok := connection.Labels[constants.WorkloadKey]
		if !ok {
			return ctrl.Result{}, fmt.Errorf("missing workload label")
		}

		workload := &tfv1.TensorFusionWorkload{}
		if err := r.Get(ctx, client.ObjectKey{Name: workloadName, Namespace: connection.Namespace}, workload); err != nil {
			return ctrl.Result{}, fmt.Errorf("get TensorFusionWorkload: %w", err)
		}

		r.Recorder.Eventf(connection, v1.EventTypeNormal, "SelectingWorker", "Selecting worker for connection %s", connection.Name)
		s, err := worker.SelectWorker(ctx, r.Client, workload, 1)
		if err != nil {
			r.Recorder.Eventf(connection, v1.EventTypeWarning, "WorkerSelectionFailed", "Failed to select worker: %v", err)
			// Update the status to WorkerPending when worker selection fails
			connection.Status.Phase = tfv1.WorkerPending
			connection.Status.WorkerName = ""
			connection.Status.ConnectionURL = ""
			if updateErr := r.Status().Update(ctx, connection); updateErr != nil {
				return ctrl.Result{}, fmt.Errorf("failed to select worker: %w, failed to update status: %v", err, updateErr)
			}
			return ctrl.Result{}, err
		}
		r.Recorder.Eventf(connection, v1.EventTypeNormal, "WorkerSelected", "Worker %s successfully selected for connection", s.WorkerName)
		connection.Status.Phase = s.WorkerPhase
		connection.Status.WorkerName = s.WorkerName
		resourceVersion := s.ResourceVersion
		if resourceVersion == "" {
			resourceVersion = "0"
		}

		connection.Status.ConnectionURL = fmt.Sprintf("native+%s+%d+%s-%s", s.WorkerIp, s.WorkerPort, s.WorkerName, resourceVersion)
		if err := r.Status().Update(ctx, connection); err != nil {
			return ctrl.Result{}, fmt.Errorf("update connection status: %w", err)
		}
		r.Recorder.Eventf(connection, v1.EventTypeNormal, "ConnectionReady", "Connection URL: %s", connection.Status.ConnectionURL)
	}

	// continuous check if worker is failed or not, if failed, trigger re-select worker
	return ctrl.Result{RequeueAfter: time.Second * 10}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *TensorFusionConnectionReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.TensorFusionConnection{}).
		Named("tensorfusionconnection").
		Complete(r)
}
