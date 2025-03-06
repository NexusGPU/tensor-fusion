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
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
	"github.com/NexusGPU/tensor-fusion-operator/internal/worker"
	"github.com/samber/lo"
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

	workloadName, ok := connection.Labels[constants.WorkloadKey]
	if !ok {
		return ctrl.Result{}, fmt.Errorf("missing workload label")
	}

	workload := &tfv1.TensorFusionWorkload{}
	if err := r.Get(ctx, client.ObjectKey{Name: workloadName, Namespace: connection.Namespace}, workload); err != nil {
		return ctrl.Result{}, fmt.Errorf("get TensorFusionWorkload: %w", err)
	}

	needReSelectWorker, workerStatus := r.needReSelectWorker(connection, workload.Status.WorkerStatuses)
	if needReSelectWorker {
		r.Recorder.Eventf(connection, corev1.EventTypeNormal, "SelectingWorker", "Selecting worker for connection %s", connection.Name)
		s, err := worker.SelectWorker(ctx, r.Client, workloadName, workload.Status.WorkerStatuses)
		if err != nil {
			r.Recorder.Eventf(connection, corev1.EventTypeWarning, "WorkerSelectionFailed", "Failed to select worker: %v", err)
			return ctrl.Result{}, err
		}
		workerStatus = *s
		r.Recorder.Eventf(connection, corev1.EventTypeNormal, "WorkerSelected", "Worker %s successfully selected for connection", workerStatus.WorkerName)
	}

	connection.Status.Phase = workerStatus.WorkerPhase
	connection.Status.WorkerName = workerStatus.WorkerName
	connection.Status.ConnectionURL = fmt.Sprintf("native+%s+%d", workerStatus.WorkerIp, workerStatus.WorkerPort)
	if err := r.Status().Update(ctx, connection); err != nil {
		return ctrl.Result{}, fmt.Errorf("update connection status: %w", err)
	}
	r.Recorder.Eventf(connection, corev1.EventTypeNormal, "ConnectionReady", "Connection URL: %s", connection.Status.ConnectionURL)
	return ctrl.Result{}, nil
}

func (r *TensorFusionConnectionReconciler) needReSelectWorker(connection *tfv1.TensorFusionConnection, workerStatuses []tfv1.WorkerStatus) (bool, tfv1.WorkerStatus) {
	workerStatus, ok := lo.Find(workerStatuses, func(workerStatus tfv1.WorkerStatus) bool {
		return workerStatus.WorkerName == connection.Status.WorkerName
	})
	return !ok || workerStatus.WorkerPhase == tfv1.WorkerFailed, workerStatus
}

// SetupWithManager sets up the controller with the Manager.
func (r *TensorFusionConnectionReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.TensorFusionConnection{}).
		Named("tensorfusionconnection").
		Complete(r)
}
