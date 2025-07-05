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

	goerrors "errors"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
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

	workloadName := connection.Labels[constants.WorkloadKey]
	workload := &tfv1.TensorFusionWorkload{}
	if err := r.Get(ctx, client.ObjectKey{Name: workloadName, Namespace: connection.Namespace}, workload); err != nil {
		return ctrl.Result{}, fmt.Errorf("can not found TensorFusionWorkload: %w", err)
	}

	if workload.Spec.IsDynamicReplica() {
		err := r.createDedicatedWorker(ctx, workload, connection)
		if err != nil {
			return ctrl.Result{}, err
		}
	}

	needSelectWorker, err := r.shouldSelectWorker(ctx, connection)
	if err != nil {
		return ctrl.Result{}, err
	}

	// connection is fine and worker status is fine, do nothing
	if !needSelectWorker {
		return ctrl.Result{}, nil
	}

	log.Info("Selecting worker for connection", "connection", connection.Name, "namespace", connection.Namespace)
	if workload.Spec.IsDynamicReplica() {
		// 1st MODE: select the dedicated worker if it's running, otherwise wait utils it's becoming ready
		return ctrl.Result{}, r.syncDedicatedWorkerStatus(ctx, connection)
	} else {
		// 2nd MODE: fixed worker replicas, select a worker from workload's all workers
		return r.selectWorkerAndSyncStatusFromWorkerPool(ctx, connection, workload)
	}
}

func (r *TensorFusionConnectionReconciler) syncDedicatedWorkerStatus(ctx context.Context, connection *tfv1.TensorFusionConnection) error {
	pod := &v1.Pod{}
	if err := r.Get(ctx, client.ObjectKey{Name: connection.Name, Namespace: connection.Namespace}, pod); err != nil {
		return fmt.Errorf("failed to get dedicated worker pod for connection %w", err)
	}
	if pod.Status.Phase != v1.PodRunning {
		// dedicated worker pod is not running, wait for it to be running,
		// pod watcher will trigger reconcile, no need to requeue
		return nil
	} else {
		connection.Status.Phase = tfv1.WorkerRunning
		connection.Status.WorkerName = pod.Name
		revision := pod.ResourceVersion
		if revision == "" {
			revision = "0"
		}
		setConnectionWorkerURL(connection, pod.Status.PodIP, pod.Name, revision)
		if err := r.Status().Update(ctx, connection); err != nil {
			return fmt.Errorf("failed to update connection status: %w", err)
		}
		return nil
	}
}

func setConnectionWorkerURL(connection *tfv1.TensorFusionConnection, podIp string, podName string, revision string) {
	connection.Status.ConnectionURL = fmt.Sprintf("native+%s+%d+%s-%s", podIp, constants.TensorFusionRemoteWorkerPortNumber, podName, revision)
}

func (r *TensorFusionConnectionReconciler) selectWorkerAndSyncStatusFromWorkerPool(
	ctx context.Context,
	connection *tfv1.TensorFusionConnection,
	workload *tfv1.TensorFusionWorkload,
) (ctrl.Result, error) {
	if workload.Spec.Replicas == nil || *workload.Spec.Replicas <= 0 {
		return ctrl.Result{}, fmt.Errorf("invalid workload, replicas less than 1")
	}
	workloadName, ok := connection.Labels[constants.WorkloadKey]
	if !ok {
		return ctrl.Result{}, fmt.Errorf("missing workload label")
	}
	s, err := worker.SelectWorker(ctx, r.Client, workload, 1)
	if err != nil {
		if goerrors.Is(err, worker.ErrNoAvailableWorker) {
			// no available worker, wait for it to be available
			return ctrl.Result{RequeueAfter: constants.StatusCheckInterval}, nil
		}
		r.Recorder.Eventf(connection, v1.EventTypeWarning, "WorkerSelectionFailed", "Failed to select worker: %v", err)
		return ctrl.Result{}, err
	}
	r.Recorder.Eventf(connection, v1.EventTypeNormal, "WorkerSelected", "Worker %s successfully selected for connection", s.WorkerName)
	connection.Status.Phase = s.WorkerPhase
	connection.Labels[constants.WorkloadKey] = workloadName
	connection.Status.WorkerName = s.WorkerName
	connection.Labels[constants.LabelWorkerName] = s.WorkerName
	resourceVersion := s.ResourceVersion
	if resourceVersion == "" {
		resourceVersion = "0"
	}
	setConnectionWorkerURL(connection, s.WorkerIp, s.WorkerName, resourceVersion)
	if err := r.Status().Update(ctx, connection); err != nil {
		return ctrl.Result{}, fmt.Errorf("update connection status: %w", err)
	}

	err = r.patchMatchedWorkerLabel(ctx, connection, s.WorkerName)
	if err != nil {
		return ctrl.Result{}, err
	}
	r.Recorder.Eventf(connection, v1.EventTypeNormal, "ConnectionReady", "Connection URL: %s", connection.Status.ConnectionURL)
	return ctrl.Result{}, nil
}

// patch worker label so that when Pod status changed, the reconcile request could be correctly triggered
func (r *TensorFusionConnectionReconciler) patchMatchedWorkerLabel(ctx context.Context, connection *tfv1.TensorFusionConnection, latestWorkerName string) error {
	if workerName, ok := connection.Labels[constants.LabelWorkerName]; !ok {
		patch := []byte(`[{
					"op": "add",
					"path": "/metadata/labels/` + utils.EscapeJSONPointer(constants.LabelWorkerName) + `",
					"value": "` + latestWorkerName + `"}]`)
		if err := r.Patch(ctx, connection, client.RawPatch(types.JSONPatchType, patch)); err != nil {
			return fmt.Errorf("failed to update connection label: %w", err)
		}

	} else if workerName != latestWorkerName {
		patch := []byte(`[{
					"op": "replace", 
					"path": "/metadata/labels/` + utils.EscapeJSONPointer(constants.LabelWorkerName) + `", 
					"value": "` + latestWorkerName + `"
				}]`)
		if err := r.Patch(ctx, connection, client.RawPatch(types.JSONPatchType, patch)); err != nil {
			return fmt.Errorf("failed to update connection label: %w", err)
		}
	}
	return nil
}

func (r *TensorFusionConnectionReconciler) shouldSelectWorker(
	ctx context.Context, connection *tfv1.TensorFusionConnection,
) (needSelectWorker bool, err error) {
	if connection.Status.WorkerName != "" {
		// check if worker pod is still running
		pod := &v1.Pod{}
		if err := r.Get(ctx, client.ObjectKey{Name: connection.Status.WorkerName, Namespace: connection.Namespace}, pod); err != nil {
			if errors.IsNotFound(err) {
				needSelectWorker = true
			} else {
				return needSelectWorker, fmt.Errorf("failed to get worker pod: %w", err)
			}
		}
		// NOTE: no need to handle pod deleting since connection should be deleted at first, sync running status with Pod
		if pod.Status.Phase != v1.PodRunning {
			connection.Status.WorkerName = ""
			connection.Status.Phase = tfv1.WorkerFailed
			connection.Status.ConnectionURL = ""
			// set worker name to empty to trigger select worker again
			if updateErr := r.Status().Update(ctx, connection); updateErr != nil {
				return false, fmt.Errorf("failed to update connection status: %w", updateErr)
			}
			// let next reconcile loop to trigger select worker
			return false, nil
		} else if connection.Status.Phase != tfv1.WorkerRunning {
			// pod is running now, but connection is not running, update connection to running
			connection.Status.Phase = tfv1.WorkerRunning
			setConnectionWorkerURL(connection, pod.Status.PodIP, pod.Name, pod.ResourceVersion)
			if updateErr := r.Status().Update(ctx, connection); updateErr != nil {
				return false, fmt.Errorf("failed to update connection status: %w", updateErr)
			}
			// current worker is working again, no need to select another worker
			return false, nil
		}
	} else {
		if connection.Status.Phase == "" {
			connection.Status.Phase = tfv1.WorkerPending
			if updateErr := r.Status().Update(ctx, connection); updateErr != nil {
				return false, fmt.Errorf("failed to update connection status: %w", updateErr)
			}
		}
		needSelectWorker = true
	}
	return needSelectWorker, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *TensorFusionConnectionReconciler) SetupWithManager(mgr ctrl.Manager) error {
	p, err := predicate.LabelSelectorPredicate(metav1.LabelSelector{
		MatchLabels: map[string]string{
			constants.LabelComponent: constants.ComponentWorker,
		},
	})
	if err != nil {
		return err
	}

	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.TensorFusionConnection{}).
		Named("tensorfusionconnection").
		Watches(&v1.Pod{}, handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []reconcile.Request {
			pod := obj.(*v1.Pod)
			if pod.Annotations != nil && pod.Annotations[constants.IsLocalGPUAnnotation] == constants.TrueStringValue {
				// local GPU pod, do not trigger re-select worker since its same process forever
				return nil
			}

			// MODE 1: dedicated worker, only trigger reconcile for this dedicated worker
			if pod.Annotations != nil && pod.Annotations[constants.DedicatedWorkerAnnotation] == constants.TrueStringValue {
				return []reconcile.Request{
					{
						NamespacedName: client.ObjectKey{
							Namespace: pod.Namespace,
							Name:      pod.Name,
						},
					},
				}
			}

			// MODE 2: fixed worker replicas, try to find worker pod's related connections, trigger reconcile requests
			connectionList := &tfv1.TensorFusionConnectionList{}
			if err := r.Client.List(ctx, connectionList, client.MatchingLabels{
				constants.LabelWorkerName: pod.Name,
			}); err != nil {
				log.FromContext(ctx).Error(err, "Failed to list TensorFusionConnection of changed Pod", "pod", pod.Name)
				return nil
			}
			var requests []reconcile.Request
			for _, connection := range connectionList.Items {
				requests = append(requests, reconcile.Request{
					NamespacedName: client.ObjectKey{Namespace: connection.Namespace, Name: connection.Name},
				})
			}
			return requests
		}), builder.WithPredicates(p), builder.WithPredicates(predicate.Funcs{
			UpdateFunc: func(e event.UpdateEvent) bool {
				oldObj, ok1 := e.ObjectOld.(*v1.Pod)
				newObj, ok2 := e.ObjectNew.(*v1.Pod)
				if !ok1 || !ok2 {
					return false
				}
				return oldObj.Status.Phase != newObj.Status.Phase
			},
		})).
		Complete(r)
}

func (r *TensorFusionConnectionReconciler) createDedicatedWorker(ctx context.Context, workload *tfv1.TensorFusionWorkload, connection *tfv1.TensorFusionConnection) error {
	if !connection.DeletionTimestamp.IsZero() {
		// do nothing when connection being deleted
		return nil
	}

	// find GPU pool config to create worker generator
	gpuPoolName := workload.Spec.PoolName
	gpuPool := &tfv1.GPUPool{}
	if err := r.Get(ctx, client.ObjectKey{Name: gpuPoolName}, gpuPool); err != nil {
		return fmt.Errorf("gpu pool(%s) does not exist", gpuPoolName)
	}
	workerGenerator := &worker.WorkerGenerator{
		WorkerConfig:     gpuPool.Spec.ComponentConfig.Worker,
		HypervisorConfig: gpuPool.Spec.ComponentConfig.Hypervisor,
	}
	podTemplateHash, err := workerGenerator.PodTemplateHash(workload.Spec)
	if err != nil {
		return fmt.Errorf("get pod template hash: %w", err)
	}

	// check existing worker pod
	existingPod := &v1.Pod{}
	if err := r.Get(ctx, client.ObjectKey{Name: connection.Name, Namespace: connection.Namespace}, existingPod); err != nil {
		if errors.IsNotFound(err) {
			// dedicated worker Pod not found, create it
			if err := r.startDedicatedWorkerPod(ctx, workerGenerator, workload, podTemplateHash, connection); err != nil {
				return err
			} else {
				return nil
			}
		} else {
			return fmt.Errorf("failed to get dedicated worker  pod for connection %w", err)
		}
	}

	if !existingPod.DeletionTimestamp.IsZero() {
		// worker being deleting but connection is still there, should start another worker to avoid connection failed
		return fmt.Errorf("dedicated worker deleting but connection still there, reconcile later after deletion")
	}

	// pod exists, check if template hash is the same, if not delete current one,
	// it automatically triggers reconcile to create another
	if existingPod.Labels[constants.LabelKeyPodTemplateHash] != podTemplateHash {
		// template hash is different, delete the pod
		log.FromContext(ctx).Info("Template hash is different, delete the pod", "pod", existingPod.Name)
		if err := r.Delete(ctx, existingPod); err != nil {
			return fmt.Errorf("failed to delete dedicated worker pod for connection %w", err)
		}
	}
	return nil
}

func (r *TensorFusionConnectionReconciler) startDedicatedWorkerPod(ctx context.Context, workerGenerator *worker.WorkerGenerator, workload *tfv1.TensorFusionWorkload, podTemplateHash string, connection *tfv1.TensorFusionConnection) error {
	pod, err := workerGenerator.GenerateWorkerPod(ctx, workload)
	if err != nil {
		return fmt.Errorf("generate worker pod %w", err)
	}
	pod.GenerateName = ""
	pod.Name = connection.Name
	pod.Labels[constants.WorkloadKey] = workload.Name
	pod.Labels[constants.LabelKeyPodTemplateHash] = podTemplateHash
	pod.Annotations[constants.DedicatedWorkerAnnotation] = constants.TrueStringValue

	// Add finalizer for GPU resource cleanup
	pod.Finalizers = append(pod.Finalizers, constants.Finalizer)

	if err := ctrl.SetControllerReference(connection, pod, r.Scheme); err != nil {
		return fmt.Errorf("set owner reference %w", err)
	}
	if err := r.Create(ctx, pod); err != nil {
		return fmt.Errorf("create pod %w", err)
	}
	return nil
}
