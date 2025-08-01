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
	"strconv"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	v1 "github.com/NexusGPU/tensor-fusion/internal/webhook/v1"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// PodReconciler reconciles a Pod object
type PodReconciler struct {
	client.Client
	Scheme        *runtime.Scheme
	Allocator     *gpuallocator.GpuAllocator
	PortAllocator *portallocator.PortAllocator
}

// +kubebuilder:rbac:groups=core,resources=*,verbs=get;list;watch
// +kubebuilder:rbac:groups=storage.k8s.io,resources=*,verbs=get;list;watch
// +kubebuilder:rbac:groups=policy,resources=*,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=pods/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=core,resources=pods/exec,verbs=create;get;update;patch
// +kubebuilder:rbac:groups=core,resources=pods/finalizers,verbs=create;get;update;patch
// +kubebuilder:rbac:groups=core,resources=pods/binding,verbs=create;get;update;patch

// Add GPU connection for Pods using GPU
// Have to create TensorFusion connection here because pod UID not available in MutatingWebhook
func (r *PodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)
	pod := &corev1.Pod{}
	if err := r.Get(ctx, req.NamespacedName, pod); err != nil {
		if errors.IsNotFound(err) {
			r.Allocator.DeallocByPodIdentifier(ctx, req.NamespacedName)
			log.Info("Released GPU resources when pod deleted", "pod", req.NamespacedName)
			return ctrl.Result{}, nil
		}
		log.Error(err, "Failed to get Pod")
		return ctrl.Result{}, err
	}
	// avoid possible nil pointer error
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}

	// check if need to set owner reference, when standalone Pod created and no owner,
	// the related workload's owner should be the Pod so that to be deleted automatically along with Pod deletion
	if ownedWorkloadName, ok := pod.Annotations[constants.SetPendingOwnedWorkloadAnnotation]; ok {
		log.Info("Setting pending owned workload", "pod", pod.Name, "ownedWorkload", ownedWorkloadName)
		if err := r.setPendingOwnedWorkload(ctx, pod, ownedWorkloadName); err != nil {
			if errors.IsNotFound(err) {
				log.Error(err, "Orphaned pod, failed to set pending owned workload because owner not found",
					"pod", pod.Name, "ownedWorkload", ownedWorkloadName)
				return ctrl.Result{}, nil
			}
			return ctrl.Result{}, err
		}
		delete(pod.Annotations, constants.SetPendingOwnedWorkloadAnnotation)
		log.Info("Pending owned workload set", "pod", pod.Name, "ownedWorkload", ownedWorkloadName)
		if err := r.Update(ctx, pod); err != nil {
			return ctrl.Result{}, err
		}
	}

	// Release cluster level port when Pod deleted
	if !pod.DeletionTimestamp.IsZero() {
		if pod.Annotations[constants.GenHostPortLabel] == constants.GenHostPortLabelValue {
			podPortNumber, _ := strconv.Atoi(pod.Annotations[constants.GenPortNumberAnnotation])
			_ = r.PortAllocator.ReleaseClusterLevelHostPort(pod.Name, podPortNumber, false)
			log.Info("Released port", "pod", pod.Name, "port", podPortNumber)
		}
	}

	if pod.Labels[constants.LabelComponent] == constants.ComponentWorker {
		metrics.SetWorkerMetricsByWorkload(pod)

		shouldReturn, err := r.handleWorkerPodFinalizer(ctx, pod)
		if err != nil {
			return ctrl.Result{}, err
		}
		if shouldReturn {
			return ctrl.Result{}, nil
		}
	}

	// generate tensor fusion connections and apply to cluster
	if pod.Labels[constants.LabelComponent] == constants.ComponentClient {

		if controllerutil.ContainsFinalizer(pod, constants.Finalizer) {
			log.Info("tensor-fusion client pod should not have tensor fusion finalizer, remove it", "pod", pod.Name)
			controllerutil.RemoveFinalizer(pod, constants.Finalizer)
			return ctrl.Result{}, r.Update(ctx, pod)
		}

		tfConnection := buildTensorFusionConnectionObj(pod)
		if tfConnection == nil {
			log.Info("Pod is not a TensorFusion client, skipped, this should never happen", "pod", pod.Name)
			return ctrl.Result{}, nil
		}

		existConn := &tfv1.TensorFusionConnection{}
		if err := r.Get(ctx, types.NamespacedName{Name: tfConnection.Name, Namespace: tfConnection.Namespace}, existConn); err != nil {
			if errors.IsNotFound(err) {
				if err := r.Create(ctx, tfConnection); err != nil {
					return ctrl.Result{}, fmt.Errorf("create connection(%s) : %w", tfConnection.Namespace+"/"+tfConnection.Name, err)
				}
			}
		}
	}

	return ctrl.Result{}, nil
}

func (r *PodReconciler) handleWorkerPodFinalizer(ctx context.Context, pod *corev1.Pod) (bool, error) {
	// Handle our GPU resource cleanup finalizer
	shouldReturn, err := utils.HandleFinalizer(ctx, pod, r.Client, func(ctx context.Context, obj *corev1.Pod) (bool, error) {
		metrics.RemoveWorkerMetrics(pod.Name, pod.DeletionTimestamp.Time)
		counter := &v1.TensorFusionPodCounter{Client: r.Client}
		if err := counter.Decrease(ctx, pod); err != nil {
			return false, err
		}
		return true, nil
	})
	if err != nil {
		return false, err
	}
	return shouldReturn, nil
}

func (r *PodReconciler) setPendingOwnedWorkload(ctx context.Context, pod *corev1.Pod, ownedWorkloadName string) error {
	tfWorkload := &tfv1.TensorFusionWorkload{}
	if err := r.Get(ctx, types.NamespacedName{Name: ownedWorkloadName, Namespace: pod.Namespace}, tfWorkload); err != nil {
		return err
	}
	if err := controllerutil.SetControllerReference(pod, tfWorkload, r.Scheme); err != nil {
		return err
	}
	return r.Update(ctx, tfWorkload)
}

func buildTensorFusionConnectionObj(pod *corev1.Pod) *tfv1.TensorFusionConnection {
	workloadName, ok := pod.Annotations[constants.SelectedWorkloadAnnotation]
	if !ok {
		return nil
	}
	nameNamespace := findConnectionNameNamespace(pod)
	if nameNamespace.Name == "" || nameNamespace.Namespace == "" {
		return nil
	}
	connection := &tfv1.TensorFusionConnection{
		ObjectMeta: metav1.ObjectMeta{
			Name:      nameNamespace.Name,
			Namespace: nameNamespace.Namespace,
			Labels: map[string]string{
				constants.WorkloadKey: workloadName,
			},
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "v1",
					Kind:       "Pod",
					Name:       pod.Name,
					UID:        pod.UID,
				},
			},
		},
		Spec: tfv1.TensorFusionConnectionSpec{
			WorkloadName: workloadName,
			ClientPod:    pod.Name,
		},
	}
	return connection
}

// SetupWithManager sets up the controller with the Manager.
func (r *PodReconciler) SetupWithManager(mgr ctrl.Manager) error {
	p, err := predicate.LabelSelectorPredicate(metav1.LabelSelector{
		MatchExpressions: []metav1.LabelSelectorRequirement{
			{
				Key:      constants.LabelComponent,
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{constants.ComponentClient, constants.ComponentWorker},
			},
		},
	})
	if err != nil {
		return fmt.Errorf("unable to create predicate: %w", err)
	}
	return ctrl.NewControllerManagedBy(mgr).
		For(&corev1.Pod{}, builder.WithPredicates(p)).
		Named("pod").
		Complete(r)
}

// findConnectionNameNamespace extracts the connection name and namespace from the container's environment variables
func findConnectionNameNamespace(pod *corev1.Pod) client.ObjectKey {
	connectionNameNamespace := client.ObjectKey{}

	for _, container := range pod.Spec.Containers {
		connectionName, ok := lo.Find(container.Env, func(env corev1.EnvVar) bool {
			return env.Name == constants.ConnectionNameEnv
		})
		if !ok {
			continue
		}
		connectionNamespace, ok := lo.Find(container.Env, func(env corev1.EnvVar) bool {
			return env.Name == constants.ConnectionNamespaceEnv
		})
		if !ok {
			continue
		}
		connectionNameNamespace.Name = connectionName.Value
		connectionNameNamespace.Namespace = connectionNamespace.Value
		break
	}
	return connectionNameNamespace
}
