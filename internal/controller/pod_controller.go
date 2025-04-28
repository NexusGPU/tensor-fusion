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

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
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
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// PodReconciler reconciles a Pod object
type PodReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;update;patch;delete;deletecollection
// +kubebuilder:rbac:groups=core,resources=pods/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=core,resources=pods/exec,verbs=create;get;update;patch
// +kubebuilder:rbac:groups=core,resources=pods/finalizers,verbs=update

// Add GPU connection for Pods using GPU
func (r *PodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)
	pod := &corev1.Pod{}
	if err := r.Get(ctx, req.NamespacedName, pod); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		log.Error(err, "Failed to get Pod")
		return ctrl.Result{}, err
	}
	// generate tensor fusion connections and apply to cluster
	tfConnection := generateTensorFusionConnection(pod)
	if tfConnection == nil {
		// not a tf pod skipped
		return ctrl.Result{}, nil
	}

	if _, ok := pod.Annotations[constants.TensorFusionEnabledReplicasAnnotation]; ok {
		shouldReturn, err := utils.HandleFinalizer(ctx, pod, r.Client, func(context context.Context, pod *corev1.Pod) (bool, error) {
			counter := &v1.TensorFusionPodCounter{Client: r.Client}
			if err := counter.Decrease(ctx, pod); err != nil {
				return false, err
			}
			return true, nil
		})
		if err != nil {
			return ctrl.Result{}, err
		}
		if shouldReturn {
			return ctrl.Result{}, nil
		}
	}

	existConn := &tfv1.TensorFusionConnection{}
	if err := r.Get(ctx, types.NamespacedName{Name: tfConnection.Name, Namespace: tfConnection.Namespace}, existConn); err != nil {
		if errors.IsNotFound(err) {
			if err := r.Client.Create(ctx, tfConnection); err != nil {
				return ctrl.Result{}, fmt.Errorf("create connection(%s) : %w", tfConnection.Namespace+"/"+tfConnection.Name, err)
			}
		}
	}

	return ctrl.Result{}, nil
}

func generateTensorFusionConnection(pod *corev1.Pod) *tfv1.TensorFusionConnection {
	workloadName, ok := pod.Annotations[constants.WorkloadKey]
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
		},
	}
	return connection
}

// SetupWithManager sets up the controller with the Manager.
func (r *PodReconciler) SetupWithManager(mgr ctrl.Manager) error {
	p, err := predicate.LabelSelectorPredicate(metav1.LabelSelector{
		MatchLabels: map[string]string{
			constants.Domain + "/enabled": "true",
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
