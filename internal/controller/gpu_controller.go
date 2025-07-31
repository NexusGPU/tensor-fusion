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
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// GPUReconciler reconciles a GPU object
type GPUReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpus,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpus/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpus/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *GPUReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	gpu := &tfv1.GPU{}
	if err := r.Get(ctx, req.NamespacedName, gpu); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	gpunode := &tfv1.GPUNode{}
	if err := r.Get(ctx, client.ObjectKey{Name: gpu.Labels[constants.LabelKeyOwner]}, gpunode); err != nil {
		return ctrl.Result{}, fmt.Errorf("can not get node %s: %w", gpu.Labels[constants.LabelKeyOwner], err)
	}

	// Fix old version issue when discovery job not set UsedBy field
	if gpu.Status.UsedBy == "" && gpu.Status.UUID != "" {
		patch := client.MergeFrom(gpu.DeepCopy())
		gpu.Status.UsedBy = tfv1.UsedByTensorFusion
		if err := r.Status().Patch(ctx, gpu, patch); err != nil {
			return ctrl.Result{}, fmt.Errorf("patch gpu %s: %w", gpu.Name, err)
		}
		return ctrl.Result{}, nil
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *GPUReconciler) SetupWithManager(ctx context.Context, mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.GPU{}).
		Named("gpu").
		Complete(r)
}
