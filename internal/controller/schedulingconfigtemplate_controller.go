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

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/config"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// SchedulingConfigTemplateReconciler reconciles a SchedulingConfigTemplate object
type SchedulingConfigTemplateReconciler struct {
	client.Client
	Scheme                *runtime.Scheme
	ScheduleTemplateState config.ScheduleTemplateState
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=schedulingconfigtemplates,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=schedulingconfigtemplates/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=schedulingconfigtemplates/finalizers,verbs=update

// When deleted, need check if any GPU pool is using this template, if so, add warning event and requeue
// When updated, trigger the GPU pool status update for scheduling part
func (r *SchedulingConfigTemplateReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)

	// TODO(user): your logic here

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *SchedulingConfigTemplateReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.SchedulingConfigTemplate{}).
		Named("schedulingconfigtemplate").
		WithEventFilter(
			predicate.Funcs{
				CreateFunc: func(e event.CreateEvent) bool {
					r.ScheduleTemplateState.Set(e.Object.(*tfv1.SchedulingConfigTemplate).Name, e.Object.(*tfv1.SchedulingConfigTemplate))
					return true
				},
				UpdateFunc: func(e event.UpdateEvent) bool {
					r.ScheduleTemplateState.Set(e.ObjectNew.(*tfv1.SchedulingConfigTemplate).Name, e.ObjectNew.(*tfv1.SchedulingConfigTemplate))
					return true
				},
				DeleteFunc: func(e event.DeleteEvent) bool {
					r.ScheduleTemplateState.Delete(e.Object.(*tfv1.SchedulingConfigTemplate).Name)
					return true
				},
			},
		).
		Complete(r)
}
