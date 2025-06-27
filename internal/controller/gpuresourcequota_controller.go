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
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/quota"
)

// GPUResourceQuotaReconciler reconciles a GPUResourceQuota object
type GPUResourceQuotaReconciler struct {
	client.Client
	Scheme     *runtime.Scheme
	Recorder   record.EventRecorder
	QuotaStore *quota.QuotaStore
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpuresourcequotas,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpuresourcequotas/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpuresourcequotas/finalizers,verbs=update
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *GPUResourceQuotaReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	quota := &tfv1.GPUResourceQuota{}
	if err := r.Get(ctx, req.NamespacedName, quota); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		log.FromContext(ctx).Error(err, "Failed to get GPUResourceQuota")
	}
	// Check for alert conditions (but don't update status - QuotaStore handles that)
	r.checkAlertConditions(quota)
	// each update status will trigger a requeue when mark quota as dirty by gpuallocator
	return ctrl.Result{}, nil
}

// checkAlertConditions checks and triggers alert events
func (r *GPUResourceQuotaReconciler) checkAlertConditions(quota *tfv1.GPUResourceQuota) {
	alertThreshold := int32(95)
	if quota.Spec.Total.AlertThresholdPercent != nil {
		alertThreshold = *quota.Spec.Total.AlertThresholdPercent
	}

	// Use calculator to get usage percentages
	percentages := r.QuotaStore.CalculateUsagePercent(quota.Namespace)

	// Check each resource type for alert threshold
	for resourceType, usagePercent := range percentages {
		if usagePercent >= int64(alertThreshold) {
			var message string
			switch resourceType {
			case "requests.tflops":
				message = fmt.Sprintf("Requests TFlops usage (%d%%) has reached alert threshold (%d%%)", usagePercent, alertThreshold)
			case "requests.vram":
				message = fmt.Sprintf("Requests VRAM usage (%d%%) has reached alert threshold (%d%%)", usagePercent, alertThreshold)
			case "workers":
				message = fmt.Sprintf("Workers usage (%d%%) has reached alert threshold (%d%%)", usagePercent, alertThreshold)
			}
			r.Recorder.Event(quota, "Warning", "AlertThresholdReached", message)
		}
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *GPUResourceQuotaReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.GPUResourceQuota{}).
		Complete(r)
}
