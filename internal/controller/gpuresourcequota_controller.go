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

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	tensorfusionv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/quota"
)

// GPUResourceQuotaReconciler reconciles a GPUResourceQuota object
type GPUResourceQuotaReconciler struct {
	client.Client
	Scheme     *runtime.Scheme
	Recorder   record.EventRecorder
	calculator *quota.Calculator
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpuresourcequotas,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpuresourcequotas/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpuresourcequotas/finalizers,verbs=update
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionworkloads,verbs=get;list;watch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=workloadprofiles,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch

// NewGPUResourceQuotaReconciler creates a new GPUResourceQuotaReconciler
func NewGPUResourceQuotaReconciler(client client.Client, scheme *runtime.Scheme, recorder record.EventRecorder) *GPUResourceQuotaReconciler {
	return &GPUResourceQuotaReconciler{
		Client:     client,
		Scheme:     scheme,
		Recorder:   recorder,
		calculator: quota.NewCalculator(),
	}
}

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *GPUResourceQuotaReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Fetch the GPUResourceQuota instance
	quota := &tensorfusionv1.GPUResourceQuota{}
	if err := r.Get(ctx, req.NamespacedName, quota); err != nil {
		if errors.IsNotFound(err) {
			logger.Info("GPUResourceQuota resource not found. Ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get GPUResourceQuota")
		return ctrl.Result{}, err
	}

	// Calculate current usage for alert checking only
	usage, err := r.calculateUsage(ctx, quota.Namespace)
	if err != nil {
		logger.Error(err, "Failed to calculate resource usage")
		return ctrl.Result{}, err
	}

	// Check for alert conditions (but don't update status - QuotaStore handles that)
	r.checkAlertConditions(quota, usage)

	// Requeue after 30 seconds to continuously monitor usage
	return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
}

// calculateUsage calculates current resource usage in the namespace
func (r *GPUResourceQuotaReconciler) calculateUsage(ctx context.Context, namespace string) (*tensorfusionv1.GPUResourceUsage, error) {
	// List all TensorFusionWorkloads in the namespace
	workloadList := &tensorfusionv1.TensorFusionWorkloadList{}
	if err := r.List(ctx, workloadList, client.InNamespace(namespace)); err != nil {
		return nil, fmt.Errorf("failed to list TensorFusionWorkloads: %w", err)
	}

	// Use calculator to create zero usage
	usage := r.calculator.CreateZeroUsage()

	for _, workload := range workloadList.Items {
		// Skip workloads that are not running
		if workload.Status.Phase != "Running" && workload.Status.Phase != "Pending" {
			continue
		}

		// Add workload resources to usage directly from spec
		r.addWorkloadResources(usage, &workload)
	}

	return usage, nil
}

// addWorkloadResources adds workload resources to the total usage
func (r *GPUResourceQuotaReconciler) addWorkloadResources(usage *tensorfusionv1.GPUResourceUsage, workload *tensorfusionv1.TensorFusionWorkload) {
	replicas := int32(1)
	if workload.Spec.Replicas != nil {
		replicas = *workload.Spec.Replicas
	}

	// Use calculator to handle all resource aggregation
	requestsOperation := r.calculator.NewResourceOperation(workload.Spec.Resources.Requests, replicas)
	limitsOperation := r.calculator.NewResourceOperation(workload.Spec.Resources.Limits, replicas)

	// Apply requests to usage directly using calculator's logic
	r.calculator.SafeAdd(usage.RequestsTFlops, requestsOperation.TFlops)
	r.calculator.SafeAdd(usage.RequestsVRAM, requestsOperation.VRAM)
	r.calculator.SafeAdd(usage.LimitsTFlops, limitsOperation.TFlops)
	r.calculator.SafeAdd(usage.LimitsVRAM, limitsOperation.VRAM)
	*usage.Workers += replicas
}

// checkAlertConditions checks and triggers alert events
func (r *GPUResourceQuotaReconciler) checkAlertConditions(quota *tensorfusionv1.GPUResourceQuota, usage *tensorfusionv1.GPUResourceUsage) {
	alertThreshold := int32(95)
	if quota.Spec.Total.AlertThresholdPercent != nil {
		alertThreshold = *quota.Spec.Total.AlertThresholdPercent
	}

	// Use calculator to get usage percentages
	percentages := r.calculator.CalculateUsagePercent(quota, usage)

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
		For(&tensorfusionv1.GPUResourceQuota{}).
		Complete(r)
}
