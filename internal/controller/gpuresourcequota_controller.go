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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	tensorfusionv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// GPUResourceQuotaReconciler reconciles a GPUResourceQuota object
type GPUResourceQuotaReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpuresourcequotas,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpuresourcequotas/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpuresourcequotas/finalizers,verbs=update
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=tensorfusionworkloads,verbs=get;list;watch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=workloadprofiles,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch

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

	// Calculate current usage
	usage, err := r.calculateUsage(ctx, quota.Namespace)
	if err != nil {
		logger.Error(err, "Failed to calculate resource usage")
		return ctrl.Result{}, err
	}

	// Update status
	if err := r.updateStatus(ctx, quota, usage); err != nil {
		logger.Error(err, "Failed to update status")
		return ctrl.Result{}, err
	}

	// Check for alert conditions
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

	usage := &tensorfusionv1.GPUResourceUsage{
		RequestsTFlops: resource.NewQuantity(0, resource.DecimalSI),
		RequestsVRAM:   resource.NewQuantity(0, resource.BinarySI),
		LimitsTFlops:   resource.NewQuantity(0, resource.DecimalSI),
		LimitsVRAM:     resource.NewQuantity(0, resource.BinarySI),
		Workers:        new(int32),
	}

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

	// Add requests
	requestsTFlops := workload.Spec.Resources.Requests.Tflops.DeepCopy()
	requestsTFlops.Set(requestsTFlops.Value() * int64(replicas))
	usage.RequestsTFlops.Add(requestsTFlops)

	requestsVRAM := workload.Spec.Resources.Requests.Vram.DeepCopy()
	requestsVRAM.Set(requestsVRAM.Value() * int64(replicas))
	usage.RequestsVRAM.Add(requestsVRAM)

	// Add limits
	limitsTFlops := workload.Spec.Resources.Limits.Tflops.DeepCopy()
	limitsTFlops.Set(limitsTFlops.Value() * int64(replicas))
	usage.LimitsTFlops.Add(limitsTFlops)

	limitsVRAM := workload.Spec.Resources.Limits.Vram.DeepCopy()
	limitsVRAM.Set(limitsVRAM.Value() * int64(replicas))
	usage.LimitsVRAM.Add(limitsVRAM)

	*usage.Workers += replicas
}

// updateStatus updates the GPUResourceQuota status
func (r *GPUResourceQuotaReconciler) updateStatus(ctx context.Context, quota *tensorfusionv1.GPUResourceQuota, usage *tensorfusionv1.GPUResourceUsage) error {
	// Calculate available percentages
	availablePercent := r.calculateAvailablePercent(quota, usage)

	// Update status
	quota.Status.Used = *usage
	quota.Status.AvailablePercent = *availablePercent
	now := metav1.Now()
	quota.Status.LastUpdateTime = &now

	// Update conditions
	r.updateConditions(quota, usage, availablePercent)

	return r.Status().Update(ctx, quota)
}

// calculateAvailablePercent calculates available percentage for each resource
func (r *GPUResourceQuotaReconciler) calculateAvailablePercent(quota *tensorfusionv1.GPUResourceQuota, usage *tensorfusionv1.GPUResourceUsage) *tensorfusionv1.GPUResourceAvailablePercent {
	percent := &tensorfusionv1.GPUResourceAvailablePercent{}

	// Calculate requests.tflops percentage
	if quota.Spec.Total.RequestsTFlops != nil && usage.RequestsTFlops != nil {
		total := quota.Spec.Total.RequestsTFlops.Value()
		used := usage.RequestsTFlops.Value()
		if total > 0 {
			available := (total - used) * 100 / total
			if available < 0 {
				available = 0
			}
			percent.RequestsTFlops = &available
		}
	}

	// Calculate requests.vram percentage
	if quota.Spec.Total.RequestsVRAM != nil && usage.RequestsVRAM != nil {
		total := quota.Spec.Total.RequestsVRAM.Value()
		used := usage.RequestsVRAM.Value()
		if total > 0 {
			available := (total - used) * 100 / total
			if available < 0 {
				available = 0
			}
			percent.RequestsVRAM = &available
		}
	}

	// Calculate limits.tflops percentage
	if quota.Spec.Total.LimitsTFlops != nil && usage.LimitsTFlops != nil {
		total := quota.Spec.Total.LimitsTFlops.Value()
		used := usage.LimitsTFlops.Value()
		if total > 0 {
			available := (total - used) * 100 / total
			if available < 0 {
				available = 0
			}
			percent.LimitsTFlops = &available
		}
	}

	// Calculate limits.vram percentage
	if quota.Spec.Total.LimitsVRAM != nil && usage.LimitsVRAM != nil {
		total := quota.Spec.Total.LimitsVRAM.Value()
		used := usage.LimitsVRAM.Value()
		if total > 0 {
			available := (total - used) * 100 / total
			if available < 0 {
				available = 0
			}
			percent.LimitsVRAM = &available
		}
	}

	// Calculate workers percentage
	if quota.Spec.Total.Workers != nil && usage.Workers != nil {
		total := int64(*quota.Spec.Total.Workers)
		used := int64(*usage.Workers)
		if total > 0 {
			available := (total - used) * 100 / total
			if available < 0 {
				available = 0
			}
			percent.Workers = &available
		}
	}

	return percent
}

// updateConditions updates the conditions based on current usage
func (r *GPUResourceQuotaReconciler) updateConditions(quota *tensorfusionv1.GPUResourceQuota, usage *tensorfusionv1.GPUResourceUsage, availablePercent *tensorfusionv1.GPUResourceAvailablePercent) {
	now := metav1.Now()

	// Check if quota is exceeded
	exceeded := r.isQuotaExceeded(quota, usage)

	// Check if alert threshold is reached
	alertThreshold := int32(95)
	if quota.Spec.Total.AlertThresholdPercent != nil {
		alertThreshold = *quota.Spec.Total.AlertThresholdPercent
	}
	alertReached := r.isAlertThresholdReached(availablePercent, alertThreshold)

	// Update Ready condition
	readyCondition := metav1.Condition{
		Type:               string(tensorfusionv1.GPUResourceQuotaConditionReady),
		Status:             metav1.ConditionTrue,
		LastTransitionTime: now,
		Reason:             "QuotaReady",
		Message:            "GPUResourceQuota is ready and monitoring resource usage",
	}

	// Update Exceeded condition
	exceededCondition := metav1.Condition{
		Type:               string(tensorfusionv1.GPUResourceQuotaConditionExceeded),
		Status:             metav1.ConditionFalse,
		LastTransitionTime: now,
		Reason:             "QuotaNotExceeded",
		Message:            "Resource usage is within quota limits",
	}
	if exceeded {
		exceededCondition.Status = metav1.ConditionTrue
		exceededCondition.Reason = "QuotaExceeded"
		exceededCondition.Message = "Resource usage has exceeded quota limits"
	}

	// Update AlertThresholdReached condition
	alertCondition := metav1.Condition{
		Type:               string(tensorfusionv1.GPUResourceQuotaConditionAlertThresholdReached),
		Status:             metav1.ConditionFalse,
		LastTransitionTime: now,
		Reason:             "AlertThresholdNotReached",
		Message:            fmt.Sprintf("Resource usage is below alert threshold (%d%%)", alertThreshold),
	}
	if alertReached {
		alertCondition.Status = metav1.ConditionTrue
		alertCondition.Reason = "AlertThresholdReached"
		alertCondition.Message = fmt.Sprintf("Resource usage has reached alert threshold (%d%%)", alertThreshold)
	}

	// Update conditions
	quota.Status.Conditions = []metav1.Condition{readyCondition, exceededCondition, alertCondition}
}

// isQuotaExceeded checks if any quota limit is exceeded
func (r *GPUResourceQuotaReconciler) isQuotaExceeded(quota *tensorfusionv1.GPUResourceQuota, usage *tensorfusionv1.GPUResourceUsage) bool {
	if quota.Spec.Total.RequestsTFlops != nil && usage.RequestsTFlops != nil {
		if usage.RequestsTFlops.Cmp(*quota.Spec.Total.RequestsTFlops) > 0 {
			return true
		}
	}
	if quota.Spec.Total.RequestsVRAM != nil && usage.RequestsVRAM != nil {
		if usage.RequestsVRAM.Cmp(*quota.Spec.Total.RequestsVRAM) > 0 {
			return true
		}
	}
	if quota.Spec.Total.LimitsTFlops != nil && usage.LimitsTFlops != nil {
		if usage.LimitsTFlops.Cmp(*quota.Spec.Total.LimitsTFlops) > 0 {
			return true
		}
	}
	if quota.Spec.Total.LimitsVRAM != nil && usage.LimitsVRAM != nil {
		if usage.LimitsVRAM.Cmp(*quota.Spec.Total.LimitsVRAM) > 0 {
			return true
		}
	}
	if quota.Spec.Total.Workers != nil && usage.Workers != nil {
		if *usage.Workers > *quota.Spec.Total.Workers {
			return true
		}
	}
	return false
}

// isAlertThresholdReached checks if alert threshold is reached for any resource
func (r *GPUResourceQuotaReconciler) isAlertThresholdReached(availablePercent *tensorfusionv1.GPUResourceAvailablePercent, threshold int32) bool {
	thresholdInt64 := int64(100 - threshold)
	if availablePercent.RequestsTFlops != nil && *availablePercent.RequestsTFlops < thresholdInt64 {
		return true
	}
	if availablePercent.RequestsVRAM != nil && *availablePercent.RequestsVRAM < thresholdInt64 {
		return true
	}
	if availablePercent.LimitsTFlops != nil && *availablePercent.LimitsTFlops < thresholdInt64 {
		return true
	}
	if availablePercent.LimitsVRAM != nil && *availablePercent.LimitsVRAM < thresholdInt64 {
		return true
	}
	if availablePercent.Workers != nil && *availablePercent.Workers < thresholdInt64 {
		return true
	}
	return false
}

// checkAlertConditions checks and triggers alert events
func (r *GPUResourceQuotaReconciler) checkAlertConditions(quota *tensorfusionv1.GPUResourceQuota, usage *tensorfusionv1.GPUResourceUsage) {
	alertThreshold := int32(95)
	if quota.Spec.Total.AlertThresholdPercent != nil {
		alertThreshold = *quota.Spec.Total.AlertThresholdPercent
	}

	// Check each resource type for alert threshold
	if quota.Spec.Total.RequestsTFlops != nil && usage.RequestsTFlops != nil {
		usagePercent := usage.RequestsTFlops.Value() * 100 / quota.Spec.Total.RequestsTFlops.Value()
		if usagePercent >= int64(alertThreshold) {
			r.Recorder.Event(quota, "Warning", "AlertThresholdReached",
				fmt.Sprintf("Requests TFlops usage (%d%%) has reached alert threshold (%d%%)", usagePercent, alertThreshold))
		}
	}

	if quota.Spec.Total.RequestsVRAM != nil && usage.RequestsVRAM != nil {
		usagePercent := usage.RequestsVRAM.Value() * 100 / quota.Spec.Total.RequestsVRAM.Value()
		if usagePercent >= int64(alertThreshold) {
			r.Recorder.Event(quota, "Warning", "AlertThresholdReached",
				fmt.Sprintf("Requests VRAM usage (%d%%) has reached alert threshold (%d%%)", usagePercent, alertThreshold))
		}
	}

	if quota.Spec.Total.Workers != nil && usage.Workers != nil {
		usagePercent := int64(*usage.Workers) * 100 / int64(*quota.Spec.Total.Workers)
		if usagePercent >= int64(alertThreshold) {
			r.Recorder.Event(quota, "Warning", "AlertThresholdReached",
				fmt.Sprintf("Workers usage (%d%%) has reached alert threshold (%d%%)", usagePercent, alertThreshold))
		}
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *GPUResourceQuotaReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tensorfusionv1.GPUResourceQuota{}).
		Complete(r)
}
