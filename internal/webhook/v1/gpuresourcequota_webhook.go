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

package v1

import (
	"context"
	"fmt"
	"net/http"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	tensorfusionv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// GPUResourceQuotaValidator validates TensorFusionWorkload against GPUResourceQuota
type GPUResourceQuotaValidator struct {
	Client  client.Client
	decoder admission.Decoder
}

//+kubebuilder:webhook:path=/validate-tensor-fusion-ai-v1-tensorfusionworkload,mutating=false,failurePolicy=fail,sideEffects=None,groups=tensor-fusion.ai,resources=tensorfusionworkloads,verbs=create;update,versions=v1,name=vgpuresourcequota.kb.io,admissionReviewVersions=v1

// Handle validates TensorFusionWorkload against GPUResourceQuota
func (v *GPUResourceQuotaValidator) Handle(ctx context.Context, req admission.Request) admission.Response {
	logger := log.FromContext(ctx)

	workload := &tensorfusionv1.TensorFusionWorkload{}
	if err := v.decoder.Decode(req, workload); err != nil {
		logger.Error(err, "Failed to decode TensorFusionWorkload")
		return admission.Errored(http.StatusBadRequest, err)
	}

	// Get the GPUResourceQuota for this namespace
	quota := &tensorfusionv1.GPUResourceQuota{}
	quotaKey := types.NamespacedName{
		Name:      workload.Namespace,
		Namespace: workload.Namespace,
	}

	if err := v.Client.Get(ctx, quotaKey, quota); err != nil {
		if errors.IsNotFound(err) {
			// No quota defined for this namespace, allow the workload
			logger.Info("No GPUResourceQuota found for namespace, allowing workload", "namespace", workload.Namespace)
			return admission.Allowed("No quota defined")
		}
		logger.Error(err, "Failed to get GPUResourceQuota")
		return admission.Errored(http.StatusInternalServerError, err)
	}

	// Apply default values if needed
	if err := v.applyDefaults(workload, quota); err != nil {
		logger.Error(err, "Failed to apply defaults")
		return admission.Errored(http.StatusInternalServerError, err)
	}

	// Validate against single workload limits
	if err := v.validateSingleWorkloadLimits(workload, quota); err != nil {
		logger.Info("Workload violates single workload limits", "error", err)
		return admission.Denied(err.Error())
	}

	// Calculate current usage in the namespace
	currentUsage, err := v.calculateCurrentUsage(ctx, workload.Namespace, workload.Name)
	if err != nil {
		logger.Error(err, "Failed to calculate current usage")
		return admission.Errored(http.StatusInternalServerError, err)
	}

	// Calculate new workload resource requirements
	newUsage := v.calculateWorkloadUsage(workload)

	// Check if adding this workload would exceed quota
	if err := v.validateTotalQuota(currentUsage, newUsage, quota); err != nil {
		logger.Info("Workload would exceed total quota", "error", err)
		return admission.Denied(err.Error())
	}

	logger.Info("Workload validation passed", "workload", workload.Name, "namespace", workload.Namespace)
	return admission.Allowed("Quota validation passed")
}

// applyDefaults applies default values from quota to workload if not specified
func (v *GPUResourceQuotaValidator) applyDefaults(workload *tensorfusionv1.TensorFusionWorkload, quota *tensorfusionv1.GPUResourceQuota) error {
	// Apply default TFlops if not specified
	if workload.Spec.Resources.Requests.Tflops.IsZero() && quota.Spec.Single.DefaultRequest != nil && quota.Spec.Single.DefaultRequest.TFlops != nil {
		workload.Spec.Resources.Requests.Tflops = *quota.Spec.Single.DefaultRequest.TFlops
	}

	// Apply default VRAM if not specified
	if workload.Spec.Resources.Requests.Vram.IsZero() && quota.Spec.Single.DefaultRequest != nil && quota.Spec.Single.DefaultRequest.VRAM != nil {
		workload.Spec.Resources.Requests.Vram = *quota.Spec.Single.DefaultRequest.VRAM
	}

	// Apply default limits if not specified
	if workload.Spec.Resources.Limits.Tflops.IsZero() && quota.Spec.Single.Default != nil && quota.Spec.Single.Default.TFlops != nil {
		workload.Spec.Resources.Limits.Tflops = *quota.Spec.Single.Default.TFlops
	}

	if workload.Spec.Resources.Limits.Vram.IsZero() && quota.Spec.Single.Default != nil && quota.Spec.Single.Default.VRAM != nil {
		workload.Spec.Resources.Limits.Vram = *quota.Spec.Single.Default.VRAM
	}

	return nil
}

// validateSingleWorkloadLimits validates workload against per-workload limits
func (v *GPUResourceQuotaValidator) validateSingleWorkloadLimits(workload *tensorfusionv1.TensorFusionWorkload, quota *tensorfusionv1.GPUResourceQuota) error {
	if quota.Spec.Single.Max == nil {
		return nil
	}

	replicas := int32(1)
	if workload.Spec.Replicas != nil {
		replicas = *workload.Spec.Replicas
	}

	// Check max TFlops
	if quota.Spec.Single.Max.TFlops != nil {
		totalTFlops := workload.Spec.Resources.Requests.Tflops.DeepCopy()
		totalTFlops.Set(totalTFlops.Value() * int64(replicas))
		if totalTFlops.Cmp(*quota.Spec.Single.Max.TFlops) > 0 {
			return fmt.Errorf("workload TFlops (%s) exceeds maximum allowed (%s)", totalTFlops.String(), quota.Spec.Single.Max.TFlops.String())
		}
	}

	// Check max VRAM
	if quota.Spec.Single.Max.VRAM != nil {
		totalVRAM := workload.Spec.Resources.Requests.Vram.DeepCopy()
		totalVRAM.Set(totalVRAM.Value() * int64(replicas))
		if totalVRAM.Cmp(*quota.Spec.Single.Max.VRAM) > 0 {
			return fmt.Errorf("workload VRAM (%s) exceeds maximum allowed (%s)", totalVRAM.String(), quota.Spec.Single.Max.VRAM.String())
		}
	}

	// Check max workers
	if quota.Spec.Single.Max.Workers != nil && replicas > *quota.Spec.Single.Max.Workers {
		return fmt.Errorf("workload replicas (%d) exceeds maximum allowed (%d)", replicas, *quota.Spec.Single.Max.Workers)
	}

	// Check min TFlops
	if quota.Spec.Single.Min != nil && quota.Spec.Single.Min.TFlops != nil {
		if workload.Spec.Resources.Requests.Tflops.Cmp(*quota.Spec.Single.Min.TFlops) < 0 {
			return fmt.Errorf("workload TFlops (%s) is below minimum required (%s)", workload.Spec.Resources.Requests.Tflops.String(), quota.Spec.Single.Min.TFlops.String())
		}
	}

	// Check min VRAM
	if quota.Spec.Single.Min != nil && quota.Spec.Single.Min.VRAM != nil {
		if workload.Spec.Resources.Requests.Vram.Cmp(*quota.Spec.Single.Min.VRAM) < 0 {
			return fmt.Errorf("workload VRAM (%s) is below minimum required (%s)", workload.Spec.Resources.Requests.Vram.String(), quota.Spec.Single.Min.VRAM.String())
		}
	}

	return nil
}

// calculateCurrentUsage calculates current resource usage in the namespace
func (v *GPUResourceQuotaValidator) calculateCurrentUsage(ctx context.Context, namespace, excludeWorkload string) (*tensorfusionv1.GPUResourceUsage, error) {
	workloadList := &tensorfusionv1.TensorFusionWorkloadList{}
	if err := v.Client.List(ctx, workloadList, client.InNamespace(namespace)); err != nil {
		return nil, fmt.Errorf("failed to list workloads: %w", err)
	}

	usage := &tensorfusionv1.GPUResourceUsage{
		RequestsTFlops: resource.NewQuantity(0, resource.DecimalSI),
		RequestsVRAM:   resource.NewQuantity(0, resource.BinarySI),
		LimitsTFlops:   resource.NewQuantity(0, resource.DecimalSI),
		LimitsVRAM:     resource.NewQuantity(0, resource.BinarySI),
		Workers:        new(int32),
	}

	for _, workload := range workloadList.Items {
		// Skip the workload being validated (for updates)
		if workload.Name == excludeWorkload {
			continue
		}

		// Skip workloads that are not running or pending
		if workload.Status.Phase != "Running" && workload.Status.Phase != "Pending" {
			continue
		}

		// Add workload resources to usage
		v.addWorkloadToUsage(usage, &workload)
	}

	return usage, nil
}

// calculateWorkloadUsage calculates resource usage for a single workload
func (v *GPUResourceQuotaValidator) calculateWorkloadUsage(workload *tensorfusionv1.TensorFusionWorkload) *tensorfusionv1.GPUResourceUsage {
	replicas := int32(1)
	if workload.Spec.Replicas != nil {
		replicas = *workload.Spec.Replicas
	}

	requestsTFlops := workload.Spec.Resources.Requests.Tflops.DeepCopy()
	requestsVRAM := workload.Spec.Resources.Requests.Vram.DeepCopy()
	limitsTFlops := workload.Spec.Resources.Limits.Tflops.DeepCopy()
	limitsVRAM := workload.Spec.Resources.Limits.Vram.DeepCopy()

	usage := &tensorfusionv1.GPUResourceUsage{
		RequestsTFlops: &requestsTFlops,
		RequestsVRAM:   &requestsVRAM,
		LimitsTFlops:   &limitsTFlops,
		LimitsVRAM:     &limitsVRAM,
		Workers:        &replicas,
	}

	// Scale by replicas
	usage.RequestsTFlops.Set(usage.RequestsTFlops.Value() * int64(replicas))
	usage.RequestsVRAM.Set(usage.RequestsVRAM.Value() * int64(replicas))
	usage.LimitsTFlops.Set(usage.LimitsTFlops.Value() * int64(replicas))
	usage.LimitsVRAM.Set(usage.LimitsVRAM.Value() * int64(replicas))

	return usage
}

// addWorkloadToUsage adds a workload's resource usage to the total
func (v *GPUResourceQuotaValidator) addWorkloadToUsage(usage *tensorfusionv1.GPUResourceUsage, workload *tensorfusionv1.TensorFusionWorkload) {
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

	// Add workers
	*usage.Workers += replicas
}

// validateTotalQuota validates that current + new usage doesn't exceed quota
func (v *GPUResourceQuotaValidator) validateTotalQuota(currentUsage, newUsage *tensorfusionv1.GPUResourceUsage, quota *tensorfusionv1.GPUResourceQuota) error {
	// Calculate total usage after adding new workload
	requestsTFlops := currentUsage.RequestsTFlops.DeepCopy()
	requestsVRAM := currentUsage.RequestsVRAM.DeepCopy()
	limitsTFlops := currentUsage.LimitsTFlops.DeepCopy()
	limitsVRAM := currentUsage.LimitsVRAM.DeepCopy()

	totalUsage := &tensorfusionv1.GPUResourceUsage{
		RequestsTFlops: &requestsTFlops,
		RequestsVRAM:   &requestsVRAM,
		LimitsTFlops:   &limitsTFlops,
		LimitsVRAM:     &limitsVRAM,
		Workers:        new(int32),
	}
	*totalUsage.Workers = *currentUsage.Workers

	// Add new workload usage
	totalUsage.RequestsTFlops.Add(*newUsage.RequestsTFlops)
	totalUsage.RequestsVRAM.Add(*newUsage.RequestsVRAM)
	totalUsage.LimitsTFlops.Add(*newUsage.LimitsTFlops)
	totalUsage.LimitsVRAM.Add(*newUsage.LimitsVRAM)
	*totalUsage.Workers += *newUsage.Workers

	// Check against quota limits
	if quota.Spec.Total.RequestsTFlops != nil && totalUsage.RequestsTFlops.Cmp(*quota.Spec.Total.RequestsTFlops) > 0 {
		return fmt.Errorf("total requests.tflops (%s) would exceed quota (%s)", totalUsage.RequestsTFlops.String(), quota.Spec.Total.RequestsTFlops.String())
	}

	if quota.Spec.Total.RequestsVRAM != nil && totalUsage.RequestsVRAM.Cmp(*quota.Spec.Total.RequestsVRAM) > 0 {
		return fmt.Errorf("total requests.vram (%s) would exceed quota (%s)", totalUsage.RequestsVRAM.String(), quota.Spec.Total.RequestsVRAM.String())
	}

	if quota.Spec.Total.LimitsTFlops != nil && totalUsage.LimitsTFlops.Cmp(*quota.Spec.Total.LimitsTFlops) > 0 {
		return fmt.Errorf("total limits.tflops (%s) would exceed quota (%s)", totalUsage.LimitsTFlops.String(), quota.Spec.Total.LimitsTFlops.String())
	}

	if quota.Spec.Total.LimitsVRAM != nil && totalUsage.LimitsVRAM.Cmp(*quota.Spec.Total.LimitsVRAM) > 0 {
		return fmt.Errorf("total limits.vram (%s) would exceed quota (%s)", totalUsage.LimitsVRAM.String(), quota.Spec.Total.LimitsVRAM.String())
	}

	if quota.Spec.Total.Workers != nil && *totalUsage.Workers > *quota.Spec.Total.Workers {
		return fmt.Errorf("total workers (%d) would exceed quota (%d)", *totalUsage.Workers, *quota.Spec.Total.Workers)
	}

	return nil
}

// ValidateCreate implements admission.CustomValidator
func (v *GPUResourceQuotaValidator) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

// ValidateUpdate implements admission.CustomValidator
func (v *GPUResourceQuotaValidator) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

// ValidateDelete implements admission.CustomValidator
func (v *GPUResourceQuotaValidator) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

// InjectDecoder injects the decoder
func (v *GPUResourceQuotaValidator) InjectDecoder(d admission.Decoder) error {
	v.decoder = d
	return nil
}

// SetupWithManager sets up the webhook with the Manager
// SetupGPUResourceQuotaWebhookWithManager registers the webhook for GPUResourceQuota validation
func SetupGPUResourceQuotaWebhookWithManager(mgr ctrl.Manager) error {
	validator := &GPUResourceQuotaValidator{
		Client:  mgr.GetClient(),
		decoder: admission.NewDecoder(mgr.GetScheme()),
	}

	webhookServer := mgr.GetWebhookServer()
	webhookServer.Register("/validate-tensor-fusion-ai-v1-tensorfusionworkload",
		&admission.Webhook{
			Handler: validator,
		})
	return nil
}
