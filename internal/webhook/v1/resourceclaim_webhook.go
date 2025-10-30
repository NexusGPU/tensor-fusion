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
	"encoding/json"
	"fmt"
	"net/http"

	"gomodules.xyz/jsonpatch/v2"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

// SetupResourceClaimWebhookWithManager registers the webhook for ResourceClaim in the manager.
func SetupResourceClaimWebhookWithManager(mgr ctrl.Manager) error {
	webhookServer := mgr.GetWebhookServer()

	mutator := &ResourceClaimMutator{
		decoder: admission.NewDecoder(runtime.NewScheme()),
		Client:  mgr.GetClient(),
	}

	webhookServer.Register("/mutate-resource-v1-resourceclaim", &admission.Webhook{Handler: mutator})
	return nil
}

type ResourceClaimMutator struct {
	Client  client.Client
	decoder admission.Decoder
}

// Handle implements admission.Handler interface.
func (m *ResourceClaimMutator) Handle(ctx context.Context, req admission.Request) admission.Response {
	resourceClaim := &resourcev1.ResourceClaim{}
	if err := m.decoder.Decode(req, resourceClaim); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}

	log := log.FromContext(ctx)
	log.Info("Mutating ResourceClaim", "name", resourceClaim.Name, "namespace", resourceClaim.Namespace)

	// Check if this ResourceClaim is created from our ResourceClaimTemplate
	if resourceClaim.Labels == nil {
		log.Info("ResourceClaim has no labels, skipping")
		return admission.Allowed("no labels")
	}

	labelValue, exists := resourceClaim.Labels[constants.TensorFusionResourceClaimTemplateLabel]
	if !exists || labelValue != constants.TrueStringValue {
		log.Info("Not a TensorFusion ResourceClaim, skipping")
		return admission.Allowed("not a TensorFusion ResourceClaim")
	}

	// Find the owner Pod to get the CEL expression annotation
	var ownerPod *corev1.Pod
	for _, ownerRef := range resourceClaim.OwnerReferences {
		if ownerRef.Kind == "Pod" && ownerRef.APIVersion == "v1" {
			pod := &corev1.Pod{}
			err := m.Client.Get(ctx, types.NamespacedName{
				Name:      ownerRef.Name,
				Namespace: resourceClaim.Namespace,
			}, pod)
			if err != nil {
				log.Error(err, "Failed to get owner Pod")
				return admission.Errored(http.StatusInternalServerError, fmt.Errorf("failed to get owner Pod: %w", err))
			}
			ownerPod = pod
			break
		}
	}

	if ownerPod == nil {
		log.Info("Owner Pod not found, skipping mutation")
		return admission.Allowed("no owner Pod found")
	}

	log.Info("Found owner Pod", "podName", ownerPod.Name)

	// Marshal current state
	currentBytes, err := json.Marshal(resourceClaim)
	if err != nil {
		return admission.Errored(http.StatusBadRequest, fmt.Errorf("failed to marshal current ResourceClaim: %w", err))
	}

	// Update ResourceClaim with CEL expression
	if err := m.updateResourceClaimCEL(resourceClaim, ownerPod); err != nil {
		log.Error(err, "Failed to update ResourceClaim CEL expression")
		return admission.Errored(http.StatusInternalServerError, err)
	}

	// Update ResourceClaim with capacity request
	if err := m.updateCapacityRequest(resourceClaim, ownerPod); err != nil {
		log.Error(err, "Failed to update ResourceClaim capacity request")
		return admission.Errored(http.StatusInternalServerError, err)
	}

	// Update ResourceClaim with GPU count
	if err := m.updateDeviceCount(resourceClaim, ownerPod); err != nil {
		log.Error(err, "Failed to update ResourceClaim device count")
		return admission.Errored(http.StatusInternalServerError, err)
	}

	// Marshal patched state
	patchedBytes, err := json.Marshal(resourceClaim)
	if err != nil {
		return admission.Errored(http.StatusBadRequest, fmt.Errorf("failed to marshal patched ResourceClaim: %w", err))
	}

	// Generate JSON patch
	patches, err := jsonpatch.CreatePatch(currentBytes, patchedBytes)
	if err != nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("failed to create patch: %w", err))
	}

	log.Info("Successfully mutated ResourceClaim", "patchCount", len(patches))
	return admission.Patched("TensorFusion ResourceClaim mutated", patches...)
}

// InjectDecoder injects the decoder.
func (m *ResourceClaimMutator) InjectDecoder(d admission.Decoder) error {
	m.decoder = d
	return nil
}

// updateResourceClaimCEL updates the ResourceClaim's CEL selector expression
func (m *ResourceClaimMutator) updateResourceClaimCEL(resourceClaim *resourcev1.ResourceClaim, pod *corev1.Pod) error {
	if len(resourceClaim.Spec.Devices.Requests) == 0 {
		return fmt.Errorf("no device requests found in ResourceClaim")
	}

	deviceReq := &resourceClaim.Spec.Devices.Requests[0]
	if deviceReq.Exactly == nil {
		return fmt.Errorf("no ExactDeviceRequest found")
	}

	// Get CEL expression from Pod annotation
	celExpression := pod.Annotations[constants.DRACelExpressionAnnotation]
	if celExpression == "" {
		return nil
	}

	// Update the CEL expression
	if len(deviceReq.Exactly.Selectors) == 0 {
		deviceReq.Exactly.Selectors = []resourcev1.DeviceSelector{{}}
	}

	if deviceReq.Exactly.Selectors[0].CEL == nil {
		deviceReq.Exactly.Selectors[0].CEL = &resourcev1.CELDeviceSelector{}
	}

	deviceReq.Exactly.Selectors[0].CEL.Expression = celExpression

	return nil
}

// updateCapacityRequest updates the ResourceClaim's capacity requests
func (m *ResourceClaimMutator) updateCapacityRequest(resourceClaim *resourcev1.ResourceClaim, pod *corev1.Pod) error {
	if len(resourceClaim.Spec.Devices.Requests) == 0 {
		return fmt.Errorf("no device requests found in ResourceClaim")
	}

	deviceReq := &resourceClaim.Spec.Devices.Requests[0]
	if deviceReq.Exactly == nil {
		return fmt.Errorf("no ExactDeviceRequest found")
	}

	gpuRequestResource, err := utils.GetGPUResource(pod, true)
	if err != nil {
		return fmt.Errorf("failed to get GPU resource: %w", err)
	}

	// Initialize Capacity if nil
	if deviceReq.Exactly.Capacity == nil {
		deviceReq.Exactly.Capacity = &resourcev1.CapacityRequirements{}
	}

	// Initialize Capacity.Requests map if nil
	if deviceReq.Exactly.Capacity.Requests == nil {
		deviceReq.Exactly.Capacity.Requests = make(map[resourcev1.QualifiedName]resource.Quantity)
	}

	// Update capacity requests using constants
	deviceReq.Exactly.Capacity.Requests[constants.DRACapacityTFlops] = gpuRequestResource.Tflops
	deviceReq.Exactly.Capacity.Requests[constants.DRACapacityVRAM] = gpuRequestResource.Vram

	return nil
}

// updateDeviceCount updates the ResourceClaim's device count based on Pod's GPU count annotation
func (m *ResourceClaimMutator) updateDeviceCount(resourceClaim *resourcev1.ResourceClaim, pod *corev1.Pod) error {
	if len(resourceClaim.Spec.Devices.Requests) == 0 {
		return fmt.Errorf("no device requests found in ResourceClaim")
	}

	deviceReq := &resourceClaim.Spec.Devices.Requests[0]
	if deviceReq.Exactly == nil {
		return fmt.Errorf("no ExactDeviceRequest found")
	}

	// Get GPU count from Pod annotation (defaults to 1 if not specified)
	gpuCountStr, exists := pod.Annotations[constants.GpuCountAnnotation]
	if !exists || gpuCountStr == "" {
		// No GPU count annotation, default to 1
		gpuCountStr = "1"
	}

	// Parse GPU count
	var gpuCount int64
	if _, err := fmt.Sscanf(gpuCountStr, "%d", &gpuCount); err != nil {
		return fmt.Errorf("invalid GPU count annotation value %q: %w", gpuCountStr, err)
	}

	// Validate GPU count (must be positive)
	if gpuCount <= 0 {
		return fmt.Errorf("GPU count must be positive, got %d", gpuCount)
	}

	// Update device count
	deviceReq.Exactly.Count = gpuCount

	return nil
}
