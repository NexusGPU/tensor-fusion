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

package dra

import (
	"context"
	"fmt"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

// ResourceClaimReconciler reconciles ResourceClaim objects
type ResourceClaimReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

//+kubebuilder:rbac:groups=resource.k8s.io,resources=resourceclaims,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *ResourceClaimReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	// Fetch the ResourceClaim instance
	resourceClaim := &resourcev1beta2.ResourceClaim{}
	if err := r.Get(ctx, req.NamespacedName, resourceClaim); err != nil {
		if errors.IsNotFound(err) {
			// Request object not found, could have been deleted after reconcile request.
			// Owned objects are automatically garbage collected. For additional cleanup logic use finalizers.
			// Return and don't requeue
			log.Info("ResourceClaim resource not found. Ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		// Error reading the object - requeue the request.
		log.Error(err, "Failed to get ResourceClaim")
		return ctrl.Result{}, err
	}

	// Check if this ResourceClaim is created from our ResourceClaimTemplate
	if resourceClaim.Labels == nil {
		// No labels, not our ResourceClaim
		return ctrl.Result{}, nil
	}

	labelValue, exists := resourceClaim.Labels[constants.TensorFusionResourceClaimTemplateLabel]
	if !exists || labelValue != constants.TrueStringValue {
		// Not our ResourceClaim, ignore
		return ctrl.Result{}, nil
	}

	log.Info("Processing TensorFusion ResourceClaim", "name", resourceClaim.Name, "namespace", resourceClaim.Namespace)

	// Check if ResourceClaim is already allocated (idempotency check)
	if resourceClaim.Status.Allocation != nil {
		log.Info("ResourceClaim already allocated, skipping update", "name", resourceClaim.Name)
		return ctrl.Result{}, nil
	}

	// Find the owner Pod to get the CEL expression annotation
	ownerPod, err := r.findOwnerPod(ctx, resourceClaim)
	if err != nil {
		log.Error(err, "Failed to find owner Pod")
		return ctrl.Result{}, err
	}

	if ownerPod == nil {
		log.Info("Owner Pod not found, ResourceClaim may not have OwnerReference yet")
		return ctrl.Result{RequeueAfter: constants.PendingRequeueDuration}, nil
	}

	// Track if any changes were made
	needsUpdate := false

	// Update ResourceClaim with CEL expression
	celUpdated, err := r.updateResourceClaimCEL(resourceClaim, ownerPod)
	if err != nil {
		log.Error(err, "Failed to update ResourceClaim CEL expression")
		return ctrl.Result{}, err
	}
	needsUpdate = needsUpdate || celUpdated

	// Update ResourceClaim with capacity request
	capacityUpdated, err := r.updateCapacityRequest(resourceClaim, ownerPod)
	if err != nil {
		log.Error(err, "Failed to update ResourceClaim capacity request")
		return ctrl.Result{}, err
	}
	needsUpdate = needsUpdate || capacityUpdated

	// Update ResourceClaim with GPU count
	countUpdated, err := r.updateDeviceCount(resourceClaim, ownerPod)
	if err != nil {
		log.Error(err, "Failed to update ResourceClaim device count")
		return ctrl.Result{}, err
	}
	needsUpdate = needsUpdate || countUpdated

	// Only update if there were actual changes
	if needsUpdate {
		if err := r.Update(ctx, resourceClaim); err != nil {
			log.Error(err, "Failed to update ResourceClaim")
			return ctrl.Result{}, err
		}
		log.Info("Successfully updated ResourceClaim")
	} else {
		log.Info("No updates needed for ResourceClaim")
	}

	return ctrl.Result{}, nil
}

// findOwnerPod finds the Pod that owns this ResourceClaim
func (r *ResourceClaimReconciler) findOwnerPod(ctx context.Context, resourceClaim *resourcev1beta2.ResourceClaim) (*corev1.Pod, error) {
	// Find the Pod OwnerReference (there should be exactly one)
	var podOwnerRef *metav1.OwnerReference
	for i, ownerRef := range resourceClaim.OwnerReferences {
		if ownerRef.Kind == "Pod" && ownerRef.APIVersion == "v1" {
			podOwnerRef = &resourceClaim.OwnerReferences[i]
			break
		}
	}

	if podOwnerRef == nil {
		return nil, nil // No Pod owner found
	}

	// Get the Pod by name and namespace (UID is automatically verified by Kubernetes)
	pod := &corev1.Pod{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      podOwnerRef.Name,
		Namespace: resourceClaim.Namespace,
	}, pod)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil, nil // Pod was deleted
		}
		return nil, fmt.Errorf("failed to get owner Pod %s/%s: %w", resourceClaim.Namespace, podOwnerRef.Name, err)
	}

	// Verify the UID matches (additional safety check)
	if pod.UID != podOwnerRef.UID {
		return nil, fmt.Errorf("Pod UID mismatch: expected %s, got %s", podOwnerRef.UID, pod.UID)
	}

	return pod, nil
}

// updateResourceClaimCEL updates the ResourceClaim's CEL selector expression
// Returns true if changes were made, false otherwise
func (r *ResourceClaimReconciler) updateResourceClaimCEL(resourceClaim *resourcev1beta2.ResourceClaim, pod *corev1.Pod) (bool, error) {
	// Check if we need to update
	if len(resourceClaim.Spec.Devices.Requests) == 0 {
		return false, fmt.Errorf("no device requests found in ResourceClaim")
	}

	deviceReq := &resourceClaim.Spec.Devices.Requests[0]
	if deviceReq.Exactly == nil {
		return false, fmt.Errorf("no ExactDeviceRequest found")
	}

	// Get CEL expression from Pod annotation
	celExpression := pod.Annotations[constants.DRACelExpressionAnnotation]

	if celExpression == "" {
		return false, nil
	}

	// Check if CEL expression is already set correctly
	if len(deviceReq.Exactly.Selectors) > 0 &&
		deviceReq.Exactly.Selectors[0].CEL != nil &&
		deviceReq.Exactly.Selectors[0].CEL.Expression == celExpression {
		// Already updated
		return false, nil
	}

	// Update the CEL expression
	if len(deviceReq.Exactly.Selectors) == 0 {
		deviceReq.Exactly.Selectors = []resourcev1beta2.DeviceSelector{{}}
	}

	if deviceReq.Exactly.Selectors[0].CEL == nil {
		deviceReq.Exactly.Selectors[0].CEL = &resourcev1beta2.CELDeviceSelector{}
	}

	deviceReq.Exactly.Selectors[0].CEL.Expression = celExpression

	return true, nil
}

// updateCapacityRequest updates the ResourceClaim's capacity requests
// Returns true if changes were made, false otherwise
func (r *ResourceClaimReconciler) updateCapacityRequest(resourceClaim *resourcev1beta2.ResourceClaim, pod *corev1.Pod) (bool, error) {
	if len(resourceClaim.Spec.Devices.Requests) == 0 {
		return false, fmt.Errorf("no device requests found in ResourceClaim")
	}

	deviceReq := &resourceClaim.Spec.Devices.Requests[0]
	if deviceReq.Exactly == nil {
		return false, fmt.Errorf("no ExactDeviceRequest found")
	}

	gpuRequestResource, err := utils.GetGPUResource(pod, true)
	if err != nil {
		return false, fmt.Errorf("failed to get GPU resource: %w", err)
	}

	// Initialize Capacity if nil
	if deviceReq.Exactly.Capacity == nil {
		deviceReq.Exactly.Capacity = &resourcev1beta2.CapacityRequirements{}
	}

	// Initialize Capacity.Requests map if nil
	if deviceReq.Exactly.Capacity.Requests == nil {
		deviceReq.Exactly.Capacity.Requests = make(map[resourcev1beta2.QualifiedName]resource.Quantity)
	}

	// Check if capacity requests are already set correctly
	tflopsAlreadySet := false
	vramAlreadySet := false

	if existingTflops, ok := deviceReq.Exactly.Capacity.Requests[constants.DRACapacityTFlops]; ok {
		tflopsAlreadySet = existingTflops.Equal(gpuRequestResource.Tflops)
	}

	if existingVram, ok := deviceReq.Exactly.Capacity.Requests[constants.DRACapacityVRAM]; ok {
		vramAlreadySet = existingVram.Equal(gpuRequestResource.Vram)
	}

	// If already set correctly, no need to update
	if tflopsAlreadySet && vramAlreadySet {
		return false, nil
	}

	// Update capacity requests using constants
	deviceReq.Exactly.Capacity.Requests[constants.DRACapacityTFlops] = gpuRequestResource.Tflops
	deviceReq.Exactly.Capacity.Requests[constants.DRACapacityVRAM] = gpuRequestResource.Vram

	return true, nil
}

// updateDeviceCount updates the ResourceClaim's device count based on Pod's GPU count annotation
// Returns true if changes were made, false otherwise
func (r *ResourceClaimReconciler) updateDeviceCount(resourceClaim *resourcev1beta2.ResourceClaim, pod *corev1.Pod) (bool, error) {
	if len(resourceClaim.Spec.Devices.Requests) == 0 {
		return false, fmt.Errorf("no device requests found in ResourceClaim")
	}

	deviceReq := &resourceClaim.Spec.Devices.Requests[0]
	if deviceReq.Exactly == nil {
		return false, fmt.Errorf("no ExactDeviceRequest found")
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
		return false, fmt.Errorf("invalid GPU count annotation value %q: %w", gpuCountStr, err)
	}

	// Validate GPU count (must be positive)
	if gpuCount <= 0 {
		return false, fmt.Errorf("GPU count must be positive, got %d", gpuCount)
	}

	// Check if count is already set correctly (idempotency)
	if deviceReq.Exactly.Count == gpuCount {
		return false, nil
	}

	// Update device count
	deviceReq.Exactly.Count = gpuCount

	return true, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *ResourceClaimReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&resourcev1beta2.ResourceClaim{}).
		Complete(r)
}
