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

	// Update ResourceClaim with CEL expression
	if err := r.updateResourceClaimCEL(resourceClaim, ownerPod); err != nil {
		log.Error(err, "Failed to update ResourceClaim CEL expression")
		return ctrl.Result{}, err
	}
	// Update ResourceClaim with capacity request
	if err := r.updateCapacityRequest(resourceClaim, ownerPod); err != nil {
		log.Error(err, "Failed to update ResourceClaim capacity request")
		return ctrl.Result{}, err
	}

	if err := r.Update(ctx, resourceClaim); err != nil {
		log.Error(err, "Failed to update ResourceClaim")
		return ctrl.Result{}, err
	}

	log.Info("Successfully updated ResourceClaim")
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
func (r *ResourceClaimReconciler) updateResourceClaimCEL(resourceClaim *resourcev1beta2.ResourceClaim, pod *corev1.Pod) error {
	// Check if we need to update
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

	// Check if CEL expression is already set correctly
	if len(deviceReq.Exactly.Selectors) > 0 &&
		deviceReq.Exactly.Selectors[0].CEL != nil &&
		deviceReq.Exactly.Selectors[0].CEL.Expression == celExpression {
		// Already updated
		return nil
	}

	// Update the CEL expression
	if len(deviceReq.Exactly.Selectors) == 0 {
		deviceReq.Exactly.Selectors = []resourcev1beta2.DeviceSelector{{}}
	}

	if deviceReq.Exactly.Selectors[0].CEL == nil {
		deviceReq.Exactly.Selectors[0].CEL = &resourcev1beta2.CELDeviceSelector{}
	}

	deviceReq.Exactly.Selectors[0].CEL.Expression = celExpression

	return nil
}

func (r *ResourceClaimReconciler) updateCapacityRequest(resourceClaim *resourcev1beta2.ResourceClaim, pod *corev1.Pod) error {
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
	//TODO extract to constants
	deviceReq.Exactly.Capacity.Requests["tflops"] = gpuRequestResource.Tflops
	deviceReq.Exactly.Capacity.Requests["vram"] = gpuRequestResource.Vram

	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *ResourceClaimReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&resourcev1beta2.ResourceClaim{}).
		Complete(r)
}
