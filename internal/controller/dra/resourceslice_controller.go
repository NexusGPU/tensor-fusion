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

	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

// ResourceSliceReconciler reconciles ResourceSlice objects based on GPUNode and GPU changes
type ResourceSliceReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

//+kubebuilder:rbac:groups=resource.k8s.io,resources=resourceslices,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpunodes,verbs=get;list;watch
//+kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpus,verbs=get;list;watch
//+kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpupools,verbs=get;list;watch

// Reconcile processes GPUNode changes and generates/updates corresponding ResourceSlices
func (r *ResourceSliceReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Reconciling ResourceSlice for GPUNode", "name", req.Name)

	// Fetch the GPUNode
	gpuNode := &tfv1.GPUNode{}
	if err := r.Get(ctx, req.NamespacedName, gpuNode); err != nil {
		if errors.IsNotFound(err) {
			// GPUNode was deleted, clean up associated ResourceSlice
			return r.cleanupResourceSlice(ctx, req.Name)
		}
		log.Error(err, "Failed to get GPUNode")
		return ctrl.Result{}, err
	}

	// If GPUNode is being deleted, clean up ResourceSlice
	if !gpuNode.DeletionTimestamp.IsZero() {
		return r.cleanupResourceSlice(ctx, gpuNode.Name)
	}
	// Get all GPUs owned by this node
	gpuList := &tfv1.GPUList{}
	if err := r.List(ctx, gpuList, client.MatchingLabels{constants.LabelKeyOwner: gpuNode.Name}); err != nil {
		log.Error(err, "Failed to list GPUs for node")
		return ctrl.Result{}, err
	}

	// Skip if no GPUs discovered yet
	if len(gpuList.Items) == 0 {
		log.Info("No GPUs discovered for node yet, skipping ResourceSlice generation")
		return ctrl.Result{}, nil
	}

	// Generate/update ResourceSlice for this node
	if err := r.reconcileResourceSlice(ctx, gpuNode, gpuList.Items); err != nil {
		log.Error(err, "Failed to reconcile ResourceSlice")
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// reconcileResourceSlice creates or updates the ResourceSlice for a GPUNode
func (r *ResourceSliceReconciler) reconcileResourceSlice(ctx context.Context, gpuNode *tfv1.GPUNode, gpus []tfv1.GPU) error {
	log := log.FromContext(ctx)

	resourceSliceName := fmt.Sprintf(constants.DRAResourceSliceName, gpuNode.Name)
	resourceSlice := &resourcev1beta2.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: resourceSliceName,
		},
	}

	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, resourceSlice, func() error {
		// Set basic spec fields
		resourceSlice.Spec.Driver = constants.DRADriverName
		resourceSlice.Spec.NodeName = &gpuNode.Name
		resourceSlice.Spec.Pool = resourcev1beta2.ResourcePool{
			Name:               gpuNode.Labels[constants.GpuPoolKey],
			Generation:         gpuNode.Generation,
			ResourceSliceCount: 1,
		}

		// Generate devices list
		devices, err := r.generateDevices(ctx, gpus)
		if err != nil {
			return fmt.Errorf("failed to generate devices: %w", err)
		}
		resourceSlice.Spec.Devices = devices

		// Set labels for easy identification
		if resourceSlice.Labels == nil {
			resourceSlice.Labels = make(map[string]string)
		}
		resourceSlice.Labels[constants.LabelKeyOwner] = gpuNode.Name
		resourceSlice.Labels[constants.KubernetesHostNameLabel] = gpuNode.Name
		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to create or update ResourceSlice: %w", err)
	}

	log.Info("Successfully reconciled ResourceSlice", "resourceSlice", resourceSliceName)
	return nil
}

// generateDevices creates the device list for ResourceSlice based on physical GPUs
func (r *ResourceSliceReconciler) generateDevices(_ context.Context, gpus []tfv1.GPU) ([]resourcev1beta2.Device, error) {
	devices := make([]resourcev1beta2.Device, 0, len(gpus))

	// Calculate virtual capacities for proportional allocation

	for _, gpu := range gpus {
		if gpu.Status.Capacity == nil {
			continue
		}
		//TODO extract to constants
		//TODO quota support
		poolName := gpu.Labels[constants.GpuPoolKey]
		device := resourcev1beta2.Device{
			Name: gpu.Status.UUID,
			Attributes: map[resourcev1beta2.QualifiedName]resourcev1beta2.DeviceAttribute{
				"model": {
					StringValue: &gpu.Status.GPUModel,
				},
				"pool_name": {
					StringValue: &poolName,
				},
				"pod_namespace": {
					StringValue: &gpu.Namespace,
				},
			},
			Capacity: map[resourcev1beta2.QualifiedName]resourcev1beta2.DeviceCapacity{
				"tflops": {
					Value: gpu.Status.Capacity.Tflops,
				},
				"vram": {
					Value: gpu.Status.Capacity.Vram,
				},
			},
			AllowMultipleAllocations: func() *bool { b := true; return &b }(),
		}

		devices = append(devices, device)
	}

	return devices, nil
}

// cleanupResourceSlice removes the ResourceSlice associated with a deleted GPUNode
func (r *ResourceSliceReconciler) cleanupResourceSlice(ctx context.Context, nodeName string) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	resourceSliceName := fmt.Sprintf(constants.DRAResourceSliceName, nodeName)
	resourceSlice := &resourcev1beta2.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: resourceSliceName,
		},
	}

	err := r.Delete(ctx, resourceSlice)
	if err != nil && !errors.IsNotFound(err) {
		log.Error(err, "Failed to delete ResourceSlice", "name", resourceSliceName)
		return ctrl.Result{}, err
	}

	log.Info("Successfully cleaned up ResourceSlice", "name", resourceSliceName)
	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager
func (r *ResourceSliceReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.GPUNode{}).
		Watches(&tfv1.GPU{}, handler.EnqueueRequestsFromMapFunc(
			func(ctx context.Context, obj client.Object) []reconcile.Request {
				// Get the owner GPUNode name from GPU labels
				if labels := obj.GetLabels(); labels != nil {
					if nodeName, ok := labels[constants.LabelKeyOwner]; ok {
						return []reconcile.Request{
							{NamespacedName: types.NamespacedName{Name: nodeName}},
						}
					}
				}
				return nil
			})).
		Complete(r)
}
