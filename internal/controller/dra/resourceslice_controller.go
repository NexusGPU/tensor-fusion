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

	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
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

// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceslices,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpunodes,verbs=get;list;watch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpus,verbs=get;list;watch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpupools,verbs=get;list;watch

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
	resourceSlice := &resourcev1.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: resourceSliceName,
		},
	}

	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, resourceSlice, func() error {
		// Validate pool name exists
		poolName := gpuNode.Labels[constants.GpuPoolKey]
		if poolName == "" {
			return fmt.Errorf("GPUNode %s missing pool label %s", gpuNode.Name, constants.GpuPoolKey)
		}

		// Get GPUPool to retrieve default QoS
		gpuPool := &tfv1.GPUPool{}
		if err := r.Get(ctx, client.ObjectKey{Name: poolName}, gpuPool); err != nil {
			log.V(1).Info("Failed to get GPUPool for default QoS, will skip QoS attribute", "pool", poolName, "error", err)
			// Continue without QoS - it's optional
		}

		// Set basic spec fields
		resourceSlice.Spec.Driver = constants.DRADriverName
		resourceSlice.Spec.NodeName = &gpuNode.Name
		resourceSlice.Spec.Pool = resourcev1.ResourcePool{
			Name:               poolName,
			Generation:         gpuNode.Generation,
			ResourceSliceCount: 1,
		}

		// Generate devices list with QoS information
		devices := r.generateDevices(ctx, gpuNode, gpus, gpuPool)
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
func (r *ResourceSliceReconciler) generateDevices(_ context.Context, gpuNode *tfv1.GPUNode, gpus []tfv1.GPU, gpuPool *tfv1.GPUPool) []resourcev1.Device {
	devices := make([]resourcev1.Device, 0, len(gpus))

	if len(gpus) == 0 {
		return devices
	}

	// Get default QoS from GPUPool (if available)
	defaultQoS := string(tfv1.QoSMedium) // Default to medium if not specified
	if gpuPool != nil && gpuPool.Spec.QosConfig != nil && gpuPool.Spec.QosConfig.DefaultQoS != "" {
		defaultQoS = string(gpuPool.Spec.QosConfig.DefaultQoS)
	}

	// Get node-level totals and virtual capacities from GPUNode Status
	nodeTotalTFlops := gpuNode.Status.TotalTFlops
	nodeTotalVRAM := gpuNode.Status.TotalVRAM
	nodeTotalGPUs := gpuNode.Status.TotalGPUs
	nodeManagedGPUs := gpuNode.Status.ManagedGPUs
	nodeVirtualTFlops := gpuNode.Status.VirtualTFlops
	nodeVirtualVRAM := gpuNode.Status.VirtualVRAM

	// Calculate per-GPU virtual capacity (equal distribution)
	var virtualTFlopsPerGPU, virtualVRAMPerGPU *resource.Quantity
	if nodeManagedGPUs > 0 && !nodeVirtualTFlops.IsZero() && !nodeVirtualVRAM.IsZero() {
		// Virtual TFlops per GPU - equally distributed
		vTFlopsFloat := float64(nodeVirtualTFlops.AsApproximateFloat64()) / float64(nodeManagedGPUs)
		virtualTFlopsPerGPU = resource.NewQuantity(int64(vTFlopsFloat), resource.DecimalSI)

		// Virtual VRAM per GPU - equally distributed
		vramFloat := float64(nodeVirtualVRAM.AsApproximateFloat64()) / float64(nodeManagedGPUs)
		virtualVRAMPerGPU = resource.NewQuantity(int64(vramFloat), resource.DecimalSI)
	}

	for _, gpu := range gpus {
		if gpu.Status.Capacity == nil {
			continue
		}

		// Validate required fields
		poolName := gpu.Labels[constants.GpuPoolKey]
		if poolName == "" {
			// Skip GPUs without pool label - they may not be fully initialized
			continue
		}

		nodeName := gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
		if nodeName == "" {
			// Skip GPUs without node selector - they may not be fully initialized
			continue
		}

		// Get QoS from GPU labels, fall back to pool default
		qosLevel := defaultQoS
		if gpuQoS, exists := gpu.Labels[constants.QoSLevelAnnotation]; exists && gpuQoS != "" {
			qosLevel = gpuQoS
		}

		gpuPhase := string(gpu.Status.Phase)
		usedBy := string(gpu.Status.UsedBy)

		device := resourcev1.Device{
			Name: gpu.Name,
			Attributes: map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{
				// Existing attributes
				constants.DRAAttributeModel: {
					StringValue: &gpu.Status.GPUModel,
				},
				constants.DRAAttributePoolName: {
					StringValue: &poolName,
				},
				constants.DRAAttributePodNamespace: {
					StringValue: &gpu.Namespace,
				},
				// Device identity and state attributes
				constants.DRAAttributeUUID: {
					StringValue: &gpu.Status.UUID,
				},
				constants.DRAAttributePhase: {
					StringValue: &gpuPhase,
				},
				constants.DRAAttributeUsedBy: {
					StringValue: &usedBy,
				},
				constants.DRAAttributeNodeName: {
					StringValue: &nodeName,
				},
				// QoS level attribute (from GPU labels or pool default)
				constants.DRAAttributeQoS: {
					StringValue: &qosLevel,
				},
				// Node-level total capacity attributes
				constants.DRAAttributeNodeTotalTFlops: {
					StringValue: func() *string { s := nodeTotalTFlops.String(); return &s }(),
				},
				constants.DRAAttributeNodeTotalVRAM: {
					StringValue: func() *string { s := nodeTotalVRAM.String(); return &s }(),
				},
				constants.DRAAttributeNodeTotalGPUs: {
					IntValue: func() *int64 { v := int64(nodeTotalGPUs); return &v }(),
				},
				constants.DRAAttributeNodeManagedGPUs: {
					IntValue: func() *int64 { v := int64(nodeManagedGPUs); return &v }(),
				},
				// Node-level virtual capacity attributes
				constants.DRAAttributeNodeVirtualTFlops: {
					StringValue: func() *string { s := nodeVirtualTFlops.String(); return &s }(),
				},
				constants.DRAAttributeNodeVirtualVRAM: {
					StringValue: func() *string { s := nodeVirtualVRAM.String(); return &s }(),
				},
			},
			Capacity: map[resourcev1.QualifiedName]resourcev1.DeviceCapacity{
				// Physical capacity
				constants.DRACapacityTFlops: {
					Value: gpu.Status.Capacity.Tflops,
				},
				constants.DRACapacityVRAM: {
					Value: gpu.Status.Capacity.Vram,
				},
			},
			AllowMultipleAllocations: func() *bool { b := true; return &b }(),
		}

		// Add virtual capacity if available
		if virtualTFlopsPerGPU != nil {
			device.Capacity[constants.DRACapacityVirtualTFlops] = resourcev1.DeviceCapacity{
				Value: *virtualTFlopsPerGPU,
			}
		}
		if virtualVRAMPerGPU != nil {
			device.Capacity[constants.DRACapacityVirtualVRAM] = resourcev1.DeviceCapacity{
				Value: *virtualVRAMPerGPU,
			}
		}

		devices = append(devices, device)
	}

	return devices
}

// cleanupResourceSlice removes the ResourceSlice associated with a deleted GPUNode
func (r *ResourceSliceReconciler) cleanupResourceSlice(ctx context.Context, nodeName string) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	resourceSliceName := fmt.Sprintf(constants.DRAResourceSliceName, nodeName)
	resourceSlice := &resourcev1.ResourceSlice{
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
	// Setup field indexer for ResourceSlice by nodeName to enable efficient queries
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(),
		&resourcev1.ResourceSlice{},
		"spec.nodeName",
		func(obj client.Object) []string {
			rs := obj.(*resourcev1.ResourceSlice)
			if rs.Spec.NodeName != nil && *rs.Spec.NodeName != "" {
				return []string{*rs.Spec.NodeName}
			}
			return nil
		},
	); err != nil {
		return fmt.Errorf("failed to setup field indexer for ResourceSlice.spec.nodeName: %w", err)
	}

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
