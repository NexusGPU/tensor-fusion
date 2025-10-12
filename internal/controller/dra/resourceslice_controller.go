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
		devices, err := r.generateDevices(ctx, gpuNode, gpus)
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
func (r *ResourceSliceReconciler) generateDevices(ctx context.Context, gpuNode *tfv1.GPUNode, gpus []tfv1.GPU) ([]resourcev1beta2.Device, error) {
	devices := make([]resourcev1beta2.Device, 0, len(gpus))

	if len(gpus) == 0 {
		return devices, nil
	}

	// Get GPUPool for virtual capacity calculation
	poolName := gpuNode.Labels[constants.GpuPoolKey]
	pool := &tfv1.GPUPool{}
	if err := r.Get(ctx, client.ObjectKey{Name: poolName}, pool); err != nil {
		return nil, fmt.Errorf("failed to get GPUPool %s: %w", poolName, err)
	}

	// Calculate node-level totals and virtual capacities
	nodeTotalTFlops := gpuNode.Status.TotalTFlops
	nodeTotalVRAM := gpuNode.Status.TotalVRAM
	nodeTotalGPUs := gpuNode.Status.TotalGPUs
	nodeManagedGPUs := gpuNode.Status.ManagedGPUs

	// Calculate node-level virtual capacities
	nodeVirtualVRAM, nodeVirtualTFlops := r.calculateNodeVirtualCapacity(gpuNode, pool)

	// Calculate per-GPU virtual capacity (proportional allocation)
	var virtualTFlopsPerGPU, virtualVRAMPerGPU *resource.Quantity
	if nodeManagedGPUs > 0 {
		// Virtual TFlops per GPU
		vTFlopsFloat := float64(nodeVirtualTFlops.AsApproximateFloat64()) / float64(nodeManagedGPUs)
		vTFlopsPerGPU := resource.NewQuantity(int64(vTFlopsFloat), resource.DecimalSI)
		virtualTFlopsPerGPU = vTFlopsPerGPU

		// Virtual VRAM per GPU (proportional to physical capacity)
		// VRAM expansion is distributed proportionally based on each GPU's physical VRAM
		virtualVRAMPerGPU = &resource.Quantity{}
	}

	for _, gpu := range gpus {
		if gpu.Status.Capacity == nil {
			continue
		}

		// Calculate this GPU's proportional virtual VRAM
		var gpuVirtualVRAM *resource.Quantity
		if virtualVRAMPerGPU != nil && nodeTotalVRAM.Value() > 0 {
			// Calculate this GPU's share of total node VRAM
			gpuShare := float64(gpu.Status.Capacity.Vram.AsApproximateFloat64()) / float64(nodeTotalVRAM.AsApproximateFloat64())
			// Apply the share to virtual VRAM
			vramFloat := nodeVirtualVRAM.AsApproximateFloat64() * gpuShare
			gpuVirtualVRAM = resource.NewQuantity(int64(vramFloat), resource.DecimalSI)
		}

		poolName := gpu.Labels[constants.GpuPoolKey]
		nodeName := gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
		gpuPhase := string(gpu.Status.Phase)
		usedBy := string(gpu.Status.UsedBy)

		device := resourcev1beta2.Device{
			Name: gpu.Name,
			Attributes: map[resourcev1beta2.QualifiedName]resourcev1beta2.DeviceAttribute{
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
			Capacity: map[resourcev1beta2.QualifiedName]resourcev1beta2.DeviceCapacity{
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
			device.Capacity[constants.DRACapacityVirtualTFlops] = resourcev1beta2.DeviceCapacity{
				Value: *virtualTFlopsPerGPU,
			}
		}
		if gpuVirtualVRAM != nil {
			device.Capacity[constants.DRACapacityVirtualVRAM] = resourcev1beta2.DeviceCapacity{
				Value: *gpuVirtualVRAM,
			}
		}

		devices = append(devices, device)
	}

	return devices, nil
}

// calculateNodeVirtualCapacity calculates the virtual capacity for a node based on oversubscription config
func (r *ResourceSliceReconciler) calculateNodeVirtualCapacity(node *tfv1.GPUNode, pool *tfv1.GPUPool) (resource.Quantity, resource.Quantity) {
	diskSize, _ := node.Status.NodeInfo.DataDiskSize.AsInt64()
	ramSize, _ := node.Status.NodeInfo.RAMSize.AsInt64()

	virtualVRAM := node.Status.TotalVRAM.DeepCopy()

	// If no oversubscription config, return physical capacity
	if pool.Spec.CapacityConfig == nil || pool.Spec.CapacityConfig.Oversubscription == nil {
		return virtualVRAM, node.Status.TotalTFlops.DeepCopy()
	}

	// Calculate virtual TFlops with oversell ratio
	vTFlops := node.Status.TotalTFlops.AsApproximateFloat64() * (float64(pool.Spec.CapacityConfig.Oversubscription.TFlopsOversellRatio) / 100.0)

	// Expand VRAM to host disk
	virtualVRAM.Add(*resource.NewQuantity(
		int64(float64(diskSize)*float64(pool.Spec.CapacityConfig.Oversubscription.VRAMExpandToHostDisk)/100.0),
		resource.DecimalSI),
	)

	// Expand VRAM to host memory
	virtualVRAM.Add(*resource.NewQuantity(
		int64(float64(ramSize)*float64(pool.Spec.CapacityConfig.Oversubscription.VRAMExpandToHostMem)/100.0),
		resource.DecimalSI),
	)

	return virtualVRAM, *resource.NewQuantity(int64(vTFlops), resource.DecimalSI)
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
	// Setup field indexer for ResourceSlice by nodeName to enable efficient queries
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(),
		&resourcev1beta2.ResourceSlice{},
		"spec.nodeName",
		func(obj client.Object) []string {
			rs := obj.(*resourcev1beta2.ResourceSlice)
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
