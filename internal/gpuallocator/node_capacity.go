package gpuallocator

import (
	"context"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/util/taints"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func RefreshGPUNodeCapacity(
	ctx context.Context, k8sClient client.Client,
	node *tfv1.GPUNode, pool *tfv1.GPUPool,
	allocator *GpuAllocator,
	coreNode *corev1.Node,
) ([]string, error) {
	gpuList := &tfv1.GPUList{}
	if err := k8sClient.List(ctx, gpuList, client.MatchingLabels{constants.LabelKeyOwner: node.Name}); err != nil {
		return nil, fmt.Errorf("failed to list GPUs: %w", err)
	}
	if len(gpuList.Items) == 0 {
		// node discovery job not completed, or GPU CRs have been removed externally.
		// In the latter case TF owns nothing on this node, so reconcile the taint as idle
		// to avoid a stale `tensor-fusion.ai/used-by` taint blocking native scheduling.
		if err := reconcileProgressiveTaint(ctx, k8sClient, coreNode, true); err != nil {
			return nil, err
		}
		return nil, nil
	}

	statusCopy := node.Status.DeepCopy()

	node.Status.AvailableVRAM = resource.Quantity{}
	node.Status.AvailableTFlops = resource.Quantity{}
	node.Status.TotalTFlops = resource.Quantity{}
	node.Status.TotalVRAM = resource.Quantity{}
	node.Status.AllocatedPods = make(map[string][]*tfv1.PodGPUInfo)

	nodeGPUPodSet := make(map[string]struct{})
	gpuModels := []string{}

	for _, gpu := range gpuList.Items {
		if gpu.Status.Available == nil || gpu.Status.Capacity == nil {
			continue
		}
		node.Status.AvailableVRAM.Add(gpu.Status.Available.Vram)
		node.Status.AvailableTFlops.Add(gpu.Status.Available.Tflops)
		node.Status.TotalVRAM.Add(gpu.Status.Capacity.Vram)
		node.Status.TotalTFlops.Add(gpu.Status.Capacity.Tflops)
		gpuModels = append(gpuModels, gpu.Status.GPUModel)

		if _, ok := node.Status.AllocatedPods[gpu.Name]; !ok {
			node.Status.AllocatedPods[gpu.Name] = []*tfv1.PodGPUInfo{}
		}
		for _, runningApp := range gpu.Status.RunningApps {
			for _, pod := range runningApp.Pods {
				if _, ok := nodeGPUPodSet[pod.UID]; !ok {
					nodeGPUPodSet[pod.UID] = struct{}{}
				}
				node.Status.AllocatedPods[gpu.Name] = append(node.Status.AllocatedPods[gpu.Name], pod)
			}
		}
	}

	virtualVRAM, virtualTFlops := calculateVirtualCapacity(node, pool)
	node.Status.VirtualTFlops = virtualTFlops
	node.Status.VirtualVRAM = virtualVRAM
	node.Status.TotalGPUPods = int32(len(nodeGPUPodSet))

	vramAvailable := virtualVRAM.DeepCopy()
	tflopsAvailable := virtualTFlops.DeepCopy()

	allocRequests := allocator.GetAllocationReqByNodeName(node.Name)
	for _, allocRequest := range allocRequests {
		vramAvailable.Sub(allocRequest.Limit.Vram)
		// Get actual TFLOPs value, converting from ComputePercent if needed
		var limitTflops resource.Quantity
		if len(allocRequest.GPUNames) > 0 {
			// Try to find the GPU to get capacity for conversion
			for _, gpu := range gpuList.Items {
				if gpu.Name == allocRequest.GPUNames[0] && gpu.Status.Capacity != nil {
					if !allocRequest.Limit.ComputePercent.IsZero() {
						requiredTflops := utils.ComputePercentToTflops(gpu.Status.Capacity.Tflops, allocRequest.Limit)
						limitTflops = *requiredTflops
					} else {
						limitTflops = allocRequest.Limit.Tflops
					}
					break
				}
			}
			// If GPU not found, fallback to direct TFLOPs
			if limitTflops.IsZero() && !allocRequest.Limit.Tflops.IsZero() {
				limitTflops = allocRequest.Limit.Tflops
			}
		} else {
			limitTflops = allocRequest.Limit.Tflops
		}
		if !limitTflops.IsZero() {
			tflopsAvailable.Sub(limitTflops)
		}
	}
	node.Status.VirtualAvailableVRAM = &vramAvailable
	node.Status.VirtualAvailableTFlops = &tflopsAvailable

	node.Status.Phase = tfv1.TensorFusionGPUNodePhaseRunning

	if !equality.Semantic.DeepEqual(node.Status, statusCopy) {
		if err := k8sClient.Status().Update(ctx, node); err != nil {
			return nil, fmt.Errorf("failed to update GPU node status: %w", err)
		}
	}

	// Guard against a transient state where all GPU CRs on this node are missing
	// Status.Capacity / Status.Available (e.g. fresh discovery, hypervisor restart,
	// external status reset). The aggregation loop above skips such GPUs, so
	// Total/Available both end up 0 and "Available.Equal(Total)" yields a
	// meaningless isIdle=true — without this guard we'd wrongly remove the taint
	// from a node that may still be serving TF workers. The next reconcile round
	// will converge once GPU status is populated.
	if node.Status.TotalVRAM.IsZero() && node.Status.TotalTFlops.IsZero() {
		return gpuModels, nil
	}

	// Reconcile the isolation taint every round, independent of status diff.
	// A transient Node update failure otherwise leaves the taint stuck because the next
	// round's DeepEqual would be true and the block would be skipped.
	isIdle := node.Status.AvailableVRAM.Equal(node.Status.TotalVRAM) &&
		node.Status.AvailableTFlops.Equal(node.Status.TotalTFlops)
	if err := reconcileProgressiveTaint(ctx, k8sClient, coreNode, isIdle); err != nil {
		return nil, err
	}
	return gpuModels, nil
}

// reconcileProgressiveTaint converges the `tensor-fusion.ai/used-by` PreferNoSchedule
// taint on the core k8s Node to the desired state derived from isIdle:
//   - isIdle=true  + has taint  -> remove
//   - isIdle=false + no taint   -> add
//
// No-op in all other combinations. Only runs when progressive migration is enabled
// and coreNode is available.
func reconcileProgressiveTaint(ctx context.Context, k8sClient client.Client, coreNode *corev1.Node, isIdle bool) error {
	if !utils.IsProgressiveMigration() || coreNode == nil {
		return nil
	}
	taint := &corev1.Taint{
		Key:    constants.NodeUsedByTaintKey,
		Effect: corev1.TaintEffectPreferNoSchedule,
		Value:  constants.TensorFusionSystemName,
	}
	hasTaint := taints.TaintExists(coreNode.Spec.Taints, taint)
	var (
		updated *corev1.Node
		changed bool
	)
	switch {
	case isIdle && hasTaint:
		updated, changed, _ = taints.RemoveTaint(coreNode, taint)
	case !isIdle && !hasTaint:
		updated, changed, _ = taints.AddOrUpdateTaint(coreNode, taint)
	}
	if !changed {
		return nil
	}
	log.FromContext(ctx).Info("Reconciling TensorFusion isolation taint on node",
		"node", updated.Name, "taint", taint.Key, "idle", isIdle)
	if err := k8sClient.Update(ctx, updated); err != nil {
		return fmt.Errorf("failed to update K8S node: %w", err)
	}
	return nil
}

func calculateVirtualCapacity(node *tfv1.GPUNode, pool *tfv1.GPUPool) (resource.Quantity, resource.Quantity) {
	diskSize := node.Status.NodeInfo.DataDiskSize.Value()
	ramSize := node.Status.NodeInfo.RAMSize.Value()

	virtualVRAM := node.Status.TotalVRAM.DeepCopy()
	if pool.Spec.CapacityConfig == nil || pool.Spec.CapacityConfig.Oversubscription == nil {
		return virtualVRAM, node.Status.TotalTFlops.DeepCopy()
	}
	vTFlops := node.Status.TotalTFlops.AsApproximateFloat64() * (float64(pool.Spec.CapacityConfig.Oversubscription.TFlopsOversellRatio) / 100.0)

	virtualVRAM.Add(*resource.NewQuantity(
		int64(float64(float64(diskSize)*float64(pool.Spec.CapacityConfig.Oversubscription.VRAMExpandToHostDisk)/100.0)),
		resource.DecimalSI),
	)
	virtualVRAM.Add(*resource.NewQuantity(
		int64(float64(float64(ramSize)*float64(pool.Spec.CapacityConfig.Oversubscription.VRAMExpandToHostMem)/100.0)),
		resource.DecimalSI),
	)

	return virtualVRAM, *resource.NewQuantity(int64(vTFlops), resource.DecimalSI)
}
