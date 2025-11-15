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
		// node discovery job not completed, wait next reconcile loop to check again
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
		tflopsAvailable.Sub(allocRequest.Limit.Tflops)
	}
	node.Status.VirtualAvailableVRAM = &vramAvailable
	node.Status.VirtualAvailableTFlops = &tflopsAvailable

	node.Status.Phase = tfv1.TensorFusionGPUNodePhaseRunning

	if !equality.Semantic.DeepEqual(node.Status, statusCopy) {
		err := k8sClient.Status().Update(ctx, node)
		if err != nil {
			return nil, fmt.Errorf("failed to update GPU node status: %w", err)
		}

		// check if need to update K8S node label
		if utils.IsProgressiveMigration() && coreNode != nil {
			taint := &corev1.Taint{
				Key:    constants.NodeUsedByTaintKey,
				Effect: corev1.TaintEffectNoSchedule,
				Value:  constants.TensorFusionSystemName,
			}
			needUpdateNode := false
			if node.Status.AvailableVRAM.Equal(node.Status.TotalVRAM) && node.Status.AvailableTFlops.Equal(node.Status.TotalTFlops) {
				// check if need to remove the taint
				coreNode, needUpdateNode, _ = taints.RemoveTaint(coreNode, taint)
			} else if !taints.TaintExists(coreNode.Spec.Taints, taint) {
				// check if need to add the taint
				coreNode, needUpdateNode, _ = taints.AddOrUpdateTaint(coreNode, taint)
			}
			if needUpdateNode {
				log.FromContext(ctx).Info("Updating K8S node taints for isolation of tensor-fusion and non-tensor-fusion used nodes",
					"node", coreNode.Name, "taint", taint.Key)
				err := k8sClient.Update(ctx, coreNode)
				if err != nil {
					return nil, fmt.Errorf("failed to update K8S node: %w", err)
				}
			}
		}
	}
	return gpuModels, nil
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
