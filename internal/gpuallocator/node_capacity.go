package gpuallocator

import (
	"context"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func RefreshGPUNodeCapacity(ctx context.Context, k8sClient client.Client, node *tfv1.GPUNode, pool *tfv1.GPUPool) ([]string, error) {
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
	node.Status.AllocationInfo = []*tfv1.RunningAppDetail{}

	gpuModels := []string{}
	deduplicationMap := make(map[string]struct{})

	for _, gpu := range gpuList.Items {
		if gpu.Status.Available == nil || gpu.Status.Capacity == nil {
			continue
		}
		node.Status.AvailableVRAM.Add(gpu.Status.Available.Vram)
		node.Status.AvailableTFlops.Add(gpu.Status.Available.Tflops)
		node.Status.TotalVRAM.Add(gpu.Status.Capacity.Vram)
		node.Status.TotalTFlops.Add(gpu.Status.Capacity.Tflops)
		gpuModels = append(gpuModels, gpu.Status.GPUModel)

		for _, runningApp := range gpu.Status.RunningApps {
			if _, ok := deduplicationMap[runningApp.Name+"_"+runningApp.Namespace]; !ok {
				node.Status.AllocationInfo = append(node.Status.AllocationInfo, runningApp.DeepCopy())
				deduplicationMap[runningApp.Name+"_"+runningApp.Namespace] = struct{}{}
			}
		}
	}

	virtualVRAM, virtualTFlops := calculateVirtualCapacity(node, pool)
	node.Status.VirtualTFlops = virtualTFlops
	node.Status.VirtualVRAM = virtualVRAM

	node.Status.Phase = tfv1.TensorFusionGPUNodePhaseRunning

	if !equality.Semantic.DeepEqual(node.Status, statusCopy) {
		err := k8sClient.Status().Update(ctx, node)
		if err != nil {
			return nil, fmt.Errorf("failed to update GPU node status: %w", err)
		}
	}
	return gpuModels, nil
}

func calculateVirtualCapacity(node *tfv1.GPUNode, pool *tfv1.GPUPool) (resource.Quantity, resource.Quantity) {
	diskSize, _ := node.Status.NodeInfo.DataDiskSize.AsInt64()
	ramSize, _ := node.Status.NodeInfo.RAMSize.AsInt64()

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
