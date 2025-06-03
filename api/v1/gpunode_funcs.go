package v1

import (
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
)

func (node *GPUNode) InitializeStatus(initTFlops, initVRAM resource.Quantity, initGPUs int32) {
	node.Status = GPUNodeStatus{
		Phase:               TensorFusionGPUNodePhasePending,
		TotalTFlops:         initTFlops,
		TotalVRAM:           initVRAM,
		TotalGPUs:           initGPUs,
		AllocationInfo:      []*RunningAppDetail{},
		LoadedModels:        &[]string{},
		ManagedGPUDeviceIDs: []string{},
		ObservedGeneration:  node.Generation,
	}
}

func (node *GPUNode) SetAnnotationToTriggerNodeSync() {
	if node.Annotations == nil {
		node.Annotations = make(map[string]string)
	}
	node.Annotations["tensor-fusion.ai/refresh-node-state"] = time.Now().String()
}
