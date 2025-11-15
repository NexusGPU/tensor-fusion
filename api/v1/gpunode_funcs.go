package v1

import (
	"k8s.io/apimachinery/pkg/api/resource"
)

func (node *GPUNode) InitializeStatus(initTFlops, initVRAM resource.Quantity, initGPUs int32) {
	node.Status = GPUNodeStatus{
		Phase:               TensorFusionGPUNodePhasePending,
		TotalTFlops:         initTFlops,
		TotalVRAM:           initVRAM,
		TotalGPUs:           initGPUs,
		AllocatedPods:       make(map[string][]*PodGPUInfo),
		LoadedModels:        &[]string{},
		ManagedGPUDeviceIDs: []string{},
		ObservedGeneration:  node.Generation,
	}
}
