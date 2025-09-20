package workload

import (
	"fmt"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
)

type State struct {
	Namespace             string
	Name                  string
	Spec                  tfv1.WorkloadProfileSpec
	Status                tfv1.TensorFusionWorkloadStatus
	CurrentActiveWorkers  map[string]*corev1.Pod
	WorkerUsageSamplers   map[string]*metrics.WorkerUsageSampler
	WorkerUsageAggregator *metrics.WorkerUsageAggregator
}

func NewWorkloadState() *State {
	return &State{
		WorkerUsageSamplers:   make(map[string]*metrics.WorkerUsageSampler),
		WorkerUsageAggregator: metrics.NewWorkerUsageAggregator(),
	}
}

func (w *State) GetOriginalResourcesSpec() *tfv1.Resources {
	return &w.Spec.Resources
}

func (w *State) GetCurrentResourcesSpec() *tfv1.Resources {
	if w.Status.Recommendation != nil {
		return w.Status.Recommendation
	}
	return w.GetOriginalResourcesSpec()
}

func (w *State) IsAutoSetResourcesEnabled() bool {
	return w.Spec.AutoScalingConfig.AutoSetResources.Enable &&
		w.Spec.AutoScalingConfig.AutoSetResources.TargetResource != ""
}

func (w *State) ShouldScaleResource(name tfv1.ResourceName) bool {
	target := w.Spec.AutoScalingConfig.AutoSetResources.TargetResource
	// Do not scale when TargetResouce is empty
	return strings.EqualFold(target, "all") || strings.EqualFold(string(name), target)
}

func (w *State) IsRecommendationAppliedToAllWorkers() bool {
	if w.Status.Recommendation == nil {
		return true
	}

	if int32(len(w.CurrentActiveWorkers)) != w.Status.AppliedRecommendedReplicas {
		return false
	}

	curRes := w.GetCurrentResourcesSpec()
	for _, worker := range w.CurrentActiveWorkers {
		if isWorkerHasDedicatedGPU(worker) {
			continue
		}
		workerRes, _ := utils.GPUResourcesFromAnnotations(worker.Annotations)
		if !curRes.Equal(workerRes) {
			return false
		}
	}

	return true
}

func (w *State) updateCurrentActiveWorkers(podList *corev1.PodList) {
	w.CurrentActiveWorkers = map[string]*corev1.Pod{}
	for _, worker := range podList.Items {
		if !worker.DeletionTimestamp.IsZero() {
			continue
		}
		if _, exists := w.WorkerUsageSamplers[worker.Name]; !exists {
			w.WorkerUsageSamplers[worker.Name] = metrics.NewWorkerUsageSampler()
		}
		w.CurrentActiveWorkers[worker.Name] = &worker
	}

	for key := range w.WorkerUsageSamplers {
		if _, exists := w.CurrentActiveWorkers[key]; !exists {
			delete(w.WorkerUsageSamplers, key)
		}
	}
}

func (w *State) AddSample(sample *metrics.WorkerUsage) {
	sampler, exists := w.WorkerUsageSamplers[sample.WorkerName]
	if !exists {
		sampler = metrics.NewWorkerUsageSampler()
		w.WorkerUsageSamplers[sample.WorkerName] = sampler
	}
	sampler.AddSample(w.WorkerUsageAggregator, sample)
}

func (w *State) GetWorkerMaxGPUAllocation(allocator *gpuallocator.GpuAllocator) (*tfv1.Resource, error) {
	if len(w.CurrentActiveWorkers) <= 0 {
		return nil, nil
	}

	gpuStore, _, allocRequests := allocator.GetAllocationInfo()
	gpuToWorkers := map[*tfv1.GPU][]*corev1.Pod{}
	for _, worker := range w.CurrentActiveWorkers {
		allocated, exists := allocRequests[string(worker.UID)]
		if !exists || allocated == nil {
			return nil, fmt.Errorf("worker %s has not allocated GPUs", worker.Name)
		}
		for _, gpuName := range allocated.GPUNames {
			gpuNameNs := types.NamespacedName{Name: gpuName}
			gpu, exists := gpuStore[gpuNameNs]
			if !exists {
				return nil, fmt.Errorf("GPU not found in allocator store %s", gpuName)
			}
			gpuToWorkers[gpu] = append(gpuToWorkers[gpu], worker)
		}
	}

	var (
		maxTflops int64 = -1
		maxVram   int64 = -1
	)
	for gpu, workers := range gpuToWorkers {
		avaiableTflops := gpu.Status.Available.Tflops
		avaiableVram := gpu.Status.Available.Vram
		for _, worker := range workers {
			avaiableTflops.Add(allocRequests[string(worker.UID)].Request.Tflops)
			avaiableVram.Add(allocRequests[string(worker.UID)].Request.Vram)
		}

		allTflops, _ := avaiableTflops.AsInt64()
		allVram, _ := avaiableVram.AsInt64()
		workerCount := int64(len(workers))
		tflopsPerWorker := allTflops / workerCount
		vramPerWorker := allVram / workerCount
		if maxTflops == -1 || tflopsPerWorker < maxTflops {
			maxTflops = tflopsPerWorker
		}
		if maxVram == -1 || vramPerWorker < maxVram {
			maxVram = vramPerWorker
		}
	}

	return &tfv1.Resource{
		Tflops: *resource.NewQuantity(maxTflops, resource.DecimalSI),
		Vram:   *resource.NewQuantity(maxVram, resource.BinarySI),
	}, nil
}

func isWorkerHasDedicatedGPU(worker *corev1.Pod) bool {
	return worker.Annotations[constants.DedicatedGPUAnnotation] == constants.TrueStringValue
}
