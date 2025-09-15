package workload

import (
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
)

type State struct {
	Namespace             string
	Name                  string
	Annotations           map[string]string
	Spec                  tfv1.WorkloadProfileSpec
	Status                tfv1.TensorFusionWorkloadStatus
	CurrentReplicas       int32
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
	return w.Spec.AutoScalingConfig.AutoSetResources.Enable
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

	if w.CurrentReplicas != w.Status.AppliedRecommendedReplicas {
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

	w.CurrentReplicas = int32(len(w.CurrentActiveWorkers))
}

func (w *State) AddSample(sample *metrics.WorkerUsage) {
	sampler, exists := w.WorkerUsageSamplers[sample.WorkerName]
	if !exists {
		sampler = metrics.NewWorkerUsageSampler()
		w.WorkerUsageSamplers[sample.WorkerName] = sampler
	}
	sampler.AddSample(w.WorkerUsageAggregator, sample)
}

func isWorkerHasDedicatedGPU(worker *corev1.Pod) bool {
	return worker.Annotations[constants.DedicatedGPUAnnotation] == constants.TrueStringValue
}
