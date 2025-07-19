package workload

import (
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/recommender"
	corev1 "k8s.io/api/core/v1"
)

type WorkloadState struct {
	Namespace             string
	Name                  string
	Spec                  tfv1.WorkloadProfileSpec
	Recommendation        recommender.RecommendedResources
	Workers               map[string]*WorkerState
	WorkerUsageAggregator *metrics.WorkerUsageAggregator
}

func NewWorkloadState(name string) *WorkloadState {
	return &WorkloadState{
		Name:                  name,
		Workers:               make(map[string]*WorkerState),
		WorkerUsageAggregator: metrics.NewWorkerUsageAggregator(),
	}
}

func (w *WorkloadState) UpdateRecommendation(recommendation recommender.RecommendedResources) {
	w.Recommendation = recommendation
}

func (w *WorkloadState) IsAutoScalingEnabled() bool {
	return w.Spec.AutoScalingConfig.AutoSetResources.Enable
}

func (w *WorkloadState) ShouldScaleResource(name tfv1.ResourceName) bool {
	target := w.Spec.AutoScalingConfig.AutoSetResources.TargetResource
	return target == "" || strings.EqualFold(target, "all") || strings.EqualFold(string(name), target)
}

func (w *WorkloadState) UpdateWorkers(podList *corev1.PodList) {
	observedWorkers := map[string]bool{}
	for _, worker := range podList.Items {
		if !worker.DeletionTimestamp.IsZero() {
			continue
		}
		if _, exists := w.Workers[worker.Name]; !exists {
			w.Workers[worker.Name] = NewWorkerState(worker.Name, w.Name)
		}
		observedWorkers[worker.Name] = true
	}

	for key, worker := range w.Workers {
		if worker.WorkloadName == w.Name && !observedWorkers[key] {
			delete(w.Workers, key)
		}
	}
}

func (w *WorkloadState) AddSample(sample *metrics.WorkerUsage) {
	worker, exists := w.Workers[sample.WorkerName]
	if !exists {
		worker = NewWorkerState(sample.WorkerName, sample.WorkloadName)
		w.Workers[sample.WorkerName] = worker
	}
	worker.AddSample(w.WorkerUsageAggregator, sample)
}
