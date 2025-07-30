package workload

import (
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
)

type State struct {
	Namespace             string
	Name                  string
	Annotations           map[string]string
	Spec                  tfv1.WorkloadProfileSpec
	Recommendation        tfv1.Resources
	Workers               map[string]*WorkerState
	WorkerUsageAggregator *metrics.WorkerUsageAggregator
}

func NewWorkloadState(name string) *State {
	return &State{
		Name:                  name,
		Workers:               make(map[string]*WorkerState),
		WorkerUsageAggregator: metrics.NewWorkerUsageAggregator(),
	}
}

func (w *State) GetLastResourcesSpec() (*tfv1.Resources, error) {
	return utils.LastResourcesFromAnnotations(w.Annotations)
}

func (w *State) GetCurrentResourcesSpec() (*tfv1.Resources, error) {
	return utils.CurrentResourcesFromAnnotations(w.Annotations)
}

func (w *State) IsAutoSetResourcesEnabled() bool {
	return w.Spec.AutoScalingConfig.AutoSetResources.Enable
}

func (w *State) ShouldScaleResource(name tfv1.ResourceName) bool {
	target := w.Spec.AutoScalingConfig.AutoSetResources.TargetResource
	return strings.EqualFold(target, "all") || strings.EqualFold(string(name), target)
}

func (w *State) updateWorkers(podList *corev1.PodList) {
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

func (w *State) AddSample(sample *metrics.WorkerUsage) {
	worker, exists := w.Workers[sample.WorkerName]
	if !exists {
		worker = NewWorkerState(sample.WorkerName, sample.WorkloadName)
		w.Workers[sample.WorkerName] = worker
	}
	worker.AddSample(w.WorkerUsageAggregator, sample)
}
