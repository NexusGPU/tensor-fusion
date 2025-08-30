package workload

import (
	"fmt"
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
	ScalingAnnotations    map[string]string
	Spec                  tfv1.WorkloadProfileSpec
	Status                tfv1.TensorFusionWorkloadStatus
	Workers               map[string]*WorkerState
	WorkerUsageAggregator *metrics.WorkerUsageAggregator
}

func NewWorkloadState() *State {
	return &State{
		Workers:               make(map[string]*WorkerState),
		ScalingAnnotations:    make(map[string]string),
		WorkerUsageAggregator: metrics.NewWorkerUsageAggregator(),
	}
}

func (w *State) GetResourcesSpec() *tfv1.Resources {
	return &w.Spec.Resources
}

func (w *State) GetCurrentResourcesSpec() (*tfv1.Resources, error) {
	resources, err := utils.GPUResourcesFromAnnotations(w.Annotations)
	if err != nil {
		return nil, fmt.Errorf("failed to get resources from annotations: %v", err)
	}
	if resources == nil {
		return &w.Spec.Resources, nil
	}
	return resources, nil
}

func (w *State) SetScalingAnnotation(key string, value string) {
	w.ScalingAnnotations[key] = value
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
