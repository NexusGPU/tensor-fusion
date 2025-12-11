package workload

import (
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type State struct {
	Namespace             string
	Name                  string
	Spec                  tfv1.WorkloadProfileSpec
	Status                tfv1.TensorFusionWorkloadStatus
	CreationTimestamp     metav1.Time
	CurrentActiveWorkers  map[string]*corev1.Pod
	WorkerUsageSamplers   map[string]*metrics.WorkerUsageSampler
	WorkerUsageAggregator *metrics.WorkerUsageAggregator
	HistoryPeriod         time.Duration
}

func NewWorkloadState() *State {
	return &State{
		// Default history period is 2 hours, decay to half in 1 hour
		HistoryPeriod:         2 * time.Hour,
		WorkerUsageSamplers:   make(map[string]*metrics.WorkerUsageSampler),
		WorkerUsageAggregator: metrics.NewWorkerUsageAggregator(time.Hour),
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
	asr := w.Spec.AutoScalingConfig.AutoSetResources
	if asr == nil {
		return false
	}
	target := asr.TargetResource
	// Do not scale when TargetResource is empty
	if target == "" {
		return false
	}
	if strings.EqualFold(string(target), "all") {
		return true
	}
	// Map ResourceName to ScalingTargetResource: "tflops" -> "compute"
	resourceNameStr := string(name)
	if resourceNameStr == "tflops" {
		resourceNameStr = "compute"
	}
	return strings.EqualFold(resourceNameStr, string(target))
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

func (w *State) updateHistoryPeriod(historyDataPeriod string) {
	if historyDataPeriod == "" {
		return
	}
	period, err := time.ParseDuration(historyDataPeriod)
	if err != nil {
		return
	}
	if w.HistoryPeriod == period {
		return
	}
	w.HistoryPeriod = period
	w.WorkerUsageAggregator = metrics.NewWorkerUsageAggregator(period / 2)
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

func isWorkerHasDedicatedGPU(worker *corev1.Pod) bool {
	return worker.Annotations[constants.DedicatedGPUAnnotation] == constants.TrueStringValue
}
