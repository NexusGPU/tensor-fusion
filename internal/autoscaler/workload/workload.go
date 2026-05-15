package workload

import (
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type State struct {
	Namespace string
	Name      string
	Spec      tfv1.WorkloadProfileSpec
	// Status holds the autoscaler's authoritative view of the recommendation /
	// applied-replicas / active-cron-rule / conditions fields. It is seeded
	// from the API CR on the first UpdateWorkloadState call and then owned
	// in-memory by the autoscaler — subsequent UpdateWorkloadState calls do
	// NOT overwrite Status, otherwise concurrent loadWorkloads ticks would
	// blow away pending recommender writes between Recommend and
	// UpdateWorkloadStatus (lost-write transaction race).
	Status                tfv1.TensorFusionWorkloadStatus
	CreationTimestamp     metav1.Time
	CurrentActiveWorkers  map[string]*corev1.Pod
	WorkerUsageSamplers   map[string]*metrics.WorkerUsageSampler
	WorkerUsageAggregator *metrics.WorkerUsageAggregator
	HistoryPeriod         time.Duration
	Mu                    sync.RWMutex
	// statusSeeded marks whether Status has been populated from the API CR.
	// Toggled true the first time UpdateWorkloadState observes this workload.
	statusSeeded bool
	// uid is the API UID of the workload this State is bound to. Used by
	// loadWorkloads to detect same-namespace/name-but-recreated workloads and
	// drop the stale State so its Recommendation / ActiveCronScalingRule /
	// AppliedRecommendedReplicas / conditions don't pollute the new workload.
	uid string
}

// MatchesUID reports whether this State is currently bound to the given API
// UID. A fresh State (uid == "") matches any UID, so the first
// UpdateWorkloadState call gets to claim it.
func (w *State) MatchesUID(uid string) bool {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	return w.uid == "" || w.uid == uid
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
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	return w.Spec.Resources.DeepCopy()
}

func (w *State) GetCurrentResourcesSpec() *tfv1.Resources {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	if w.Status.Recommendation != nil {
		return w.Status.Recommendation.DeepCopy()
	}
	return w.Spec.Resources.DeepCopy()
}

func (w *State) IsAutoSetResourcesEnabled() bool {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	asr := w.Spec.AutoScalingConfig.AutoSetResources
	return asr != nil && asr.Enable && asr.TargetResource != ""
}

func (w *State) ShouldScaleResource(name tfv1.ResourceName) bool {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	return w.shouldScaleResourceLocked(name)
}

func (w *State) shouldScaleResourceLocked(name tfv1.ResourceName) bool {
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
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	if w.Status.Recommendation == nil {
		return true
	}

	// Only non-dedicated-GPU workers receive recommendations, so the denominator
	// must exclude dedicated workers — otherwise AppliedRecommendedReplicas can
	// never reach len(CurrentActiveWorkers) and we'd retry forever.
	scalable := int32(0)
	for _, worker := range w.CurrentActiveWorkers {
		if !isWorkerHasDedicatedGPU(worker) {
			scalable++
		}
	}
	if scalable != w.Status.AppliedRecommendedReplicas {
		return false
	}

	curRes := w.Status.Recommendation
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

// ScalableWorkerCount returns the number of current active workers that are
// eligible to receive recommendations (i.e. excluding dedicated-GPU workers).
func (w *State) ScalableWorkerCount() int32 {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	var n int32
	for _, worker := range w.CurrentActiveWorkers {
		if !isWorkerHasDedicatedGPU(worker) {
			n++
		}
	}
	return n
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
	for i := range podList.Items {
		worker := &podList.Items[i]
		if !worker.DeletionTimestamp.IsZero() {
			continue
		}
		if _, exists := w.WorkerUsageSamplers[worker.Name]; !exists {
			w.WorkerUsageSamplers[worker.Name] = metrics.NewWorkerUsageSampler()
		}
		w.CurrentActiveWorkers[worker.Name] = worker.DeepCopy()
	}

	for key := range w.WorkerUsageSamplers {
		if _, exists := w.CurrentActiveWorkers[key]; !exists {
			delete(w.WorkerUsageSamplers, key)
		}
	}
}

func (w *State) AddSample(sample *metrics.WorkerUsage) {
	w.Mu.Lock()
	defer w.Mu.Unlock()
	sampler, exists := w.WorkerUsageSamplers[sample.WorkerName]
	if !exists {
		sampler = metrics.NewWorkerUsageSampler()
		w.WorkerUsageSamplers[sample.WorkerName] = sampler
	}
	sampler.AddSample(w.WorkerUsageAggregator, sample)
}

func (w *State) LatestRecommendation() *tfv1.Resources {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	if w.Status.Recommendation == nil {
		return nil
	}
	return w.Status.Recommendation.DeepCopy()
}

// SetRecommendation stores the latest computed recommendation in the
// autoscaler-owned in-memory Status so subsequent reads observe it across
// reconcile cycles. Must be called after every successful GetRecommendation
// that returns a non-nil value — otherwise GetCurrentResourcesSpec falls back
// to the original spec, LatestRecommendation cannot service retries, and
// IsRecommendationAppliedToAllWorkers short-circuits to true.
//
// The deep copy keeps callers' references independent of State internals.
// Pass nil to clear (e.g., when reverting to original spec).
func (w *State) SetRecommendation(rec *tfv1.Resources) {
	w.Mu.Lock()
	defer w.Mu.Unlock()
	if rec == nil {
		w.Status.Recommendation = nil
		return
	}
	w.Status.Recommendation = rec.DeepCopy()
}

// SetAppliedRecommendedReplicas overwrites AppliedRecommendedReplicas in one locked
// section. Prefer this over Reset+Inc-per-iteration to avoid lock churn in
// hot apply loops.
func (w *State) SetAppliedRecommendedReplicas(n int32) {
	w.Mu.Lock()
	defer w.Mu.Unlock()
	w.Status.AppliedRecommendedReplicas = n
}

func (w *State) ActiveWorkersSnapshot() []*corev1.Pod {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	workers := make([]*corev1.Pod, 0, len(w.CurrentActiveWorkers))
	for _, worker := range w.CurrentActiveWorkers {
		if worker == nil {
			continue
		}
		workers = append(workers, worker.DeepCopy())
	}
	return workers
}

func (w *State) StatusSnapshot() (string, string, tfv1.TensorFusionWorkloadStatus) {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	return w.Namespace, w.Name, *w.Status.DeepCopy()
}

func isWorkerHasDedicatedGPU(worker *corev1.Pod) bool {
	return worker.Annotations[constants.DedicatedGPUAnnotation] == constants.TrueStringValue
}

// Coordinates returns the workload (namespace, name). Both are immutable
// after the first UpdateWorkloadState but locked for completeness so external
// callers don't reach into the struct directly.
func (w *State) Coordinates() (string, string) {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	return w.Namespace, w.Name
}

// CreationTime returns a copy of CreationTimestamp.Time under lock.
func (w *State) CreationTime() time.Time {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	return w.CreationTimestamp.Time
}

// SpecSnapshot returns a deep copy of the WorkloadProfileSpec.
func (w *State) SpecSnapshot() tfv1.WorkloadProfileSpec {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	return *w.Spec.DeepCopy()
}

// AutoScalingConfigSnapshot returns a deep copy of the auto-scaling config.
func (w *State) AutoScalingConfigSnapshot() tfv1.AutoScalingConfig {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	return *w.Spec.AutoScalingConfig.DeepCopy()
}

// AutoSetResources returns a deep copy of the AutoSetResources config, or nil
// when unset. Callers checking ".Enable" or ".TargetResource" must handle nil.
func (w *State) AutoSetResources() *tfv1.AutoSetResources {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	if w.Spec.AutoScalingConfig.AutoSetResources == nil {
		return nil
	}
	return w.Spec.AutoScalingConfig.AutoSetResources.DeepCopy()
}

// ExternalScalerConfig returns a deep copy of the ExternalScaler config, or nil.
func (w *State) ExternalScalerConfig() *tfv1.ExternalScalerConfig {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	if w.Spec.AutoScalingConfig.ExternalScaler == nil {
		return nil
	}
	return w.Spec.AutoScalingConfig.ExternalScaler.DeepCopy()
}

// Qos returns the QoS level (immutable copy).
func (w *State) Qos() tfv1.QoSLevel {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	return w.Spec.Qos
}

// ActiveCronScalingRuleSnapshot returns a deep copy of the rule, or nil when unset.
func (w *State) ActiveCronScalingRuleSnapshot() *tfv1.CronScalingRule {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	if w.Status.ActiveCronScalingRule == nil {
		return nil
	}
	return w.Status.ActiveCronScalingRule.DeepCopy()
}

// SetActiveCronScalingRule replaces Status.ActiveCronScalingRule under lock.
// Pass a deep copy if the caller wants to retain its own reference.
func (w *State) SetActiveCronScalingRule(rule *tfv1.CronScalingRule) {
	w.Mu.Lock()
	defer w.Mu.Unlock()
	w.Status.ActiveCronScalingRule = rule
}

// UpsertStatusCondition runs meta.SetStatusCondition on Status.Conditions
// under the write lock.
func (w *State) UpsertStatusCondition(cond metav1.Condition) {
	w.Mu.Lock()
	defer w.Mu.Unlock()
	meta.SetStatusCondition(&w.Status.Conditions, cond)
}

// ConsumeAggregator runs fn while holding the State read lock so callers can
// safely query the WorkerUsageAggregator without exposing the pointer outside
// the lock window. The aggregator MUST NOT be retained after fn returns.
func (w *State) ConsumeAggregator(fn func(*metrics.WorkerUsageAggregator)) {
	w.Mu.RLock()
	defer w.Mu.RUnlock()
	fn(w.WorkerUsageAggregator)
}

// UpdateWorkerAnnotations merges the given annotations into the tracked
// worker's in-memory annotations under the State write lock. Call this after a
// successful API Patch so subsequent ActiveWorkersSnapshot reads see the new
// state without requiring a re-list. No-op when the worker is not tracked.
func (w *State) UpdateWorkerAnnotations(workerName string, annotations map[string]string) {
	if len(annotations) == 0 {
		return
	}
	w.Mu.Lock()
	defer w.Mu.Unlock()
	worker, ok := w.CurrentActiveWorkers[workerName]
	if !ok || worker == nil {
		return
	}
	if worker.Annotations == nil {
		worker.Annotations = make(map[string]string, len(annotations))
	}
	for k, v := range annotations {
		worker.Annotations[k] = v
	}
}
