package workload

import (
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
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

	// Mu guards every field above except WorkerUsageAggregator, whose own
	// internal lock makes it safe to call concurrently with State.Mu held or
	// not. All exported State methods that read/write these fields take Mu
	// themselves; private helpers (e.g. updateCurrentActiveWorkers,
	// updateHistoryPeriod) assume the caller already holds Mu.
	//
	// Recommender / status-update paths must not read Spec / Status / map
	// fields directly. Use Snapshot() to obtain an immutable StateView and
	// drive logic from it; mutation intents flow back through ApplyIntents.
	Mu sync.Mutex
}

func NewWorkloadState() *State {
	return &State{
		// Default history period is 2 hours, decay to half in 1 hour
		HistoryPeriod:         2 * time.Hour,
		WorkerUsageSamplers:   make(map[string]*metrics.WorkerUsageSampler),
		WorkerUsageAggregator: metrics.NewWorkerUsageAggregator(time.Hour),
	}
}

// StateView is an immutable snapshot of State at a point in time. Recommenders
// and status-update helpers operate on a view so that they no longer race
// with UpdateWorkloadState / AddSample mutating State concurrently.
//
// Spec / Status / Workers are deep-copied at snapshot time so mutating the
// view (or the underlying informer object) cannot leak back into State.
// Aggregator stays as a pointer because it has its own internal lock and
// cloning the histograms would be expensive; if updateHistoryPeriod swaps
// the aggregator on State, the view simply keeps reading the old one for the
// rest of the tick — that's accepted staleness, not a race.
type StateView struct {
	Namespace         string
	Name              string
	Spec              tfv1.WorkloadProfileSpec
	Status            tfv1.TensorFusionWorkloadStatus
	CreationTimestamp metav1.Time
	HistoryPeriod     time.Duration
	Workers           []*corev1.Pod
	Aggregator        *metrics.WorkerUsageAggregator
}

// Snapshot returns an immutable view of the current State. The returned view
// is safe to read concurrently with further AddSample / UpdateWorkloadState
// calls on State.
func (w *State) Snapshot() *StateView {
	w.Mu.Lock()
	defer w.Mu.Unlock()

	workers := make([]*corev1.Pod, 0, len(w.CurrentActiveWorkers))
	for _, pod := range w.CurrentActiveWorkers {
		workers = append(workers, pod.DeepCopy())
	}

	return &StateView{
		Namespace:         w.Namespace,
		Name:              w.Name,
		Spec:              *w.Spec.DeepCopy(),
		Status:            *w.Status.DeepCopy(),
		CreationTimestamp: w.CreationTimestamp,
		HistoryPeriod:     w.HistoryPeriod,
		Workers:           workers,
		Aggregator:        w.WorkerUsageAggregator,
	}
}

// Intent describes a recommender side-effect that must be merged back into
// State.Status under State.Mu. nil-valued fields mean "no change".
type Intent struct {
	Condition         *metav1.Condition
	SetActiveCronRule bool
	ActiveCronRule    *tfv1.CronScalingRule
}

// ApplyIntents merges recommender side-effects into Status under State.Mu.
func (w *State) ApplyIntents(intents []Intent) {
	if len(intents) == 0 {
		return
	}
	w.Mu.Lock()
	defer w.Mu.Unlock()
	for _, in := range intents {
		if in.Condition != nil {
			meta.SetStatusCondition(&w.Status.Conditions, *in.Condition)
		}
		if in.SetActiveCronRule {
			w.Status.ActiveCronScalingRule = in.ActiveCronRule
		}
	}
}

// SetAppliedRecommendedReplicas updates the in-memory replica counter under Mu.
func (w *State) SetAppliedRecommendedReplicas(n int32) {
	w.Mu.Lock()
	defer w.Mu.Unlock()
	w.Status.AppliedRecommendedReplicas = n
}

// IncAppliedRecommendedReplicas atomically increments the counter under Mu.
func (w *State) IncAppliedRecommendedReplicas() {
	w.Mu.Lock()
	defer w.Mu.Unlock()
	w.Status.AppliedRecommendedReplicas++
}

// setWorkerAnnotations mirrors a freshly patched worker's annotations into
// the in-memory state under Mu. Without this, a deep-copied per-tick worker
// snapshot would diverge from the K8s pod after a successful Patch and
// subsequent ticks would compare recommendation against pre-patch values.
//
// The annotations slice is copied, so the caller may continue mutating its
// own map without affecting state.
func (w *State) setWorkerAnnotations(workerName string, annotations map[string]string) {
	w.Mu.Lock()
	defer w.Mu.Unlock()
	pod, ok := w.CurrentActiveWorkers[workerName]
	if !ok {
		return
	}
	cloned := make(map[string]string, len(annotations))
	for k, v := range annotations {
		cloned[k] = v
	}
	pod.Annotations = cloned
}

func (v *StateView) GetOriginalResourcesSpec() *tfv1.Resources {
	return &v.Spec.Resources
}

func (v *StateView) GetCurrentResourcesSpec() *tfv1.Resources {
	if v.Status.Recommendation != nil {
		return v.Status.Recommendation
	}
	return v.GetOriginalResourcesSpec()
}

func (v *StateView) IsAutoSetResourcesEnabled() bool {
	asr := v.Spec.AutoScalingConfig.AutoSetResources
	if asr == nil {
		return false
	}
	return asr.Enable && asr.TargetResource != ""
}

func (v *StateView) ShouldScaleResource(name tfv1.ResourceName) bool {
	asr := v.Spec.AutoScalingConfig.AutoSetResources
	if asr == nil {
		return false
	}
	target := asr.TargetResource
	if target == "" {
		return false
	}
	if strings.EqualFold(string(target), "all") {
		return true
	}
	resourceNameStr := string(name)
	if resourceNameStr == "tflops" {
		resourceNameStr = "compute"
	}
	return strings.EqualFold(resourceNameStr, string(target))
}

// IsRecommendationAppliedToAllWorkers checks under Mu whether every active
// worker has the latest recommendation in its annotations.
func (w *State) IsRecommendationAppliedToAllWorkers() bool {
	w.Mu.Lock()
	defer w.Mu.Unlock()

	if w.Status.Recommendation == nil {
		return true
	}

	if int32(len(w.CurrentActiveWorkers)) != w.Status.AppliedRecommendedReplicas {
		return false
	}

	curRes := w.currentResourcesSpecLocked()
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

// LatestRecommendation returns Status.Recommendation under Mu.
func (w *State) LatestRecommendation() *tfv1.Resources {
	w.Mu.Lock()
	defer w.Mu.Unlock()
	if w.Status.Recommendation == nil {
		return nil
	}
	return w.Status.Recommendation.DeepCopy()
}

// ShouldScaleResource is the *State-side helper used inside the
// status-update / apply path. It takes Mu itself; callers that already hold
// Mu (e.g. ApplyRecommendationToWorkload, while it is also pulling
// Namespace/Name out of the same critical section) should use
// shouldScaleResourceLocked to avoid re-locking.
func (w *State) ShouldScaleResource(name tfv1.ResourceName) bool {
	w.Mu.Lock()
	defer w.Mu.Unlock()
	return w.shouldScaleResourceLocked(name)
}

// shouldScaleResourceLocked is the lock-free body of ShouldScaleResource;
// callers must already hold State.Mu.
func (w *State) shouldScaleResourceLocked(name tfv1.ResourceName) bool {
	asr := w.Spec.AutoScalingConfig.AutoSetResources
	if asr == nil {
		return false
	}
	target := asr.TargetResource
	if target == "" {
		return false
	}
	if strings.EqualFold(string(target), "all") {
		return true
	}
	resourceNameStr := string(name)
	if resourceNameStr == "tflops" {
		resourceNameStr = "compute"
	}
	return strings.EqualFold(resourceNameStr, string(target))
}

// IsAutoSetResourcesEnabled is the *State-side helper that reads Spec under Mu.
func (w *State) IsAutoSetResourcesEnabled() bool {
	w.Mu.Lock()
	defer w.Mu.Unlock()
	asr := w.Spec.AutoScalingConfig.AutoSetResources
	if asr == nil {
		return false
	}
	return asr.Enable && asr.TargetResource != ""
}

func (w *State) currentResourcesSpecLocked() *tfv1.Resources {
	if w.Status.Recommendation != nil {
		return w.Status.Recommendation
	}
	return &w.Spec.Resources
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
		// DeepCopy to fully detach from the informer-owned slice; the recommender /
		// applyRecommendationToWorker path may mutate Annotations etc.
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

func isWorkerHasDedicatedGPU(worker *corev1.Pod) bool {
	return worker.Annotations[constants.DedicatedGPUAnnotation] == constants.TrueStringValue
}
