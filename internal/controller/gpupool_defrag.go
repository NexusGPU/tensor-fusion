// Strategy #2 for GPUPool compaction: move TF workers off low-utilization
// nodes so Strategy #1 can later reclaim the emptied nodes.
//
// +kubebuilder:rbac:groups=core,resources=pods/eviction,verbs=create
// +kubebuilder:rbac:groups=core,resources=nodes,verbs=get;list;patch;update
package controller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator/filter"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

const (
	// How long the watcher waits for allocator state to observe an empty node.
	drainConfirmationTimeout = 10 * time.Minute

	// Fallback when MaxDuration is empty or invalid.
	defaultDefragMaxDuration = 2 * time.Hour

	// Independent timeout for persisting LastDefragTime.
	defragStatusPersistTimeout = 30 * time.Second

	// Bound how long we wait for scheduler cache to observe the drain label.
	defragSchedulerCacheSyncTimeout = 30 * time.Second
	defragSchedulerCacheSyncPoll    = 200 * time.Millisecond
)

const (
	defragEventStarted           = "DefragStarted"
	defragEventNodeDraining      = "DefragNodeDraining"
	defragEventPodEvicted        = "DefragPodEvicted"
	defragEventPodEvictFailed    = "DefragPodEvictFailed"
	defragEventAbortNode         = "DefragAbortNode"
	defragEventSkipUnschedulable = "DefragSkipUnschedulable"
	defragEventNodeDrained       = "DefragNodeDrained"
	defragEventDrainTimeout      = "DefragDrainTimeout"
	defragEventStaleLabelCleared = "DefragStaleLabelCleared"
	defragEventFinished          = "DefragFinished"
)

// Mirrored from the scheduler expander to avoid an import cycle.
type schedulerFitPodAPI interface {
	FindNodesThatFitPod(
		ctx context.Context,
		schedFramework framework.Framework,
		state fwk.CycleState,
		pod *corev1.Pod,
	) ([]fwk.NodeInfo, framework.Diagnosis, error)
}

// Snapshot of the most recent defrag run for a pool.
type defragRunStats struct {
	StartTime          time.Time `json:"startTime"`
	EndTime            time.Time `json:"endTime"`
	CandidateNodes     int       `json:"candidateNodes"`
	ProcessedNodes     int       `json:"processedNodes"`
	EvictedPods        int       `json:"evictedPods"`
	EvictionFailures   int       `json:"evictionFailures"`
	UnmovableNodes     int       `json:"unmovableNodes"`
	FreshPodSkips      int       `json:"freshPodSkips"`
	DeadlineExceeded   bool      `json:"deadlineExceeded"`
	DrainingCandidates []string  `json:"drainingCandidates,omitempty"`
}

// sync.Map keyed by pool name -> *defragRunStats.
var defragLastRunStats sync.Map

func GetDefragLastRunStats() map[string]defragRunStats {
	out := map[string]defragRunStats{}
	defragLastRunStats.Range(func(k, v any) bool {
		name, _ := k.(string)
		stats, ok := v.(*defragRunStats)
		if !ok || stats == nil {
			return true
		}
		out[name] = *stats
		return true
	})
	return out
}

// ----- cron gating ------------------------------------------------------

// maybeStartDefragRun validates the config, checks the schedule, and starts
// a background run when the pool is due.
func (r *GPUPoolCompactionReconciler) maybeStartDefragRun(ctx context.Context, pool *tfv1.GPUPool) {
	logger := log.FromContext(ctx).WithValues("pool", pool.Name, "component", "defrag")

	cfg := getDefragConfig(pool)
	if cfg == nil || !cfg.Enabled {
		return
	}
	if r.Scheduler == nil || r.KubeClient == nil {
		logger.V(4).Info("defrag skipped: scheduler or kube client not wired; set ENABLE_SCHEDULER=true and inject KubeClient")
		return
	}
	if r.Allocator == nil {
		return
	}
	if !r.Allocator.IsReady() {
		logger.V(4).Info("defrag skipped: allocator not ready")
		return
	}

	schedule, err := r.defragParser.Parse(cfg.Schedule)
	if err != nil {
		logger.Error(err, "invalid defrag cron schedule; defrag disabled", "schedule", cfg.Schedule)
		return
	}
	maxDuration := parseDefragMaxDuration(cfg.MaxDuration, logger)

	// Treat a run as due once a scheduled tick has elapsed since the anchor.
	now := time.Now()
	anchor := now.Add(-maxDuration).Add(-time.Minute)
	if pool.Status.LastDefragTime != nil {
		anchor = pool.Status.LastDefragTime.Time
	}
	next := schedule.Next(anchor)
	if next.After(now) {
		logger.V(5).Info("defrag not due yet", "next", next, "anchor", anchor)
		return
	}

	flagAny, _ := r.defragRunning.LoadOrStore(pool.Name, &atomic.Bool{})
	flag, _ := flagAny.(*atomic.Bool)
	if !flag.CompareAndSwap(false, true) {
		logger.V(4).Info("defrag already running for this pool; skipping")
		return
	}

	runStart := time.Now()
	poolCopy := pool.DeepCopy()
	go func() {
		defer flag.Store(false)
		runCtx, cancel := context.WithTimeout(context.Background(), maxDuration)
		defer cancel()
		r.runDefrag(runCtx, poolCopy, runStart)
	}()
}

func getDefragConfig(pool *tfv1.GPUPool) *tfv1.NodeDefragConfig {
	if pool.Spec.NodeManagerConfig == nil ||
		pool.Spec.NodeManagerConfig.NodeCompaction == nil {
		return nil
	}
	return pool.Spec.NodeManagerConfig.NodeCompaction.Defrag
}

func parseDefragMaxDuration(raw string, logger interface{ Info(string, ...any) }) time.Duration {
	if raw == "" {
		return defaultDefragMaxDuration
	}
	d, err := time.ParseDuration(raw)
	if err != nil || d <= 0 {
		logger.Info("invalid defrag maxDuration, falling back to default", "input", raw, "default", defaultDefragMaxDuration)
		return defaultDefragMaxDuration
	}
	return d
}

// ----- main run ---------------------------------------------------------

// runDefrag processes one pool under a MaxDuration deadline.
func (r *GPUPoolCompactionReconciler) runDefrag(
	ctx context.Context,
	pool *tfv1.GPUPool,
	runStart time.Time,
) {
	l := log.FromContext(ctx).WithValues("pool", pool.Name, "component", "defrag", "runStart", runStart)
	r.Recorder.Eventf(pool, corev1.EventTypeNormal, defragEventStarted,
		"defrag run started at %s", runStart.Format(time.RFC3339))

	stats := &defragRunStats{StartTime: runStart}
	defer func() {
		stats.EndTime = time.Now()
		defragLastRunStats.Store(pool.Name, stats)
		// Use a fresh context so timeouts still persist the schedule anchor.
		persistCtx, persistCancel := context.WithTimeout(
			context.Background(), defragStatusPersistTimeout)
		defer persistCancel()
		if err := r.patchPoolLastDefragTime(persistCtx, pool, runStart); err != nil {
			l.Error(err, "failed to patch pool.Status.LastDefragTime")
		}
		r.Recorder.Eventf(pool, corev1.EventTypeNormal, defragEventFinished,
			"defrag run finished: candidates=%d processed=%d evicted=%d failed=%d unmovable=%d freshSkip=%d deadline=%t",
			stats.CandidateNodes, stats.ProcessedNodes, stats.EvictedPods, stats.EvictionFailures,
			stats.UnmovableNodes, stats.FreshPodSkips, stats.DeadlineExceeded)
	}()

	candidates, err := r.collectDefragCandidates(ctx, pool)
	if err != nil {
		l.Error(err, "collect defrag candidates failed")
		return
	}
	stats.CandidateNodes = len(candidates)
	if len(candidates) == 0 {
		l.Info("no defrag candidates")
		return
	}

	maxWorkerPerNode := r.Allocator.MaxWorkerPerNode()

	for _, cand := range candidates {
		if ctxDone(ctx) {
			stats.DeadlineExceeded = true
			l.Info("defrag deadline exceeded during candidate loop", "remaining", len(candidates)-stats.ProcessedNodes)
			return
		}
		stats.ProcessedNodes++
		r.processDefragCandidate(ctx, pool, cand, runStart, maxWorkerPerNode, stats)
	}
}

func ctxDone(ctx context.Context) bool {
	select {
	case <-ctx.Done():
		return true
	default:
		return false
	}
}

// ----- candidate filtering + sorting ------------------------------------

// defragCandidate holds the node state needed for simulation and eviction.
type defragCandidate struct {
	nodeName         string
	gpuNodeName      string
	totalPoolGPUs    int
	usedPoolGPUs     int
	utilizationScore float64 // 0..100 for stable sort
	workerPods       []*corev1.Pod
}

func (r *GPUPoolCompactionReconciler) collectDefragCandidates(ctx context.Context, pool *tfv1.GPUPool) ([]*defragCandidate, error) {
	cfg := getDefragConfig(pool)
	if cfg == nil {
		return nil, nil
	}

	allNodes := &tfv1.GPUNodeList{}
	if err := r.List(ctx, allNodes, client.MatchingLabels(map[string]string{
		fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name): constants.TrueStringValue,
	})); err != nil {
		return nil, fmt.Errorf("list gpu nodes: %w", err)
	}

	nodeGpuStore := r.Allocator.GetNodeGpuStore()
	nodeWorkerStore := r.Allocator.GetNodeWorkerStoreSnapshot()

	threshold := int64(cfg.UtilizationThresholdPercent)
	out := make([]*defragCandidate, 0, len(allNodes.Items))

	for i := range allNodes.Items {
		gpuNode := &allNodes.Items[i]
		if gpuNode.Labels[constants.SchedulingDoNotDisruptLabel] == constants.TrueStringValue {
			continue
		}
		if !gpuNode.DeletionTimestamp.IsZero() {
			continue
		}
		if gpuNode.Status.Phase != tfv1.TensorFusionGPUNodePhaseRunning {
			continue
		}

		k8sNodeName := gpuNode.Name
		k8sNode := &corev1.Node{}
		if err := r.Get(ctx, client.ObjectKey{Name: k8sNodeName}, k8sNode); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			continue
		}
		if k8sNode.Spec.Unschedulable {
			continue
		}
		if k8sNode.Labels[constants.DefragDrainingLabel] == constants.TrueStringValue {
			continue
		}

		nodeGPUs := nodeGpuStore[k8sNodeName]
		if len(nodeGPUs) == 0 {
			continue
		}
		total, used := countPoolGPUUsage(nodeGPUs, pool.Name)
		if total == 0 {
			continue
		}
		if used == 0 {
			continue
		}
		utilization := float64(used) * 100 / float64(total)
		if int64(utilization) > threshold {
			continue
		}

		pods, err := r.listNodeWorkerPods(ctx, k8sNodeName, nodeWorkerStore[k8sNodeName])
		if err != nil {
			continue
		}
		if len(pods) == 0 {
			continue
		}

		out = append(out, &defragCandidate{
			nodeName:         k8sNodeName,
			gpuNodeName:      gpuNode.Name,
			totalPoolGPUs:    total,
			usedPoolGPUs:     used,
			utilizationScore: utilization,
			workerPods:       pods,
		})
	}

	// Drain the emptiest nodes first.
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].utilizationScore < out[j].utilizationScore
	})
	return out, nil
}

func countPoolGPUUsage(gpus map[string]*tfv1.GPU, poolName string) (total, used int) {
	for _, g := range gpus {
		if g == nil {
			continue
		}
		if g.Labels[constants.GpuPoolKey] != poolName {
			continue
		}
		if g.Status.UsedBy != "" && g.Status.UsedBy != tfv1.UsedByTensorFusion {
			continue
		}
		if g.Status.Capacity == nil || g.Status.Available == nil {
			continue
		}
		total++
		// Count the GPU as used once either dimension is partially consumed.
		if g.Status.Available.Tflops.Cmp(g.Status.Capacity.Tflops) < 0 ||
			g.Status.Available.Vram.Cmp(g.Status.Capacity.Vram) < 0 {
			used++
		}
	}
	return
}

// listNodeWorkerPods fetches TF worker pods from allocator-owned keys.
func (r *GPUPoolCompactionReconciler) listNodeWorkerPods(
	ctx context.Context,
	nodeName string,
	workerSet map[types.NamespacedName]struct{},
) ([]*corev1.Pod, error) {
	pods := make([]*corev1.Pod, 0, len(workerSet))
	for nn := range workerSet {
		pod := &corev1.Pod{}
		if err := r.Get(ctx, nn, pod); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			return nil, fmt.Errorf("get pod %s: %w", nn.String(), err)
		}
		if pod.Spec.NodeName != nodeName {
			continue
		}
		if !pod.DeletionTimestamp.IsZero() {
			continue
		}
		if !utils.IsTensorFusionWorker(pod) {
			continue
		}
		pods = append(pods, pod)
	}
	return pods, nil
}

// ----- per-candidate processing ----------------------------------------

func (r *GPUPoolCompactionReconciler) processDefragCandidate(
	ctx context.Context,
	pool *tfv1.GPUPool,
	cand *defragCandidate,
	runStart time.Time,
	maxWorkerPerNode int,
	stats *defragRunStats,
) {
	l := log.FromContext(ctx).WithValues("pool", pool.Name, "node", cand.nodeName,
		"utilization", cand.utilizationScore, "workerCount", len(cand.workerPods))

	// Label the source node before simulation so it cannot be chosen again.
	if err := r.applyDefragDrainingLabel(ctx, cand.nodeName, pool.Name); err != nil {
		l.Error(err, "patch draining label failed; skip candidate")
		return
	}
	r.Recorder.Eventf(pool, corev1.EventTypeNormal, defragEventNodeDraining,
		"node %s labeled for defrag draining (utilization=%.1f%%, workers=%d)",
		cand.nodeName, cand.utilizationScore, len(cand.workerPods))

	if err := r.waitSchedulerObservedDefragLabel(ctx, cand); err != nil {
		l.Error(err, "scheduler cache did not observe drain label; skip candidate")
		if clearErr := r.clearDefragDrainingLabel(ctx, cand.nodeName); clearErr != nil {
			l.Error(clearErr, "clear draining label failed after scheduler cache wait")
		}
		return
	}

	refreshedPods, err := r.currentNodeWorkerPods(ctx, cand.nodeName)
	if err != nil {
		l.Error(err, "refresh worker pods failed after drain label; skip candidate")
		if clearErr := r.clearDefragDrainingLabel(ctx, cand.nodeName); clearErr != nil {
			l.Error(clearErr, "clear draining label failed after worker refresh error")
		}
		return
	}
	if len(refreshedPods) == 0 {
		if clearErr := r.clearDefragDrainingLabel(ctx, cand.nodeName); clearErr != nil {
			l.Error(clearErr, "clear draining label failed after empty worker refresh")
		}
		return
	}
	if freshPod, ok := findNewOrFreshDefragWorker(runStart, cand.workerPods, refreshedPods); ok {
		stats.FreshPodSkips++
		l.Info("skip node: contains TF worker that appeared after candidate collection",
			"pod", freshPod.Namespace+"/"+freshPod.Name, "createdAt", freshPod.CreationTimestamp.Time)
		if clearErr := r.clearDefragDrainingLabel(ctx, cand.nodeName); clearErr != nil {
			l.Error(clearErr, "clear draining label failed after fresh worker detection")
		}
		return
	}
	cand.workerPods = refreshedPods

	canRelocate, simErr := r.simulateJointPlacement(ctx, pool, cand, maxWorkerPerNode)
	if simErr != nil {
		l.Error(simErr, "joint placement simulation errored; treating as unmovable")
	}
	if !canRelocate {
		stats.UnmovableNodes++
		r.Recorder.Eventf(pool, corev1.EventTypeNormal, defragEventSkipUnschedulable,
			"node %s skipped: current TF workers cannot be jointly placed on other nodes", cand.nodeName)
		if err := r.clearDefragDrainingLabel(ctx, cand.nodeName); err != nil {
			l.Error(err, "clear draining label failed after unmovable verdict")
		}
		return
	}

	// Abort the node on the first eviction failure and leave stale cleanup
	// to remove the label later.
	allSucceeded := r.evictWorkerPods(ctx, pool, cand, stats, l)
	if allSucceeded {
		r.defragDrainingNodes.Store(cand.nodeName, time.Now())
		stats.DrainingCandidates = append(stats.DrainingCandidates, cand.nodeName)
	} else {
		r.Recorder.Eventf(pool, corev1.EventTypeWarning, defragEventAbortNode,
			"node %s: eviction aborted; label retained for stale cleanup", cand.nodeName)
	}
}

// ----- simulation: joint placement using local budget -------------------

// nodeBudget is the per-target-node virtual accounting used by simulation.
type nodeBudget struct {
	gpus        map[string]*tfv1.GPU
	workerCount int
	nodeInfo    fwk.NodeInfo
}

func (r *GPUPoolCompactionReconciler) simulateJointPlacement(
	ctx context.Context,
	pool *tfv1.GPUPool,
	cand *defragCandidate,
	maxWorkerPerNode int,
) (bool, error) {
	fitAPI, ok := any(r.Scheduler).(schedulerFitPodAPI)
	if !ok {
		return false, errors.New("scheduler vendor patch missing: FindNodesThatFitPod not available")
	}

	// Build a local budget for every target node except the source.
	nodeGpuStore := r.Allocator.GetNodeGpuStore()
	nodeWorkerStore := r.Allocator.GetNodeWorkerStoreSnapshot()
	profileName := constants.SchedulerName
	if len(cand.workerPods) > 0 && cand.workerPods[0].Spec.SchedulerName != "" {
		profileName = cand.workerPods[0].Spec.SchedulerName
	}
	schedFramework := r.Scheduler.Profiles[profileName]
	if schedFramework == nil {
		return false, fmt.Errorf("scheduler framework not found for scheduler %q", profileName)
	}

	budget, err := buildDefragNodeBudgets(
		pool.Name,
		cand.nodeName,
		nodeGpuStore,
		nodeWorkerStore,
		schedFramework.SnapshotSharedLister().NodeInfos(),
		r.snapshotDefragDrainingNodes(),
	)
	if err != nil {
		return false, err
	}
	if len(budget) == 0 {
		return false, nil
	}

	for _, srcPod := range cand.workerPods {
		if ctxDone(ctx) {
			return false, ctx.Err()
		}
		placed, err := r.placeSinglePod(ctx, fitAPI, srcPod, budget, maxWorkerPerNode)
		if err != nil {
			return false, err
		}
		if !placed {
			return false, nil
		}
	}
	return true, nil
}

// placeSinglePod asks the scheduler for feasible nodes, intersects that
// result with the local budget, and commits the chosen virtual placement.
func (r *GPUPoolCompactionReconciler) placeSinglePod(
	ctx context.Context,
	fitAPI schedulerFitPodAPI,
	pod *corev1.Pod,
	budget map[string]*nodeBudget,
	maxWorkerPerNode int,
) (bool, error) {
	podCopy := prepareDefragSimulationPod(pod)

	fwkInstance := r.Scheduler.Profiles[podCopy.Spec.SchedulerName]
	if fwkInstance == nil {
		return false, fmt.Errorf("scheduler framework not found for scheduler %q", podCopy.Spec.SchedulerName)
	}

	state := framework.NewCycleState()
	state.SetRecordPluginMetrics(false)
	podsToActivate := framework.NewPodsToActivate()
	state.Write(framework.PodsToActivateKey, podsToActivate)
	// Tell internal plugins this is a dry run.
	state.Write(fwk.StateKey(constants.SchedulerSimulationKey), &gpuallocator.SimulateSchedulingFilterDetail{
		FilterStageDetails: []filter.FilterDetail{},
	})

	feasible, _, err := fitAPI.FindNodesThatFitPod(ctx, fwkInstance, state, podCopy)
	if err != nil {
		return false, fmt.Errorf("find feasible nodes: %w", err)
	}
	if len(feasible) == 0 {
		return false, nil
	}

	req, msg, err := r.Allocator.ComposeAllocationRequest(podCopy)
	if err != nil {
		return false, fmt.Errorf("compose alloc request: %s: %w", msg, err)
	}
	podInfo, _ := framework.NewPodInfo(podCopy)

	// Defrag always prefers compact placement when breaking node ties.
	compactScorer := gpuallocator.NewStrategy(
		tfv1.PlacementModeCompactFirst,
		&config.GPUFitConfig{VramWeight: 0.5, TflopsWeight: 0.5},
		nil,
	)

	// Walk feasible ∩ budget and choose the highest-scoring target.
	type fitCandidate struct {
		node     string
		selected []*tfv1.GPU
		nodeInfo fwk.NodeInfo
	}
	var best *fitCandidate
	var bestScore int

	for _, info := range feasible {
		node := info.Node()
		if node == nil {
			continue
		}
		tgt := node.Name
		if tgt == "" {
			continue
		}
		nb, ok := budget[tgt]
		if !ok {
			continue
		}

		// Mirror CheckQuotaAndFilter: only reject nodes already over the cap.
		if maxWorkerPerNode > 0 && nb.workerCount > maxWorkerPerNode {
			continue
		}
		candidateInfo := cloneNodeInfoWithVirtualPod(nb.nodeInfo, podInfo)
		if candidateInfo == nil || !fitsVirtualNodeAllocatable(candidateInfo) {
			continue
		}
		// Re-run filters against the virtualized node snapshot so resource-based
		// plugins see the capacity already consumed by prior simulated placements.
		if status := fwkInstance.RunFilterPluginsWithNominatedPods(ctx, state.Clone(), podCopy, candidateInfo); status != nil && !status.IsSuccess() {
			continue
		}

		gpuList := make([]*tfv1.GPU, 0, len(nb.gpus))
		for _, g := range nb.gpus {
			gpuList = append(gpuList, g)
		}
		filtered, _, ferr := r.Allocator.Filter(req, gpuList, true /* isSimulateSchedule */)
		if ferr != nil || len(filtered) == 0 || uint(len(filtered)) < req.Count {
			continue
		}

		selected, serr := r.Allocator.Select(req, filtered)
		if serr != nil || len(selected) != int(req.Count) {
			continue
		}

		// Score on the budget copies so tie-breaks match the simulated state.
		score := 0
		for _, g := range selected {
			if orig, ok := nb.gpus[g.Name]; ok {
				score += compactScorer.Score(orig, false)
			}
		}
		if best == nil || score > bestScore {
			best = &fitCandidate{node: tgt, selected: selected, nodeInfo: candidateInfo}
			bestScore = score
		}
	}

	if best == nil {
		return false, nil
	}

	// Commit the placement into the local budget.
	nb := budget[best.node]
	for _, g := range best.selected {
		live := nb.gpus[g.Name]
		if live == nil || live.Status.Available == nil {
			continue
		}
		subtractGPURequest(live, req)
	}
	nb.nodeInfo = best.nodeInfo
	nb.workerCount++
	return true, nil
}

func prepareDefragSimulationPod(pod *corev1.Pod) *corev1.Pod {
	podCopy := pod.DeepCopy()

	// Clear source-node and assigned-GPU state before asking the scheduler
	// whether this worker can be placed somewhere else.
	podCopy.Spec.NodeName = ""
	podCopy.Status.NominatedNodeName = ""
	if podCopy.Annotations != nil {
		delete(podCopy.Annotations, constants.GPUDeviceIDsAnnotation)
		delete(podCopy.Annotations, constants.ContainerGPUsAnnotation)
	}
	return podCopy
}

func canFitVirtualNodeResources(nodeInfo fwk.NodeInfo, pod *corev1.Pod) bool {
	podInfo, _ := framework.NewPodInfo(pod)
	virtualNode := cloneNodeInfoWithVirtualPod(nodeInfo, podInfo)
	if virtualNode == nil {
		return false
	}
	return fitsVirtualNodeAllocatable(virtualNode)
}

func cloneNodeInfoWithVirtualPod(nodeInfo fwk.NodeInfo, podInfo fwk.PodInfo) fwk.NodeInfo {
	if nodeInfo == nil || podInfo == nil {
		return nil
	}
	virtualNode := nodeInfo.Snapshot()
	virtualNode.AddPodInfo(podInfo)
	return virtualNode
}

func fitsVirtualNodeAllocatable(nodeInfo fwk.NodeInfo) bool {
	if nodeInfo == nil {
		return false
	}
	requested := nodeInfo.GetRequested()
	allocatable := nodeInfo.GetAllocatable()
	if requested.GetMilliCPU() > allocatable.GetMilliCPU() {
		return false
	}
	if requested.GetMemory() > allocatable.GetMemory() {
		return false
	}
	if requested.GetEphemeralStorage() > allocatable.GetEphemeralStorage() {
		return false
	}
	if allocatable.GetAllowedPodNumber() > 0 && len(nodeInfo.GetPods()) > allocatable.GetAllowedPodNumber() {
		return false
	}
	for name, qty := range requested.GetScalarResources() {
		if qty > allocatable.GetScalarResources()[name] {
			return false
		}
	}
	return true
}

func buildDefragNodeBudgets(
	poolName string,
	sourceNode string,
	nodeGpuStore map[string]map[string]*tfv1.GPU,
	nodeWorkerStore map[string]map[types.NamespacedName]struct{},
	nodeInfoLister framework.NodeInfoLister,
	drainingNodes map[string]struct{},
) (map[string]*nodeBudget, error) {
	budget := make(map[string]*nodeBudget, len(nodeGpuStore))
	for nodeName, gpus := range nodeGpuStore {
		if nodeName == sourceNode {
			continue
		}
		if _, draining := drainingNodes[nodeName]; draining {
			continue
		}
		nodeInfo, err := nodeInfoLister.Get(nodeName)
		if err != nil {
			return nil, fmt.Errorf("get scheduler node info for %s: %w", nodeName, err)
		}
		if node := nodeInfo.Node(); node == nil || node.Labels[constants.NodeDeletionMark] == constants.TrueStringValue {
			continue
		}

		copied := make(map[string]*tfv1.GPU, len(gpus))
		for name, g := range gpus {
			if g == nil {
				continue
			}
			if g.Labels[constants.GpuPoolKey] != poolName {
				continue
			}
			copied[name] = g.DeepCopy()
		}
		if len(copied) == 0 {
			continue
		}
		budget[nodeName] = &nodeBudget{
			gpus:        copied,
			workerCount: len(nodeWorkerStore[nodeName]),
			nodeInfo:    nodeInfo.Snapshot(),
		}
	}
	return budget, nil
}

func (r *GPUPoolCompactionReconciler) snapshotDefragDrainingNodes() map[string]struct{} {
	out := map[string]struct{}{}
	r.defragDrainingNodes.Range(func(k, _ any) bool {
		nodeName, _ := k.(string)
		if nodeName != "" {
			out[nodeName] = struct{}{}
		}
		return true
	})
	return out
}

func (r *GPUPoolCompactionReconciler) currentNodeWorkerPods(ctx context.Context, nodeName string) ([]*corev1.Pod, error) {
	nodeWorkerStore := r.Allocator.GetNodeWorkerStoreSnapshot()
	return r.listNodeWorkerPods(ctx, nodeName, nodeWorkerStore[nodeName])
}

func findNewOrFreshDefragWorker(
	runStart time.Time,
	originalPods []*corev1.Pod,
	currentPods []*corev1.Pod,
) (*corev1.Pod, bool) {
	original := make(map[types.NamespacedName]struct{}, len(originalPods))
	for _, p := range originalPods {
		if p == nil {
			continue
		}
		original[types.NamespacedName{Namespace: p.Namespace, Name: p.Name}] = struct{}{}
	}
	for _, p := range currentPods {
		if p == nil {
			continue
		}
		if p.CreationTimestamp.After(runStart) {
			return p, true
		}
		key := types.NamespacedName{Namespace: p.Namespace, Name: p.Name}
		if _, existed := original[key]; !existed {
			return p, true
		}
	}
	return nil, false
}

func (r *GPUPoolCompactionReconciler) waitSchedulerObservedDefragLabel(
	ctx context.Context,
	cand *defragCandidate,
) error {
	profileName := constants.SchedulerName
	if len(cand.workerPods) > 0 && cand.workerPods[0].Spec.SchedulerName != "" {
		profileName = cand.workerPods[0].Spec.SchedulerName
	}
	schedFramework := r.Scheduler.Profiles[profileName]
	if schedFramework == nil {
		return fmt.Errorf("scheduler framework not found for scheduler %q", profileName)
	}
	waitCtx, cancel := context.WithTimeout(ctx, defragSchedulerCacheSyncTimeout)
	defer cancel()
	return waitForSchedulerNodeDefragLabel(
		waitCtx,
		schedFramework.SnapshotSharedLister().NodeInfos(),
		cand.nodeName,
		defragSchedulerCacheSyncPoll,
	)
}

func waitForSchedulerNodeDefragLabel(
	ctx context.Context,
	nodeInfoLister framework.NodeInfoLister,
	nodeName string,
	pollInterval time.Duration,
) error {
	if nodeInfoLister == nil {
		return fmt.Errorf("scheduler node info lister is nil")
	}
	if pollInterval <= 0 {
		pollInterval = defragSchedulerCacheSyncPoll
	}

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		nodeInfo, err := nodeInfoLister.Get(nodeName)
		if err == nil {
			if node := nodeInfo.Node(); node != nil &&
				node.Labels[constants.DefragDrainingLabel] == constants.TrueStringValue {
				return nil
			}
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("wait for scheduler cache to observe defrag drain label on node %s: %w", nodeName, ctx.Err())
		case <-ticker.C:
		}
	}
}

// subtractGPURequest mirrors GpuAllocator.Bind.
func subtractGPURequest(gpu *tfv1.GPU, req *tfv1.AllocRequest) {
	if gpu == nil || gpu.Status.Available == nil || gpu.Status.Capacity == nil {
		return
	}
	if !req.Request.ComputePercent.IsZero() {
		t := utils.ComputePercentToTflops(gpu.Status.Capacity.Tflops, req.Request)
		if t != nil {
			gpu.Status.Available.Tflops.Sub(*t)
		}
	} else {
		gpu.Status.Available.Tflops.Sub(req.Request.Tflops)
	}
	gpu.Status.Available.Vram.Sub(req.Request.Vram)
}

// ----- eviction ---------------------------------------------------------

func (r *GPUPoolCompactionReconciler) evictWorkerPods(
	ctx context.Context,
	pool *tfv1.GPUPool,
	cand *defragCandidate,
	stats *defragRunStats,
	logger interface {
		Info(string, ...any)
		Error(error, string, ...any)
	},
) bool {
	for _, pod := range cand.workerPods {
		if ctxDone(ctx) {
			stats.DeadlineExceeded = true
			return false
		}
		eviction := &policyv1.Eviction{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pod.Name,
				Namespace: pod.Namespace,
			},
		}
		err := r.KubeClient.CoreV1().Pods(pod.Namespace).EvictV1(ctx, eviction)
		if err == nil {
			stats.EvictedPods++
			r.Recorder.Eventf(pool, corev1.EventTypeNormal, defragEventPodEvicted,
				"evicted pod %s/%s on node %s for defrag", pod.Namespace, pod.Name, cand.nodeName)
			continue
		}
		if apierrors.IsNotFound(err) {
			stats.EvictedPods++
			continue
		}
		stats.EvictionFailures++
		r.Recorder.Eventf(pool, corev1.EventTypeWarning, defragEventPodEvictFailed,
			"failed to evict pod %s/%s on node %s: %v (node eviction aborted)",
			pod.Namespace, pod.Name, cand.nodeName, err)
		logger.Error(err, "eviction failed; aborting remaining pods on node",
			"pod", pod.Namespace+"/"+pod.Name, "node", cand.nodeName)
		return false
	}
	return true
}

// ----- drain watcher / stale cleanup -----------------------------------

// defragDrainWatcherTick clears labels once allocator state sees a node as
// empty. Timed-out nodes stay labeled until stale cleanup removes them.
func (r *GPUPoolCompactionReconciler) defragDrainWatcherTick(ctx context.Context, pool *tfv1.GPUPool) {
	if r.Allocator == nil {
		return
	}
	if !r.Allocator.IsReady() {
		return
	}
	nodeWorkerStore := r.Allocator.GetNodeWorkerStoreSnapshot()

	r.defragDrainingNodes.Range(func(k, v any) bool {
		nodeName, _ := k.(string)
		start, _ := v.(time.Time)
		node := &corev1.Node{}
		if err := r.Get(ctx, client.ObjectKey{Name: nodeName}, node); err != nil {
			if apierrors.IsNotFound(err) {
				r.defragDrainingNodes.Delete(nodeName)
				return true
			}
			log.FromContext(ctx).Error(err, "get draining node in watcher", "node", nodeName)
			return true
		}
		belongs, err := r.defragNodeBelongsToPool(ctx, pool.Name, node)
		if err != nil {
			log.FromContext(ctx).Error(err, "check draining node pool ownership",
				"node", nodeName, "pool", pool.Name)
			return true
		}
		if !belongs {
			return true
		}

		if len(nodeWorkerStore[nodeName]) == 0 {
			if err := r.clearDefragDrainingLabel(ctx, nodeName); err != nil {
				log.FromContext(ctx).Error(err, "clear draining label in watcher",
					"node", nodeName)
				return true
			}
			r.defragDrainingNodes.Delete(nodeName)
			r.Recorder.Eventf(pool, corev1.EventTypeNormal, defragEventNodeDrained,
				"node %s fully drained after defrag", nodeName)
			return true
		}
		if time.Since(start) > drainConfirmationTimeout {
			r.Recorder.Eventf(pool, corev1.EventTypeWarning, defragEventDrainTimeout,
				"node %s still has %d TF workers after %s; relying on stale cleanup",
				nodeName, len(nodeWorkerStore[nodeName]), drainConfirmationTimeout)
		}
		return true
	})
}

// defragStaleLabelCleanupTick removes old drain labels for the current pool.
func (r *GPUPoolCompactionReconciler) defragStaleLabelCleanupTick(ctx context.Context, pool *tfv1.GPUPool) {
	cfg := getDefragConfig(pool)
	if cfg == nil {
		return
	}
	maxDuration := parseDefragMaxDuration(cfg.MaxDuration, log.FromContext(ctx))
	staleAfter := 2 * maxDuration

	nodeList := &corev1.NodeList{}
	if err := r.List(ctx, nodeList, client.MatchingLabels{
		constants.DefragDrainingLabel: constants.TrueStringValue,
	}); err != nil {
		log.FromContext(ctx).Error(err, "list draining nodes for stale cleanup")
		return
	}
	now := time.Now()
	for i := range nodeList.Items {
		n := &nodeList.Items[i]
		belongs, err := r.defragNodeBelongsToPool(ctx, pool.Name, n)
		if err != nil {
			log.FromContext(ctx).Error(err, "check stale draining node pool ownership",
				"node", n.Name, "pool", pool.Name)
			continue
		}
		if !belongs {
			continue
		}

		raw := n.Annotations[constants.DefragDrainingSinceAnnotation]
		if raw == "" {
			_ = r.clearDefragDrainingLabel(ctx, n.Name)
			r.defragDrainingNodes.Delete(n.Name)
			continue
		}
		since, err := time.Parse(time.RFC3339, raw)
		if err != nil {
			_ = r.clearDefragDrainingLabel(ctx, n.Name)
			r.defragDrainingNodes.Delete(n.Name)
			continue
		}
		if now.Sub(since) > staleAfter {
			if err := r.clearDefragDrainingLabel(ctx, n.Name); err != nil {
				log.FromContext(ctx).Error(err, "clear stale draining label", "node", n.Name)
				continue
			}
			r.defragDrainingNodes.Delete(n.Name)
			r.Recorder.Eventf(pool, corev1.EventTypeNormal, defragEventStaleLabelCleared,
				"node %s defrag draining label cleared after %s", n.Name, now.Sub(since))
		}
	}
}

// scheduleDefragDrainWatcherBootstrap rebuilds in-memory drain state after
// the manager cache is ready.
func (r *GPUPoolCompactionReconciler) scheduleDefragDrainWatcherBootstrap(mgr manager.Manager) {
	_ = mgr.Add(manager.RunnableFunc(func(ctx context.Context) error {
		if !mgr.GetCache().WaitForCacheSync(ctx) {
			return nil
		}
		nodeList := &corev1.NodeList{}
		if err := r.List(ctx, nodeList, client.MatchingLabels{
			constants.DefragDrainingLabel: constants.TrueStringValue,
		}); err != nil {
			log.FromContext(ctx).Error(err, "bootstrap draining nodes list failed")
			return nil
		}
		if r.Allocator == nil {
			return nil
		}
		if !r.Allocator.WaitUntilReady(ctx) {
			return nil
		}
		nodeWorkerStore := r.Allocator.GetNodeWorkerStoreSnapshot()
		for i := range nodeList.Items {
			n := &nodeList.Items[i]
			if len(nodeWorkerStore[n.Name]) == 0 {
				_ = r.clearDefragDrainingLabel(ctx, n.Name)
				continue
			}
			since := n.Annotations[constants.DefragDrainingSinceAnnotation]
			start := time.Now()
			if since != "" {
				if t, err := time.Parse(time.RFC3339, since); err == nil {
					start = t
				}
			}
			r.defragDrainingNodes.Store(n.Name, start)
		}
		return nil
	}))
}

// ----- node label / annotation helpers ---------------------------------

func (r *GPUPoolCompactionReconciler) defragNodeBelongsToPool(ctx context.Context, poolName string, node *corev1.Node) (bool, error) {
	if node == nil {
		return false, nil
	}
	if node.Annotations != nil {
		if owner := node.Annotations[constants.DefragDrainingPoolAnnotation]; owner != "" {
			return owner == poolName, nil
		}
	}

	poolKey := fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, poolName)
	if node.Labels != nil && node.Labels[poolKey] == constants.TrueStringValue {
		return true, nil
	}

	gpuNode := &tfv1.GPUNode{}
	if err := r.Get(ctx, client.ObjectKey{Name: node.Name}, gpuNode); err != nil {
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return gpuNode.Labels[poolKey] == constants.TrueStringValue, nil
}

// applyDefragDrainingLabel writes the drain label and annotations.
func (r *GPUPoolCompactionReconciler) applyDefragDrainingLabel(ctx context.Context, nodeName string, poolName string) error {
	patch := map[string]any{
		"metadata": map[string]any{
			"labels": map[string]any{
				constants.DefragDrainingLabel: constants.TrueStringValue,
			},
			"annotations": map[string]any{
				constants.DefragDrainingSinceAnnotation: time.Now().Format(time.RFC3339),
				constants.DefragDrainingPoolAnnotation:  poolName,
			},
		},
	}
	raw, err := json.Marshal(patch)
	if err != nil {
		return err
	}
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}
	return r.Patch(ctx, node, client.RawPatch(types.MergePatchType, raw))
}

// clearDefragDrainingLabel removes the label and annotation.
func (r *GPUPoolCompactionReconciler) clearDefragDrainingLabel(ctx context.Context, nodeName string) error {
	patch := map[string]any{
		"metadata": map[string]any{
			"labels": map[string]any{
				constants.DefragDrainingLabel: nil,
			},
			"annotations": map[string]any{
				constants.DefragDrainingSinceAnnotation: nil,
				constants.DefragDrainingPoolAnnotation:  nil,
			},
		},
	}
	raw, err := json.Marshal(patch)
	if err != nil {
		return err
	}
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}
	if err := r.Patch(ctx, node, client.RawPatch(types.MergePatchType, raw)); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	return nil
}

// patchPoolLastDefragTime persists the schedule anchor in status.
func (r *GPUPoolCompactionReconciler) patchPoolLastDefragTime(ctx context.Context, pool *tfv1.GPUPool, runStart time.Time) error {
	live := &tfv1.GPUPool{}
	if err := r.Get(ctx, client.ObjectKey{Name: pool.Name}, live); err != nil {
		return err
	}
	patch := client.MergeFrom(live.DeepCopy())
	now := metav1.NewTime(runStart)
	live.Status.LastDefragTime = &now
	return r.Status().Patch(ctx, live, patch)
}

var _ = ctrl.Result{}
