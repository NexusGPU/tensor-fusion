// Strategy #2 for GPUPool compaction: move TF workers off low-utilization
// nodes so Strategy #1 can later reclaim the emptied nodes.
//
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;patch
// +kubebuilder:rbac:groups=core,resources=pods/eviction,verbs=create
// +kubebuilder:rbac:groups=core,resources=nodes,verbs=get;list;patch;update
// +kubebuilder:rbac:groups=policy,resources=poddisruptionbudgets,verbs=get;list
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
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// Fallback when MaxDuration is empty or invalid.
	defaultDefragMaxDuration = 2 * time.Hour

	// Independent timeout for persisting LastDefragTime.
	defragStatusPersistTimeout = 30 * time.Second
)

const (
	defragEventStarted           = "DefragStarted"
	defragEventPodEvicted        = "DefragPodEvicted"
	defragEventPodEvictFailed    = "DefragPodEvictFailed"
	defragEventAbortNode         = "DefragAbortNode"
	defragEventSkipUnschedulable = "DefragSkipUnschedulable"
	defragEventSkipPDBBlocked    = "DefragSkipPDBBlocked"
	defragEventFinished          = "DefragFinished"
)

// Mirrored from the scheduler expander to avoid an import cycle.
type schedulerSnapshotRefreshAPI interface {
	UpdateNodeInfoSnapshot(ctx context.Context) error
}

type schedulerFitPodAPI interface {
	schedulerSnapshotRefreshAPI
	FindNodesThatFitPod(
		ctx context.Context,
		schedFramework framework.Framework,
		state fwk.CycleState,
		pod *corev1.Pod,
	) ([]fwk.NodeInfo, framework.Diagnosis, error)
}

// Snapshot of the most recent defrag run for a pool.
type defragRunStats struct {
	StartTime        time.Time `json:"startTime"`
	EndTime          time.Time `json:"endTime"`
	CandidateNodes   int       `json:"candidateNodes"`
	ProcessedNodes   int       `json:"processedNodes"`
	EvictedPods      int       `json:"evictedPods"`
	EvictionFailures int       `json:"evictionFailures"`
	UnmovableNodes   int       `json:"unmovableNodes"`
	PDBBlockedNodes  int       `json:"pdbBlockedNodes"`
	FreshPodSkips    int       `json:"freshPodSkips"`
	DeadlineExceeded bool      `json:"deadlineExceeded"`
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

// maybeRunDefragStep validates the config, checks the schedule, and processes
// at most one defrag source node when the current campaign is due.
func (r *GPUPoolCompactionReconciler) maybeRunDefragStep(ctx context.Context, pool *tfv1.GPUPool) {
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

	// Treat a campaign as due once a scheduled tick has elapsed since the anchor.
	now := time.Now()
	anchor := now.Add(-maxDuration).Add(-time.Minute)
	if pool.Status.LastDefragTime != nil {
		anchor = pool.Status.LastDefragTime.Time
	}
	campaignStart := schedule.Next(anchor)
	if campaignStart.After(now) {
		logger.V(5).Info("defrag not due yet", "next", campaignStart, "anchor", anchor)
		return
	}
	if now.Sub(campaignStart) > maxDuration {
		logger.Info("defrag campaign expired before this reconcile; advancing schedule anchor",
			"campaignStart", campaignStart, "maxDuration", maxDuration)
		persistCtx, persistCancel := context.WithTimeout(ctx, defragStatusPersistTimeout)
		defer persistCancel()
		if err := r.patchPoolLastDefragTime(persistCtx, pool, campaignStart); err != nil {
			logger.Error(err, "failed to patch pool.Status.LastDefragTime after expired defrag campaign")
		}
		return
	}

	flagAny, _ := r.defragRunning.LoadOrStore(pool.Name, &atomic.Bool{})
	flag, _ := flagAny.(*atomic.Bool)
	if !flag.CompareAndSwap(false, true) {
		logger.V(4).Info("defrag already running for this pool; skipping")
		return
	}
	defer flag.Store(false)

	runCtx, cancel := context.WithDeadline(ctx, campaignStart.Add(maxDuration))
	defer cancel()
	result := r.runDefragStep(runCtx, pool.DeepCopy(), campaignStart)
	if result.finishCampaign {
		persistCtx, persistCancel := context.WithTimeout(ctx, defragStatusPersistTimeout)
		defer persistCancel()
		if err := r.patchPoolLastDefragTime(persistCtx, pool, campaignStart); err != nil {
			logger.Error(err, "failed to patch pool.Status.LastDefragTime after defrag step")
		}
	}
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

type defragStepResult struct {
	evictedNode    bool
	finishCampaign bool
}

// runDefragStep processes one pool under the current campaign deadline and
// emits evictions for at most one source node.
func (r *GPUPoolCompactionReconciler) runDefragStep(
	ctx context.Context,
	pool *tfv1.GPUPool,
	runStart time.Time,
) defragStepResult {
	l := log.FromContext(ctx).WithValues("pool", pool.Name, "component", "defrag", "runStart", runStart)
	r.Recorder.Eventf(pool, corev1.EventTypeNormal, defragEventStarted,
		"defrag step started at %s", runStart.Format(time.RFC3339))

	stats := &defragRunStats{StartTime: runStart}
	defer func() {
		stats.EndTime = time.Now()
		defragLastRunStats.Store(pool.Name, stats)
		r.Recorder.Eventf(pool, corev1.EventTypeNormal, defragEventFinished,
			"defrag step finished: candidates=%d processed=%d evicted=%d failed=%d unmovable=%d pdbBlocked=%d freshSkip=%d deadline=%t",
			stats.CandidateNodes, stats.ProcessedNodes, stats.EvictedPods, stats.EvictionFailures,
			stats.UnmovableNodes, stats.PDBBlockedNodes, stats.FreshPodSkips, stats.DeadlineExceeded)
	}()

	blocked, err := r.hasDefragEvictedPods(ctx)
	if err != nil {
		l.Error(err, "list defrag-evicted pods failed")
		return defragStepResult{}
	}
	if blocked {
		l.Info("defrag paused: defrag-evicted pod still exists")
		return defragStepResult{}
	}

	candidates, err := r.collectDefragCandidates(ctx, pool)
	if err != nil {
		l.Error(err, "collect defrag candidates failed")
		return defragStepResult{}
	}
	stats.CandidateNodes = len(candidates)
	if len(candidates) == 0 {
		l.Info("no defrag candidates")
		return defragStepResult{finishCampaign: true}
	}

	maxWorkerPerNode := r.Allocator.MaxWorkerPerNode()
	result := runDefragCandidateLoop(ctx, candidates, stats, func(cand *defragCandidate) defragCandidateOutcome {
		return r.processDefragCandidate(ctx, pool, cand, maxWorkerPerNode, stats)
	})
	if stats.DeadlineExceeded {
		l.Info("defrag deadline exceeded during candidate loop", "remaining", len(candidates)-stats.ProcessedNodes)
	}
	return result
}

func ctxDone(ctx context.Context) bool {
	select {
	case <-ctx.Done():
		return true
	default:
		return false
	}
}

func runDefragCandidateLoop(
	ctx context.Context,
	candidates []*defragCandidate,
	stats *defragRunStats,
	process func(*defragCandidate) defragCandidateOutcome,
) defragStepResult {
	for _, cand := range candidates {
		if ctxDone(ctx) {
			stats.DeadlineExceeded = true
			return defragStepResult{finishCampaign: true}
		}
		stats.ProcessedNodes++
		switch process(cand) {
		case defragCandidateSkipped:
			continue
		case defragCandidateEvicted:
			return defragStepResult{evictedNode: true}
		case defragCandidateAborted:
			return defragStepResult{}
		}
	}
	return defragStepResult{finishCampaign: true}
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
	minPodAge := parseDefragMaxDuration(cfg.MaxDuration, log.FromContext(ctx))
	now := time.Now()

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
		if freshPod, ok := findFreshDefragWorker(now, minPodAge, pods); ok {
			log.FromContext(ctx).V(4).Info("skip defrag candidate: contains fresh TF worker",
				"node", k8sNodeName,
				"pod", freshPod.Namespace+"/"+freshPod.Name,
				"createdAt", freshPod.CreationTimestamp.Time,
				"minAge", minPodAge)
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

func (r *GPUPoolCompactionReconciler) hasDefragEvictedPods(ctx context.Context) (bool, error) {
	podList := &corev1.PodList{}
	if err := r.List(ctx, podList, client.MatchingLabels{
		constants.DefragEvictedPodLabel: constants.TrueStringValue,
	}); err != nil {
		return false, fmt.Errorf("list defrag-evicted pods: %w", err)
	}
	return len(podList.Items) > 0, nil
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

type defragCandidateOutcome int

const (
	defragCandidateSkipped defragCandidateOutcome = iota
	defragCandidateEvicted
	defragCandidateAborted
)

func (r *GPUPoolCompactionReconciler) processDefragCandidate(
	ctx context.Context,
	pool *tfv1.GPUPool,
	cand *defragCandidate,
	maxWorkerPerNode int,
	stats *defragRunStats,
) defragCandidateOutcome {
	l := log.FromContext(ctx).WithValues("pool", pool.Name, "node", cand.nodeName,
		"utilization", cand.utilizationScore, "workerCount", len(cand.workerPods))

	refreshedPods, err := r.currentNodeWorkerPods(ctx, cand.nodeName)
	if err != nil {
		l.Error(err, "refresh worker pods failed; skip candidate")
		return defragCandidateSkipped
	}
	if len(refreshedPods) == 0 {
		return defragCandidateSkipped
	}
	cfg := getDefragConfig(pool)
	minPodAge := defaultDefragMaxDuration
	if cfg != nil {
		minPodAge = parseDefragMaxDuration(cfg.MaxDuration, l)
	}
	if freshPod, ok := findFreshDefragWorker(time.Now(), minPodAge, refreshedPods); ok {
		stats.FreshPodSkips++
		l.Info("skip node: contains fresh TF worker",
			"pod", freshPod.Namespace+"/"+freshPod.Name,
			"createdAt", freshPod.CreationTimestamp.Time,
			"minAge", minPodAge)
		return defragCandidateSkipped
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
		return defragCandidateSkipped
	}

	pdbBlocked, pdbReason, err := r.checkDefragPDBPreflight(ctx, cand)
	if err != nil {
		stats.EvictionFailures++
		r.Recorder.Eventf(pool, corev1.EventTypeWarning, defragEventAbortNode,
			"node %s: eviction preflight failed: %v", cand.nodeName, err)
		return defragCandidateAborted
	}
	if pdbBlocked {
		stats.PDBBlockedNodes++
		r.Recorder.Eventf(pool, corev1.EventTypeNormal, defragEventSkipPDBBlocked,
			"node %s skipped before eviction: %s", cand.nodeName, pdbReason)
		return defragCandidateSkipped
	}

	// Abort the node on the first eviction or post-eviction label failure.
	allSucceeded := r.evictWorkerPods(ctx, pool, cand, stats, l)
	if allSucceeded {
		return defragCandidateEvicted
	} else {
		r.Recorder.Eventf(pool, corev1.EventTypeWarning, defragEventAbortNode,
			"node %s: eviction aborted", cand.nodeName)
	}
	return defragCandidateAborted
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
		return false, errors.New("scheduler vendor patch missing: FindNodesThatFitPod/UpdateNodeInfoSnapshot not available")
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
	if err := fitAPI.UpdateNodeInfoSnapshot(ctx); err != nil {
		return false, fmt.Errorf("refresh scheduler snapshot before defrag simulation: %w", err)
	}

	budget := buildDefragNodeBudgets(
		pool.Name,
		cand.nodeName,
		nodeGpuStore,
		nodeWorkerStore,
		schedFramework.SnapshotSharedLister().NodeInfos(),
	)
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
) map[string]*nodeBudget {
	budget := make(map[string]*nodeBudget, len(nodeGpuStore))
	for nodeName, gpus := range nodeGpuStore {
		if nodeName == sourceNode {
			continue
		}
		nodeInfo, err := nodeInfoLister.Get(nodeName)
		if err != nil {
			// Allocator state can briefly outlive scheduler/cache state for
			// deleted nodes. A missing target node is unusable for defrag, but
			// it must not make the whole source candidate look unmovable.
			continue
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
	return budget
}

func (r *GPUPoolCompactionReconciler) currentNodeWorkerPods(ctx context.Context, nodeName string) ([]*corev1.Pod, error) {
	nodeWorkerStore := r.Allocator.GetNodeWorkerStoreSnapshot()
	return r.listNodeWorkerPods(ctx, nodeName, nodeWorkerStore[nodeName])
}

func findFreshDefragWorker(
	now time.Time,
	minAge time.Duration,
	pods []*corev1.Pod,
) (*corev1.Pod, bool) {
	if minAge <= 0 {
		return nil, false
	}
	for _, p := range pods {
		if p == nil || p.CreationTimestamp.IsZero() {
			continue
		}
		if now.Sub(p.CreationTimestamp.Time) < minAge {
			return p, true
		}
	}
	return nil, false
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

func (r *GPUPoolCompactionReconciler) checkDefragPDBPreflight(
	ctx context.Context,
	cand *defragCandidate,
) (bool, string, error) {
	if r.KubeClient == nil {
		return false, "", errors.New("kube client is nil")
	}
	pdbsByNamespace := map[string][]policyv1.PodDisruptionBudget{}
	for _, pod := range cand.workerPods {
		if pod == nil {
			continue
		}
		if pod.Namespace == "" {
			return false, "", fmt.Errorf("pod %s has empty namespace", pod.Name)
		}
		pdbs, ok := pdbsByNamespace[pod.Namespace]
		if !ok {
			pdbList, err := r.KubeClient.PolicyV1().
				PodDisruptionBudgets(pod.Namespace).
				List(ctx, metav1.ListOptions{})
			if err != nil {
				return false, "", fmt.Errorf("list PDBs for namespace %s: %w", pod.Namespace, err)
			}
			pdbs = pdbList.Items
			pdbsByNamespace[pod.Namespace] = pdbs
		}
		for i := range pdbs {
			pdb := &pdbs[i]
			selector, err := metav1.LabelSelectorAsSelector(pdb.Spec.Selector)
			if err != nil {
				return false, "", fmt.Errorf("parse PDB selector %s/%s: %w", pdb.Namespace, pdb.Name, err)
			}
			if selector == nil {
				selector = labels.Nothing()
			}
			if !selector.Matches(labels.Set(pod.Labels)) {
				continue
			}
			if pdb.Status.DisruptionsAllowed <= 0 {
				return true, fmt.Sprintf(
					"pod %s/%s is blocked by PDB %s/%s: disruptionsAllowed=%d",
					pod.Namespace, pod.Name, pdb.Namespace, pdb.Name, pdb.Status.DisruptionsAllowed,
				), nil
			}
		}
	}
	return false, "", nil
}

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
			if markErr := r.markPodDefragEvicted(ctx, pod); markErr != nil {
				stats.EvictionFailures++
				r.Recorder.Eventf(pool, corev1.EventTypeWarning, defragEventPodEvictFailed,
					"evicted pod %s/%s on node %s but failed to mark defrag label: %v (node eviction aborted)",
					pod.Namespace, pod.Name, cand.nodeName, markErr)
				logger.Error(markErr, "failed to mark evicted pod; aborting remaining pods on node",
					"pod", pod.Namespace+"/"+pod.Name, "node", cand.nodeName)
				return false
			}
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

func (r *GPUPoolCompactionReconciler) markPodDefragEvicted(ctx context.Context, pod *corev1.Pod) error {
	if r.KubeClient == nil {
		return errors.New("kube client is nil")
	}
	if pod == nil {
		return errors.New("pod is nil")
	}
	patch := map[string]any{
		"metadata": map[string]any{
			"labels": map[string]any{
				constants.DefragEvictedPodLabel: constants.TrueStringValue,
			},
		},
	}
	raw, err := json.Marshal(patch)
	if err != nil {
		return err
	}
	_, err = r.KubeClient.CoreV1().Pods(pod.Namespace).Patch(
		ctx,
		pod.Name,
		types.MergePatchType,
		raw,
		metav1.PatchOptions{},
	)
	if apierrors.IsNotFound(err) {
		return nil
	}
	return err
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
