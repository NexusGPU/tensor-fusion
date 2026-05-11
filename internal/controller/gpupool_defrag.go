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
	defaultDefragMaxDuration = 2 * time.Hour

	// Marker TTLs are deliberately decoupled from MaxDuration so a long
	// campaign window never traps a stuck eviction marker and a short
	// window never releases one while a drain is still in flight.
	defaultDefragMarkerTTL = 30 * time.Minute

	defragStatusPersistTimeout = 30 * time.Second
	defragActiveStepRequeue    = 10 * time.Second
)

const (
	defragEventStarted           = "DefragStarted"
	defragEventPodEvicted        = "DefragPodEvicted"
	defragEventPodEvictFailed    = "DefragPodEvictFailed"
	defragEventAbortNode         = "DefragAbortNode"
	defragEventSkipUnschedulable = "DefragSkipUnschedulable"
	defragEventSkipPDBBlocked    = "DefragSkipPDBBlocked"
	defragEventSkipMissingPDB    = "DefragSkipMissingPDB"
	defragEventBlockedSourceNode = "DefragBlockedBySourceNode"
	defragEventBlockedEvictedPod = "DefragBlockedByEvictedPod"
	defragEventEvictSkip         = "DefragNodeEvictSkip"
	defragEventFinished          = "DefragFinished"
)

// schedulerFitPodAPI is satisfied by *scheduler.Scheduler only after the
// vendor patch (scripts/patch-scheduler.sh) lifts the private
// findNodesThatFitPod into a public wrapper. We assert it at runtime so an
// un-patched build still compiles in -mod=mod and only fails defrag.
type schedulerFitPodAPI interface {
	UpdateNodeInfoSnapshot(ctx context.Context) error
	FindNodesThatFitPod(
		ctx context.Context,
		schedFramework framework.Framework,
		state fwk.CycleState,
		pod *corev1.Pod,
	) ([]fwk.NodeInfo, framework.Diagnosis, error)
}

// defragCompactScorer ties broken toward already-loaded GPUs to consolidate.
var defragCompactScorer = gpuallocator.NewStrategy(
	tfv1.PlacementModeCompactFirst,
	&config.GPUFitConfig{VramWeight: 0.5, TflopsWeight: 0.5},
	nil,
)

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
	MissingPDBNodes  int       `json:"missingPdbNodes"`
	FreshPodSkips    int       `json:"freshPodSkips"`
	DeadlineExceeded bool      `json:"deadlineExceeded"`
}

// keyed by pool name -> *defragRunStats.
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

// maybeRunDefragStep processes at most one source node per cron tick. Returns
// a non-zero requeue when the next wake-up should be sooner than the normal
// compaction interval.
func (r *GPUPoolCompactionReconciler) maybeRunDefragStep(ctx context.Context, pool *tfv1.GPUPool, normalRequeue time.Duration) time.Duration {
	logger := log.FromContext(ctx).WithValues("pool", pool.Name, "component", "defrag")

	cfg := getDefragConfig(pool)
	if cfg == nil || !cfg.Enabled {
		return 0
	}
	if r.Scheduler == nil || r.KubeClient == nil {
		logger.V(4).Info("defrag skipped: scheduler or kube client not wired; set ENABLE_SCHEDULER=true and inject KubeClient")
		return 0
	}
	if r.Allocator == nil {
		return 0
	}
	if !r.Allocator.IsReady() {
		logger.V(4).Info("defrag skipped: allocator not ready")
		return 0
	}

	schedule, err := r.defragParser.Parse(cfg.Schedule)
	if err != nil {
		logger.Error(err, "invalid defrag cron schedule; defrag disabled", "schedule", cfg.Schedule)
		return 0
	}
	maxDuration := parseDefragMaxDuration(cfg.MaxDuration, logger)

	// Sweep stale markers every reconcile so an interrupted defrag
	// (controller restart, ungraceful exit) doesn't permanently exclude
	// nodes from target selection until the next cron tick.
	r.runDefragSafetySweep(ctx, pool)

	// Guards pause the pool when a same-pool evicted pod is still pending
	// reschedule or a source node is still draining. Letting multiple
	// source nodes accumulate would shrink capacity (scheduler excludes
	// source nodes as targets).
	guard := r.evaluateDefragGuards(ctx, pool)
	if guard.err != nil {
		logger.Error(guard.err, "evaluate defrag guards failed")
		return 0
	}
	if guard.blockedByEvictedPod {
		logger.V(4).Info("defrag paused: defrag-evicted pod still exists")
		r.Recorder.Eventf(pool, corev1.EventTypeNormal, defragEventBlockedEvictedPod,
			"defrag step paused: defrag-evicted pod still pending reschedule")
		return defragRequeueAfter(defragStepResult{blockedByEvictedPod: true}, normalRequeue)
	}
	if guard.blockedBySourceNode {
		logger.V(4).Info("defrag paused: defrag source node still active")
		r.Recorder.Eventf(pool, corev1.EventTypeNormal, defragEventBlockedSourceNode,
			"defrag step paused: existing defrag source node is still draining")
		return defragRequeueAfter(defragStepResult{blockedBySourceNode: true}, normalRequeue)
	}

	now := time.Now()
	anchor := now.Add(-maxDuration).Add(-time.Minute)
	if pool.Status.LastDefragTime != nil {
		anchor = pool.Status.LastDefragTime.Time
	}
	campaign := selectDefragCampaign(schedule, anchor, now, maxDuration)
	if !campaign.due {
		logger.V(5).Info("defrag not due yet", "next", campaign.start, "anchor", anchor)
		return 0
	}
	if campaign.skipMissed {
		logger.Info("defrag campaign expired before this reconcile; skipping missed schedule window",
			"advancedAnchor", campaign.start, "maxDuration", maxDuration)
		persistCtx, persistCancel := context.WithTimeout(ctx, defragStatusPersistTimeout)
		defer persistCancel()
		if err := r.patchPoolLastDefragTime(persistCtx, pool, campaign.start); err != nil {
			logger.Error(err, "failed to patch pool.Status.LastDefragTime after expired defrag campaign")
		}
		// A missed window counts as a finished cycle, so drop the
		// campaign-scoped evict-skip list before the next tick.
		if err := r.clearAllDefragEvictSkipMarkersForPool(persistCtx, pool); err != nil {
			logger.Error(err, "failed to clear defrag evict-skip markers after expired defrag campaign")
		}
		return 0
	}

	flagAny, _ := r.defragRunning.LoadOrStore(pool.Name, &atomic.Bool{})
	flag, _ := flagAny.(*atomic.Bool)
	if !flag.CompareAndSwap(false, true) {
		logger.V(4).Info("defrag already running for this pool; skipping")
		return defragActiveStepRequeue
	}
	defer flag.Store(false)

	runCtx, cancel := context.WithDeadline(ctx, campaign.start.Add(maxDuration))
	defer cancel()
	result := r.runDefragStep(runCtx, pool.DeepCopy(), campaign.start)
	if result.finishCampaign {
		persistCtx, persistCancel := context.WithTimeout(ctx, defragStatusPersistTimeout)
		defer persistCancel()
		if err := r.patchPoolLastDefragTime(persistCtx, pool, campaign.start); err != nil {
			logger.Error(err, "failed to patch pool.Status.LastDefragTime after defrag step")
		}
		// Reset the campaign-scoped evict-skip list. Stuck workers will
		// re-trigger the marker on the next campaign, but defrag is never
		// permanently blind to them.
		if err := r.clearAllDefragEvictSkipMarkersForPool(persistCtx, pool); err != nil {
			logger.Error(err, "failed to clear defrag evict-skip markers after defrag campaign")
		}
	}
	return defragRequeueAfter(result, normalRequeue)
}

type defragSchedule interface {
	Next(time.Time) time.Time
}

type defragCampaignDecision struct {
	start      time.Time
	due        bool
	skipMissed bool
}

func selectDefragCampaign(schedule defragSchedule, anchor, now time.Time, maxDuration time.Duration) defragCampaignDecision {
	campaignStart := schedule.Next(anchor)
	if campaignStart.After(now) {
		return defragCampaignDecision{start: campaignStart}
	}
	if now.Sub(campaignStart) <= maxDuration {
		return defragCampaignDecision{start: campaignStart, due: true}
	}

	latestActive, ok := latestScheduledDefragTick(schedule, now.Add(-maxDuration).Add(-time.Nanosecond), now)
	if ok {
		return defragCampaignDecision{start: latestActive, due: true}
	}
	return defragCampaignDecision{start: now, due: true, skipMissed: true}
}

func latestScheduledDefragTick(schedule defragSchedule, after, now time.Time) (time.Time, bool) {
	next := schedule.Next(after)
	if next.After(now) {
		return time.Time{}, false
	}
	latest := next
	for {
		candidate := schedule.Next(latest)
		if !candidate.After(latest) || candidate.After(now) {
			return latest, true
		}
		latest = candidate
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

// parseDefragMarkerTTL falls back to defaultDefragMarkerTTL on empty / invalid
// input so an operator cannot accidentally disable marker expiry.
func parseDefragMarkerTTL(raw, fieldName string, logger interface{ Info(string, ...any) }) time.Duration {
	if raw == "" {
		return defaultDefragMarkerTTL
	}
	d, err := time.ParseDuration(raw)
	if err != nil || d <= 0 {
		logger.Info("invalid defrag marker TTL, falling back to default",
			"field", fieldName, "input", raw, "default", defaultDefragMarkerTTL)
		return defaultDefragMarkerTTL
	}
	return d
}

// defragMinPodAge is the minimum age a TF worker must reach before defrag may
// evict it. Reuses MaxDuration: a worker younger than the campaign window is
// treated as "still settling" to avoid re-evicting fresh reschedules.
func defragMinPodAge(cfg *tfv1.NodeDefragConfig, logger interface{ Info(string, ...any) }) time.Duration {
	if cfg == nil {
		return defaultDefragMaxDuration
	}
	return parseDefragMaxDuration(cfg.MaxDuration, logger)
}

func evictedPodMarkerTTL(cfg *tfv1.NodeDefragConfig, logger interface{ Info(string, ...any) }) time.Duration {
	if cfg == nil {
		return defaultDefragMarkerTTL
	}
	return parseDefragMarkerTTL(cfg.EvictedPodMarkerTTL, "evictedPodMarkerTTL", logger)
}

func sourceNodeMarkerTTL(cfg *tfv1.NodeDefragConfig, logger interface{ Info(string, ...any) }) time.Duration {
	if cfg == nil {
		return defaultDefragMarkerTTL
	}
	return parseDefragMarkerTTL(cfg.SourceNodeMarkerTTL, "sourceNodeMarkerTTL", logger)
}

// ----- main run ---------------------------------------------------------

type defragStepResult struct {
	evictedNode         bool
	finishCampaign      bool
	blockedByEvictedPod bool
	blockedBySourceNode bool
}

func defragRequeueAfter(result defragStepResult, normalRequeue time.Duration) time.Duration {
	if normalRequeue <= 0 {
		normalRequeue = defaultCompactionDuration
	}
	if result.evictedNode || result.blockedByEvictedPod || result.blockedBySourceNode {
		if defragActiveStepRequeue < normalRequeue {
			return defragActiveStepRequeue
		}
	}
	return normalRequeue
}

// runDefragStep evicts at most one source node within the campaign deadline.
// Caller must have already run the safety sweep and confirmed guards are
// clear; this function does not repeat them.
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
			"defrag step finished: candidates=%d processed=%d evicted=%d failed=%d unmovable=%d pdbBlocked=%d missingPdb=%d freshSkip=%d deadline=%t",
			stats.CandidateNodes, stats.ProcessedNodes, stats.EvictedPods, stats.EvictionFailures,
			stats.UnmovableNodes, stats.PDBBlockedNodes, stats.MissingPDBNodes, stats.FreshPodSkips, stats.DeadlineExceeded)
	}()

	candidates, err := r.collectDefragCandidates(ctx, pool, stats)
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

type defragCandidate struct {
	nodeName         string
	gpuNodeName      string
	totalPoolGPUs    int
	usedPoolGPUs     int
	utilizationScore float64 // 0..100 for stable sort
	workerPods       []*corev1.Pod
}

func (r *GPUPoolCompactionReconciler) collectDefragCandidates(ctx context.Context, pool *tfv1.GPUPool, stats *defragRunStats) ([]*defragCandidate, error) {
	cfg := getDefragConfig(pool)
	if cfg == nil {
		return nil, nil
	}
	minPodAge := defragMinPodAge(cfg, log.FromContext(ctx))
	now := time.Now()

	allNodes := &tfv1.GPUNodeList{}
	if err := r.List(ctx, allNodes, client.MatchingLabels(map[string]string{
		fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name): constants.TrueStringValue,
	})); err != nil {
		return nil, fmt.Errorf("list gpu nodes: %w", err)
	}

	nodeGpuStore := r.Allocator.GetNodeGpuStore()
	nodeWorkerStore := r.Allocator.GetNodeWorkerStoreSnapshot()

	threshold := float64(cfg.UtilizationThresholdPercent)
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
			if !apierrors.IsNotFound(err) {
				log.FromContext(ctx).V(4).Info("skip defrag candidate: get node failed",
					"node", k8sNodeName, "pool", pool.Name, "err", err.Error())
			}
			continue
		}
		if k8sNode.Spec.Unschedulable {
			continue
		}
		// Scheduler intentionally does not honor evict-skip, so an
		// already-evicted pod may land back on the same node. That's
		// acceptable: defrag just stops re-evicting it this campaign.
		if k8sNode.Labels[constants.DefragEvictSkipNodeLabel] == constants.TrueStringValue &&
			defragEvictSkipNodeBelongsToPool(pool.Name, k8sNode) {
			log.FromContext(ctx).V(4).Info("skip defrag candidate: node on defrag evict-skip list",
				"node", k8sNodeName, "pool", pool.Name)
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
		utilization := gpuUtilizationPercent(used, total)
		if utilization > threshold {
			continue
		}

		pods, err := r.listNodeWorkerPods(ctx, k8sNodeName, nodeWorkerStore[k8sNodeName])
		if err != nil {
			log.FromContext(ctx).V(4).Info("skip defrag candidate: list worker pods failed",
				"node", k8sNodeName, "pool", pool.Name, "err", err.Error())
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

		// Defrag refuses to evict workers without a real PDB guardrail.
		missingPod, pdbErr := r.findWorkerMissingPDB(ctx, pods)
		if pdbErr != nil {
			log.FromContext(ctx).Error(pdbErr, "skip defrag candidate: list PDB failed",
				"node", k8sNodeName, "pool", pool.Name)
			continue
		}
		if missingPod != nil {
			if stats != nil {
				stats.MissingPDBNodes++
			}
			r.Recorder.Eventf(pool, corev1.EventTypeWarning, defragEventSkipMissingPDB,
				"node %s skipped from defrag: pod %s/%s has no PodDisruptionBudget covering it; please add a PDB to enable safe defrag",
				k8sNodeName, missingPod.Namespace, missingPod.Name)
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

// sweepStaleDefragEvictedPodMarkers is best-effort: a stale pod must not block
// cleanup of the rest, so per-pod failures only log.
func (r *GPUPoolCompactionReconciler) sweepStaleDefragEvictedPodMarkers(ctx context.Context, pool *tfv1.GPUPool) error {
	if pool == nil {
		return errors.New("pool is nil")
	}
	podList := &corev1.PodList{}
	if err := r.List(ctx, podList, client.MatchingLabels{
		constants.DefragEvictedPodLabel: constants.TrueStringValue,
	}); err != nil {
		return fmt.Errorf("list defrag-evicted pods: %w", err)
	}
	staleAfter := evictedPodMarkerTTL(getDefragConfig(pool), log.FromContext(ctx))
	now := time.Now()
	for i := range podList.Items {
		pod := &podList.Items[i]
		if !defragEvictedPodBelongsToPool(pool.Name, pod) {
			continue
		}
		if !isDefragEvictedPodMarkerStale(pod, now, staleAfter) {
			continue
		}
		if err := r.clearPodDefragEvictedMarker(ctx, pod); err != nil {
			log.FromContext(ctx).Error(err, "clear stale defrag-evicted pod marker",
				"pod", pod.Namespace+"/"+pod.Name, "pool", pool.Name)
		}
	}
	return nil
}

// hasActiveDefragEvictedPods is a pure read; the caller must have already
// run sweepStaleDefragEvictedPodMarkers so a stale leftover does not block
// the pool forever.
func (r *GPUPoolCompactionReconciler) hasActiveDefragEvictedPods(ctx context.Context, pool *tfv1.GPUPool) (bool, error) {
	if pool == nil {
		return false, errors.New("pool is nil")
	}
	podList := &corev1.PodList{}
	if err := r.List(ctx, podList, client.MatchingLabels{
		constants.DefragEvictedPodLabel: constants.TrueStringValue,
	}); err != nil {
		return false, fmt.Errorf("list defrag-evicted pods: %w", err)
	}
	staleAfter := evictedPodMarkerTTL(getDefragConfig(pool), log.FromContext(ctx))
	now := time.Now()
	for i := range podList.Items {
		pod := &podList.Items[i]
		if !defragEvictedPodBelongsToPool(pool.Name, pod) {
			continue
		}
		if isDefragEvictedPodMarkerStale(pod, now, staleAfter) {
			continue
		}
		return true, nil
	}
	return false, nil
}

// runDefragSafetySweep runs the three independent stale-marker cleanups.
// Each sub-sweep is best-effort: a failure must not block the others.
func (r *GPUPoolCompactionReconciler) runDefragSafetySweep(ctx context.Context, pool *tfv1.GPUPool) {
	if pool == nil {
		return
	}
	logger := log.FromContext(ctx)
	if err := r.sweepStaleDefragEvictedPodMarkers(ctx, pool); err != nil {
		logger.Error(err, "safety sweep: stale defrag-evicted pod markers", "pool", pool.Name)
	}
	if err := r.cleanupStaleDefragSourceMarkers(ctx, pool); err != nil {
		logger.Error(err, "safety sweep: stale defrag source-node markers", "pool", pool.Name)
	}
	if err := r.cleanupStaleDefragEvictSkipMarkers(ctx, pool); err != nil {
		logger.Error(err, "safety sweep: stale defrag evict-skip markers", "pool", pool.Name)
	}
}

type defragGuardDecision struct {
	blockedByEvictedPod bool
	blockedBySourceNode bool
	err                 error
}

// evaluateDefragGuards is read-only; the sweep must run first so a
// stale-but-not-yet-cleaned marker does not falsely block the pool.
func (r *GPUPoolCompactionReconciler) evaluateDefragGuards(ctx context.Context, pool *tfv1.GPUPool) defragGuardDecision {
	if pool == nil {
		return defragGuardDecision{err: errors.New("pool is nil")}
	}
	blockedByPod, err := r.hasActiveDefragEvictedPods(ctx, pool)
	if err != nil {
		return defragGuardDecision{err: fmt.Errorf("check defrag-evicted pods: %w", err)}
	}
	if blockedByPod {
		return defragGuardDecision{blockedByEvictedPod: true}
	}
	blockedByNode, err := r.hasActiveDefragSourceNodes(ctx, pool)
	if err != nil {
		return defragGuardDecision{err: fmt.Errorf("check active defrag source nodes: %w", err)}
	}
	if blockedByNode {
		return defragGuardDecision{blockedBySourceNode: true}
	}
	return defragGuardDecision{}
}

// hasActiveDefragSourceNodes reports whether any node still carries the defrag
// hasActiveDefragSourceNodes pauses new evictions while any same-pool source
// node is still draining. Sweep must run first or stale markers will falsely
// block the pool.
func (r *GPUPoolCompactionReconciler) hasActiveDefragSourceNodes(ctx context.Context, pool *tfv1.GPUPool) (bool, error) {
	if pool == nil {
		return false, errors.New("pool is nil")
	}
	nodeList := &corev1.NodeList{}
	if err := r.List(ctx, nodeList, client.MatchingLabels{
		constants.DefragSourceNodeLabel: constants.TrueStringValue,
	}); err != nil {
		return false, fmt.Errorf("list defrag source nodes: %w", err)
	}
	for i := range nodeList.Items {
		node := &nodeList.Items[i]
		if defragSourceNodeBelongsToPool(pool.Name, node) {
			return true, nil
		}
	}
	return false, nil
}

func defragEvictedPodBelongsToPool(poolName string, pod *corev1.Pod) bool {
	if pod == nil || poolName == "" {
		return false
	}
	if evictedPodMarker.belongsToPool(pod, poolName) {
		return true
	}
	// Legacy fallback for markers written before
	// DefragEvictedPodPoolAnnotation existed.
	return pod.Annotations[constants.GpuPoolKey] == poolName
}

func isDefragEvictedPodMarkerStale(pod *corev1.Pod, now time.Time, staleAfter time.Duration) bool {
	if pod == nil || staleAfter <= 0 || pod.Annotations == nil {
		return false
	}
	raw := pod.Annotations[constants.DefragEvictedPodSinceAnnotation]
	if raw == "" {
		return false
	}
	since, err := time.Parse(time.RFC3339, raw)
	if err != nil {
		return false
	}
	return now.Sub(since) > staleAfter
}

func (r *GPUPoolCompactionReconciler) cleanupStaleDefragSourceMarkers(ctx context.Context, pool *tfv1.GPUPool) error {
	if pool == nil {
		return errors.New("pool is nil")
	}
	logger := log.FromContext(ctx)
	staleAfter := sourceNodeMarkerTTL(getDefragConfig(pool), logger)
	now := time.Now()

	nodeList := &corev1.NodeList{}
	if err := r.List(ctx, nodeList, client.MatchingLabels{
		constants.DefragSourceNodeLabel: constants.TrueStringValue,
	}); err != nil {
		return fmt.Errorf("list defrag source nodes: %w", err)
	}

	// Best-effort: a per-node failure must not block cleanup of the rest,
	// otherwise one stuck node freezes source-marker cleanup pool-wide.
	for i := range nodeList.Items {
		node := &nodeList.Items[i]
		if !defragSourceNodeBelongsToPool(pool.Name, node) {
			continue
		}
		hasWorkers, err := r.hasActiveTensorFusionWorkerOnNode(ctx, node.Name)
		if err != nil {
			logger.Error(err, "skip source-marker cleanup: list active TF workers failed",
				"node", node.Name, "pool", pool.Name)
			continue
		}
		if !isDefragSourceNodeMarkerStale(node, now, staleAfter, hasWorkers) {
			continue
		}
		if err := r.clearNodeDefragSourceMarker(ctx, node.Name); err != nil {
			logger.Error(err, "clear stale defrag source marker failed",
				"node", node.Name, "pool", pool.Name)
			continue
		}
	}
	return nil
}

func (r *GPUPoolCompactionReconciler) hasActiveTensorFusionWorkerOnNode(ctx context.Context, nodeName string) (bool, error) {
	pods, err := r.currentNodeWorkerPods(ctx, nodeName)
	if err != nil {
		return false, err
	}
	return len(pods) > 0, nil
}

func defragSourceNodeBelongsToPool(poolName string, node *corev1.Node) bool {
	return sourceNodeMarker.belongsToPool(node, poolName)
}

// isDefragSourceNodeMarkerStale releases the source-node marker either when
// the node has no active TF worker (happy path) or when the marker has
// outlived staleAfter (safety net for stuck terminating workers). Callers
// must ensure no same-pool defrag-evicted pod is still in flight, otherwise
// the source node may be selected as a target while its original workers
// are still being rescheduled.
func isDefragSourceNodeMarkerStale(node *corev1.Node, now time.Time, staleAfter time.Duration, hasActiveWorkers bool) bool {
	if node == nil || node.Annotations == nil {
		return false
	}
	raw := node.Annotations[constants.DefragSourceNodeSinceAnnotation]
	if raw == "" {
		return false
	}
	if !hasActiveWorkers {
		return true
	}
	if staleAfter <= 0 {
		return false
	}
	since, err := time.Parse(time.RFC3339, raw)
	if err != nil {
		return false
	}
	return now.Sub(since) > staleAfter
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
		if !isGPUFullyAvailable(g) {
			used++
		}
	}
	return
}

// isGPUFullyAvailable is the inverse of the "used" predicate inside
// countPoolGPUUsage so source and budget sides share one definition.
func isGPUFullyAvailable(g *tfv1.GPU) bool {
	if g == nil || g.Status.Available == nil || g.Status.Capacity == nil {
		return false
	}
	return g.Status.Available.Tflops.Cmp(g.Status.Capacity.Tflops) >= 0 &&
		g.Status.Available.Vram.Cmp(g.Status.Capacity.Vram) >= 0
}

// gpuUtilizationPercent is shared by the candidate source-side view and the
// simulation budget side; keeping both on one helper is what makes the
// monotonicity gate sound.
func gpuUtilizationPercent(used, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(used) * 100 / float64(total)
}

func budgetUtilizationPercent(nb *nodeBudget) float64 {
	if nb == nil {
		return 0
	}
	return gpuUtilizationPercent(nb.usedGPUs, nb.totalGPUs)
}

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
	minPodAge := defragMinPodAge(getDefragConfig(pool), l)
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
		l.Error(simErr, "joint placement simulation errored; abort candidate")
		r.Recorder.Eventf(pool, corev1.EventTypeWarning, defragEventAbortNode,
			"node %s aborted: simulation error: %v", cand.nodeName, simErr)
		return defragCandidateAborted
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

	if r.evictWorkerPods(ctx, pool, cand, stats, l) {
		return defragCandidateEvicted
	}
	r.Recorder.Eventf(pool, corev1.EventTypeWarning, defragEventAbortNode,
		"node %s: eviction aborted", cand.nodeName)
	return defragCandidateAborted
}

// ----- simulation: joint placement using local budget -------------------

// nodeBudget is the simulation's virtual accounting per target node.
// totalGPUs / usedGPUs mirror countPoolGPUUsage so the monotonicity gate can
// require target utilization >= source. usedGPUs is mutated on commit so
// subsequent placements in the same step see the updated value.
type nodeBudget struct {
	gpus        map[string]*tfv1.GPU
	workerCount int
	nodeInfo    fwk.NodeInfo
	totalGPUs   int
	usedGPUs    int
}

func (r *GPUPoolCompactionReconciler) simulateJointPlacement(
	ctx context.Context,
	pool *tfv1.GPUPool,
	cand *defragCandidate,
	maxWorkerPerNode int,
) (bool, error) {
	fitAPI, ok := any(r.Scheduler).(schedulerFitPodAPI)
	if !ok {
		return false, errors.New("scheduler vendor patch missing: run scripts/patch-scheduler.sh")
	}
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
		r.Allocator.GetNodeGpuStore(),
		r.Allocator.GetNodeWorkerStoreSnapshot(),
		schedFramework.SnapshotSharedLister().NodeInfos(),
	)
	if len(budget) == 0 {
		return false, nil
	}

	for _, srcPod := range cand.workerPods {
		if ctxDone(ctx) {
			return false, ctx.Err()
		}
		placed, err := r.placeSinglePod(ctx, fitAPI, srcPod, budget, maxWorkerPerNode, cand.utilizationScore)
		if err != nil {
			return false, err
		}
		if !placed {
			return false, nil
		}
	}
	return true, nil
}

// placeSinglePod intersects scheduler-feasible nodes with the local budget
// and commits one virtual placement. (false, nil) is the "no fit" path that
// callers should treat as unmovable. sourceUtilization (0..100) gates
// targets to at-least-as-loaded ones; the gate reads the dynamic budget so
// prior placements within the same step are visible.
func (r *GPUPoolCompactionReconciler) placeSinglePod(
	ctx context.Context,
	fitAPI schedulerFitPodAPI,
	pod *corev1.Pod,
	budget map[string]*nodeBudget,
	maxWorkerPerNode int,
	sourceUtilization float64,
) (bool, error) {
	podCopy := cloneAsUnscheduledWorker(pod)

	fwkInstance := r.Scheduler.Profiles[podCopy.Spec.SchedulerName]
	if fwkInstance == nil {
		return false, fmt.Errorf("scheduler framework not found for scheduler %q", podCopy.Spec.SchedulerName)
	}

	state := framework.NewCycleState()
	state.SetRecordPluginMetrics(false)
	state.Write(framework.PodsToActivateKey, framework.NewPodsToActivate())
	// Mark this as a dry run so gpuresources / allocator plugins skip
	// real-state mutations and use the simulation-only code paths.
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

		// Mirror CheckQuotaAndFilter: only reject nodes already over cap.
		if maxWorkerPerNode > 0 && nb.workerCount > maxWorkerPerNode {
			continue
		}
		// Monotonicity gate: workers may only flow toward at-least-as-
		// loaded targets, evaluated against the dynamic budget.
		targetUtil := budgetUtilizationPercent(nb)
		if targetUtil < sourceUtilization {
			log.FromContext(ctx).V(5).Info("defrag reject lower-utilization target",
				"target", tgt, "targetUtil", targetUtil, "sourceUtil", sourceUtilization)
			continue
		}
		candidateInfo := cloneNodeInfoWithVirtualPod(nb.nodeInfo, podInfo)
		if candidateInfo == nil || !fitsVirtualNodeAllocatable(candidateInfo) {
			continue
		}
		// Re-run filters on the virtualized snapshot so resource plugins
		// see capacity already consumed by prior simulated placements.
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

		// Score the budget copies so tie-breaks reflect simulated state.
		score := 0
		for _, g := range selected {
			if orig, ok := nb.gpus[g.Name]; ok {
				score += defragCompactScorer.Score(orig, false)
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

	nb := budget[best.node]
	applyGPUPlacementToBudget(nb, best.selected, req)
	nb.nodeInfo = best.nodeInfo
	nb.workerCount++
	return true, nil
}

// applyGPUPlacementToBudget subtracts req from each selected GPU and bumps
// nb.usedGPUs on each "fully free -> partially used" transition, keeping
// the budget aligned with countPoolGPUUsage.
func applyGPUPlacementToBudget(nb *nodeBudget, selected []*tfv1.GPU, req *tfv1.AllocRequest) {
	if nb == nil {
		return
	}
	for _, g := range selected {
		live := nb.gpus[g.Name]
		if live == nil || live.Status.Available == nil {
			continue
		}
		wasFree := isGPUFullyAvailable(live)
		subtractGPURequest(live, req)
		if wasFree && !isGPUFullyAvailable(live) {
			nb.usedGPUs++
		}
	}
}

// cloneAsUnscheduledWorker returns a deep copy with allocator-assigned
// scheduling state cleared so the scheduler treats it as a fresh candidate.
func cloneAsUnscheduledWorker(pod *corev1.Pod) *corev1.Pod {
	podCopy := pod.DeepCopy()
	podCopy.Spec.NodeName = ""
	podCopy.Status.NominatedNodeName = ""
	if podCopy.Annotations != nil {
		delete(podCopy.Annotations, constants.GPUDeviceIDsAnnotation)
		delete(podCopy.Annotations, constants.ContainerGPUsAnnotation)
	}
	return podCopy
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
	nodeInfoLister fwk.NodeInfoLister,
) map[string]*nodeBudget {
	budget := make(map[string]*nodeBudget, len(nodeGpuStore))
	for nodeName, gpus := range nodeGpuStore {
		if nodeName == sourceNode {
			continue
		}
		nodeInfo, err := nodeInfoLister.Get(nodeName)
		if err != nil {
			// Allocator state can briefly outlive scheduler cache for
			// deleted nodes; drop only this target, not the candidate.
			continue
		}
		if node := nodeInfo.Node(); node == nil ||
			node.Labels[constants.NodeDeletionMark] == constants.TrueStringValue ||
			node.Labels[constants.DefragSourceNodeLabel] == constants.TrueStringValue {
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
		// Skip fully empty target nodes: relocating onto one swaps which
		// machine is occupied without reducing total occupancy. Strategy
		// #1 owns reclaiming empty nodes, defrag stays out of its way.
		total, used := countPoolGPUUsage(copied, poolName)
		if used == 0 {
			continue
		}
		budget[nodeName] = &nodeBudget{
			gpus:        copied,
			workerCount: len(nodeWorkerStore[nodeName]),
			nodeInfo:    nodeInfo.Snapshot(),
			totalGPUs:   total,
			usedGPUs:    used,
		}
	}
	return budget
}

func (r *GPUPoolCompactionReconciler) currentNodeWorkerPods(ctx context.Context, nodeName string) ([]*corev1.Pod, error) {
	if r.Allocator == nil {
		return nil, errors.New("allocator is nil")
	}
	return r.listNodeWorkerPods(ctx, nodeName, r.Allocator.GetNodeWorkerStoreSnapshot()[nodeName])
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

// pdbSelectorMode picks the policy for PDBs with empty / match-everything
// selectors. Both modes exist because K8s' LabelSelectorAsSelector(nil)
// defaults to "matches everything", which is unsafe in our context.
type pdbSelectorMode int

const (
	// Skip empty selectors entirely (used by the "every pod must be
	// covered by a real PDB" precheck so misconfigured PDBs are not
	// trusted as guardrails).
	pdbRequireExplicitSelector pdbSelectorMode = iota
	// Rewrite empty selectors to labels.Nothing() (used by the "is
	// anyone blocking eviction" runtime check so an empty PDB never
	// falsely blocks an unrelated pod).
	pdbTreatEmptyAsMatchNothing
)

// forEachMatchingPDB caches PDBs per namespace and calls fn for each
// (pod, matched PDB) pair. fn returning stop=true halts iteration.
func (r *GPUPoolCompactionReconciler) forEachMatchingPDB(
	ctx context.Context,
	pods []*corev1.Pod,
	mode pdbSelectorMode,
	fn func(pod *corev1.Pod, pdb *policyv1.PodDisruptionBudget) (stop bool),
) (matched map[types.NamespacedName]bool, err error) {
	if r.KubeClient == nil {
		return nil, errors.New("kube client is nil")
	}
	matched = map[types.NamespacedName]bool{}
	pdbsByNamespace := map[string][]policyv1.PodDisruptionBudget{}

	for _, pod := range pods {
		if pod == nil {
			continue
		}
		if pod.Namespace == "" {
			return nil, fmt.Errorf("pod %s has empty namespace", pod.Name)
		}
		pdbs, ok := pdbsByNamespace[pod.Namespace]
		if !ok {
			list, listErr := r.KubeClient.PolicyV1().
				PodDisruptionBudgets(pod.Namespace).
				List(ctx, metav1.ListOptions{})
			if listErr != nil {
				return nil, fmt.Errorf("list PDBs for namespace %s: %w", pod.Namespace, listErr)
			}
			pdbs = list.Items
			pdbsByNamespace[pod.Namespace] = pdbs
		}

		for i := range pdbs {
			pdb := &pdbs[i]
			selector, selErr := pdbSelectorFor(pdb, mode)
			if selErr != nil {
				return nil, selErr
			}
			if selector == nil || !selector.Matches(labels.Set(pod.Labels)) {
				continue
			}
			matched[types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}] = true
			if fn != nil && fn(pod, pdb) {
				return matched, nil
			}
		}
	}
	return matched, nil
}

func pdbSelectorFor(pdb *policyv1.PodDisruptionBudget, mode pdbSelectorMode) (labels.Selector, error) {
	sel := pdb.Spec.Selector
	empty := sel == nil || (len(sel.MatchLabels) == 0 && len(sel.MatchExpressions) == 0)
	switch mode {
	case pdbRequireExplicitSelector:
		if empty {
			return nil, nil
		}
	case pdbTreatEmptyAsMatchNothing:
		if empty {
			return labels.Nothing(), nil
		}
	}
	selector, err := metav1.LabelSelectorAsSelector(sel)
	if err != nil {
		return nil, fmt.Errorf("parse PDB selector %s/%s: %w", pdb.Namespace, pdb.Name, err)
	}
	return selector, nil
}

// findWorkerMissingPDB returns the first worker pod that no explicit-selector
// PDB covers. Empty / match-everything selectors are treated as missing.
func (r *GPUPoolCompactionReconciler) findWorkerMissingPDB(
	ctx context.Context,
	pods []*corev1.Pod,
) (*corev1.Pod, error) {
	matched, err := r.forEachMatchingPDB(ctx, pods, pdbRequireExplicitSelector, nil)
	if err != nil {
		return nil, err
	}
	for _, pod := range pods {
		if pod == nil {
			continue
		}
		if !matched[types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}] {
			return pod, nil
		}
	}
	return nil, nil
}

// checkDefragPDBPreflight returns (blocked, reason) when any worker pod is
// covered by a PDB whose DisruptionsAllowed has hit zero.
func (r *GPUPoolCompactionReconciler) checkDefragPDBPreflight(
	ctx context.Context,
	cand *defragCandidate,
) (bool, string, error) {
	var (
		blocked bool
		reason  string
	)
	_, err := r.forEachMatchingPDB(ctx, cand.workerPods, pdbTreatEmptyAsMatchNothing,
		func(pod *corev1.Pod, pdb *policyv1.PodDisruptionBudget) bool {
			if pdb.Status.DisruptionsAllowed > 0 {
				return false
			}
			blocked = true
			reason = fmt.Sprintf(
				"pod %s/%s is blocked by PDB %s/%s: disruptionsAllowed=%d",
				pod.Namespace, pod.Name, pdb.Namespace, pdb.Name, pdb.Status.DisruptionsAllowed,
			)
			return true
		})
	if err != nil {
		return false, "", err
	}
	return blocked, reason, nil
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
	if ctxDone(ctx) {
		stats.DeadlineExceeded = true
		return false
	}
	if err := r.markNodeDefragSource(ctx, pool.Name, cand.nodeName); err != nil {
		stats.EvictionFailures++
		r.Recorder.Eventf(pool, corev1.EventTypeWarning, defragEventPodEvictFailed,
			"failed to mark node %s as defrag source: %v (node eviction aborted)", cand.nodeName, err)
		logger.Error(err, "failed to mark defrag source node; aborting node eviction", "node", cand.nodeName)
		return false
	}
	// The deferred clear only fires when nothing was evicted (otherwise
	// the marker must stay to guard already-evicted pods). suppressDeferred
	// covers branches that already cleared the marker themselves.
	var anySuccess, suppressDeferred bool
	defer func() {
		if anySuccess || suppressDeferred {
			return
		}
		if clearErr := r.clearNodeDefragSourceMarker(context.Background(), cand.nodeName); clearErr != nil {
			logger.Error(clearErr, "failed to clear unused defrag source node marker", "node", cand.nodeName)
		}
	}()

	for _, pod := range cand.workerPods {
		if ctxDone(ctx) {
			stats.DeadlineExceeded = true
			return false
		}
		eviction := &policyv1.Eviction{
			ObjectMeta: metav1.ObjectMeta{Name: pod.Name, Namespace: pod.Namespace},
		}
		err := r.KubeClient.CoreV1().Pods(pod.Namespace).EvictV1(ctx, eviction)

		if err == nil {
			anySuccess = true
			// Detach from ctx so a parent deadline expiring between Evict
			// and Patch does not leave the just-evicted pod untracked.
			markCtx, markCancel := context.WithTimeout(context.Background(), defragStatusPersistTimeout)
			markErr := r.markPodDefragEvicted(markCtx, pool.Name, pod)
			markCancel()
			if markErr != nil {
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
			anySuccess = true
			stats.EvictedPods++
			continue
		}

		// EvictV1 refused by api-server (webhook, finalizer, PDB
		// contention). Mark evict-skip so this node is left alone next
		// step, and release the source marker so any already-evicted
		// pods can be rescheduled normally.
		stats.EvictionFailures++
		r.Recorder.Eventf(pool, corev1.EventTypeWarning, defragEventPodEvictFailed,
			"failed to evict pod %s/%s on node %s: %v (node eviction aborted)",
			pod.Namespace, pod.Name, cand.nodeName, err)
		logger.Error(err, "eviction failed; aborting remaining pods on node",
			"pod", pod.Namespace+"/"+pod.Name, "node", cand.nodeName)

		reason := fmt.Sprintf("evict pod %s/%s failed: %v", pod.Namespace, pod.Name, err)
		if skipErr := r.markNodeDefragEvictSkip(ctx, pool.Name, cand.nodeName, reason); skipErr != nil {
			logger.Error(skipErr, "failed to mark node defrag evict-skip", "node", cand.nodeName)
		} else {
			r.Recorder.Eventf(pool, corev1.EventTypeWarning, defragEventEvictSkip,
				"node %s added to defrag evict-skip list after eviction failure: %v", cand.nodeName, err)
		}
		if clearErr := r.clearNodeDefragSourceMarker(context.Background(), cand.nodeName); clearErr != nil {
			logger.Error(clearErr, "failed to clear source marker after evict-skip", "node", cand.nodeName)
		}
		suppressDeferred = true
		return false
	}
	return true
}

// ----- marker helpers ---------------------------------------------------

// markerSpec captures one defrag marker's label + annotation contract.
// reasonAnno is optional and only carried by the evict-skip marker.
type markerSpec struct {
	label      string
	poolAnno   string
	sinceAnno  string
	reasonAnno string
}

var (
	sourceNodeMarker = markerSpec{
		label:     constants.DefragSourceNodeLabel,
		poolAnno:  constants.DefragSourceNodePoolAnnotation,
		sinceAnno: constants.DefragSourceNodeSinceAnnotation,
	}
	evictSkipMarker = markerSpec{
		label:      constants.DefragEvictSkipNodeLabel,
		poolAnno:   constants.DefragEvictSkipNodePoolAnnotation,
		sinceAnno:  constants.DefragEvictSkipNodeSinceAnnotation,
		reasonAnno: constants.DefragEvictSkipNodeReasonAnnotation,
	}
	evictedPodMarker = markerSpec{
		label:     constants.DefragEvictedPodLabel,
		poolAnno:  constants.DefragEvictedPodPoolAnnotation,
		sinceAnno: constants.DefragEvictedPodSinceAnnotation,
	}
)

func (m markerSpec) belongsToPool(obj metav1.Object, poolName string) bool {
	if obj == nil || poolName == "" {
		return false
	}
	annos := obj.GetAnnotations()
	if annos == nil {
		return false
	}
	return annos[m.poolAnno] == poolName
}

func (m markerSpec) markPatch(poolName, reason string) ([]byte, error) {
	annotations := map[string]any{
		m.poolAnno:  poolName,
		m.sinceAnno: time.Now().Format(time.RFC3339),
	}
	if m.reasonAnno != "" && reason != "" {
		const maxReasonLen = 256
		if len(reason) > maxReasonLen {
			reason = reason[:maxReasonLen]
		}
		annotations[m.reasonAnno] = reason
	}
	return metadataMergePatch(
		map[string]any{m.label: constants.TrueStringValue},
		annotations,
	)
}

func (m markerSpec) clearPatch() ([]byte, error) {
	annotations := map[string]any{
		m.poolAnno:  nil,
		m.sinceAnno: nil,
	}
	if m.reasonAnno != "" {
		annotations[m.reasonAnno] = nil
	}
	return metadataMergePatch(
		map[string]any{m.label: nil},
		annotations,
	)
}

func (r *GPUPoolCompactionReconciler) markNode(ctx context.Context, m markerSpec, nodeName, poolName, reason string) error {
	if r.Client == nil {
		return errors.New("controller client is nil")
	}
	if nodeName == "" {
		return errors.New("node name is empty")
	}
	raw, err := m.markPatch(poolName, reason)
	if err != nil {
		return err
	}
	return r.Patch(ctx, &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}},
		client.RawPatch(types.MergePatchType, raw))
}

func (r *GPUPoolCompactionReconciler) clearNodeMarker(ctx context.Context, m markerSpec, nodeName string) error {
	if r.Client == nil || nodeName == "" {
		return nil
	}
	raw, err := m.clearPatch()
	if err != nil {
		return err
	}
	err = r.Patch(ctx, &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}},
		client.RawPatch(types.MergePatchType, raw))
	if apierrors.IsNotFound(err) {
		return nil
	}
	return err
}

func (r *GPUPoolCompactionReconciler) markNodeDefragSource(ctx context.Context, poolName, nodeName string) error {
	return r.markNode(ctx, sourceNodeMarker, nodeName, poolName, "")
}

func (r *GPUPoolCompactionReconciler) clearNodeDefragSourceMarker(ctx context.Context, nodeName string) error {
	return r.clearNodeMarker(ctx, sourceNodeMarker, nodeName)
}

func (r *GPUPoolCompactionReconciler) markNodeDefragEvictSkip(ctx context.Context, poolName, nodeName, reason string) error {
	return r.markNode(ctx, evictSkipMarker, nodeName, poolName, reason)
}

func (r *GPUPoolCompactionReconciler) clearNodeDefragEvictSkipMarker(ctx context.Context, nodeName string) error {
	return r.clearNodeMarker(ctx, evictSkipMarker, nodeName)
}

func defragEvictSkipNodeBelongsToPool(poolName string, node *corev1.Node) bool {
	return evictSkipMarker.belongsToPool(node, poolName)
}

// forEachPoolEvictSkipNode lists same-pool evict-skip nodes and invokes fn.
// When fn returns an error, iteration stops and the error is propagated.
func (r *GPUPoolCompactionReconciler) forEachPoolEvictSkipNode(
	ctx context.Context, pool *tfv1.GPUPool, fn func(node *corev1.Node) error,
) error {
	if pool == nil {
		return errors.New("pool is nil")
	}
	nodeList := &corev1.NodeList{}
	if err := r.List(ctx, nodeList, client.MatchingLabels{
		constants.DefragEvictSkipNodeLabel: constants.TrueStringValue,
	}); err != nil {
		return fmt.Errorf("list defrag evict-skip nodes: %w", err)
	}
	for i := range nodeList.Items {
		node := &nodeList.Items[i]
		if !defragEvictSkipNodeBelongsToPool(pool.Name, node) {
			continue
		}
		if err := fn(node); err != nil {
			return err
		}
	}
	return nil
}

// cleanupStaleDefragEvictSkipMarkers releases evict-skip markers whose nodes
// have already drained, letting the next step re-evaluate them. Per-node
// failures only log so a single stuck node never blocks the rest.
func (r *GPUPoolCompactionReconciler) cleanupStaleDefragEvictSkipMarkers(ctx context.Context, pool *tfv1.GPUPool) error {
	logger := log.FromContext(ctx)
	return r.forEachPoolEvictSkipNode(ctx, pool, func(node *corev1.Node) error {
		hasWorkers, err := r.hasActiveTensorFusionWorkerOnNode(ctx, node.Name)
		if err != nil {
			logger.Error(err, "skip evict-skip cleanup: list active TF workers failed",
				"node", node.Name, "pool", pool.Name)
			return nil
		}
		if hasWorkers {
			return nil
		}
		if err := r.clearNodeDefragEvictSkipMarker(ctx, node.Name); err != nil {
			logger.Error(err, "clear evict-skip marker failed",
				"node", node.Name, "pool", pool.Name)
		}
		return nil
	})
}

// clearAllDefragEvictSkipMarkersForPool drops every same-pool marker on
// campaign end, even when workers are still stuck, so the next cron tick
// gives each node a fresh chance. Best-effort per node.
func (r *GPUPoolCompactionReconciler) clearAllDefragEvictSkipMarkersForPool(ctx context.Context, pool *tfv1.GPUPool) error {
	logger := log.FromContext(ctx)
	return r.forEachPoolEvictSkipNode(ctx, pool, func(node *corev1.Node) error {
		if err := r.clearNodeDefragEvictSkipMarker(ctx, node.Name); err != nil {
			logger.Error(err, "clear evict-skip marker failed",
				"node", node.Name, "pool", pool.Name)
		}
		return nil
	})
}

func (r *GPUPoolCompactionReconciler) markPodDefragEvicted(ctx context.Context, poolName string, pod *corev1.Pod) error {
	if r.KubeClient == nil {
		return errors.New("kube client is nil")
	}
	if pod == nil {
		return errors.New("pod is nil")
	}
	raw, err := evictedPodMarker.markPatch(poolName, "")
	if err != nil {
		return err
	}
	_, err = r.KubeClient.CoreV1().Pods(pod.Namespace).Patch(
		ctx, pod.Name, types.MergePatchType, raw, metav1.PatchOptions{},
	)
	if apierrors.IsNotFound(err) {
		return nil
	}
	return err
}

func (r *GPUPoolCompactionReconciler) clearPodDefragEvictedMarker(ctx context.Context, pod *corev1.Pod) error {
	if pod == nil {
		return nil
	}
	raw, err := evictedPodMarker.clearPatch()
	if err != nil {
		return err
	}
	target := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: pod.Namespace, Name: pod.Name}}
	if err := r.Patch(ctx, target, client.RawPatch(types.MergePatchType, raw)); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	return nil
}

func metadataMergePatch(labels, annotations map[string]any) ([]byte, error) {
	metadata := map[string]any{}
	if len(labels) > 0 {
		metadata["labels"] = labels
	}
	if len(annotations) > 0 {
		metadata["annotations"] = annotations
	}
	return json.Marshal(map[string]any{"metadata": metadata})
}

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
