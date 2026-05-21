/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package gang

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	gocache "github.com/patrickmn/go-cache"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	listerv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2"
	"k8s.io/kube-scheduler/framework"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

const (
	// DefaultBackoffDuration is the default backoff duration after gang scheduling failure
	DefaultBackoffDuration = 10 * time.Second

	// Event reasons
	EventReasonGangScheduled        = "GangScheduled"
	EventReasonGangTimeout          = "GangSchedulingTimeout"
	EventReasonGangWaiting          = "GangWaitingForMembers"
	EventReasonGangInsufficientPods = "GangInsufficientPods"
)

var log = ctrl.Log.WithName("gang-scheduler")

// Manager handles gang scheduling logic
type Manager struct {
	// podLister is used to list pods
	podLister listerv1.PodLister

	// eventRecorder records events for gang scheduling
	eventRecorder events.EventRecorder

	// frameworkHandle is used to access waiting pods in scheduler
	frameworkHandle framework.Handle

	// client is only used for persisting gang status to TensorFusionWorkload.
	// Gang scheduling decisions are made entirely from pod annotations.
	// Read via getClient() / written via SetClient() — both serialized on
	// clientMu so the statusFlushLoop reading concurrently with a late
	// SetClient call does not race.
	clientMu sync.RWMutex
	client   client.Client

	// pluginName is used when calling Allow/Reject on waiting pods
	pluginName string

	// podGroups stores pod group information
	// Key: PodGroupKey (namespace/<group-name>)
	podGroups map[PodGroupKey]*PodGroupInfo

	// backedOffGroups stores groups that failed recently and should be backed off
	backedOffGroups *gocache.Cache

	// statusQueue receives gang status updates to be flushed asynchronously,
	// keeping API server writes off the Permit hot path.
	statusQueue chan statusUpdate

	// pendingTerminal holds terminal status updates (TimedOut/Failed) that could
	// not be enqueued because the channel was full. The statusFlushLoop merges
	// these into the pending map on every tick so they are retried until success,
	// without ever blocking the scheduling goroutine.
	pendingTerminal   map[client.ObjectKey]statusUpdate
	pendingTerminalMu sync.Mutex

	mu sync.RWMutex
}

// statusUpdate is an item enqueued for async gang status persistence.
type statusUpdate struct {
	key    client.ObjectKey
	status *tfv1.GangSchedulingStatus
}

const (
	// statusQueueSize is the buffer size for async status updates.
	statusQueueSize = 256
	// statusFlushInterval is how often the background worker flushes pending updates.
	statusFlushInterval = 500 * time.Millisecond
)

// NewManager creates a new gang scheduling manager and starts the background
// status flush loop.
func NewManager(podLister listerv1.PodLister, eventRecorder events.EventRecorder, pluginName string) *Manager {
	m := newManager(podLister, eventRecorder, pluginName, statusQueueSize)
	go m.statusFlushLoop()
	return m
}

// newManager builds a manager *without* starting the flush loop. Used by
// tests that need to fault-inject queue state before the loop has a chance
// to drain. Production callers must use NewManager.
func newManager(
	podLister listerv1.PodLister,
	eventRecorder events.EventRecorder,
	pluginName string,
	queueSize int,
) *Manager {
	return &Manager{
		podLister:       podLister,
		eventRecorder:   eventRecorder,
		pluginName:      pluginName,
		podGroups:       make(map[PodGroupKey]*PodGroupInfo),
		backedOffGroups: gocache.New(DefaultBackoffDuration, DefaultBackoffDuration),
		statusQueue:     make(chan statusUpdate, queueSize),
		pendingTerminal: make(map[client.ObjectKey]statusUpdate),
	}
}

// statusFlushLoop drains the statusQueue, coalescing updates per workload key,
// and flushes them to the API server at a bounded rate.
func (m *Manager) statusFlushLoop() {
	ticker := time.NewTicker(statusFlushInterval)
	defer ticker.Stop()

	// pending holds the latest status per workload key (coalesces rapid updates).
	pending := make(map[client.ObjectKey]statusUpdate)

	for {
		select {
		case u, ok := <-m.statusQueue:
			if !ok {
				// Channel closed — merge overflow and flush remaining, then exit.
				m.drainTerminalOverflow(pending)
				m.flushStatusUpdates(pending)
				return
			}
			// Always keep only the latest update per workload.
			pending[u.key] = u

		case <-ticker.C:
			// Merge any terminal updates that overflowed the channel.
			m.drainTerminalOverflow(pending)
			m.flushStatusUpdates(pending)
		}
	}
}

// drainTerminalOverflow merges any terminal updates from the overflow map into pending.
func (m *Manager) drainTerminalOverflow(pending map[client.ObjectKey]statusUpdate) {
	m.pendingTerminalMu.Lock()
	defer m.pendingTerminalMu.Unlock()
	for k, u := range m.pendingTerminal {
		pending[k] = u
		delete(m.pendingTerminal, k)
	}
}

func (m *Manager) flushStatusUpdates(pending map[client.ObjectKey]statusUpdate) {
	if m.getClient() == nil || len(pending) == 0 {
		return
	}
	for key, u := range pending {
		status := u.status
		if err := m.patchWorkloadGangStatus(context.Background(), key, func(latest *tfv1.TensorFusionWorkload) bool {
			// Value-copy to avoid mutating the shared status pointer in pending map.
			updatedStatus := *status
			if latest.Status.Gang != nil && SameGangStatusState(latest.Status.Gang, &updatedStatus) {
				updatedStatus.LastTransitionTime = latest.Status.Gang.LastTransitionTime
			}
			if latest.Status.Gang != nil && SameGangStatus(latest.Status.Gang, &updatedStatus) {
				return false
			}
			latest.Status.Gang = &updatedStatus
			return true
		}); err != nil {
			// Keep the entry in pending so it is retried on the next flush tick.
			// This prevents permanent loss of TimedOut/Failed status updates when
			// the API server is temporarily unreachable or returns a conflict.
			log.Error(err, "Failed to flush gang status, will retry",
				"workload", fmt.Sprintf("%s/%s", key.Namespace, key.Name),
				"phase", status.Phase)
			continue
		}
		delete(pending, key)
	}
}

// SetFrameworkHandle sets the scheduler framework handle (called after scheduler initialization)
func (m *Manager) SetFrameworkHandle(handle framework.Handle) {
	m.frameworkHandle = handle
	// The manager is created before scheduler framework is fully initialized.
	// Populate pod lister here so PreFilter can validate gang member cardinality.
	if m.podLister == nil && handle != nil {
		m.podLister = handle.SharedInformerFactory().Core().V1().Pods().Lister()
	}
}

// SetClient sets the controller-runtime client used to read workload spec and persist gang status.
func (m *Manager) SetClient(c client.Client) {
	m.clientMu.Lock()
	m.client = c
	m.clientMu.Unlock()
}

// getClient returns the controller-runtime client under read lock. Returns
// nil if SetClient was never called (gang manager constructed without a
// client, e.g. in tests that do not exercise the status flush path).
func (m *Manager) getClient() client.Client {
	m.clientMu.RLock()
	defer m.clientMu.RUnlock()
	return m.client
}

// QuorumReachable reports whether the gang has at least RequiredMembers
// active peers visible in the pod cache. Used by PostFilter to distinguish
// a transient PreFilter rejection (peers still arriving) from a genuine
// Filter-stage failure that warrants group rejection. Permissive on
// uncertainty: returns true (assume reachable) when gang is disabled, lister
// is unavailable, or list errors out — those paths must not block a real
// Filter-stage strict reject.
func (m *Manager) QuorumReachable(pod *corev1.Pod) bool {
	config, groupKey, mode := m.resolveGangConfigWithContext(context.TODO(), pod)
	if !config.Enabled || m.podLister == nil {
		return true
	}
	active, err := m.countActivePodsInGang(pod, groupKey, mode)
	if err != nil {
		return true
	}
	return int32(active) >= config.RequiredMembers
}

// RejectGroupOnUnschedulable marks the gang cycle invalid and rejects all
// currently-waiting peers. Called from PostFilter when one member is found
// unschedulable in Strict mode: DefaultPreemption has already failed to
// rescue the pod, so the whole group cannot make progress — fail fast rather
// than letting peers waste the Permit timeout window. The reject path writes
// GangSchedulingPhaseFailed (not TimedOut, since this is not a wait expiry).
//
// Safe to call on non-gang pods (no-op) and on already-rejected groups
// (idempotent: backoff cache absorbs repeat calls, waiting peers map shrinks
// to empty).
func (m *Manager) RejectGroupOnUnschedulable(ctx context.Context, pod *corev1.Pod) {
	config, _, _ := m.resolveGangConfigWithContext(ctx, pod)
	if !config.Enabled {
		return
	}

	m.mu.RLock()
	pgInfo, exists := m.podGroups[config.GroupKey]
	m.mu.RUnlock()
	if !exists {
		// Group not yet tracked — just install the backoff so subsequent
		// PreEnqueue / PreFilter calls bounce back without spinning up state.
		m.backedOffGroups.Set(string(config.GroupKey), struct{}{}, DefaultBackoffDuration)
		return
	}

	pgInfo.mu.Lock()
	pgInfo.ScheduleCycleValid = false
	totalActive := len(pgInfo.WaitingPods) + len(pgInfo.ScheduledPods)
	waitingCount := len(pgInfo.WaitingPods)
	for podUID, waitingInfo := range pgInfo.WaitingPods {
		if m.frameworkHandle != nil {
			if wp := m.frameworkHandle.GetWaitingPod(podUID); wp != nil {
				wp.Reject(m.pluginName, "gang member unschedulable")
			}
		}
		select {
		case waitingInfo.RejectCh <- "gang member unschedulable":
		default:
		}
		delete(pgInfo.WaitingPods, podUID)
	}
	pgInfo.mu.Unlock()

	m.backedOffGroups.Set(string(config.GroupKey), struct{}{}, DefaultBackoffDuration)
	backoffUntil := time.Now().Add(DefaultBackoffDuration)
	m.syncWorkloadGangStatus(
		ctx,
		pgInfo,
		tfv1.GangSchedulingPhaseFailed,
		"GangMemberUnschedulable",
		fmt.Sprintf("Gang member %s/%s unschedulable; rejected %d waiting peer(s)", pod.Namespace, pod.Name, waitingCount),
		backoffUntil,
	)

	log.Info("Gang strict-rejected by PostFilter",
		"group", config.GroupKey,
		"trigger", pod.Name,
		"waitingRejected", waitingCount,
		"totalActiveBefore", totalActive)
}

// IsPodWaiting reports whether the pod with the given UID is currently parked
// at the Permit stage waiting for gang peers. Used by the allocator's TTL
// sweep so legitimately waiting gang pods (which may stay assumed for the
// full IndefiniteGangWaitDuration) are not mistaken for orphans.
//
// Lock hierarchy:
//   - m.mu protects podGroups (which entries exist).
//   - pgInfo.mu protects each pgInfo's WaitingPods/ScheduledPods maps.
//
// We snapshot the podGroups slice under m.mu, then drop m.mu and take
// pgInfo.mu.RLock() per group before reading WaitingPods. Holding m.mu
// while iterating WaitingPods directly would race with concurrent Permit
// / MarkScheduled / Unreserve writers that only synchronize on pgInfo.mu.
func (m *Manager) IsPodWaiting(podUID types.UID) bool {
	if m == nil {
		return false
	}
	m.mu.RLock()
	groups := make([]*PodGroupInfo, 0, len(m.podGroups))
	for _, pgInfo := range m.podGroups {
		groups = append(groups, pgInfo)
	}
	m.mu.RUnlock()

	for _, pgInfo := range groups {
		pgInfo.mu.RLock()
		_, ok := pgInfo.WaitingPods[podUID]
		pgInfo.mu.RUnlock()
		if ok {
			return true
		}
	}
	return false
}

// ParseGangConfig parses gang scheduling configuration from pod
func (m *Manager) ParseGangConfig(pod *corev1.Pod) GangSchedulingConfig {
	config, _, _ := m.resolveGangConfigWithContext(context.Background(), pod)
	return config
}

type groupKeyMode int

const (
	groupKeyModeUnknown groupKeyMode = iota
	groupKeyModeWorkloadRef
	groupKeyModeWorkloadLabel
)

func (m *Manager) resolveGangConfig(pod *corev1.Pod) (GangSchedulingConfig, PodGroupKey, groupKeyMode) {
	return m.resolveGangConfigWithContext(context.Background(), pod)
}

// resolveGangConfigWithContext resolves gang scheduling configuration for a pod.
//
// The scheduler only reads pod annotations that were stamped by the admission
// webhook. It never fetches TensorFusionWorkload CRDs for scheduling decisions.
func (m *Manager) resolveGangConfigWithContext(_ context.Context, pod *corev1.Pod) (GangSchedulingConfig, PodGroupKey, groupKeyMode) {
	config := GangSchedulingConfig{Enabled: false}
	if pod == nil {
		return config, "", groupKeyModeUnknown
	}

	if !gangEnabledFromAnnotations(pod.Annotations) {
		return config, "", groupKeyModeUnknown
	}

	desiredMembers, dOk := parseGangInt32Annotation(pod.Annotations, constants.GangDesiredMembersAnnotation)
	requiredMembers, rOk := parseGangInt32Annotation(pod.Annotations, constants.GangRequiredMembersAnnotation)
	if !dOk || !rOk {
		return config, "", groupKeyModeUnknown
	}

	groupKey, mode := resolveGroupKeyFromAnnotationsOrPod(pod)
	if groupKey == "" {
		return config, "", groupKeyModeUnknown
	}

	minMembers, _ := parseGangMinMembersFromAnnotations(pod.Annotations)
	config.Enabled = true
	config.GroupKey = groupKey
	config.MinMembers = minMembers
	config.DesiredMembers = desiredMembers
	config.RequiredMembers = requiredMembers
	config.Timeout = parseGangTimeout(pod.Annotations)
	return config, groupKey, mode
}

// parseGangInt32Annotation parses an int32 annotation value, returning (value, true)
// only when the annotation exists and holds a valid integer >= 2.
func parseGangInt32Annotation(annotations map[string]string, key string) (int32, bool) {
	if annotations == nil {
		return 0, false
	}
	s := annotations[key]
	if s == "" {
		return 0, false
	}
	v, err := strconv.ParseInt(s, 10, 32)
	if err != nil || v < 2 {
		return 0, false
	}
	return int32(v), true
}

// resolveGroupKeyFromAnnotationsOrPod returns the gang group key.
// It prefers the webhook-stamped gang-group-key annotation, falling back to
// resolveGroupKey for test compatibility.
func resolveGroupKeyFromAnnotationsOrPod(pod *corev1.Pod) (PodGroupKey, groupKeyMode) {
	if pod.Annotations != nil {
		if key := pod.Annotations[constants.GangGroupKeyAnnotation]; key != "" {
			return PodGroupKey(key), groupKeyModeWorkloadLabel
		}
	}
	key, mode, ok := resolveGroupKey(pod)
	if !ok {
		return "", groupKeyModeUnknown
	}
	return key, mode
}

func gangEnabledFromAnnotations(annotations map[string]string) bool {
	if annotations == nil {
		return false
	}
	return annotations[constants.GangEnabledAnnotation] == constants.TrueStringValue
}

func parseGangMinMembersFromAnnotations(annotations map[string]string) (int32, bool) {
	if annotations == nil {
		return 0, false
	}

	minMembersStr := annotations[constants.GangMinMembersAnnotation]
	if minMembersStr == "" {
		return 0, false
	}
	minMembers, err := strconv.ParseInt(minMembersStr, 10, 32)
	if err != nil || minMembers < 2 {
		return 0, false
	}
	return int32(minMembers), true
}

func resolveGroupKey(pod *corev1.Pod) (PodGroupKey, groupKeyMode, bool) {
	if pod == nil {
		return "", groupKeyModeUnknown, false
	}

	if hasUsableWorkloadRef(pod.Spec.WorkloadRef) {
		return NewPodGroupKeyFromWorkloadRef(
			pod.Namespace,
			pod.Spec.WorkloadRef.Name,
			pod.Spec.WorkloadRef.PodGroup,
			pod.Spec.WorkloadRef.PodGroupReplicaKey,
		), groupKeyModeWorkloadRef, true
	}

	workloadName := pod.Labels[constants.WorkloadKey]
	if workloadName == "" {
		return "", groupKeyModeUnknown, false
	}
	return NewPodGroupKey(pod.Namespace, workloadName), groupKeyModeWorkloadLabel, true
}

func hasUsableWorkloadRef(workloadRef *corev1.WorkloadReference) bool {
	return workloadRef != nil && workloadRef.Name != "" && workloadRef.PodGroup != ""
}

func parseGangTimeout(annotations map[string]string) time.Duration {
	if annotations == nil {
		return 0
	}
	timeoutStr := annotations[constants.GangTimeoutAnnotation]
	if timeoutStr != "" && timeoutStr != "0" && timeoutStr != "0s" {
		if timeout, err := time.ParseDuration(timeoutStr); err == nil {
			return timeout
		}
	}
	return 0
}

// PreEnqueue is the cheap queue-admission gate for gang pods. It returns nil
// when the pod may enter the active scheduling queue, and an error otherwise
// — the caller should translate that into fwk.Unschedulable so the framework
// keeps the pod in the unschedulable queue until a QueueingHint fires.
//
// The check is intentionally lighter than PreFilter: no pod-group state
// mutation, no lazy reset, no metric writes. Three rejection reasons:
//  1. Group is backed off (recent PostFilter strict-fail or Permit timeout).
//  2. Group has not previously reached quorum AND the live count of valid
//     gang peers is below RequiredMembers.
//
// Once the group has reached quorum once (OnceResourceSatisfied) we always
// admit, matching PreFilter's "restarted pod skips wait" policy.
func (m *Manager) PreEnqueue(ctx context.Context, pod *corev1.Pod) error {
	config, groupKey, mode := m.resolveGangConfigWithContext(ctx, pod)
	if !config.Enabled {
		return nil
	}

	if _, backed := m.backedOffGroups.Get(string(config.GroupKey)); backed {
		return fmt.Errorf("pod group %s is backed off", config.GroupKey)
	}

	m.mu.RLock()
	if pgInfo, exists := m.podGroups[config.GroupKey]; exists {
		pgInfo.mu.RLock()
		onceSatisfied := pgInfo.OnceResourceSatisfied
		pgInfo.mu.RUnlock()
		m.mu.RUnlock()
		if onceSatisfied {
			return nil
		}
	} else {
		m.mu.RUnlock()
	}

	if m.podLister == nil {
		// Without a lister we cannot count peers; defer admission to PreFilter.
		return nil
	}
	activePods, err := m.countActivePodsInGang(pod, groupKey, mode)
	if err != nil {
		return err
	}
	if activePods < int(config.RequiredMembers) {
		return fmt.Errorf("gang peers not ready: have %d, need %d", activePods, config.RequiredMembers)
	}
	return nil
}

// PreFilter checks if the pod can proceed to scheduling
// Returns error if gang requirements cannot be met
func (m *Manager) PreFilter(ctx context.Context, pod *corev1.Pod) error {
	config, groupKey, mode := m.resolveGangConfigWithContext(ctx, pod)
	if !config.Enabled {
		return nil // Not a gang pod, skip
	}

	m.reconcilePodGroupState(pod, groupKey, mode)

	// Backoff gate: the group was rejected recently (PostFilter strict-fail or
	// Permit timeout). The gocache entry's TTL is the authoritative cooldown
	// window; the ScheduleCycleValid flag mirrors it for in-flight pods so
	// they do not need to consult the cache directly.
	_, backed := m.backedOffGroups.Get(string(config.GroupKey))
	if backed {
		return fmt.Errorf("pod group %s is backed off, retry later", config.GroupKey)
	}

	m.mu.RLock()
	pgInfo, pgExists := m.podGroups[config.GroupKey]
	m.mu.RUnlock()
	if pgExists {
		pgInfo.mu.Lock()
		// Lazy reset: backoff has expired (cache miss above) ⇒ next cycle is
		// allowed to proceed. Without this, a group that was strict-failed
		// once would have ScheduleCycleValid=false stuck across cycles.
		if !pgInfo.ScheduleCycleValid {
			pgInfo.ScheduleCycleValid = true
		}
		onceSatisfied := pgInfo.OnceResourceSatisfied
		pgInfo.mu.Unlock()

		if onceSatisfied {
			// Restarted pods skip the quorum gate so they do not block forever
			// waiting for peers that may never come back.
			return nil
		}
	}

	// Check if there are enough pods in the group
	if m.podLister != nil {
		activePods, err := m.countActivePodsInGang(pod, groupKey, mode)
		if err != nil {
			return err
		}
		if activePods < int(config.RequiredMembers) {
			return fmt.Errorf("not enough pods in group %s: have %d active, need %d",
				config.GroupKey, activePods, config.RequiredMembers)
		}
	}

	return nil
}

// gangLabelSelector returns a label selector that narrows the pod listing
// to likely gang members, avoiding a full namespace scan.
func gangLabelSelector(pod *corev1.Pod, mode groupKeyMode) labels.Selector {
	switch mode {
	case groupKeyModeWorkloadLabel, groupKeyModeWorkloadRef:
		if pod.Labels != nil {
			if name := pod.Labels[constants.WorkloadKey]; name != "" {
				return labels.SelectorFromSet(labels.Set{constants.WorkloadKey: name})
			}
		}
		return labels.Everything()
	default:
		return labels.Everything()
	}
}

func (m *Manager) countActivePodsInGang(pod *corev1.Pod, targetKey PodGroupKey, targetMode groupKeyMode) (int, error) {
	activePodUIDs, err := m.activeGangPodUIDs(pod, targetKey, targetMode)
	if err != nil {
		return 0, err
	}
	return len(activePodUIDs), nil
}

// activeGangPodUIDs returns UIDs of pods that are active gang peers in the
// target group. A pod qualifies only when:
//   - it is active (not deleted/failed/succeeded),
//   - it has gang-enabled=true,
//   - it carries valid (>=2) desired/required-members annotations, and
//   - its resolved group key + mode matches the target.
//
// The full resolveGangConfigWithContext check is necessary because the label
// selector alone (workload label) would also match rollout siblings that
// belong to an older non-gang revision; those pods would inflate the count
// and let PreFilter falsely admit a gang that cannot reach quorum.
func (m *Manager) activeGangPodUIDs(pod *corev1.Pod, targetKey PodGroupKey, targetMode groupKeyMode) (map[types.UID]struct{}, error) {
	if m.podLister == nil {
		return nil, nil
	}
	pods, err := m.podLister.Pods(pod.Namespace).List(gangLabelSelector(pod, targetMode))
	if err != nil {
		return nil, fmt.Errorf("list pods for gang %s/%s: %w", pod.Namespace, targetKey, err)
	}

	matches := func(p *corev1.Pod) bool {
		if !isActivePod(p) {
			return false
		}
		cfg, key, mode := m.resolveGangConfigWithContext(context.TODO(), p)
		return cfg.Enabled && mode == targetMode && key == targetKey
	}

	activePodUIDs := make(map[types.UID]struct{}, len(pods))
	for _, p := range pods {
		if matches(p) {
			activePodUIDs[p.UID] = struct{}{}
		}
	}
	if matches(pod) {
		activePodUIDs[pod.UID] = struct{}{}
	}
	return activePodUIDs, nil
}

func isActivePod(pod *corev1.Pod) bool {
	return pod != nil &&
		pod.DeletionTimestamp.IsZero() &&
		pod.Status.Phase != corev1.PodSucceeded &&
		pod.Status.Phase != corev1.PodFailed
}

func (m *Manager) reconcilePodGroupState(pod *corev1.Pod, groupKey PodGroupKey, mode groupKeyMode) {
	if pod == nil || groupKey == "" || m.podLister == nil {
		return
	}

	activePodUIDs, err := m.activeGangPodUIDs(pod, groupKey, mode)
	if err != nil {
		log.Error(err, "Failed to reconcile pod group state",
			"pod", pod.Name,
			"namespace", pod.Namespace,
			"group", groupKey)
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	pgInfo, exists := m.podGroups[groupKey]
	if !exists {
		return
	}

	pgInfo.mu.Lock()
	defer pgInfo.mu.Unlock()

	removedWaiting := 0
	for podUID, waitingInfo := range pgInfo.WaitingPods {
		if _, ok := activePodUIDs[podUID]; ok {
			continue
		}
		if m.frameworkHandle != nil {
			if wp := m.frameworkHandle.GetWaitingPod(podUID); wp != nil {
				wp.Reject(m.pluginName, "pod no longer active")
			}
		}
		select {
		case waitingInfo.RejectCh <- "pod no longer active":
		default:
		}
		delete(pgInfo.WaitingPods, podUID)
		removedWaiting++
	}

	removedScheduled := 0
	for podUID := range pgInfo.ScheduledPods {
		if _, ok := activePodUIDs[podUID]; ok {
			continue
		}
		delete(pgInfo.ScheduledPods, podUID)
		removedScheduled++
	}

	if removedWaiting == 0 && removedScheduled == 0 {
		return
	}

	log.Info("Reconciled stale pod group members",
		"group", groupKey,
		"removedWaiting", removedWaiting,
		"removedScheduled", removedScheduled,
		"activePods", len(activePodUIDs))

	if len(pgInfo.WaitingPods) == 0 && len(pgInfo.ScheduledPods) == 0 {
		delete(m.podGroups, groupKey)
		m.backedOffGroups.Delete(string(groupKey))
	}
}

// Permit decides if a pod can proceed or must wait for gang members
// Returns PermitStatus and wait duration
//
// Key insight: When returning PermitWait, the scheduler creates a waitingPod with a timer.
// Before that timer expires, we MUST call framework.Handle.GetWaitingPod(uid).Allow(pluginName)
// to signal that the pod can proceed. Otherwise, the pod will be rejected when timeout.
func (m *Manager) Permit(
	ctx context.Context,
	pod *corev1.Pod,
	nodeName string,
	allocReq *tfv1.AllocRequest,
) (PermitStatus, time.Duration, *WaitingPodInfo) {
	config, groupKey, mode := m.resolveGangConfigWithContext(ctx, pod)
	if !config.Enabled {
		return PermitAllow, 0, nil // Not a gang pod, allow immediately
	}

	m.mu.Lock()

	// Get or create pod group
	pgInfo, exists := m.podGroups[config.GroupKey]
	if !exists {
		pgInfo = NewPodGroupInfo(config.GroupKey, config.MinMembers, config.DesiredMembers, config.RequiredMembers, config.Timeout)
		m.podGroups[config.GroupKey] = pgInfo
	}
	pgInfo.MinMembers = config.MinMembers
	pgInfo.DesiredMembers = config.DesiredMembers
	pgInfo.RequiredMembers = config.RequiredMembers
	m.rememberStatusTarget(pgInfo, pod)

	pgInfo.mu.Lock()
	m.mu.Unlock()

	// Check if already timed out
	if pgInfo.IsTimedOut() {
		m.handleTimeoutLocked(ctx, pgInfo)
		pgInfo.mu.Unlock()
		m.cleanupPodGroupIfEmpty(config.GroupKey)
		return PermitReject, 0, nil
	}
	defer pgInfo.mu.Unlock()

	// If the gang was previously satisfied (once-satisfied policy),
	// allow the pod immediately without waiting for the full quorum.
	if pgInfo.OnceResourceSatisfied {
		log.Info("Gang previously satisfied, allowing pod immediately (once-satisfied)",
			"pod", pod.Name,
			"namespace", pod.Namespace,
			"group", config.GroupKey)
		return PermitAllow, 0, nil
	}

	// Create waiting pod info
	waitingInfo := NewWaitingPodInfo(
		pod.UID,
		pod.Name,
		pod.Namespace,
		nodeName,
		allocReq.GPUNames,
		allocReq,
	)
	pgInfo.WaitingPods[pod.UID] = waitingInfo

	log.Info("Pod added to gang scheduling",
		"pod", pod.Name,
		"namespace", pod.Namespace,
		"group", config.GroupKey,
		"waiting", len(pgInfo.WaitingPods),
		"requiredMembers", config.RequiredMembers)

	if int32(len(pgInfo.WaitingPods)) >= config.RequiredMembers {
		// All members ready!
		pgInfo.OnceResourceSatisfied = true
		// For previous waiting pods: call scheduler's Allow API
		// For the current pod (the last one): return PermitAllow directly
		m.allowAllWaitingPodsViaSchedulerLocked(ctx, pgInfo, pod.UID)
		m.syncWorkloadGangStatus(
			ctx,
			pgInfo,
			tfv1.GangSchedulingPhaseScheduling,
			"GangQuorumSatisfied",
			fmt.Sprintf("Gang quorum reached: %d/%d members ready", len(pgInfo.WaitingPods), int(config.RequiredMembers)),
			time.Time{},
		)
		return PermitAllow, 0, waitingInfo
	}

	// Not enough pods yet, calculate wait time
	waitTime := pgInfo.RemainingTimeout()
	activatedPods := m.activateUnscheduledPods(ctx, pod, groupKey, mode)

	// Record event
	if m.eventRecorder != nil {
		m.eventRecorder.Eventf(pod, nil, corev1.EventTypeNormal, EventReasonGangWaiting, "Waiting",
			"Waiting for gang members: %d/%d ready", len(pgInfo.WaitingPods), int(config.RequiredMembers))
	}
	if activatedPods > 0 {
		log.Info("Activated unscheduled gang pods",
			"group", config.GroupKey,
			"pod", pod.Name,
			"activated", activatedPods)
	}
	m.syncWorkloadGangStatus(
		ctx,
		pgInfo,
		tfv1.GangSchedulingPhaseWaiting,
		"GangWaitingForMembers",
		fmt.Sprintf("Waiting for gang members: %d/%d ready", len(pgInfo.WaitingPods), int(config.RequiredMembers)),
		time.Time{},
	)

	return PermitWait, waitTime, waitingInfo
}

// WaitForGang blocks until the gang is ready or timeout
// Returns true if gang is ready, false if rejected/timeout
func (m *Manager) WaitForGang(ctx context.Context, waitingInfo *WaitingPodInfo, timeout time.Duration) (bool, string) {
	if waitingInfo == nil {
		return true, ""
	}

	var timer *time.Timer
	var timerCh <-chan time.Time

	if timeout > 0 {
		timer = time.NewTimer(timeout)
		timerCh = timer.C
		defer timer.Stop()
	}

	select {
	case <-waitingInfo.AllowCh:
		return true, ""
	case reason := <-waitingInfo.RejectCh:
		return false, reason
	case <-timerCh:
		return false, "gang scheduling timeout"
	case <-ctx.Done():
		return false, "context cancelled"
	}
}

// MarkScheduled marks a pod as successfully scheduled
func (m *Manager) MarkScheduled(pod *corev1.Pod) {
	groupKey, ok := m.groupKeyForLifecycle(pod)
	if !ok {
		return
	}

	m.mu.RLock()
	pgInfo, exists := m.podGroups[groupKey]
	m.mu.RUnlock()

	if !exists {
		return
	}

	pgInfo.mu.Lock()
	defer pgInfo.mu.Unlock()

	pgInfo.ScheduledPods[pod.UID] = struct{}{}
	delete(pgInfo.WaitingPods, pod.UID)

	log.Info("Pod marked as scheduled",
		"pod", pod.Name,
		"namespace", pod.Namespace,
		"group", groupKey,
		"scheduled", len(pgInfo.ScheduledPods))

	m.syncWorkloadGangStatus(
		context.Background(),
		pgInfo,
		tfv1.GangSchedulingPhaseScheduling,
		"GangBinding",
		fmt.Sprintf("Gang members binding: %d/%d scheduled", len(pgInfo.ScheduledPods), int(pgInfo.DesiredMembers)),
		time.Time{},
	)
}

// Unreserve handles when a pod fails after Permit
func (m *Manager) Unreserve(ctx context.Context, pod *corev1.Pod) {
	groupKey, ok := m.groupKeyForLifecycle(pod)
	if !ok {
		return
	}

	m.mu.Lock()
	pgInfo, exists := m.podGroups[groupKey]
	if !exists {
		m.mu.Unlock()
		return
	}

	pgInfo.mu.Lock()
	m.mu.Unlock()

	// Remove from waiting pods
	if waitingInfo, ok := pgInfo.WaitingPods[pod.UID]; ok {
		if m.frameworkHandle != nil {
			if wp := m.frameworkHandle.GetWaitingPod(pod.UID); wp != nil {
				wp.Reject(m.pluginName, "unreserved")
			}
		}
		// Signal rejection to the waiting goroutine
		select {
		case waitingInfo.RejectCh <- "unreserved":
		default:
		}
		delete(pgInfo.WaitingPods, pod.UID)
	}

	// Remove from scheduled pods
	delete(pgInfo.ScheduledPods, pod.UID)

	pgInfo.mu.Unlock()

	log.Info("Pod unreserved from gang",
		"pod", pod.Name,
		"namespace", pod.Namespace,
		"group", groupKey)

	// If this causes the gang to fail, reject all waiting pods
	m.checkAndRejectGangIfNeeded(ctx, groupKey)
	m.cleanupPodGroupIfEmpty(groupKey)
}

func (m *Manager) groupKeyForLifecycle(pod *corev1.Pod) (PodGroupKey, bool) {
	config, _, _ := m.resolveGangConfig(pod)
	if config.Enabled {
		return config.GroupKey, true
	}
	// Fallback for non-gang pods that still have a group key (e.g., lifecycle cleanup)
	key, _ := resolveGroupKeyFromAnnotationsOrPod(pod)
	return key, key != ""
}

// HandleTimeout handles timeout for a pod group
func (m *Manager) HandleTimeout(ctx context.Context, groupKey PodGroupKey) {
	m.mu.RLock()
	pgInfo, exists := m.podGroups[groupKey]
	m.mu.RUnlock()

	if !exists {
		return
	}

	pgInfo.mu.Lock()
	m.handleTimeoutLocked(ctx, pgInfo)
	pgInfo.mu.Unlock()
	m.cleanupPodGroupIfEmpty(groupKey)
}

// handleTimeoutLocked handles timeout while holding the lock
func (m *Manager) handleTimeoutLocked(ctx context.Context, pgInfo *PodGroupInfo) {
	waitingCount := len(pgInfo.WaitingPods)

	log.Info("Gang scheduling timeout",
		"group", pgInfo.Key,
		"waiting", waitingCount,
		"requiredMembers", pgInfo.RequiredMembers)

	// Capture a representative pod for event recording before clearing the map.
	var representativePod *WaitingPodInfo
	for _, info := range pgInfo.WaitingPods {
		representativePod = info
		break
	}

	// Reject all waiting pods
	for podUID, waitingInfo := range pgInfo.WaitingPods {
		if m.frameworkHandle != nil {
			if wp := m.frameworkHandle.GetWaitingPod(podUID); wp != nil {
				wp.Reject(m.pluginName, "gang scheduling timeout")
			}
		}
		select {
		case waitingInfo.RejectCh <- "gang scheduling timeout":
		default:
		}
		delete(pgInfo.WaitingPods, podUID)
	}

	// Set backoff for this group
	m.backedOffGroups.Set(string(pgInfo.Key), struct{}{}, DefaultBackoffDuration)
	backoffUntil := time.Now().Add(DefaultBackoffDuration)
	m.syncWorkloadGangStatus(
		ctx,
		pgInfo,
		tfv1.GangSchedulingPhaseTimedOut,
		"GangSchedulingTimeout",
		fmt.Sprintf("Pod group %s timed out", string(pgInfo.Key)),
		backoffUntil,
	)

	// Record event using a representative pod from the group.
	if m.eventRecorder != nil && representativePod != nil {
		refPod := &corev1.Pod{}
		refPod.Name = representativePod.PodName
		refPod.Namespace = representativePod.Namespace
		refPod.UID = representativePod.PodUID
		m.eventRecorder.Eventf(refPod, nil, corev1.EventTypeWarning, EventReasonGangTimeout, "Timeout",
			"Pod group %s timed out with %d/%d members ready",
			string(pgInfo.Key), waitingCount, int(pgInfo.RequiredMembers))
	}
}

// allowAllWaitingPodsViaSchedulerLocked allows waiting pods via scheduler's framework handle
// This is the correct way to unblock pods waiting in scheduler's Permit phase.
// The currentPodUID is excluded because it's the pod that triggered this (will return PermitAllow directly).
func (m *Manager) allowAllWaitingPodsViaSchedulerLocked(_ context.Context, pgInfo *PodGroupInfo, currentPodUID types.UID) {
	log.Info("Gang scheduling complete, allowing all waiting pods via scheduler",
		"group", pgInfo.Key,
		"count", len(pgInfo.WaitingPods))

	for podUID, waitingInfo := range pgInfo.WaitingPods {
		if podUID == currentPodUID {
			continue // Skip current pod, it will return PermitAllow directly
		}

		// Call scheduler's Allow API to unblock the waiting pod
		if m.frameworkHandle != nil {
			if wp := m.frameworkHandle.GetWaitingPod(podUID); wp != nil {
				wp.Allow(m.pluginName)
				log.Info("Allowed waiting pod via scheduler",
					"pod", waitingInfo.PodName,
					"namespace", waitingInfo.Namespace,
					"group", pgInfo.Key)
			} else {
				log.Info("Waiting pod not found in scheduler (may have been removed)",
					"pod", waitingInfo.PodName,
					"podUID", podUID)
			}
		}

		// Also signal internal channel for backward compatibility
		select {
		case waitingInfo.AllowCh <- struct{}{}:
		default:
		}
	}

	// Record event using a representative pod from the group.
	if m.eventRecorder != nil {
		var refPod *corev1.Pod
		for _, info := range pgInfo.WaitingPods {
			refPod = &corev1.Pod{}
			refPod.Name = info.PodName
			refPod.Namespace = info.Namespace
			refPod.UID = info.PodUID
			break
		}
		if refPod != nil {
			m.eventRecorder.Eventf(refPod, nil, corev1.EventTypeNormal, EventReasonGangScheduled, "Scheduled",
				"Pod group %s scheduled with %d members", string(pgInfo.Key), len(pgInfo.WaitingPods))
		}
	}
}

// checkAndRejectGangIfNeeded checks if gang can still be fulfilled and rejects if not
func (m *Manager) checkAndRejectGangIfNeeded(ctx context.Context, groupKey PodGroupKey) {
	m.mu.RLock()
	pgInfo, exists := m.podGroups[groupKey]
	m.mu.RUnlock()

	if !exists {
		return
	}

	pgInfo.mu.Lock()
	defer pgInfo.mu.Unlock()

	// If the gang was previously satisfied, don't reject on transient failures.
	if pgInfo.OnceResourceSatisfied {
		return
	}

	// If PostFilter has already invalidated this cycle (strict-fail), it has
	// written the Failed status, rejected all waiting peers, and set backoff.
	// Subsequent Unreserve callbacks for the same group are noise — return so
	// we do not overwrite the more-specific PostFilter reason with a generic
	// "unfulfillable" message.
	if !pgInfo.ScheduleCycleValid {
		return
	}

	// If there are still enough pods, don't reject
	totalActive := len(pgInfo.WaitingPods) + len(pgInfo.ScheduledPods)
	if int32(totalActive) >= pgInfo.RequiredMembers {
		m.syncWorkloadGangStatus(
			ctx,
			pgInfo,
			tfv1.GangSchedulingPhaseScheduling,
			"GangProgressing",
			fmt.Sprintf("Gang progressing: %d/%d members active", totalActive, int(pgInfo.RequiredMembers)),
			time.Time{},
		)
		return
	}

	// Distinguish timeout from unfulfillable so external observers can tell
	// "we waited the configured window and gave up" apart from "we lost
	// peers and can no longer reach quorum".
	timedOut := pgInfo.IsTimedOut()
	phase := tfv1.GangSchedulingPhaseFailed
	reason := "GangUnfulfillable"
	rejectMsg := "gang cannot be fulfilled"
	statusMsg := fmt.Sprintf("Gang cannot be fulfilled: %d/%d members active", totalActive, int(pgInfo.RequiredMembers))
	if timedOut {
		phase = tfv1.GangSchedulingPhaseTimedOut
		reason = "GangSchedulingTimeout"
		rejectMsg = "gang scheduling timeout"
		statusMsg = fmt.Sprintf("Pod group %s timed out with %d/%d members ready", string(pgInfo.Key), totalActive, int(pgInfo.RequiredMembers))
	}

	log.Info("Rejecting all waiting gang pods",
		"group", pgInfo.Key,
		"active", totalActive,
		"requiredMembers", pgInfo.RequiredMembers,
		"reason", reason)

	for podUID, waitingInfo := range pgInfo.WaitingPods {
		if m.frameworkHandle != nil {
			if wp := m.frameworkHandle.GetWaitingPod(podUID); wp != nil {
				wp.Reject(m.pluginName, rejectMsg)
			}
		}
		select {
		case waitingInfo.RejectCh <- rejectMsg:
		default:
		}
		delete(pgInfo.WaitingPods, podUID)
	}

	// Set backoff
	m.backedOffGroups.Set(string(pgInfo.Key), struct{}{}, DefaultBackoffDuration)
	backoffUntil := time.Now().Add(DefaultBackoffDuration)
	m.syncWorkloadGangStatus(ctx, pgInfo, phase, reason, statusMsg, backoffUntil)
}

func (m *Manager) cleanupPodGroupIfEmpty(groupKey PodGroupKey) {
	m.mu.Lock()
	defer m.mu.Unlock()

	pgInfo, exists := m.podGroups[groupKey]
	if !exists {
		return
	}

	pgInfo.mu.Lock()
	empty := len(pgInfo.WaitingPods) == 0 && len(pgInfo.ScheduledPods) == 0
	pgInfo.mu.Unlock()
	if empty {
		delete(m.podGroups, groupKey)
	}
}

// CleanupPodGroup removes a pod group from the manager
func (m *Manager) CleanupPodGroup(groupKey PodGroupKey) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if pgInfo, exists := m.podGroups[groupKey]; exists {
		pgInfo.mu.Lock()
		// Reject any remaining waiting pods
		for podUID, waitingInfo := range pgInfo.WaitingPods {
			if m.frameworkHandle != nil {
				if wp := m.frameworkHandle.GetWaitingPod(podUID); wp != nil {
					wp.Reject(m.pluginName, "pod group cleaned up")
				}
			}
			select {
			case waitingInfo.RejectCh <- "pod group cleaned up":
			default:
			}
		}
		pgInfo.mu.Unlock()
		delete(m.podGroups, groupKey)
	}
}

func (m *Manager) collectActivatablePods(
	currentPod *corev1.Pod,
	targetKey PodGroupKey,
	targetMode groupKeyMode,
) (map[string]*corev1.Pod, error) {
	podsToActivate := make(map[string]*corev1.Pod)
	if m.podLister == nil || currentPod == nil {
		return podsToActivate, nil
	}

	pods, err := m.podLister.Pods(currentPod.Namespace).List(gangLabelSelector(currentPod, targetMode))
	if err != nil {
		return nil, fmt.Errorf("list pods for activation in gang %s/%s: %w", currentPod.Namespace, targetKey, err)
	}

	for _, peerPod := range pods {
		if !isActivePod(peerPod) || peerPod.UID == currentPod.UID || peerPod.Spec.NodeName != "" {
			continue
		}
		key, mode := resolveGroupKeyFromAnnotationsOrPod(peerPod)
		if key == "" || mode != targetMode || key != targetKey {
			continue
		}
		podsToActivate[peerPod.Namespace+"/"+peerPod.Name] = peerPod
	}
	return podsToActivate, nil
}

func (m *Manager) activateUnscheduledPods(
	ctx context.Context,
	currentPod *corev1.Pod,
	targetKey PodGroupKey,
	targetMode groupKeyMode,
) int {
	if m.frameworkHandle == nil {
		return 0
	}

	podsToActivate, err := m.collectActivatablePods(currentPod, targetKey, targetMode)
	if err != nil {
		log.Error(err, "Failed to collect activatable gang pods",
			"pod", currentPod.Name,
			"group", targetKey)
		return 0
	}
	if len(podsToActivate) == 0 {
		return 0
	}

	m.frameworkHandle.Activate(klog.FromContext(ctx), podsToActivate)
	return len(podsToActivate)
}

// GetPodGroupInfo returns the pod group info for debugging/testing
func (m *Manager) GetPodGroupInfo(groupKey PodGroupKey) *PodGroupInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.podGroups[groupKey]
}

// IsBackedOff returns whether a group is currently backed off
func (m *Manager) IsBackedOff(groupKey PodGroupKey) bool {
	_, backed := m.backedOffGroups.Get(string(groupKey))
	return backed
}

// SetBackoff manually sets backoff for a group (for testing)
func (m *Manager) SetBackoff(groupKey PodGroupKey, duration time.Duration) {
	m.backedOffGroups.Set(string(groupKey), struct{}{}, duration)
}

// ClearBackoff clears backoff for a group (for testing)
func (m *Manager) ClearBackoff(groupKey PodGroupKey) {
	m.backedOffGroups.Delete(string(groupKey))
}

// FlushStatusForTest synchronously drains all pending status updates.
// Only for use in tests.
func (m *Manager) FlushStatusForTest() {
	// Drain the channel and the overflow map into a local map, then flush.
	pending := make(map[client.ObjectKey]statusUpdate)
	for {
		select {
		case u := <-m.statusQueue:
			pending[u.key] = u
		default:
			m.drainTerminalOverflow(pending)
			m.flushStatusUpdates(pending)
			return
		}
	}
}
