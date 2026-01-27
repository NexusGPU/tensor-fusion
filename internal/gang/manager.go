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
	"k8s.io/client-go/tools/record"
	"k8s.io/kube-scheduler/framework"
	ctrl "sigs.k8s.io/controller-runtime"

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
	eventRecorder record.EventRecorder

	// frameworkHandle is used to access waiting pods in scheduler
	frameworkHandle framework.Handle

	// pluginName is used when calling Allow/Reject on waiting pods
	pluginName string

	// podGroups stores pod group information
	// Key: PodGroupKey (namespace/workload-name)
	podGroups map[PodGroupKey]*PodGroupInfo

	// backedOffGroups stores groups that failed recently and should be backed off
	backedOffGroups *gocache.Cache

	mu sync.RWMutex
}

// NewManager creates a new gang scheduling manager
func NewManager(podLister listerv1.PodLister, eventRecorder record.EventRecorder, pluginName string) *Manager {
	return &Manager{
		podLister:       podLister,
		eventRecorder:   eventRecorder,
		pluginName:      pluginName,
		podGroups:       make(map[PodGroupKey]*PodGroupInfo),
		backedOffGroups: gocache.New(DefaultBackoffDuration, DefaultBackoffDuration),
	}
}

// SetFrameworkHandle sets the scheduler framework handle (called after scheduler initialization)
func (m *Manager) SetFrameworkHandle(handle framework.Handle) {
	m.frameworkHandle = handle
}

// ParseGangConfig parses gang scheduling configuration from pod
func (m *Manager) ParseGangConfig(pod *corev1.Pod) GangSchedulingConfig {
	config := GangSchedulingConfig{
		Enabled: false,
	}

	if pod.Labels == nil || pod.Annotations == nil {
		return config
	}

	// Get workload name from label
	workloadName := pod.Labels[constants.WorkloadKey]
	if workloadName == "" {
		return config
	}

	// Check for gang-min-members annotation
	minMembersStr := pod.Annotations[constants.GangMinMembersAnnotation]
	if minMembersStr == "" {
		return config
	}

	minMembers, err := strconv.ParseInt(minMembersStr, 10, 32)
	if err != nil || minMembers <= 0 {
		return config
	}

	config.Enabled = true
	config.MinMembers = int32(minMembers)
	config.GroupKey = NewPodGroupKey(pod.Namespace, workloadName)

	// Parse timeout (optional)
	timeoutStr := pod.Annotations[constants.GangTimeoutAnnotation]
	if timeoutStr != "" && timeoutStr != "0" && timeoutStr != "0s" {
		if timeout, err := time.ParseDuration(timeoutStr); err == nil {
			config.Timeout = timeout
		}
	}
	// Timeout = 0 means wait indefinitely

	return config
}

// PreFilter checks if the pod can proceed to scheduling
// Returns error if gang requirements cannot be met
func (m *Manager) PreFilter(ctx context.Context, pod *corev1.Pod) error {
	config := m.ParseGangConfig(pod)
	if !config.Enabled {
		return nil // Not a gang pod, skip
	}

	// Check if this group is backed off
	if _, backed := m.backedOffGroups.Get(string(config.GroupKey)); backed {
		return fmt.Errorf("pod group %s is backed off, retry later", config.GroupKey)
	}

	// Check if there are enough pods in the group
	if m.podLister != nil {
		workloadName := pod.Labels[constants.WorkloadKey]
		pods, err := m.podLister.Pods(pod.Namespace).List(
			labels.SelectorFromSet(labels.Set{constants.WorkloadKey: workloadName}),
		)
		if err != nil {
			return fmt.Errorf("list pods for gang %s: %w", config.GroupKey, err)
		}

		// Count pods that are not completed/failed
		activePods := 0
		for _, p := range pods {
			if p.Status.Phase != corev1.PodSucceeded && p.Status.Phase != corev1.PodFailed {
				activePods++
			}
		}

		if activePods < int(config.MinMembers) {
			return fmt.Errorf("not enough pods in group %s: have %d active, need %d",
				config.GroupKey, activePods, config.MinMembers)
		}
	}

	return nil
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
	config := m.ParseGangConfig(pod)
	if !config.Enabled {
		return PermitAllow, 0, nil // Not a gang pod, allow immediately
	}

	m.mu.Lock()

	// Get or create pod group
	pgInfo, exists := m.podGroups[config.GroupKey]
	if !exists {
		pgInfo = NewPodGroupInfo(config.GroupKey, config.MinMembers, config.Timeout)
		m.podGroups[config.GroupKey] = pgInfo
	}

	pgInfo.mu.Lock()
	defer pgInfo.mu.Unlock()
	m.mu.Unlock()

	// Check if already timed out
	if pgInfo.IsTimedOut() {
		m.handleTimeoutLocked(ctx, pgInfo)
		return PermitReject, 0, nil
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
		"required", config.MinMembers)

	// Check if we have enough waiting pods
	if int32(len(pgInfo.WaitingPods)) >= config.MinMembers {
		// All members ready!
		// For previous waiting pods: call scheduler's Allow API
		// For the current pod (the last one): return PermitAllow directly
		m.allowAllWaitingPodsViaSchedulerLocked(ctx, pgInfo, pod.UID)
		return PermitAllow, 0, waitingInfo
	}

	// Not enough pods yet, calculate wait time
	waitTime := pgInfo.RemainingTimeout()

	// Record event
	if m.eventRecorder != nil {
		m.eventRecorder.Eventf(pod, corev1.EventTypeNormal, EventReasonGangWaiting,
			"Waiting for gang members: %d/%d ready", len(pgInfo.WaitingPods), config.MinMembers)
	}

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
	config := m.ParseGangConfig(pod)
	if !config.Enabled {
		return
	}

	m.mu.RLock()
	pgInfo, exists := m.podGroups[config.GroupKey]
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
		"group", config.GroupKey,
		"scheduled", len(pgInfo.ScheduledPods))
}

// Unreserve handles when a pod fails after Permit
func (m *Manager) Unreserve(ctx context.Context, pod *corev1.Pod) {
	config := m.ParseGangConfig(pod)
	if !config.Enabled {
		return
	}

	m.mu.Lock()
	pgInfo, exists := m.podGroups[config.GroupKey]
	if !exists {
		m.mu.Unlock()
		return
	}

	pgInfo.mu.Lock()
	m.mu.Unlock()

	// Remove from waiting pods
	if waitingInfo, ok := pgInfo.WaitingPods[pod.UID]; ok {
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
		"group", config.GroupKey)

	// If this causes the gang to fail, reject all waiting pods
	m.checkAndRejectGangIfNeeded(ctx, config.GroupKey)
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
	defer pgInfo.mu.Unlock()

	m.handleTimeoutLocked(ctx, pgInfo)
}

// handleTimeoutLocked handles timeout while holding the lock
func (m *Manager) handleTimeoutLocked(_ context.Context, pgInfo *PodGroupInfo) {
	log.Info("Gang scheduling timeout",
		"group", pgInfo.Key,
		"waiting", len(pgInfo.WaitingPods),
		"required", pgInfo.MinMembers)

	// Reject all waiting pods
	for podUID, waitingInfo := range pgInfo.WaitingPods {
		select {
		case waitingInfo.RejectCh <- "gang scheduling timeout":
		default:
		}
		delete(pgInfo.WaitingPods, podUID)
	}

	// Set backoff for this group
	m.backedOffGroups.Set(string(pgInfo.Key), struct{}{}, DefaultBackoffDuration)

	// Record event (on the first waiting pod if available)
	if m.eventRecorder != nil {
		m.eventRecorder.Eventf(nil, corev1.EventTypeWarning, EventReasonGangTimeout,
			"Pod group %s timed out with %d/%d members ready",
			pgInfo.Key, len(pgInfo.WaitingPods), pgInfo.MinMembers)
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

	// Record event
	if m.eventRecorder != nil {
		m.eventRecorder.Eventf(nil, corev1.EventTypeNormal, EventReasonGangScheduled,
			"Pod group %s scheduled with %d members", pgInfo.Key, len(pgInfo.WaitingPods))
	}
}

// checkAndRejectGangIfNeeded checks if gang can still be fulfilled and rejects if not
func (m *Manager) checkAndRejectGangIfNeeded(_ context.Context, groupKey PodGroupKey) {
	m.mu.RLock()
	pgInfo, exists := m.podGroups[groupKey]
	m.mu.RUnlock()

	if !exists {
		return
	}

	pgInfo.mu.Lock()
	defer pgInfo.mu.Unlock()

	// If there are still enough pods, don't reject
	totalActive := len(pgInfo.WaitingPods) + len(pgInfo.ScheduledPods)
	if int32(totalActive) >= pgInfo.MinMembers {
		return
	}

	// Not enough pods, reject all waiting
	log.Info("Gang cannot be fulfilled, rejecting all waiting pods",
		"group", pgInfo.Key,
		"active", totalActive,
		"required", pgInfo.MinMembers)

	for podUID, waitingInfo := range pgInfo.WaitingPods {
		select {
		case waitingInfo.RejectCh <- "gang cannot be fulfilled":
		default:
		}
		delete(pgInfo.WaitingPods, podUID)
	}

	// Set backoff
	m.backedOffGroups.Set(string(pgInfo.Key), struct{}{}, DefaultBackoffDuration)
}

// CleanupPodGroup removes a pod group from the manager
func (m *Manager) CleanupPodGroup(groupKey PodGroupKey) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if pgInfo, exists := m.podGroups[groupKey]; exists {
		pgInfo.mu.Lock()
		// Reject any remaining waiting pods
		for _, waitingInfo := range pgInfo.WaitingPods {
			select {
			case waitingInfo.RejectCh <- "pod group cleaned up":
			default:
			}
		}
		pgInfo.mu.Unlock()
		delete(m.podGroups, groupKey)
	}
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
