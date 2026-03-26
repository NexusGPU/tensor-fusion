package gang

import (
	"context"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func workloadNameForPod(pod *corev1.Pod) string {
	if pod == nil || pod.Labels == nil {
		return ""
	}
	return pod.Labels[constants.WorkloadKey]
}

func (m *Manager) rememberStatusTarget(pgInfo *PodGroupInfo, pod *corev1.Pod) {
	if pgInfo == nil || pod == nil || pgInfo.StatusWorkloadName != "" {
		return
	}
	workloadName := workloadNameForPod(pod)
	if workloadName == "" {
		return
	}
	pgInfo.StatusNamespace = pod.Namespace
	pgInfo.StatusWorkloadName = workloadName
}

// syncWorkloadGangStatus enqueues a gang status update for async persistence.
// The actual API server write happens in the background flush loop, keeping
// the Permit hot path non-blocking.
func (m *Manager) syncWorkloadGangStatus(
	_ context.Context,
	pgInfo *PodGroupInfo,
	phase tfv1.GangSchedulingPhase,
	reason, message string,
	backoffUntil time.Time,
) {
	if m.client == nil || pgInfo == nil || pgInfo.StatusNamespace == "" || pgInfo.StatusWorkloadName == "" {
		return
	}

	status := &tfv1.GangSchedulingStatus{
		Phase:            phase,
		GroupKey:         string(pgInfo.Key),
		DesiredMembers:   pgInfo.DesiredMembers,
		WaitingMembers:   int32(len(pgInfo.WaitingPods)),
		ScheduledMembers: int32(len(pgInfo.ScheduledPods)),
		// ReadyMembers is left as 0 during scheduling — the controller sets
		// the accurate value from actual pod readiness conditions.
		ReadyMembers:       0,
		Reason:             reason,
		Message:            message,
		LastTransitionTime: metav1.Now(),
	}
	if !backoffUntil.IsZero() {
		status.BackoffUntil = backoffUntil.UTC().Format(time.RFC3339)
	}

	u := statusUpdate{
		key:    client.ObjectKey{Namespace: pgInfo.StatusNamespace, Name: pgInfo.StatusWorkloadName},
		status: status,
	}
	// Terminal phases (TimedOut, Failed) must not be dropped. Try a non-blocking
	// enqueue first; if the channel is full, store in the overflow map so the
	// statusFlushLoop picks it up on the next tick. This keeps the scheduling
	// goroutine non-blocking regardless of API server back-pressure.
	if phase == tfv1.GangSchedulingPhaseTimedOut || phase == tfv1.GangSchedulingPhaseFailed {
		select {
		case m.statusQueue <- u:
		default:
			m.pendingTerminalMu.Lock()
			m.pendingTerminal[u.key] = u
			m.pendingTerminalMu.Unlock()
		}
		return
	}
	select {
	case m.statusQueue <- u:
	default:
		// Queue full — drop the update; the next one will carry fresher state.
		log.V(4).Info("Gang status queue full, dropping update",
			"group", pgInfo.Key, "phase", phase)
	}
}

// SameGangStatusState returns true when phase, reason, message and backoffUntil match.
func SameGangStatusState(a, b *tfv1.GangSchedulingStatus) bool {
	if a == nil || b == nil {
		return false
	}
	return a.Phase == b.Phase &&
		a.Reason == b.Reason &&
		a.Message == b.Message &&
		a.BackoffUntil == b.BackoffUntil
}

// SameGangStatus returns true when all observable gang status fields match.
func SameGangStatus(a, b *tfv1.GangSchedulingStatus) bool {
	if a == nil || b == nil {
		return a == b
	}
	return SameGangStatusState(a, b) &&
		a.GroupKey == b.GroupKey &&
		a.DesiredMembers == b.DesiredMembers &&
		a.WaitingMembers == b.WaitingMembers &&
		a.ScheduledMembers == b.ScheduledMembers &&
		a.ReadyMembers == b.ReadyMembers
}

func (m *Manager) patchWorkloadGangStatus(
	ctx context.Context,
	key client.ObjectKey,
	mutate func(*tfv1.TensorFusionWorkload) bool,
) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		workload := &tfv1.TensorFusionWorkload{}
		if err := m.client.Get(ctx, key, workload); err != nil {
			if apierrors.IsNotFound(err) {
				return nil
			}
			return err
		}

		if !mutate(workload) {
			return nil
		}
		return m.client.Status().Update(ctx, workload)
	})
}
