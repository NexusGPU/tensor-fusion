package computing

import (
	"context"
	"fmt"

	"k8s.io/klog/v2"
)

// PodEventEmitter is implemented by a backend that can record a Kubernetes
// Event on a Pod, as the Kubernetes backend does.
type PodEventEmitter interface {
	EmitPodEvent(ctx context.Context, namespace, podName, podUID, eventType, reason, message string) error
}

const (
	// podEventTypeWarning is corev1.EventTypeWarning.
	podEventTypeWarning = "Warning"
	// The Event of a Pod whose bucket is left out of the split of its device.
	// Its text is fixed: nothing found in the worker's mapping is part of it.
	meterExclusionEventReason = "LimiterBalanceUnaccounted"
)

// meterExclusionEventMessage is the text of that Event.
var meterExclusionEventMessage = fmt.Sprintf(
	"The token balance of the GPU limiter of this Pod could not be accounted for during %.0f s: the tokens it spends "+
		"are not used to measure its GPU usage, its refill rate is held, and its compute limit may not be fully "+
		"enforced. This ends once the balance has been consistent for %.0f s.",
	(meterExclusionIntervals * erlUpdateInterval).Seconds(), (meterReadmissionIntervals * erlUpdateInterval).Seconds())

// SetPodEventEmitter wires where the control loop records Events on Pods; with
// nil it records none. NewQuotaController does this with its backend when the
// backend can record them. Events are written away from the control loop, one
// attempt each: an API server that is slow or away does not hold up a rate.
func (c *Controller) SetPodEventEmitter(emitter PodEventEmitter) {
	if emitter == nil {
		c.podEventFn = nil
		return
	}
	c.podEventFn = func(namespace, podName, podUID, reason, message string) {
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), podEventTimeout)
			defer cancel()
			if err := emitter.EmitPodEvent(ctx, namespace, podName, podUID, podEventTypeWarning, reason, message); err != nil {
				klog.V(2).Infof("Failed to record %s event on pod %s/%s: %v", reason, namespace, podName, err)
			}
		}()
	}
}

// reportMeterExclusion logs when a worker's bucket is left out of the split of
// its device or taken back into it (before is what it was at the previous
// look), and tells the Pod with an Event while it is left out, at most once
// per meterExclusionEventInterval and Pod.
func (c *Controller) reportMeterExclusion(member *erlMember, worker *WorkerInfoSnapshot, before bool) {
	excluded := member.meter.excluded
	if excluded && !before {
		klog.Warningf("ERL [%s] the token balance could not be accounted for at %d looks: the worker is left out of "+
			"the token split of its device and its refill rate is held", member.stateKey, meterExclusionIntervals)
	} else if before && !excluded {
		klog.Infof("ERL [%s] the token balance is accounted for again: the worker is back in the token split of "+
			"its device", member.stateKey)
	}
	// A worker without a compute limit has no limit to tell it about.
	if !excluded || member.device.UpLimit >= 100 || c.podEventFn == nil {
		return
	}
	now := c.nowFunc()
	c.erlStatesMu.Lock()
	_, removed := c.removedWorkers[member.workerUID]
	last, raised := c.exclusionEvents[member.workerUID]
	due := !removed && (!raised || now.Sub(last) >= meterExclusionEventInterval || now.Before(last))
	if due {
		c.exclusionEvents[member.workerUID] = now
	}
	c.erlStatesMu.Unlock()
	if due {
		c.podEventFn(worker.Namespace, worker.WorkerName, member.workerUID,
			meterExclusionEventReason, meterExclusionEventMessage)
	}
}
