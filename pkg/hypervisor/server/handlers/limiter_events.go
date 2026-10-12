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

package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/gin-gonic/gin"
	"k8s.io/klog/v2"
)

// Limits of POST /api/v1/limiter/events. The caller is code running inside a
// workload, so everything it sends is bounded before it is stored, logged or
// turned into a Kubernetes Event.
const (
	// LimiterEventsMaxBodyBytes is the largest report body that is read. A full
	// report (LimiterEventsMaxEvents entries with details of the maximum
	// length) is about 5 KiB.
	LimiterEventsMaxBodyBytes = 8 << 10
	// LimiterEventsMaxEvents is the number of entries of one report that are
	// looked at; further entries are dropped and counted.
	LimiterEventsMaxEvents = 16
	// LimiterEventsMaxCount is the most occurrences one entry can add to a
	// counter.
	LimiterEventsMaxCount = 1_000_000
	// LimiterEventsMaxDetailBytes is the length a detail is cut to.
	LimiterEventsMaxDetailBytes = 255
	// limiterEventsMaxKindBytes is the length a kind that is not known is cut
	// to before it is logged.
	limiterEventsMaxKindBytes = 32

	// A Pod may send LimiterEventsBurst reports at once and
	// LimiterEventsBurst per LimiterEventsWindow in the long run. A limiter
	// reports at most once per window and process, so this covers a Pod with
	// several processes and holds back one that reports far more often.
	LimiterEventsBurst  = 8
	LimiterEventsWindow = 30 * time.Second

	// LimiterEventsMaxPods is the number of Pods state is kept for. A Pod that
	// did not report for LimiterEventsPodTTL is forgotten, with its counters.
	LimiterEventsMaxPods = 1024
	LimiterEventsPodTTL  = 15 * time.Minute
	// limiterEventsSweepInterval is how often forgotten Pods are looked for
	// while there is room for new ones.
	limiterEventsSweepInterval = time.Minute

	// LimiterEventsKubernetesEventInterval is the shortest time between two
	// Kubernetes Events of the same kind on the same Pod.
	LimiterEventsKubernetesEventInterval = 5 * time.Minute
	// limiterEventsQueueSize is the number of Kubernetes Events that can wait
	// to be written; one that finds the queue full is dropped and counted.
	limiterEventsQueueSize   = 64
	limiterEventsEmitTimeout = 5 * time.Second
)

// Kinds of limiter events. A kind that is not in this list is counted as
// LimiterEventOther.
const (
	LimiterEventGraceEntered   = "grace_entered"
	LimiterEventGraceExpired   = "grace_expired"
	LimiterEventUnwrappedEntry = "unwrapped_entry"
	LimiterEventClampedCharge  = "clamped_charge"
	LimiterEventStalledBucket  = "stalled_bucket"
	LimiterEventOldHypervisor  = "old_hypervisor"
	LimiterEventOther          = "other"
)

// limiterEventKinds is the closed set of kinds counters are kept for, which
// keeps the number of series per Pod fixed whatever a workload sends.
var limiterEventKinds = [...]string{
	LimiterEventGraceEntered,
	LimiterEventGraceExpired,
	LimiterEventUnwrappedEntry,
	LimiterEventClampedCharge,
	LimiterEventStalledBucket,
	LimiterEventOldHypervisor,
	LimiterEventOther,
}

const limiterEventOtherIndex = len(limiterEventKinds) - 1

// limiterEventReasons holds the Kubernetes Event reason of the kinds that mean
// the compute limit of the Pod is not fully enforced. The other kinds are only
// counted and logged.
var limiterEventReasons = map[string]string{
	LimiterEventUnwrappedEntry: "LimiterUnwrappedEntry",
	LimiterEventClampedCharge:  "LimiterClampedCharge",
	LimiterEventStalledBucket:  "LimiterStalledBucket",
}

// limiterEventMessages holds what the Kubernetes Event of each of those kinds
// says. An Event is stored by the API server and shown to everybody who may
// read the Pod, so it is made of this text and the number of occurrences and
// of nothing the workload wrote: the detail of a report only goes to the
// hypervisor's log, at debug verbosity (see logLimiterEvent).
var limiterEventMessages = map[string]string{
	LimiterEventUnwrappedEntry: "The in-process GPU limiter reported GPU work submitted through a call it does not " +
		"intercept (unwrapped_entry)",
	LimiterEventClampedCharge: "The in-process GPU limiter reported work it could not charge in full to the token " +
		"bucket of the Pod (clamped_charge)",
	LimiterEventStalledBucket: "The in-process GPU limiter reported work it admitted from a token bucket that was " +
		"not being refilled (stalled_bucket)",
}

// limiterEventTypeWarning is corev1.EventTypeWarning.
const limiterEventTypeWarning = "Warning"

// PodEventEmitter is implemented by a backend that can record a Kubernetes
// Event on a Pod. A backend without it (single node) gets counters and logs
// only.
type PodEventEmitter interface {
	EmitPodEvent(ctx context.Context, namespace, podName, podUID, eventType, reason, message string) error
}

// LimiterEventCount is one counter: the occurrences of one kind of event a Pod
// reported.
type LimiterEventCount struct {
	Namespace string
	Pod       string
	PodUID    string
	Kind      string
	Count     uint64
}

// LimiterEventStats is a copy of the counters of the limiter events endpoint.
type LimiterEventStats struct {
	// Pods holds the counters of every Pod that reported within
	// LimiterEventsPodTTL, one entry per Pod and kind that occurred, ordered by
	// namespace, Pod and kind. It never describes more than
	// LimiterEventsMaxPods Pods.
	Pods []LimiterEventCount
	// Totals holds the occurrences per kind of all Pods since the hypervisor
	// started, including those of Pods that have been forgotten since.
	Totals map[string]uint64
	// DroppedEvents counts the entries beyond LimiterEventsMaxEvents.
	DroppedEvents uint64
	// RateLimitedRequests counts the reports answered with 429.
	RateLimitedRequests uint64
	// KubernetesEventsDropped counts the Kubernetes Events that were not
	// written because the queue was full or the API server refused them.
	KubernetesEventsDropped uint64
}

// limiterPodState is what is remembered about one Pod.
type limiterPodState struct {
	namespace, podName string
	lastSeen           time.Time

	// The token bucket of the Pod's reports.
	tokens     float64
	refilledAt time.Time

	counts [len(limiterEventKinds)]uint64
	// logged is set once the first occurrence of a kind has been logged.
	logged [len(limiterEventKinds)]bool
	// eventAt is when a Kubernetes Event of a kind was last queued.
	eventAt [len(limiterEventKinds)]time.Time
}

type limiterPodEvent struct {
	namespace, podName, podUID string
	reason, message            string
}

// LimiterEventsHandler receives the operational events the in-process limiter
// of a workload reports.
type LimiterEventsHandler struct {
	// auth is the handler of POST /api/v1/process. Its token review and its
	// lookup of the calling process are used as they are, so both routes accept
	// exactly the same callers.
	auth                 *LegacyHandler
	allocationController framework.WorkerAllocationController
	emitter              PodEventEmitter
	now                  func() time.Time

	mu      sync.Mutex
	pods    map[string]*limiterPodState
	sweptAt time.Time
	totals  [len(limiterEventKinds)]uint64
	dropped uint64
	limited uint64
	unsent  uint64
	queue   chan limiterPodEvent
}

// NewLimiterEventsHandler creates the handler of POST /api/v1/limiter/events.
// Kubernetes Events are written by one goroutine that runs until ctx is done,
// and only when the backend can write them.
func NewLimiterEventsHandler(
	ctx context.Context,
	auth *LegacyHandler,
	allocationController framework.WorkerAllocationController,
	backend framework.Backend,
) *LimiterEventsHandler {
	h := &LimiterEventsHandler{
		auth:                 auth,
		allocationController: allocationController,
		now:                  time.Now,
		pods:                 make(map[string]*limiterPodState),
	}
	if emitter, ok := backend.(PodEventEmitter); ok {
		if ctx == nil {
			ctx = context.Background()
		}
		h.emitter = emitter
		h.queue = make(chan limiterPodEvent, limiterEventsQueueSize)
		go h.emitLoop(ctx)
	}
	return h
}

// limiterEventsReport is the part of a report that is used. The version, and
// the pid and container it repeats from the query, are not: the query is what
// the caller is verified against, and a newer version is read as far as it is
// understood.
type limiterEventsReport struct {
	Events []limiterEventEntry `json:"events"`
}

type limiterEventEntry struct {
	Kind string `json:"kind"`
	// Count is a float so that any JSON number is accepted and clamped.
	Count  float64 `json:"count"`
	Detail string  `json:"detail"`
}

// HandleLimiterEvents handles POST /api/v1/limiter/events.
//
// Responses: 204 accepted, 400 malformed query or body, 401 missing or invalid
// Pod token, 403 the Pod has no allocation on this node or the process is not
// one of its own, 413 body larger than LimiterEventsMaxBodyBytes, 429 the Pod
// reports too often (or state is kept for LimiterEventsMaxPods Pods already).
func (h *LimiterEventsHandler) HandleLimiterEvents(c *gin.Context) {
	var query processInitQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		c.JSON(http.StatusBadRequest, api.ErrorResponse{Error: err.Error()})
		return
	}

	// The same two steps as POST /api/v1/process: the token is reviewed by the
	// API server, and the process has to be one whose cgroup belongs to the Pod
	// the token is bound to. There is no configuration that skips either.
	if h.auth == nil {
		c.JSON(http.StatusUnauthorized, api.ErrorResponse{Error: "pod token authentication is unavailable"})
		return
	}
	podUID, err := h.auth.authenticatePodToken(c)
	if err != nil {
		c.JSON(http.StatusUnauthorized, api.ErrorResponse{Error: err.Error()})
		return
	}
	var allocation *api.WorkerAllocation
	if h.allocationController != nil {
		allocation, _ = h.allocationController.GetWorkerAllocation(podUID)
	}
	if allocation == nil || allocation.WorkerInfo == nil {
		c.JSON(http.StatusForbidden, api.ErrorResponse{Error: "Pod has no GPU allocation on this node"})
		return
	}
	namespace, podName := allocation.WorkerInfo.Namespace, allocation.WorkerInfo.WorkerName

	// Before the process lookup, which reads /proc, and before the body.
	now := h.now()
	if !h.allow(podUID, namespace, podName, now) {
		c.JSON(http.StatusTooManyRequests, api.ErrorResponse{Error: "too many limiter event reports"})
		return
	}
	if _, err := h.auth.findHostPID(podUID, namespace, podName, query.ContainerName, query.ContainerPID); err != nil {
		c.JSON(http.StatusForbidden, api.ErrorResponse{Error: "process does not belong to the authenticated Pod"})
		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(c.Writer, c.Request.Body, LimiterEventsMaxBodyBytes))
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			c.JSON(http.StatusRequestEntityTooLarge, api.ErrorResponse{Error: "report is too large"})
			return
		}
		c.JSON(http.StatusBadRequest, api.ErrorResponse{Error: "report could not be read"})
		return
	}
	var report limiterEventsReport
	if err := json.Unmarshal(body, &report); err != nil {
		c.JSON(http.StatusBadRequest, api.ErrorResponse{Error: "report is not valid JSON of the expected shape"})
		return
	}

	h.record(podUID, namespace, podName, query, report.Events, now)
	c.Status(http.StatusNoContent)
}

// allow takes one token from the Pod's bucket and reports whether there was
// one. It is also where the state of a Pod is created, which is why a Pod that
// would exceed LimiterEventsMaxPods is turned away here.
func (h *LimiterEventsHandler) allow(podUID, namespace, podName string, now time.Time) bool {
	h.mu.Lock()
	defer h.mu.Unlock()

	if now.Sub(h.sweptAt) >= limiterEventsSweepInterval || now.Before(h.sweptAt) {
		h.sweepLocked(now)
	}
	state := h.pods[podUID]
	if state == nil {
		if len(h.pods) >= LimiterEventsMaxPods {
			h.sweepLocked(now)
		}
		if len(h.pods) >= LimiterEventsMaxPods {
			h.limited++
			return false
		}
		state = &limiterPodState{tokens: LimiterEventsBurst, refilledAt: now}
		h.pods[podUID] = state
	}
	state.namespace, state.podName, state.lastSeen = namespace, podName, now

	if elapsed := now.Sub(state.refilledAt); elapsed > 0 {
		refill := elapsed.Seconds() * LimiterEventsBurst / LimiterEventsWindow.Seconds()
		state.tokens = math.Min(LimiterEventsBurst, state.tokens+refill)
	}
	state.refilledAt = now
	if state.tokens < 1 {
		h.limited++
		return false
	}
	state.tokens--
	return true
}

// sweepLocked forgets the Pods that did not report for LimiterEventsPodTTL.
func (h *LimiterEventsHandler) sweepLocked(now time.Time) {
	h.sweptAt = now
	for uid, state := range h.pods {
		if now.Sub(state.lastSeen) >= LimiterEventsPodTTL {
			delete(h.pods, uid)
		}
	}
}

// limiterEventNote is what is done about one accepted entry once the lock is
// released.
type limiterEventNote struct {
	kind, reported, detail string
	count                  uint64
	first                  bool
}

// record counts the entries of one report, logs them and queues the Kubernetes
// Events they call for. Nothing in it waits for anything but the handler's own
// lock.
func (h *LimiterEventsHandler) record(
	podUID, namespace, podName string, query processInitQuery, entries []limiterEventEntry, now time.Time,
) {
	notes := make([]limiterEventNote, 0, min(len(entries), LimiterEventsMaxEvents))

	h.mu.Lock()
	if len(entries) > LimiterEventsMaxEvents {
		h.dropped += uint64(len(entries) - LimiterEventsMaxEvents)
		entries = entries[:LimiterEventsMaxEvents]
	}
	// allow created the state in this request; it is only missing if the clock
	// jumped by more than the TTL in between.
	state := h.pods[podUID]
	for _, entry := range entries {
		count := clampLimiterEventCount(entry.Count)
		if count == 0 {
			// Nothing happened since the last report.
			continue
		}
		index := limiterEventKindIndex(entry.Kind)
		note := limiterEventNote{
			kind: limiterEventKinds[index], count: count,
			detail: sanitizeLimiterText(entry.Detail, LimiterEventsMaxDetailBytes),
		}
		if index == limiterEventOtherIndex {
			note.reported = sanitizeLimiterText(entry.Kind, limiterEventsMaxKindBytes)
		}
		h.totals[index] += count
		if state != nil {
			state.counts[index] += count
			note.first, state.logged[index] = !state.logged[index], true
			h.queueKubernetesEventLocked(state, index, limiterPodEvent{
				namespace: namespace, podName: podName, podUID: podUID,
				message: limiterEventMessage(note.kind, note.count),
			}, now)
		}
		notes = append(notes, note)
	}
	h.mu.Unlock()

	for _, note := range notes {
		logLimiterEvent(namespace, podName, query, note)
	}
}

// queueKubernetesEventLocked hands a Kubernetes Event to the goroutine that
// writes them, for the kinds that call for one and at most once per
// LimiterEventsKubernetesEventInterval, Pod and kind. It never waits: with a
// full queue (the API server is slow or away) the Event is dropped and counted,
// and the next report of the kind raises it again.
func (h *LimiterEventsHandler) queueKubernetesEventLocked(
	state *limiterPodState, index int, event limiterPodEvent, now time.Time,
) {
	reason, degraded := limiterEventReasons[limiterEventKinds[index]]
	if !degraded || h.emitter == nil {
		return
	}
	if last := state.eventAt[index]; !last.IsZero() {
		if since := now.Sub(last); since >= 0 && since < LimiterEventsKubernetesEventInterval {
			return
		}
	}
	event.reason = reason
	select {
	case h.queue <- event:
		state.eventAt[index] = now
	default:
		h.unsent++
	}
}

// logLimiterEvent logs one accepted entry. The first occurrence of a kind in a
// Pod is worth a line; the following ones are in the counters. That line holds
// nothing the workload wrote: the detail of the entry, and the name it gave a
// kind that is not known, are logged at debug verbosity only.
func logLimiterEvent(namespace, podName string, query processInitQuery, note limiterEventNote) {
	if note.first {
		klog.Infof("Limiter event %s reported by pod %s/%s container %q pid %d: count=%d",
			note.kind, namespace, podName, query.ContainerName, query.ContainerPID, note.count)
	}
	if klog.V(4).Enabled() {
		klog.V(4).Infof("Limiter event %s detail of pod %s/%s container %q pid %d: count=%d reported kind=%q detail=%q",
			note.kind, namespace, podName, query.ContainerName, query.ContainerPID, note.count, note.reported, note.detail)
	}
}

// limiterEventMessage is the text of the Kubernetes Event of a kind that calls
// for one: the fixed sentence of the kind and how often it occurred.
func limiterEventMessage(kind string, count uint64) string {
	return fmt.Sprintf("%s, %d time(s) since its last report: the compute limit of this Pod may not be fully enforced.",
		limiterEventMessages[kind], count)
}

// emitLoop writes the queued Kubernetes Events one at a time, away from the
// goroutines that serve requests.
func (h *LimiterEventsHandler) emitLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case event := <-h.queue:
			emitCtx, cancel := context.WithTimeout(ctx, limiterEventsEmitTimeout)
			err := h.emitter.EmitPodEvent(emitCtx, event.namespace, event.podName, event.podUID,
				limiterEventTypeWarning, event.reason, event.message)
			cancel()
			if err != nil {
				h.mu.Lock()
				h.unsent++
				h.mu.Unlock()
				klog.V(2).Infof("Failed to record %s event on pod %s/%s: %v",
					event.reason, event.namespace, event.podName, err)
			}
		}
	}
}

// Stats returns a copy of the counters.
func (h *LimiterEventsHandler) Stats() LimiterEventStats {
	h.mu.Lock()
	defer h.mu.Unlock()

	stats := LimiterEventStats{
		Totals:                  make(map[string]uint64, len(limiterEventKinds)),
		DroppedEvents:           h.dropped,
		RateLimitedRequests:     h.limited,
		KubernetesEventsDropped: h.unsent,
	}
	for index, kind := range limiterEventKinds {
		stats.Totals[kind] = h.totals[index]
	}
	for uid, state := range h.pods {
		for index, count := range state.counts {
			if count == 0 {
				continue
			}
			stats.Pods = append(stats.Pods, LimiterEventCount{
				Namespace: state.namespace, Pod: state.podName, PodUID: uid,
				Kind: limiterEventKinds[index], Count: count,
			})
		}
	}
	sort.Slice(stats.Pods, func(i, j int) bool {
		a, b := stats.Pods[i], stats.Pods[j]
		if a.Namespace != b.Namespace {
			return a.Namespace < b.Namespace
		}
		if a.Pod != b.Pod {
			return a.Pod < b.Pod
		}
		if a.PodUID != b.PodUID {
			return a.PodUID < b.PodUID
		}
		return a.Kind < b.Kind
	})
	return stats
}

func limiterEventKindIndex(kind string) int {
	for index, known := range limiterEventKinds[:limiterEventOtherIndex] {
		if kind == known {
			return index
		}
	}
	return limiterEventOtherIndex
}

// clampLimiterEventCount turns a reported count into 0 (nothing to count: less
// than one occurrence) or a whole number up to LimiterEventsMaxCount.
func clampLimiterEventCount(count float64) uint64 {
	if count < 1 {
		return 0
	}
	if count >= LimiterEventsMaxCount {
		return LimiterEventsMaxCount
	}
	return uint64(count)
}

// sanitizeLimiterText cuts text written by a workload to limit bytes and
// replaces every byte that is not printable ASCII, so that it cannot break a
// log line or an Event apart or carry control sequences into a terminal.
func sanitizeLimiterText(text string, limit int) string {
	if len(text) > limit {
		text = text[:limit]
	}
	clean := true
	for i := 0; i < len(text); i++ {
		if text[i] < 0x20 || text[i] > 0x7e {
			clean = false
			break
		}
	}
	if clean {
		return text
	}
	out := []byte(text)
	for i, b := range out {
		if b < 0x20 || b > 0x7e {
			out[i] = '?'
		}
	}
	return string(out)
}
