package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	hyperapi "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/gin-gonic/gin"
	"github.com/go-logr/logr/funcr"
	"k8s.io/klog/v2"
)

const (
	limiterEventsTestContainer = "tensorfusion-worker"
	// limiterEventsTestStalledReason is the Event reason of LimiterEventStalledBucket.
	limiterEventsTestStalledReason = "LimiterStalledBucket"
)

type fakePodEvent struct {
	namespace, podName, podUID, eventType, reason, message string
}

// fakeEventBackend is a backend that authenticates Pod tokens and records the
// Kubernetes Events it is asked to write.
type fakeEventBackend struct {
	fakeBackend
	// release, when set, holds every write until it is closed, like an API
	// server that does not answer.
	release chan struct{}
	seen    chan fakePodEvent

	mu     sync.Mutex
	events []fakePodEvent
}

func newFakeEventBackend(podUID string) *fakeEventBackend {
	return &fakeEventBackend{fakeBackend: fakeBackend{podUID: podUID}, seen: make(chan fakePodEvent, 256)}
}

func (f *fakeEventBackend) EmitPodEvent(
	_ context.Context, namespace, podName, podUID, eventType, reason, message string,
) error {
	if f.release != nil {
		<-f.release
	}
	event := fakePodEvent{namespace, podName, podUID, eventType, reason, message}
	f.mu.Lock()
	f.events = append(f.events, event)
	f.mu.Unlock()
	f.seen <- event
	return nil
}

func (f *fakeEventBackend) recorded() []fakePodEvent {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]fakePodEvent(nil), f.events...)
}

// next returns the next Event written, or fails the test if none arrives.
func (f *fakeEventBackend) next(t *testing.T) fakePodEvent {
	t.Helper()
	select {
	case event := <-f.seen:
		return event
	case <-time.After(10 * time.Second):
		t.Fatal("no Kubernetes Event was written")
		return fakePodEvent{}
	}
}

// limiterEventsFixture serves the route the way the server does, for callers
// whose process (container pid 1 of limiterEventsTestContainer) is found in a
// cgroup of the Pod processPodUID names.
type limiterEventsFixture struct {
	t       *testing.T
	handler *LimiterEventsHandler
	router  *gin.Engine
	// identity is what the token review answers; changing podUID makes the
	// next request one of another Pod.
	identity *fakeBackend
	workers  *fakeWorkerController
	clock    time.Time
	// processPodUID is the Pod the calling process belongs to by its cgroup;
	// empty means the Pod the token is bound to.
	processPodUID string
	// processWithoutPodUID makes the cgroup of the calling process name no Pod.
	processWithoutPodUID bool
	// processLookups counts how often the process table was read.
	processLookups int
}

func limiterTestWorkers(count int) []*hyperapi.WorkerInfo {
	workers := make([]*hyperapi.WorkerInfo, count)
	for i := range workers {
		workers[i] = &hyperapi.WorkerInfo{
			WorkerUID: fmt.Sprintf("pod-uid-%d", i), Namespace: "team-a", WorkerName: fmt.Sprintf("pod-%d", i),
		}
	}
	return workers
}

func newLimiterEventsFixture(
	t *testing.T, backend framework.Backend, identity *fakeBackend, workers ...*hyperapi.WorkerInfo,
) *limiterEventsFixture {
	t.Helper()
	gin.SetMode(gin.TestMode)

	fx := &limiterEventsFixture{
		t: t, identity: identity, workers: &fakeWorkerController{}, clock: time.Unix(1_800_000_000, 0),
	}
	allocations := &fakeAllocationController{allocations: map[string]*hyperapi.WorkerAllocation{}}
	for _, worker := range workers {
		allocations.allocations[worker.WorkerUID] = &hyperapi.WorkerAllocation{WorkerInfo: worker}
	}
	legacy := NewLegacyHandler(fx.workers, allocations, backend, nil)
	legacy.listHostPIDsFunc = func() ([]uint32, error) {
		fx.processLookups++
		return []uint32{11, 22}, nil
	}
	legacy.processMappingFunc = func(hostPID uint32) (*framework.ProcessMappingInfo, error) {
		if hostPID != 22 {
			return nil, nil
		}
		podUID := fx.processPodUID
		if podUID == "" && fx.identity != nil {
			podUID = fx.identity.podUID
		}
		if fx.processWithoutPodUID {
			podUID = ""
		}
		return &framework.ProcessMappingInfo{
			// The names are those of the caller whatever Pod the process is in:
			// only the UID of the cgroup decides.
			Namespace: "team-a", PodName: "pod-0", ContainerName: limiterEventsTestContainer,
			GuestPID: 1, HostPID: 22, PodUID: podUID,
		}, nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	fx.handler = NewLimiterEventsHandler(ctx, legacy, allocations, backend)
	fx.handler.now = func() time.Time { return fx.clock }
	fx.router = gin.New()
	fx.router.POST("/api/v1/limiter/events", fx.handler.HandleLimiterEvents)
	return fx
}

const limiterEventsTestURL = "/api/v1/limiter/events?container_pid=1&container_name=" + limiterEventsTestContainer

func (fx *limiterEventsFixture) request(url, authorization, body string) *httptest.ResponseRecorder {
	fx.t.Helper()
	req := httptest.NewRequest(http.MethodPost, url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if authorization != "" {
		req.Header.Set("Authorization", authorization)
	}
	recorder := httptest.NewRecorder()
	fx.router.ServeHTTP(recorder, req)
	return recorder
}

// post sends a report with a Pod token.
func (fx *limiterEventsFixture) post(body string) *httptest.ResponseRecorder {
	fx.t.Helper()
	return fx.request(limiterEventsTestURL, "Bearer "+createTestJWT("team-a", "pod-0"), body)
}

func (fx *limiterEventsFixture) mustPost(body string, want int) {
	fx.t.Helper()
	if recorder := fx.post(body); recorder.Code != want {
		fx.t.Fatalf("status = %d, want %d (%s)", recorder.Code, want, recorder.Body.String())
	}
}

type limiterTestEvent struct {
	Kind   string  `json:"kind"`
	Count  float64 `json:"count"`
	Detail string  `json:"detail"`
}

func limiterTestReport(events ...limiterTestEvent) string {
	if events == nil {
		events = []limiterTestEvent{}
	}
	body, err := json.Marshal(map[string]any{
		"version": 1, "pid": 1, "container": limiterEventsTestContainer, "events": events,
	})
	if err != nil {
		panic(err)
	}
	return string(body)
}

func assertNoLimiterEventsCounted(t *testing.T, fx *limiterEventsFixture) {
	t.Helper()
	stats := fx.handler.Stats()
	if len(stats.Pods) != 0 {
		t.Errorf("events were counted for %v", stats.Pods)
	}
	for kind, count := range stats.Totals {
		if count != 0 {
			t.Errorf("%d events of kind %s were counted", count, kind)
		}
	}
	if calls := fx.workers.shmCalls; len(calls) != 0 {
		t.Errorf("the report touched the shared memory of %v", calls)
	}
}

func TestLimiterEventsAreCountedPerPodAndKind(t *testing.T) {
	workers := limiterTestWorkers(2)
	backend := &fakeBackend{podUID: workers[0].WorkerUID}
	fx := newLimiterEventsFixture(t, backend, backend, workers...)

	recorder := fx.post(limiterTestReport(
		limiterTestEvent{Kind: LimiterEventGraceEntered, Count: 2},
		limiterTestEvent{Kind: LimiterEventClampedCharge, Count: 5, Detail: "charge of 900 limited to 400"},
	))
	if recorder.Code != http.StatusNoContent || recorder.Body.Len() != 0 {
		t.Fatalf("status = %d with body %q, want 204 without a body", recorder.Code, recorder.Body.String())
	}
	fx.mustPost(limiterTestReport(limiterTestEvent{Kind: LimiterEventGraceEntered, Count: 1}), http.StatusNoContent)
	backend.podUID = workers[1].WorkerUID
	fx.mustPost(limiterTestReport(limiterTestEvent{Kind: LimiterEventOldHypervisor, Count: 1}), http.StatusNoContent)
	// A report without events is a valid one.
	fx.mustPost(limiterTestReport(), http.StatusNoContent)

	stats := fx.handler.Stats()
	want := []LimiterEventCount{
		{Namespace: "team-a", Pod: "pod-0", PodUID: "pod-uid-0", Kind: LimiterEventClampedCharge, Count: 5},
		{Namespace: "team-a", Pod: "pod-0", PodUID: "pod-uid-0", Kind: LimiterEventGraceEntered, Count: 3},
		{Namespace: "team-a", Pod: "pod-1", PodUID: "pod-uid-1", Kind: LimiterEventOldHypervisor, Count: 1},
	}
	if fmt.Sprint(stats.Pods) != fmt.Sprint(want) {
		t.Errorf("per-Pod counters = %v, want %v", stats.Pods, want)
	}
	if stats.Totals[LimiterEventGraceEntered] != 3 || stats.Totals[LimiterEventClampedCharge] != 5 ||
		stats.Totals[LimiterEventOldHypervisor] != 1 || len(stats.Totals) != len(limiterEventKinds) {
		t.Errorf("totals = %v", stats.Totals)
	}
	// Reporting is not process registration: the shared memory is left alone.
	if calls := fx.workers.shmCalls; len(calls) != 0 {
		t.Errorf("the report touched the shared memory of %v", calls)
	}
}

// The route accepts exactly the callers POST /api/v1/process accepts: without a
// token the API server confirmed there is no answer but 401, whatever backend
// the hypervisor runs with.
func TestLimiterEventsRequireVerifiedPodToken(t *testing.T) {
	report := limiterTestReport(limiterTestEvent{Kind: LimiterEventStalledBucket, Count: 1})
	token := "Bearer " + createTestJWT("team-a", "pod-0")
	for _, tc := range []struct {
		name          string
		authorization string
		backend       *fakeBackend
	}{
		{name: "no token", backend: &fakeBackend{podUID: "pod-uid-0"}},
		{name: "not a bearer token", authorization: "Basic cG9kOnBvZA==", backend: &fakeBackend{podUID: "pod-uid-0"}},
		{name: "token the API server rejects", authorization: token,
			backend: &fakeBackend{podUID: "pod-uid-0", authErr: errors.New("invalid signature")}},
		{name: "token not bound to a Pod", authorization: token, backend: &fakeBackend{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fx := newLimiterEventsFixture(t, tc.backend, tc.backend, limiterTestWorkers(1)...)
			recorder := fx.request(limiterEventsTestURL, tc.authorization, report)
			if recorder.Code != http.StatusUnauthorized {
				t.Fatalf("status = %d, want 401 (%s)", recorder.Code, recorder.Body.String())
			}
			assertNoLimiterEventsCounted(t, fx)
			if fx.processLookups != 0 || len(fx.handler.pods) != 0 {
				t.Errorf("an unauthenticated request got as far as the process lookup (%d) or left state (%d Pods)",
					fx.processLookups, len(fx.handler.pods))
			}
		})
	}

	// A backend that cannot review tokens (single node) has no caller it could
	// accept, and neither has a server without a backend.
	t.Run("backend without token review", func(t *testing.T) {
		for _, backend := range []framework.Backend{nil, unauthenticatedBackend{}} {
			fx := newLimiterEventsFixture(t, backend, nil, limiterTestWorkers(1)...)
			recorder := fx.request(limiterEventsTestURL, token, report)
			if recorder.Code != http.StatusUnauthorized {
				t.Fatalf("status = %d, want 401 (%s)", recorder.Code, recorder.Body.String())
			}
			assertNoLimiterEventsCounted(t, fx)
		}
	})
}

// unauthenticatedBackend is a backend without AuthenticatePodToken.
type unauthenticatedBackend struct{ framework.Backend }

// A valid token only speaks for the processes of its own Pod: the process named
// in the query has to be in a cgroup of that Pod.
func TestLimiterEventsRejectProcessOfAnotherPod(t *testing.T) {
	report := limiterTestReport(limiterTestEvent{Kind: LimiterEventStalledBucket, Count: 1})
	for _, tc := range []struct {
		name          string
		tokenPodUID   string
		processPodUID string
		noPodUID      bool
		url           string
	}{
		{name: "process of another Pod", tokenPodUID: "pod-uid-0", processPodUID: "pod-uid-1"},
		// The names of the process match the caller's; without the UID of a
		// cgroup they select nothing.
		{name: "process whose cgroup names no Pod", tokenPodUID: "pod-uid-0", noPodUID: true},
		{name: "another pid of the container", tokenPodUID: "pod-uid-0",
			url: "/api/v1/limiter/events?container_pid=7&container_name=" + limiterEventsTestContainer},
		{name: "another container", tokenPodUID: "pod-uid-0",
			url: "/api/v1/limiter/events?container_pid=1&container_name=sidecar"},
		{name: "Pod without an allocation on this node", tokenPodUID: "pod-uid-elsewhere"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := newFakeEventBackend(tc.tokenPodUID)
			fx := newLimiterEventsFixture(t, backend, &backend.fakeBackend, limiterTestWorkers(2)...)
			fx.processPodUID, fx.processWithoutPodUID = tc.processPodUID, tc.noPodUID
			url := tc.url
			if url == "" {
				url = limiterEventsTestURL
			}
			recorder := fx.request(url, "Bearer "+createTestJWT("team-a", "pod-0"), report)
			if recorder.Code != http.StatusForbidden {
				t.Fatalf("status = %d, want 403 (%s)", recorder.Code, recorder.Body.String())
			}
			assertNoLimiterEventsCounted(t, fx)
			if events := backend.recorded(); len(events) != 0 {
				t.Errorf("a refused report raised %v", events)
			}
		})
	}
}

func TestLimiterEventsRejectOversizeBody(t *testing.T) {
	backend := &fakeBackend{podUID: "pod-uid-0"}
	fx := newLimiterEventsFixture(t, backend, backend, limiterTestWorkers(1)...)

	// The largest report the limiter can send fits.
	full := make([]limiterTestEvent, LimiterEventsMaxEvents)
	for i := range full {
		full[i] = limiterTestEvent{
			Kind: LimiterEventClampedCharge, Count: 1, Detail: strings.Repeat("d", LimiterEventsMaxDetailBytes),
		}
	}
	body := limiterTestReport(full...)
	if len(body) > LimiterEventsMaxBodyBytes {
		t.Fatalf("a full report is %d bytes, the limit is %d", len(body), LimiterEventsMaxBodyBytes)
	}
	fx.mustPost(body, http.StatusNoContent)
	before := fx.handler.Stats().Totals[LimiterEventClampedCharge]

	// Valid JSON all the way, so that only its size can be the reason.
	oversize := limiterTestReport(limiterTestEvent{
		Kind: LimiterEventClampedCharge, Count: 1, Detail: strings.Repeat("d", LimiterEventsMaxBodyBytes),
	})
	fx.mustPost(oversize, http.StatusRequestEntityTooLarge)
	if got := fx.handler.Stats().Totals[LimiterEventClampedCharge]; got != before {
		t.Errorf("an oversize report was counted: %d -> %d", before, got)
	}
}

func TestLimiterEventsRejectMalformedReport(t *testing.T) {
	backend := &fakeBackend{podUID: "pod-uid-0"}
	fx := newLimiterEventsFixture(t, backend, backend, limiterTestWorkers(1)...)
	token := "Bearer " + createTestJWT("team-a", "pod-0")

	for name, body := range map[string]string{
		"empty body":            ``,
		"not JSON":              `events=stalled_bucket`,
		"truncated JSON":        `{"version":1,"events":[{"kind":"stalled_bucket","count":1`,
		"trailing data":         `{"version":1,"events":[]} {"events":[]}`,
		"array instead of body": `[{"kind":"stalled_bucket","count":1}]`,
		"events is not a list":  `{"version":1,"events":{"kind":"stalled_bucket","count":1}}`,
		"count is not a number": `{"version":1,"events":[{"kind":"stalled_bucket","count":"many"}]}`,
		"kind is not a string":  `{"version":1,"events":[{"kind":7,"count":1}]}`,
		"detail is not text":    `{"version":1,"events":[{"kind":"stalled_bucket","count":1,"detail":{"a":1}}]}`,
	} {
		// Each one is a request of its own Pod-minute: keep the bucket full.
		fx.clock = fx.clock.Add(LimiterEventsWindow)
		if recorder := fx.post(body); recorder.Code != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400 (%s)", name, recorder.Code, recorder.Body.String())
		}
	}
	for name, url := range map[string]string{
		"no query":          "/api/v1/limiter/events",
		"no container name": "/api/v1/limiter/events?container_pid=1",
		"no pid":            "/api/v1/limiter/events?container_name=" + limiterEventsTestContainer,
		"pid is not a pid":  "/api/v1/limiter/events?container_pid=init&container_name=" + limiterEventsTestContainer,
	} {
		if recorder := fx.request(url, token, limiterTestReport()); recorder.Code != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400 (%s)", name, recorder.Code, recorder.Body.String())
		}
	}
	assertNoLimiterEventsCounted(t, fx)
}

func TestLimiterEventsCountUnknownKindAsOther(t *testing.T) {
	backend := newFakeEventBackend("pod-uid-0")
	fx := newLimiterEventsFixture(t, backend, &backend.fakeBackend, limiterTestWorkers(1)...)

	fx.mustPost(limiterTestReport(
		limiterTestEvent{Kind: "bucket_on_fire", Count: 2},
		limiterTestEvent{Kind: "", Count: 1},
		limiterTestEvent{Kind: "STALLED_BUCKET", Count: 1},
		limiterTestEvent{Kind: strings.Repeat("k", 4000), Count: 1},
		limiterTestEvent{Kind: LimiterEventGraceExpired, Count: 1},
	), http.StatusNoContent)

	stats := fx.handler.Stats()
	if stats.Totals[LimiterEventOther] != 5 || stats.Totals[LimiterEventGraceExpired] != 1 {
		t.Errorf("totals = %v, want 5 other and 1 grace_expired", stats.Totals)
	}
	// However many names a workload makes up, a Pod has one series per known kind.
	if len(stats.Pods) != 2 || len(stats.Totals) != len(limiterEventKinds) {
		t.Errorf("counters = %v, totals = %v", stats.Pods, stats.Totals)
	}
	// A kind that is not known says nothing about isolation.
	fx.clock = fx.clock.Add(time.Second)
	fx.mustPost(limiterTestReport(limiterTestEvent{Kind: LimiterEventStalledBucket, Count: 1}), http.StatusNoContent)
	if event := backend.next(t); event.reason != limiterEventReasons[LimiterEventStalledBucket] {
		t.Errorf("first Event raised is %+v", event)
	}
}

func TestLimiterEventsCapEntriesAndCounts(t *testing.T) {
	backend := &fakeBackend{podUID: "pod-uid-0"}
	fx := newLimiterEventsFixture(t, backend, backend, limiterTestWorkers(1)...)

	events := make([]limiterTestEvent, LimiterEventsMaxEvents+9)
	for i := range events {
		events[i] = limiterTestEvent{Kind: LimiterEventGraceEntered, Count: 1}
	}
	fx.mustPost(limiterTestReport(events...), http.StatusNoContent)
	stats := fx.handler.Stats()
	if got := stats.Totals[LimiterEventGraceEntered]; got != LimiterEventsMaxEvents {
		t.Errorf("%d entries of one report were counted, want %d", got, LimiterEventsMaxEvents)
	}
	if stats.DroppedEvents != 9 {
		t.Errorf("%d entries were counted as dropped, want 9", stats.DroppedEvents)
	}

	fx.mustPost(limiterTestReport(
		limiterTestEvent{Kind: LimiterEventClampedCharge, Count: 5e12},
		limiterTestEvent{Kind: LimiterEventGraceExpired, Count: 1e300},
		limiterTestEvent{Kind: LimiterEventStalledBucket, Count: 0},
		limiterTestEvent{Kind: LimiterEventUnwrappedEntry, Count: -4},
		limiterTestEvent{Kind: LimiterEventOldHypervisor, Count: 2.9},
	), http.StatusNoContent)
	totals := fx.handler.Stats().Totals
	if totals[LimiterEventClampedCharge] != LimiterEventsMaxCount ||
		totals[LimiterEventGraceExpired] != LimiterEventsMaxCount {
		t.Errorf("counts were not clamped to %d: %v", LimiterEventsMaxCount, totals)
	}
	if totals[LimiterEventStalledBucket] != 0 || totals[LimiterEventUnwrappedEntry] != 0 {
		t.Errorf("entries without an occurrence were counted: %v", totals)
	}
	if totals[LimiterEventOldHypervisor] != 2 {
		t.Errorf("a count of 2.9 was counted as %d, want 2", totals[LimiterEventOldHypervisor])
	}
}

func isPrintableASCII(text string) bool {
	for i := 0; i < len(text); i++ {
		if text[i] < 0x20 || text[i] > 0x7e {
			return false
		}
	}
	return true
}

// What a workload writes into a detail must not be able to start a new log
// line, carry terminal control sequences or grow without bound.
func TestLimiterEventsSanitiseDetail(t *testing.T) {
	hostile := "ok\nI0101 forged line\r\t\x1b[31mred\x00 café   " + strings.Repeat("x", 600)
	clean := sanitizeLimiterText(hostile, LimiterEventsMaxDetailBytes)
	if len(clean) != LimiterEventsMaxDetailBytes || !isPrintableASCII(clean) {
		t.Fatalf("sanitised detail is %d bytes: %q", len(clean), clean)
	}
	if !strings.HasPrefix(clean, "ok?I0101 forged line???[31mred? caf?? ??? xxx") {
		t.Errorf("sanitised detail = %q", clean)
	}
	if got := sanitizeLimiterText("bucket 3 idle for 31 s", LimiterEventsMaxDetailBytes); got != "bucket 3 idle for 31 s" {
		t.Errorf("a plain detail was changed to %q", got)
	}
	// Cutting may split a multi-byte character; what is left of it is replaced.
	if got := sanitizeLimiterText("abé", 3); got != "ab?" {
		t.Errorf("cut detail = %q", got)
	}

	backend := newFakeEventBackend("pod-uid-0")
	fx := newLimiterEventsFixture(t, backend, &backend.fakeBackend, limiterTestWorkers(1)...)
	fx.mustPost(limiterTestReport(
		limiterTestEvent{Kind: LimiterEventUnwrappedEntry, Count: 3, Detail: hostile},
	), http.StatusNoContent)
	// The Event is fixed text and the count: nothing of the detail is in it,
	// sanitised or not.
	event := backend.next(t)
	if want := limiterEventMessage(LimiterEventUnwrappedEntry, 3); event.message != want {
		t.Errorf("Event message = %q, want %q", event.message, want)
	}
	for _, written := range []string{"forged", "[31m", "caf", "xxx"} {
		if strings.Contains(event.message, written) {
			t.Errorf("the Event message carries %q of the detail the workload wrote: %q", written, event.message)
		}
	}
}

// What a workload writes about an event (the detail, and the name of a kind
// that is not known) reaches neither a Kubernetes Event nor a log line of the
// default verbosity, whatever it is; it is logged at debug verbosity, sanitised.
func TestLimiterEventsKeepWorkloadTextOutOfEventsAndDefaultLogs(t *testing.T) {
	const detail, madeUpKind = "token=abc123 first\nsecond", "kind_made_up_by_workload"
	report := limiterTestReport(
		limiterTestEvent{Kind: LimiterEventStalledBucket, Count: 4, Detail: detail},
		limiterTestEvent{Kind: LimiterEventClampedCharge, Count: 1, Detail: detail},
		limiterTestEvent{Kind: LimiterEventUnwrappedEntry, Count: 2, Detail: detail},
		limiterTestEvent{Kind: madeUpKind, Count: 1, Detail: detail},
	)
	wantMessages := map[string]string{
		"LimiterStalledBucket": "The in-process GPU limiter reported work it admitted from a token bucket that was " +
			"not being refilled (stalled_bucket), 4 time(s) since its last report: " +
			"the compute limit of this Pod may not be fully enforced.",
		"LimiterClampedCharge":  limiterEventMessage(LimiterEventClampedCharge, 1),
		"LimiterUnwrappedEntry": limiterEventMessage(LimiterEventUnwrappedEntry, 2),
	}

	for _, verbosity := range []int{0, 4} {
		t.Run(fmt.Sprintf("verbosity %d", verbosity), func(t *testing.T) {
			var mu sync.Mutex
			var lines []string
			klog.SetLogger(funcr.New(func(_, args string) {
				mu.Lock()
				defer mu.Unlock()
				lines = append(lines, args)
			}, funcr.Options{Verbosity: verbosity}))
			t.Cleanup(klog.ClearLogger)
			flags := flag.NewFlagSet("klog", flag.ContinueOnError)
			klog.InitFlags(flags)
			if err := flags.Set("v", fmt.Sprint(verbosity)); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = flags.Set("v", "0") })

			backend := newFakeEventBackend("pod-uid-0")
			fx := newLimiterEventsFixture(t, backend, &backend.fakeBackend, limiterTestWorkers(1)...)
			fx.mustPost(report, http.StatusNoContent)
			for range wantMessages {
				event := backend.next(t)
				if want := wantMessages[event.reason]; event.message != want {
					t.Errorf("%s Event message = %q, want %q", event.reason, event.message, want)
				}
			}
			for _, count := range fx.handler.Stats().Pods {
				if strings.Contains(count.Kind, "made_up") {
					t.Errorf("a counter is labelled with the kind the workload made up: %+v", count)
				}
			}

			mu.Lock()
			defer mu.Unlock()
			debugLines := 0
			for _, line := range lines {
				if !strings.Contains(line, "Limiter event") {
					continue
				}
				written := strings.Contains(line, "abc123") || strings.Contains(line, "first") ||
					strings.Contains(line, madeUpKind)
				if strings.Contains(line, `"level"=0`) && written {
					t.Errorf("a log line of the default verbosity carries what the workload wrote: %s", line)
				}
				if strings.Contains(line, `"level"=4`) && strings.Contains(line, "first?second") &&
					strings.Contains(line, "abc123") {
					debugLines++
				}
			}
			// One debug line per entry, and none unless debug logging is on.
			if want := verbosity / 4 * 4; debugLines != want {
				t.Errorf("%d debug lines carry the detail at verbosity %d, want %d", debugLines, verbosity, want)
			}
		})
	}
}

// A Pod that reports far more often than a limiter does is answered with 429
// before anything is looked up or read for it, and does not hold back others.
func TestLimiterEventsRateLimitPerPod(t *testing.T) {
	workers := limiterTestWorkers(2)
	backend := &fakeBackend{podUID: workers[0].WorkerUID}
	fx := newLimiterEventsFixture(t, backend, backend, workers...)
	report := limiterTestReport(limiterTestEvent{Kind: LimiterEventGraceEntered, Count: 1})

	for i := range LimiterEventsBurst {
		if recorder := fx.post(report); recorder.Code != http.StatusNoContent {
			t.Fatalf("report %d: status = %d", i, recorder.Code)
		}
	}
	lookups := fx.processLookups
	const flood = 500
	for range flood {
		fx.mustPost(report, http.StatusTooManyRequests)
	}
	if fx.processLookups != lookups {
		t.Errorf("the process table was read %d times for refused reports", fx.processLookups-lookups)
	}
	stats := fx.handler.Stats()
	if stats.RateLimitedRequests != flood || stats.Totals[LimiterEventGraceEntered] != LimiterEventsBurst {
		t.Errorf("rate limited = %d, counted = %d", stats.RateLimitedRequests, stats.Totals[LimiterEventGraceEntered])
	}

	// The neighbor has a bucket of its own.
	backend.podUID = workers[1].WorkerUID
	fx.mustPost(report, http.StatusNoContent)
	backend.podUID = workers[0].WorkerUID

	// One report is earned per LimiterEventsWindow / LimiterEventsBurst ...
	fx.clock = fx.clock.Add(LimiterEventsWindow / LimiterEventsBurst)
	fx.mustPost(report, http.StatusNoContent)
	fx.mustPost(report, http.StatusTooManyRequests)
	// ... and never more than a burst is saved up.
	fx.clock = fx.clock.Add(10 * LimiterEventsWindow)
	for range LimiterEventsBurst {
		fx.mustPost(report, http.StatusNoContent)
	}
	fx.mustPost(report, http.StatusTooManyRequests)
}

// State is kept for a bounded number of Pods, and a Pod that stopped reporting
// is forgotten.
func TestLimiterEventsKeepStateForABoundedNumberOfPods(t *testing.T) {
	workers := limiterTestWorkers(LimiterEventsMaxPods + 1)
	backend := &fakeBackend{}
	fx := newLimiterEventsFixture(t, backend, backend, workers...)
	report := limiterTestReport(limiterTestEvent{Kind: LimiterEventGraceEntered, Count: 1})

	for _, worker := range workers[:LimiterEventsMaxPods] {
		backend.podUID = worker.WorkerUID
		fx.mustPost(report, http.StatusNoContent)
	}
	if got := len(fx.handler.pods); got != LimiterEventsMaxPods {
		t.Fatalf("state is kept for %d Pods, want %d", got, LimiterEventsMaxPods)
	}

	// One more Pod is turned away; those that are known are served.
	last := workers[LimiterEventsMaxPods]
	backend.podUID = last.WorkerUID
	fx.mustPost(report, http.StatusTooManyRequests)
	backend.podUID = workers[0].WorkerUID
	fx.mustPost(report, http.StatusNoContent)
	if got := len(fx.handler.pods); got != LimiterEventsMaxPods {
		t.Errorf("state is kept for %d Pods, want %d", got, LimiterEventsMaxPods)
	}
	if got := len(fx.handler.Stats().Pods); got != LimiterEventsMaxPods {
		t.Errorf("counters describe %d Pods, want %d", got, LimiterEventsMaxPods)
	}

	// After the TTL only the Pods that kept reporting are left.
	fx.clock = fx.clock.Add(LimiterEventsPodTTL - time.Second)
	backend.podUID = workers[0].WorkerUID
	fx.mustPost(report, http.StatusNoContent)
	fx.clock = fx.clock.Add(time.Second)
	backend.podUID = last.WorkerUID
	fx.mustPost(report, http.StatusNoContent)
	if got := len(fx.handler.pods); got != 2 {
		t.Errorf("state is kept for %d Pods after the TTL, want 2", got)
	}
	// The totals outlive the Pods.
	if got := fx.handler.Stats().Totals[LimiterEventGraceEntered]; got != LimiterEventsMaxPods+3 {
		t.Errorf("total = %d, want %d", got, LimiterEventsMaxPods+3)
	}
}

// The three kinds that mean the limit is not fully enforced raise a Warning
// Event on the Pod, once per Pod and kind in LimiterEventsKubernetesEventInterval.
func TestLimiterEventsRaiseKubernetesEventOncePerPodAndKind(t *testing.T) {
	workers := limiterTestWorkers(2)
	backend := newFakeEventBackend(workers[0].WorkerUID)
	fx := newLimiterEventsFixture(t, backend, &backend.fakeBackend, workers...)

	everything := limiterTestReport(
		limiterTestEvent{Kind: LimiterEventGraceEntered, Count: 2},
		limiterTestEvent{Kind: LimiterEventUnwrappedEntry, Count: 2},
		limiterTestEvent{Kind: LimiterEventGraceExpired, Count: 2},
		limiterTestEvent{Kind: LimiterEventClampedCharge, Count: 2},
		limiterTestEvent{Kind: LimiterEventOldHypervisor, Count: 2},
		limiterTestEvent{Kind: LimiterEventStalledBucket, Count: 2},
		limiterTestEvent{Kind: "something_new", Count: 2},
		limiterTestEvent{Kind: LimiterEventStalledBucket, Count: 2},
	)
	fx.mustPost(everything, http.StatusNoContent)
	wantReasons := []string{"LimiterUnwrappedEntry", "LimiterClampedCharge", limiterEventsTestStalledReason}
	for _, reason := range wantReasons {
		event := backend.next(t)
		want := fakePodEvent{
			namespace: "team-a", podName: "pod-0", podUID: "pod-uid-0", eventType: "Warning", reason: reason,
			message: event.message,
		}
		if event != want {
			t.Errorf("Event = %+v, want %+v", event, want)
		}
		// Fixed text of the kind and the count of the entry that raised it.
		if !strings.HasPrefix(event.message, "The in-process GPU limiter reported ") ||
			!strings.HasSuffix(event.message, ", 2 time(s) since its last report: "+
				"the compute limit of this Pod may not be fully enforced.") {
			t.Errorf("Event message = %q", event.message)
		}
	}

	// Reported again and again within the interval: counted, no further Event.
	for range 3 {
		fx.clock = fx.clock.Add(LimiterEventsWindow)
		fx.mustPost(everything, http.StatusNoContent)
	}
	// Another Pod is told about its own.
	backend.podUID = workers[1].WorkerUID
	fx.mustPost(limiterTestReport(limiterTestEvent{Kind: LimiterEventClampedCharge, Count: 1}), http.StatusNoContent)
	if event := backend.next(t); event.podUID != "pod-uid-1" || event.podName != "pod-1" ||
		event.reason != "LimiterClampedCharge" {
		t.Errorf("Event = %+v, want LimiterClampedCharge on pod-1", event)
	}
	// After the interval the first Pod is told again, about what it reports then.
	backend.podUID = workers[0].WorkerUID
	fx.clock = fx.clock.Add(LimiterEventsKubernetesEventInterval)
	fx.mustPost(limiterTestReport(limiterTestEvent{Kind: LimiterEventStalledBucket, Count: 1}), http.StatusNoContent)
	if event := backend.next(t); event.podUID != "pod-uid-0" || event.reason != limiterEventsTestStalledReason {
		t.Errorf("Event = %+v, want LimiterStalledBucket on pod-0", event)
	}
	// Events are written in order by one goroutine, so these five are all there were.
	if events := backend.recorded(); len(events) != 5 {
		t.Errorf("%d Events were written, want 5: %+v", len(events), events)
	}
	if got := fx.handler.Stats().Totals[LimiterEventStalledBucket]; got != 4*4+1 {
		t.Errorf("stalled_bucket total = %d, want 17", got)
	}
}

// A backend that cannot write Events (single node) still serves the route.
func TestLimiterEventsWorkWithoutAnEventEmitter(t *testing.T) {
	backend := &fakeBackend{podUID: "pod-uid-0"}
	fx := newLimiterEventsFixture(t, backend, backend, limiterTestWorkers(1)...)
	if fx.handler.emitter != nil || fx.handler.queue != nil {
		t.Fatal("the fixture's backend is meant to have no Event emitter")
	}
	fx.mustPost(limiterTestReport(
		limiterTestEvent{Kind: LimiterEventStalledBucket, Count: 4},
		limiterTestEvent{Kind: LimiterEventUnwrappedEntry, Count: 1},
	), http.StatusNoContent)
	stats := fx.handler.Stats()
	if stats.Totals[LimiterEventStalledBucket] != 4 || stats.Totals[LimiterEventUnwrappedEntry] != 1 ||
		stats.KubernetesEventsDropped != 0 {
		t.Errorf("stats = %+v", stats)
	}
}

// An API server that does not answer must not hold up the workloads' reports:
// Events that cannot be queued are dropped, and raised by a later report.
func TestLimiterEventsDoNotWaitForTheAPIServer(t *testing.T) {
	// Three Events per Pod: more than the queue and the one being written hold.
	workers := limiterTestWorkers(limiterEventsQueueSize/3 + 4)
	backend := newFakeEventBackend("")
	backend.release = make(chan struct{})
	released := false
	t.Cleanup(func() {
		if !released {
			close(backend.release)
		}
	})
	fx := newLimiterEventsFixture(t, backend, &backend.fakeBackend, workers...)
	report := limiterTestReport(
		limiterTestEvent{Kind: LimiterEventUnwrappedEntry, Count: 1},
		limiterTestEvent{Kind: LimiterEventClampedCharge, Count: 1},
		limiterTestEvent{Kind: LimiterEventStalledBucket, Count: 1},
	)

	done := make(chan struct{})
	go func() {
		defer close(done)
		for _, worker := range workers {
			backend.podUID = worker.WorkerUID
			if recorder := fx.post(report); recorder.Code != http.StatusNoContent {
				t.Errorf("status = %d while the API server does not answer", recorder.Code)
			}
		}
	}()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("reports wait for the API server")
	}
	dropped := fx.handler.Stats().KubernetesEventsDropped
	if dropped == 0 || dropped > uint64(3*len(workers)-limiterEventsQueueSize) {
		t.Fatalf("%d Events were dropped with a queue of %d and %d raised", dropped, limiterEventsQueueSize, 3*len(workers))
	}
	if got := fx.handler.Stats().Totals[LimiterEventStalledBucket]; got != uint64(len(workers)) {
		t.Errorf("%d reports were counted, want %d", got, len(workers))
	}

	// The API server answers again: what was queued is written, and the Pod
	// whose Events were dropped gets them with its next report.
	close(backend.release)
	released = true
	written := 3*len(workers) - int(dropped)
	for range written {
		backend.next(t)
	}
	last := workers[len(workers)-1]
	for _, event := range backend.recorded() {
		if event.podUID == last.WorkerUID && event.reason == limiterEventsTestStalledReason {
			t.Fatalf("the scenario is not the one intended: the last Pod's Event was written")
		}
	}
	backend.podUID = last.WorkerUID
	fx.mustPost(limiterTestReport(limiterTestEvent{Kind: LimiterEventStalledBucket, Count: 1}), http.StatusNoContent)
	if event := backend.next(t); event.podUID != last.WorkerUID || event.reason != limiterEventsTestStalledReason {
		t.Errorf("Event = %+v, want the dropped LimiterStalledBucket of %s", event, last.WorkerUID)
	}
}

// A newer limiter may send a higher version and fields this release does not
// know; what is understood is used.
func TestLimiterEventsAcceptNewerVersionAndUnknownFields(t *testing.T) {
	backend := &fakeBackend{podUID: "pod-uid-0"}
	fx := newLimiterEventsFixture(t, backend, backend, limiterTestWorkers(1)...)

	fx.mustPost(`{
		"version": 7, "pid": 1, "container": "tensorfusion-worker", "library": {"build": "next"},
		"events": [
			{"kind": "grace_entered", "count": 3, "detail": "d", "since_ms": 1200, "device": {"index": 0}},
			{"kind": "bucket_resized", "count": 1, "extra": [1, 2, 3]},
			{"kind": "clamped_charge", "count": 2}
		],
		"trailer": null
	}`, http.StatusNoContent)
	// The version, and the pid and container of the body, are not relied upon.
	fx.mustPost(`{"version":"2.1","pid":"one","container":7,"events":[{"kind":"grace_entered","count":1}]}`,
		http.StatusNoContent)
	fx.mustPost(`{"events":[{"kind":"grace_entered","count":1}]}`, http.StatusNoContent)

	totals := fx.handler.Stats().Totals
	if totals[LimiterEventGraceEntered] != 5 || totals[LimiterEventOther] != 1 || totals[LimiterEventClampedCharge] != 2 {
		t.Errorf("totals = %v", totals)
	}
}

// The first time a Pod reports a kind is logged at the default verbosity; the
// following reports of it are not (they are logged at V(4) and counted).
func TestLimiterEventsLogFirstOccurrencePerPodAndKind(t *testing.T) {
	var mu sync.Mutex
	var lines []string
	klog.SetLogger(funcr.New(func(_, args string) {
		mu.Lock()
		defer mu.Unlock()
		lines = append(lines, args)
	}, funcr.Options{}))
	t.Cleanup(klog.ClearLogger)
	logged := func(pod, kind string) int {
		mu.Lock()
		defer mu.Unlock()
		count := 0
		for _, line := range lines {
			if strings.Contains(line, "Limiter event "+kind+" ") && strings.Contains(line, "team-a/"+pod+" ") {
				count++
			}
		}
		return count
	}

	workers := limiterTestWorkers(2)
	backend := &fakeBackend{podUID: workers[0].WorkerUID}
	fx := newLimiterEventsFixture(t, backend, backend, workers...)
	stalled := limiterTestReport(limiterTestEvent{Kind: LimiterEventStalledBucket, Count: 1, Detail: "first\nsecond"})

	fx.mustPost(stalled, http.StatusNoContent)
	fx.mustPost(stalled, http.StatusNoContent)
	fx.mustPost(stalled, http.StatusNoContent)
	if got := logged("pod-0", LimiterEventStalledBucket); got != 1 {
		t.Errorf("stalled_bucket of pod-0 was logged %d times, want once", got)
	}
	fx.mustPost(limiterTestReport(limiterTestEvent{Kind: LimiterEventGraceEntered, Count: 1}), http.StatusNoContent)
	if got := logged("pod-0", LimiterEventGraceEntered); got != 1 {
		t.Errorf("grace_entered of pod-0 was logged %d times, want once", got)
	}
	backend.podUID = workers[1].WorkerUID
	fx.mustPost(stalled, http.StatusNoContent)
	fx.mustPost(stalled, http.StatusNoContent)
	if got := logged("pod-1", LimiterEventStalledBucket); got != 1 {
		t.Errorf("stalled_bucket of pod-1 was logged %d times, want once", got)
	}

	mu.Lock()
	defer mu.Unlock()
	for _, line := range lines {
		if strings.Contains(line, "Limiter event") && strings.Contains(line, "first") {
			t.Errorf("the detail reached the log at the default verbosity: %q", line)
		}
	}
}
