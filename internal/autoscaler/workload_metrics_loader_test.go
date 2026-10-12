package autoscaler

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type emptyMetricsProvider struct{}

func (emptyMetricsProvider) GetWorkersMetrics(ctx context.Context) ([]*metrics.WorkerUsage, error) {
	return nil, nil
}
func (emptyMetricsProvider) GetWorkloadHistoryMetrics(ctx context.Context, ns, name string, start, end time.Time) ([]*metrics.WorkerUsage, error) {
	return nil, nil
}
func (emptyMetricsProvider) GetWorkloadRealtimeMetrics(ctx context.Context, ns, name string, start, end time.Time) ([]*metrics.WorkerUsage, error) {
	return nil, nil
}

// newlyCreatedWorkloadState is a workload created just now with auto scaling
// switched on the way the pod webhook does it for tensor-fusion.ai/autoscale:
// InitialDelayPeriod is left at its 30 minute default.
func newlyCreatedWorkloadState(rules ...tfv1.CronScalingRule) *workload.State {
	ws := workload.NewWorkloadState()
	ws.Namespace = "ns"
	ws.Name = "wl"
	ws.CreationTimestamp = metav1.Now()
	ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
		AutoSetResources: &tfv1.AutoSetResources{
			Enable:         true,
			TargetResource: tfv1.ScalingTargetResourceAll,
			Interval:       "10ms",
		},
		CronScalingRules: rules,
	}
	return ws
}

// A cron rule on a workload younger than InitialDelayPeriod must be evaluated:
// the window of the rule can open minutes after the workload was created, and
// the rule needs no usage data. Before the fix nothing was evaluated for the
// first 30 minutes of a workload, so a pod living shorter than that never had
// its rule applied.
func TestWorkloadMetricsLoader_CronRuleIsEvaluatedInsideInitialDelay(t *testing.T) {
	l := newWorkloadMetricsLoader(nil, emptyMetricsProvider{})
	processed := make(chan struct{}, 1)
	l.setProcessFunc(func(ctx context.Context, state *workload.State) {
		select {
		case processed <- struct{}{}:
		default:
		}
	})

	wid := WorkloadID{"ns", "wl"}
	l.addWorkload(context.Background(), wid, newlyCreatedWorkloadState(tfv1.CronScalingRule{
		Enable: true, Name: "shrink", Start: "0 1 * * *", End: "0 2 * * *",
	}))
	defer l.removeWorkload(wid)

	select {
	case <-processed:
	case <-time.After(5 * time.Second):
		t.Fatal("workload with a cron scaling rule was not evaluated inside InitialDelayPeriod")
	}
}

// Without a usable cron rule the evaluation still waits for InitialDelayPeriod.
func TestWorkloadMetricsLoader_WithoutCronRuleWaitsForInitialDelay(t *testing.T) {
	for name, rules := range map[string][]tfv1.CronScalingRule{
		"no rule":         nil,
		"disabled rule":   {{Enable: false, Name: "shrink", Start: "0 1 * * *", End: "0 2 * * *"}},
		"incomplete rule": {{Enable: true, Name: "shrink", Start: "0 1 * * *"}},
	} {
		t.Run(name, func(t *testing.T) {
			l := newWorkloadMetricsLoader(nil, emptyMetricsProvider{})
			var processed int32
			l.setProcessFunc(func(ctx context.Context, state *workload.State) {
				atomic.AddInt32(&processed, 1)
			})

			wid := WorkloadID{"ns", "wl"}
			l.addWorkload(context.Background(), wid, newlyCreatedWorkloadState(rules...))
			defer l.removeWorkload(wid)

			time.Sleep(200 * time.Millisecond)
			assert.Zero(t, atomic.LoadInt32(&processed), "evaluated before InitialDelayPeriod")
			l.mu.RLock()
			defer l.mu.RUnlock()
			assert.NotNil(t, l.workloads[wid].initialDelayTimer, "initial delay timer must be armed")
		})
	}
}

// blockingMetricsProvider lets us hold a history-load call open long enough
// to schedule removeWorkload concurrently.
type blockingMetricsProvider struct {
	historyStarted chan struct{}
	historyRelease chan struct{}
	historyCalled  int32
}

func (b *blockingMetricsProvider) GetWorkersMetrics(ctx context.Context) ([]*metrics.WorkerUsage, error) {
	return nil, nil
}
func (b *blockingMetricsProvider) GetWorkloadHistoryMetrics(ctx context.Context, ns, name string, start, end time.Time) ([]*metrics.WorkerUsage, error) {
	atomic.AddInt32(&b.historyCalled, 1)
	select {
	case b.historyStarted <- struct{}{}:
	default:
	}
	select {
	case <-b.historyRelease:
	case <-ctx.Done():
	}
	return nil, nil
}
func (b *blockingMetricsProvider) GetWorkloadRealtimeMetrics(ctx context.Context, ns, name string, start, end time.Time) ([]*metrics.WorkerUsage, error) {
	return nil, nil
}

// TestWorkloadMetricsLoader_RemoveDuringHistoryLoadDoesNotLeakTicker covers
// the race where removeWorkload observes ticker == nil (because
// startWorkloadMetricsLoading is still blocked in loadHistoryMetricsForWorkload),
// only cancels ctx, and then the history-load returns — previously
// startWorkloadMetricsLoading would proceed to create a fresh time.Ticker
// and start a goroutine, leaking the ticker. Run with -race for the
// ticker-field synchronization check.
func TestWorkloadMetricsLoader_RemoveDuringHistoryLoadDoesNotLeakTicker(t *testing.T) {
	prov := &blockingMetricsProvider{
		historyStarted: make(chan struct{}, 1),
		historyRelease: make(chan struct{}),
	}
	l := newWorkloadMetricsLoader(nil, prov)
	l.setProcessFunc(func(ctx context.Context, state *workload.State) {})

	ws := workload.NewWorkloadState()
	ws.Namespace = "ns"
	ws.Name = "wl"
	ws.HistoryPeriod = time.Hour

	wid := WorkloadID{"ns", "wl"}

	// Manually wire a loaderState that goes into history-load right away.
	loaderCtx, cancel := context.WithCancel(context.Background())
	loaderState := &workloadMetricsState{
		workloadID:         wid,
		state:              ws,
		initialDelay:       0,
		evaluationInterval: 10 * time.Millisecond,
		historyDataPeriod:  time.Hour,
		ctx:                loaderCtx,
		cancel:             cancel,
		firstLoad:          true,
	}
	l.mu.Lock()
	l.workloads[wid] = loaderState
	l.mu.Unlock()

	done := make(chan struct{})
	go func() {
		l.startWorkloadMetricsLoading(loaderState)
		close(done)
	}()

	// Wait until we're inside the history-load (blocked on historyRelease).
	select {
	case <-prov.historyStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("history load never started")
	}

	// removeWorkload runs while history load is still blocking.
	l.removeWorkload(wid)

	// Release history-load; startWorkloadMetricsLoading should now return
	// without creating a ticker because ctx is already cancelled.
	close(prov.historyRelease)

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("startWorkloadMetricsLoading did not return after cancel")
	}

	// Ticker must NOT have been published — startWorkloadMetricsLoading bailed
	// post-history-load on ctx.Err(). If the bug came back, ticker would be
	// a non-nil leaked *time.Ticker.
	l.mu.RLock()
	defer l.mu.RUnlock()
	assert.Nil(t, loaderState.ticker, "ticker must not be allocated after removeWorkload cancelled ctx during history load")
}
