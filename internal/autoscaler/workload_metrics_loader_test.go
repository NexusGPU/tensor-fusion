package autoscaler

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/stretchr/testify/assert"
)

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
