package metrics

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

var errDatabaseDown = errors.New("dial tcp 127.0.0.1:4002: connect: connection refused")

// quickBackoff keeps the retries of these tests in the millisecond range.
func quickBackoff() ConnectBackoff {
	return ConnectBackoff{Initial: time.Millisecond, Max: 4 * time.Millisecond}
}

// The operator starts while the database is still coming up: the first
// attempts fail, a later one succeeds. Nothing is connected in between, and
// the handle is usable from the moment the attempt succeeds.
func TestConnectWithRetryConnectsOnceTheDatabaseAnswers(t *testing.T) {
	reachable, _ := newMockTSDB(t)
	var attempts atomic.Int32
	tsdb := &TimeSeriesDB{}
	tsdb.open = func(GreptimeDBConnection) (*gorm.DB, error) {
		require.False(t, tsdb.Connected(), "not connected before an attempt succeeds")
		if attempts.Add(1) <= 3 {
			return nil, errDatabaseDown
		}
		return reachable.DB, nil
	}

	require.NoError(t, tsdb.ConnectWithRetry(context.Background(), GreptimeDBConnection{}, quickBackoff()))
	require.Equal(t, int32(4), attempts.Load())
	require.True(t, tsdb.Connected())
	require.Same(t, reachable.DB, tsdb.Conn())

	// Connected is final: neither Setup nor another retry opens it again.
	require.NoError(t, tsdb.Setup(GreptimeDBConnection{}))
	require.NoError(t, tsdb.ConnectWithRetry(context.Background(), GreptimeDBConnection{}, quickBackoff()))
	require.Equal(t, int32(4), attempts.Load())
}

// A database that never answers is retried for as long as the operator runs,
// with a wait between the attempts, and is no error for anybody but the
// caller's context.
func TestConnectWithRetryKeepsTryingWhileUnreachable(t *testing.T) {
	var attempts atomic.Int32
	tsdb := &TimeSeriesDB{open: func(GreptimeDBConnection) (*gorm.DB, error) {
		attempts.Add(1)
		return nil, errDatabaseDown
	}}
	ctx, cancel := context.WithTimeout(context.Background(), 80*time.Millisecond)
	defer cancel()

	err := tsdb.ConnectWithRetry(ctx, GreptimeDBConnection{}, quickBackoff())
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.False(t, tsdb.Connected())
	require.Nil(t, tsdb.Conn())
	require.Greater(t, attempts.Load(), int32(3), "it keeps trying")
	// 80ms of waits that grow from 1ms to 4ms: a loop that did not wait would
	// make thousands of attempts.
	require.Less(t, attempts.Load(), int32(90), "it waits between attempts")
}

func TestConnectWithRetryStopsWhenContextIsCancelled(t *testing.T) {
	var attempts atomic.Int32
	firstAttempt := make(chan struct{})
	tsdb := &TimeSeriesDB{open: func(GreptimeDBConnection) (*gorm.DB, error) {
		if attempts.Add(1) == 1 {
			close(firstAttempt)
		}
		return nil, errDatabaseDown
	}}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		// The wait after the first failure would outlast the test.
		done <- tsdb.ConnectWithRetry(ctx, GreptimeDBConnection{}, ConnectBackoff{Initial: time.Hour, Max: time.Hour})
	}()
	<-firstAttempt
	cancel()

	select {
	case err := <-done:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(5 * time.Second):
		t.Fatal("ConnectWithRetry did not return after its context was cancelled")
	}
	require.Equal(t, int32(1), attempts.Load(), "no attempt after cancellation")
	require.False(t, tsdb.Connected())

	// A context that is already over makes no attempt at all.
	require.ErrorIs(t, tsdb.ConnectWithRetry(ctx, GreptimeDBConnection{}, quickBackoff()), context.Canceled)
	require.Equal(t, int32(1), attempts.Load())
}

// Retrying for minutes must not fill the log: one line per change of state.
func TestRetryUntilConnectedLogsStateChangesOnly(t *testing.T) {
	var lines []string
	backoff := quickBackoff()
	backoff.Logger = funcr.New(func(_, args string) { lines = append(lines, args) }, funcr.Options{})

	failures := []error{errDatabaseDown, errDatabaseDown, errDatabaseDown, errDatabaseDown,
		errors.New("unknown database"), errors.New("unknown database")}
	attempt := 0
	require.NoError(t, RetryUntilConnected(context.Background(), "metrics store", backoff, func() error {
		attempt++
		if attempt <= len(failures) {
			return failures[attempt-1]
		}
		return nil
	}))
	require.Equal(t, 7, attempt)
	require.Len(t, lines, 3, "first failure, the failure for another reason, the recovery: %v", lines)
	require.Contains(t, lines[0], "connection refused")
	require.Contains(t, lines[1], "unknown database")
	require.Contains(t, lines[2], "metrics store is reachable now")

	// Reachable at the first attempt: nothing changed, nothing to say.
	lines = nil
	require.NoError(t, RetryUntilConnected(context.Background(), "metrics store", backoff, func() error { return nil }))
	require.Empty(t, lines)

	// Never reachable: the failure once, and that it gave up.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	require.Error(t, RetryUntilConnected(ctx, "metrics store", backoff, func() error { return errDatabaseDown }))
	require.Len(t, lines, 2, "%v", lines)
	require.True(t, strings.Contains(lines[1], "never reachable"), lines[1])
}

// What the users of the handle get while it is not connected: an error or an
// empty answer, never a nil dereference.
func TestUnconnectedDatabaseAnswersWithoutData(t *testing.T) {
	for name, tsdb := range map[string]*TimeSeriesDB{"no handle": nil, "handle not connected yet": {}} {
		t.Run(name, func(t *testing.T) {
			require.Nil(t, tsdb.Conn())
			require.False(t, tsdb.Connected())
			require.Equal(t, DefaultTimeIndexColumn, tsdb.TimeIndexColumn(context.Background(), "tf_gpu_usage"))
			require.NoError(t, tsdb.SetTableTTL("30d"), "nothing to alter yet")
			rows, err := tsdb.FindRecentNodeMetrics()
			require.ErrorIs(t, err, ErrNotConnected)
			require.Empty(t, rows)
			require.ErrorIs(t, tsdb.SetupTables(nil), ErrNotConnected)
		})
	}
}

// The connection is set by the background connect while consumers are already
// asking for it; run with -race.
func TestConnIsSafeWhileConnecting(t *testing.T) {
	reachable, _ := newMockTSDB(t)
	var attempts atomic.Int32
	tsdb := &TimeSeriesDB{}
	tsdb.open = func(GreptimeDBConnection) (*gorm.DB, error) {
		if attempts.Add(1) <= 5 {
			return nil, errDatabaseDown
		}
		return reachable.DB, nil
	}

	stop := make(chan struct{})
	var readers sync.WaitGroup
	for range 4 {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for {
				select {
				case <-stop:
					return
				default:
					_ = tsdb.Conn()
					_ = tsdb.SetTableTTL("")
					_, _ = tsdb.FindRecentNodeMetrics()
				}
			}
		}()
	}
	require.NoError(t, tsdb.ConnectWithRetry(context.Background(), GreptimeDBConnection{}, quickBackoff()))
	close(stop)
	readers.Wait()
	require.True(t, tsdb.Connected())
}
