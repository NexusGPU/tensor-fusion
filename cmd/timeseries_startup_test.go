package main

import (
	"context"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// closedLocalPort returns a port of this host that nothing listens on.
func closedLocalPort(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	_, port, err := net.SplitHostPort(listener.Addr().String())
	require.NoError(t, err)
	require.NoError(t, listener.Close())
	return port
}

// At install the operator starts before GreptimeDB accepts connections. The
// start-up code gets a handle at once, without a connection and without
// waiting for one; the connect goes on in the background.
func TestSetupTimeSeriesDBDoesNotNeedAReachableDatabase(t *testing.T) {
	t.Setenv("TSDB_MYSQL_HOST", "127.0.0.1")
	t.Setenv("TSDB_MYSQL_PORT", closedLocalPort(t))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	started := time.Now()
	tsdb := setupTimeSeriesDB(ctx)
	require.Less(t, time.Since(started), 2*time.Second, "start-up does not wait for the database")
	require.NotNil(t, tsdb, "consumers get a handle, not nil")
	require.False(t, tsdb.Connected())
	require.NoError(t, tsdb.SetTableTTL("30d"), "a TTL update before the connection is skipped, not a failure")

	// The first attempt has failed by now and the retry is waiting.
	time.Sleep(200 * time.Millisecond)
	require.False(t, tsdb.Connected())
}
