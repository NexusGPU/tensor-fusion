package metrics

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// limiterEventTestCount is one counter of the limiter events endpoint as its
// handler reports it: per Pod UID and kind.
type limiterEventTestCount struct {
	namespace, pod, kind string
	count                uint64
}

// recordLimiterEvents writes one record and returns its rows keyed by
// "namespace/pod/kind", with the count of each.
func recordLimiterEvents(t *testing.T, recorder *HypervisorMetricsRecorder) map[string]float64 {
	t.Helper()
	var out bytes.Buffer
	recorder.RecordLimiterEventMetrics(&out)
	rows := map[string]float64{}
	for _, line := range parseUsageLines(t, out.Bytes()) {
		require.Equal(t, "tf_limiter_events", line.measurement)
		require.Equal(t, map[string]string{
			"node": "node-1", "pool": "pool-a",
			"namespace": line.tags["namespace"], "pod": line.tags["pod"], "kind": line.tags["kind"],
		}, line.tags, "tags of a tf_limiter_events row")
		require.Len(t, line.fields, 1)
		key := line.tags["namespace"] + "/" + line.tags["pod"] + "/" + line.tags["kind"]
		require.NotContains(t, rows, key, "two rows of one series in one record")
		rows[key] = line.fields["count"]
	}
	// The count is written as an integer field.
	if out.Len() > 0 {
		require.Contains(t, out.String(), " count=")
		for _, line := range strings.Split(strings.TrimSpace(out.String()), "\n") {
			require.Regexp(t, ` count=\d+i `, line)
		}
	}
	return rows
}

// The counters of the limiter events endpoint are exported as cumulative
// counts per namespace, Pod and kind, and a row is only written when its count
// changed since it was written last.
func TestLimiterEventRowsCarryCumulativeCountsAndOnlyChanges(t *testing.T) {
	recorder := &HypervisorMetricsRecorder{nodeName: "node-1", gpuPool: "pool-a"}
	// Before the server exists there is nothing to export.
	require.Empty(t, recordLimiterEvents(t, recorder))

	counts := []limiterEventTestCount{
		{"team-a", "pod-0", "stalled_bucket", 3},
		{"team-a", "pod-0", "clamped_charge", 1},
		{"team-b", "pod-1", "other", 2},
	}
	recorder.SetLimiterEventSource(func(yield func(namespace, pod, kind string, count uint64)) {
		for _, c := range counts {
			yield(c.namespace, c.pod, c.kind, c.count)
		}
	})
	require.Equal(t, map[string]float64{
		"team-a/pod-0/stalled_bucket": 3, "team-a/pod-0/clamped_charge": 1, "team-b/pod-1/other": 2,
	}, recordLimiterEvents(t, recorder))

	// Nothing new was reported: nothing is written.
	require.Empty(t, recordLimiterEvents(t, recorder))

	// One counter moved: its row carries the count since the start, not the increase.
	counts[0].count = 8
	require.Equal(t, map[string]float64{"team-a/pod-0/stalled_bucket": 8}, recordLimiterEvents(t, recorder))

	// A Pod replaced by one of the same name while the first is still
	// remembered shares its row: one series, the counts added.
	counts = append(counts, limiterEventTestCount{"team-b", "pod-1", "other", 5})
	require.Equal(t, map[string]float64{"team-b/pod-1/other": 7}, recordLimiterEvents(t, recorder))

	// The endpoint forgot a Pod: nothing is written for it, and nothing is
	// remembered about it either.
	counts = counts[:2]
	require.Empty(t, recordLimiterEvents(t, recorder))
	require.Len(t, recorder.limiterEventsWritten, 2)
	// When it reports again its counter starts over, which is written although
	// the count is one that was written before: a reset, as for any counter.
	counts = append(counts, limiterEventTestCount{"team-b", "pod-1", "other", 7})
	require.Equal(t, map[string]float64{"team-b/pod-1/other": 7}, recordLimiterEvents(t, recorder))

	// Rows that could not be written are written with the next record.
	counts[1].count = 4
	recorder.RecordLimiterEventMetrics(&failingWriter{err: errors.New("no space left on device")})
	require.Equal(t, map[string]float64{"team-a/pod-0/clamped_charge": 4}, recordLimiterEvents(t, recorder))

	// Without a source nothing is written.
	recorder.SetLimiterEventSource(nil)
	counts[0].count = 9
	require.Empty(t, recordLimiterEvents(t, recorder))
}
