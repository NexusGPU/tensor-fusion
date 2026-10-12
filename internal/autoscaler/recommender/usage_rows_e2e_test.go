package recommender

import (
	"context"
	"math"
	"regexp"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/stretchr/testify/require"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// The path from usage rows to a recommendation, with rows shaped the way the
// hypervisor writes them into tf_worker_usage: one row per worker and GPU a
// minute, the worker identified by Pod UID, compute_tflops being the worker's
// utilization share of the GPU times the GPU's TFLOPS capacity, and left out
// when the utilization was not measured.

const usageAggregates = "SELECT namespace, workload, worker, max(compute_tflops) as compute_tflops, " +
	"max(compute_percentage) as compute_percentage, max(memory_bytes) as memory_bytes, "

const usageWorkloadRange = "FROM `tf_worker_usage` " +
	"WHERE greptime_timestamp > ? and greptime_timestamp <= ? and namespace = ? and workload = ? "

const usageHistoryQuery = usageAggregates + "date_bin('1 minute'::INTERVAL, greptime_timestamp) as time_window " +
	usageWorkloadRange + "GROUP BY namespace, workload, worker, time_window ORDER BY time_window asc"

const usageRealtimeQuery = usageAggregates + "max(greptime_timestamp) as ts " +
	usageWorkloadRange + "GROUP BY namespace, workload, worker ORDER BY ts asc"

// usageRow is one aggregated minute of one worker. A nil tflops is a NULL
// compute_tflops: no row of that minute carried the field; likewise percent
// and compute_percentage, the worker's utilization of the GPU (0-100).
type usageRow struct {
	workerUID   string
	minutesAgo  int
	tflops      *float64
	memoryBytes uint64
	percent     *float64
}

// earlierReleaseRow is a minute written before the hypervisor computed
// compute_tflops: the field is 0 whatever the utilization was.
func earlierReleaseRow(workerUID string, minutesAgo int, percent float64) usageRow {
	zero := 0.0
	return usageRow{workerUID: workerUID, minutesAgo: minutesAgo, tflops: &zero, memoryBytes: 6 * gib, percent: &percent}
}

// measuredRow is a minute of a current hypervisor on an RTX 3090 (71.2 TFLOPS).
func measuredRow(workerUID string, minutesAgo int, percent float64) usageRow {
	tflops := percent / 100 * 71.2
	return usageRow{workerUID: workerUID, minutesAgo: minutesAgo, tflops: &tflops, memoryBytes: 6 * gib, percent: &percent}
}

func nullable(value *float64) any {
	if value == nil {
		return nil
	}
	return *value
}

// loadUsageRows feeds the rows to the workload as its usage history.
func loadUsageRows(t *testing.T, state *workload.State, rows []usageRow) {
	t.Helper()
	queryUsageRows(t, state, rows, true)
}

// loadRealtimeUsageRows feeds the rows to the workload the way the periodic
// query does: one row per worker for the interval since the previous query.
func loadRealtimeUsageRows(t *testing.T, state *workload.State, rows []usageRow) {
	t.Helper()
	queryUsageRows(t, state, rows, false)
}

func queryUsageRows(t *testing.T, state *workload.State, rows []usageRow, history bool) {
	t.Helper()
	conn, mock, err := sqlmock.New()
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	db, err := gorm.Open(mysql.New(mysql.Config{Conn: conn, SkipInitializeWithVersion: true}), &gorm.Config{})
	require.NoError(t, err)

	now := time.Now()
	timeColumn, query := "ts", usageRealtimeQuery
	if history {
		timeColumn, query = "time_window", usageHistoryQuery
	}
	result := sqlmock.NewRows([]string{"namespace", "workload", "worker", "compute_tflops", "compute_percentage", "memory_bytes", timeColumn})
	for _, row := range rows {
		result.AddRow("default", "inference", row.workerUID, nullable(row.tflops), nullable(row.percent), row.memoryBytes,
			now.Add(-time.Duration(row.minutesAgo)*time.Minute))
	}
	mock.ExpectQuery(regexp.QuoteMeta(query)).WillReturnRows(result)

	provider := metrics.NewProviderWithDB(db)
	var samples []*metrics.WorkerUsage
	if history {
		samples, err = provider.GetWorkloadHistoryMetrics(context.Background(), "default", "inference", now.Add(-2*time.Hour), now)
	} else {
		samples, err = provider.GetWorkloadRealtimeMetrics(context.Background(), "default", "inference", now.Add(-time.Minute), now)
	}
	require.NoError(t, err)
	require.Len(t, samples, len(rows))
	for _, sample := range samples {
		state.AddSample(sample)
	}
	require.NoError(t, mock.ExpectationsWereMet())
}

// settle makes the recommender's changes one after the other, as the
// autoscaler does over its evaluations (a change is bounded, see bounds.go),
// until the recommender has nothing left to change. It returns the resources
// the workload ends up with and the number of changes that took.
func settle(t *testing.T, state *workload.State) (tfv1.Resources, int) {
	t.Helper()
	recommender := NewPercentileRecommender(nil)
	for changes := 0; changes < 50; changes++ {
		got, err := recommender.Recommend(context.Background(), state)
		require.NoError(t, err)
		require.NotNil(t, got)
		if got.HasApplied {
			return *state.GetCurrentResourcesSpec(), changes
		}
		state.SetRecommendation(&got.Resources)
	}
	t.Fatal("the recommendation did not settle")
	return tfv1.Resources{}, 0
}

func tflopsSampleCount(state *workload.State) int {
	count := 0
	state.ConsumeAggregator(func(aggregator *metrics.WorkerUsageAggregator) { count = aggregator.TotalSamplesCount })
	return count
}

func newUsageTestWorkload(current tfv1.Resources) *workload.State {
	state := workload.NewWorkloadState()
	state.CreationTimestamp = metav1.NewTime(time.Now().Add(-3 * time.Hour))
	state.Spec.AutoScalingConfig.AutoSetResources = &tfv1.AutoSetResources{Enable: true}
	state.Spec.Resources = current
	state.CurrentActiveWorkers = map[string]*corev1.Pod{
		"inference-0": {ObjectMeta: metav1.ObjectMeta{Name: "inference-0", UID: types.UID("uid-0")}},
		"inference-1": {ObjectMeta: metav1.ObjectMeta{Name: "inference-1", UID: types.UID("uid-1")}},
	}
	return state
}

const gib = uint64(1) << 30

func TestRecommendationFromHypervisorUsageRows(t *testing.T) {
	current := tfv1.Resources{
		Requests: tfv1.Resource{Tflops: resource.MustParse("10"), Vram: resource.MustParse("4Gi")},
		Limits:   tfv1.Resource{Tflops: resource.MustParse("20"), Vram: resource.MustParse("8Gi")},
	}

	t.Run("measured usage drives the compute recommendation", func(t *testing.T) {
		state := newUsageTestWorkload(current)
		// Two workers on RTX 3090s (71.2 FP16 TFLOPS) using about 40% of a GPU:
		// 0.40 x 71.2 = 28.5 TFLOPS, with a ripple, peaking near 45% (32 TFLOPS).
		rows := make([]usageRow, 0, 240)
		for minute := 119; minute >= 0; minute-- {
			for i, uid := range []string{"uid-0", "uid-1"} {
				share := 0.40 + 0.05*math.Sin(float64(minute+17*i)/9)
				rows = append(rows, measuredRow(uid, minute, share*100))
			}
		}
		loadUsageRows(t, state, rows)

		// Rows keyed by Pod UID land in the samplers of the Pods they belong to.
		require.Len(t, state.WorkerUsageSamplers, 2)
		require.Contains(t, state.WorkerUsageSamplers, "inference-0")
		require.Contains(t, state.WorkerUsageSamplers, "inference-1")

		got, changes := settle(t, state)
		// Medium QoS: request = median usage + 15% margin, limit = p99 + margin,
		// within the 5% steps of the histogram: about 0.40 x 71.2 x 1.15 = 32.8 and
		// 0.45 x 71.2 x 1.15 = 36.8.
		request, limit := got.Requests.Tflops.AsApproximateFloat64(), got.Limits.Tflops.AsApproximateFloat64()
		require.InDelta(t, 32.8, request, 3, "TFLOPS request")
		require.InDelta(t, 36.8, limit, 3.5, "TFLOPS limit")
		require.GreaterOrEqual(t, limit, request)
		// 6 GiB in use with the same margin.
		require.InDelta(t, 6.9*float64(gib), got.Requests.Vram.AsApproximateFloat64(), 0.5*float64(gib))
		// From 10 TFLOPS that is more than one bounded change away.
		require.Greater(t, changes, 1)
	})

	t.Run("unmeasured utilization leaves compute alone", func(t *testing.T) {
		state := newUsageTestWorkload(current)
		// The provider could not supply utilization: the hypervisor wrote the rows
		// without compute_tflops. Memory was measured as usual.
		rows := make([]usageRow, 0, 240)
		for minute := 119; minute >= 0; minute-- {
			for _, uid := range []string{"uid-0", "uid-1"} {
				rows = append(rows, usageRow{workerUID: uid, minutesAgo: minute, memoryBytes: 6 * gib})
			}
		}
		loadUsageRows(t, state, rows)

		var tflopsSamples, vramEmpty bool
		state.ConsumeAggregator(func(aggregator *metrics.WorkerUsageAggregator) {
			tflopsSamples, vramEmpty = !aggregator.TflopsHistogram.IsEmpty(), aggregator.VramHistogram.IsEmpty()
		})
		require.False(t, tflopsSamples, "an unmeasured utilization must not be sampled as zero usage")
		require.False(t, vramEmpty)

		got, _ := settle(t, state)
		require.True(t, got.Requests.Tflops.Equal(current.Requests.Tflops),
			"TFLOPS request changed to %s without a single usage sample", got.Requests.Tflops.String())
		require.True(t, got.Limits.Tflops.Equal(current.Limits.Tflops))
		require.InDelta(t, 6.9*float64(gib), got.Requests.Vram.AsApproximateFloat64(), 0.5*float64(gib))
	})

	t.Run("a measured idle worker is scaled down, within the allowed ratio", func(t *testing.T) {
		state := newUsageTestWorkload(current)
		// The worker did some work two hours ago and has been idle since: a zero
		// figure next to a zero utilization, after a real figure, is a measurement.
		rows := []usageRow{measuredRow("uid-0", 119, 40)}
		for minute := 118; minute >= 0; minute-- {
			rows = append(rows, measuredRow("uid-0", minute, 0))
		}
		loadUsageRows(t, state, rows)
		require.Equal(t, 120, tflopsSampleCount(state))

		got, changes := settle(t, state)
		// Not below a tenth of the original request, and never below 1 TFLOPS.
		require.Equal(t, "1", got.Requests.Tflops.String())
		// The compute limit is never lowered below the original.
		require.True(t, got.Limits.Tflops.Equal(current.Limits.Tflops))
		// At most a quarter per change: 10, 8, 6, 5, 4, 3, 2, 1.
		require.GreaterOrEqual(t, changes, 7)
	})

	t.Run("mixed minutes: only measured ones are sampled", func(t *testing.T) {
		state := newUsageTestWorkload(current)
		measured := 0.5 * 71.2
		rows := make([]usageRow, 0, 120)
		for minute := 119; minute >= 0; minute-- {
			row := usageRow{workerUID: "uid-0", minutesAgo: minute, memoryBytes: 6 * gib}
			if minute%2 == 0 {
				row.tflops = &measured
			}
			rows = append(rows, row)
		}
		loadUsageRows(t, state, rows)

		got, _ := settle(t, state)
		require.InDelta(t, measured*1.15, got.Requests.Tflops.AsApproximateFloat64(), 3.5,
			"the unmeasured minutes must not halve the estimate")
	})
}

// Releases before the hypervisor computed compute_tflops wrote the field as 0
// into every usage row. Such rows are still in the database after an upgrade
// and must not be read as "the worker used nothing".
func TestRecommendationIgnoresZeroFiguresOfEarlierReleases(t *testing.T) {
	// The reported case: a Medium-QoS workload requesting 30 TFLOPS.
	current := tfv1.Resources{
		Requests: tfv1.Resource{Tflops: resource.MustParse("30"), Vram: resource.MustParse("4Gi")},
		Limits:   tfv1.Resource{Tflops: resource.MustParse("40"), Vram: resource.MustParse("8Gi")},
	}
	// Two hours of an earlier release: busy minutes (utilization above zero)
	// and idle ones (utilization zero), all with a zero figure.
	earlierReleaseHistory := func(fromMinute, toMinute int) []usageRow {
		rows := []usageRow{}
		for minute := fromMinute; minute >= toMinute; minute-- {
			for _, uid := range []string{"uid-0", "uid-1"} {
				percent := 0.0
				if minute%3 != 0 {
					percent = 42
				}
				rows = append(rows, earlierReleaseRow(uid, minute, percent))
			}
		}
		return rows
	}

	t.Run("a workload with only such rows keeps its compute resources", func(t *testing.T) {
		state := newUsageTestWorkload(current)
		loadUsageRows(t, state, earlierReleaseHistory(119, 0))
		require.Zero(t, tflopsSampleCount(state), "a zero figure of an earlier release is not a compute sample")

		got, _ := settle(t, state)
		// Sampled as usage, the zeros put the request at a tenth of the original: 3.
		require.Equal(t, "30", got.Requests.Tflops.String())
		require.Equal(t, "40", got.Limits.Tflops.String())
		// Memory was measured all along.
		require.InDelta(t, 6.9*float64(gib), got.Requests.Vram.AsApproximateFloat64(), 0.5*float64(gib))
	})

	t.Run("real figures after such rows are the whole compute history", func(t *testing.T) {
		state := newUsageTestWorkload(current)
		rows := earlierReleaseHistory(119, 30)
		// The hypervisors were upgraded half an hour ago: about 40% of an RTX 3090.
		for minute := 29; minute >= 0; minute-- {
			for i, uid := range []string{"uid-0", "uid-1"} {
				rows = append(rows, measuredRow(uid, minute, 40+2*math.Sin(float64(minute+5*i)/4)))
			}
		}
		loadUsageRows(t, state, rows)
		require.Equal(t, 60, tflopsSampleCount(state), "one compute sample per worker and measured minute")

		got, _ := settle(t, state)
		// Median of the real figures plus the margin: 0.40 x 71.2 x 1.15 = 32.8.
		// With the ninety minutes of zeros sampled the median would be zero.
		require.InDelta(t, 32.8, got.Requests.Tflops.AsApproximateFloat64(), 3, "TFLOPS request")
		require.GreaterOrEqual(t, got.Limits.Tflops.AsApproximateFloat64(), 40.0)
	})

	t.Run("history before a worker's first real figure is ignored per worker", func(t *testing.T) {
		state := newUsageTestWorkload(current)
		rows := []usageRow{}
		for minute := 119; minute >= 0; minute-- {
			// uid-0 was upgraded an hour ago and went idle 20 minutes ago.
			switch {
			case minute >= 60:
				rows = append(rows, earlierReleaseRow("uid-0", minute, 0))
			case minute >= 20:
				rows = append(rows, measuredRow("uid-0", minute, 40))
			default:
				rows = append(rows, measuredRow("uid-0", minute, 0))
			}
			// uid-1 has not reported a real figure yet.
			rows = append(rows, earlierReleaseRow("uid-1", minute, 0))
		}
		loadUsageRows(t, state, rows)
		// uid-0: 40 busy minutes and 20 measured idle ones; uid-1: nothing.
		require.Equal(t, 60, tflopsSampleCount(state))
		require.True(t, state.WorkerUsageSamplers["inference-0"].TflopsMeasured)
		require.False(t, state.WorkerUsageSamplers["inference-1"].TflopsMeasured)
	})

	t.Run("the periodic query follows the same rule", func(t *testing.T) {
		state := newUsageTestWorkload(current)
		loadUsageRows(t, state, earlierReleaseHistory(119, 2))
		require.Zero(t, tflopsSampleCount(state))

		// Still an earlier release, and idle: zero figure, zero utilization.
		loadRealtimeUsageRows(t, state, []usageRow{earlierReleaseRow("uid-0", 1, 0), earlierReleaseRow("uid-1", 1, 0)})
		require.Zero(t, tflopsSampleCount(state))
		got, _ := settle(t, state)
		require.Equal(t, "30", got.Requests.Tflops.String())

		// uid-0 reports its first real figure, then an idle interval.
		loadRealtimeUsageRows(t, state, []usageRow{measuredRow("uid-0", 1, 35), earlierReleaseRow("uid-1", 1, 0)})
		require.Equal(t, 1, tflopsSampleCount(state))
		loadRealtimeUsageRows(t, state, []usageRow{measuredRow("uid-0", 0, 0), earlierReleaseRow("uid-1", 0, 0)})
		require.Equal(t, 2, tflopsSampleCount(state), "a zero after a real figure is a measured idle interval")
	})
}
