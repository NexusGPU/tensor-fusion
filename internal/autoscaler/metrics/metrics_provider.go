package metrics

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"gorm.io/gorm"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	defaultQueryTimeout        time.Duration = 15 * time.Second
	defaultHistoryQueryTimeout time.Duration = 60 * time.Second
)

type WorkerUsage struct {
	Namespace    string
	WorkloadName string
	// WorkerName identifies the worker Pod as the usage rows do: by Pod name, or
	// by Pod UID (the hypervisor tags its rows with the UID).
	WorkerName string
	// TflopsUsage is the worker's TFLOPS in use per GPU, in the unit of TFLOPS
	// requests and limits: its utilization share of a GPU times that GPU's
	// TFLOPS capacity. It is meaningless when TflopsUnknown is set.
	TflopsUsage float64
	// TflopsUnknown is set when the interval has no measured compute figure: no
	// usage row carried one, or the only figure is the zero that releases before
	// the hypervisor computed it wrote next to a utilization above zero (see
	// workerUsageRow.measuredTflops). That is not zero usage and must not be
	// sampled as such.
	TflopsUnknown bool
	VramUsage     uint64
	Timestamp     time.Time
}

// workerUsageRow is one aggregated row of tf_worker_usage. The hypervisor leaves
// compute_tflops out of a row when the utilization was not measured, so the
// aggregate is NULL when no row of the group had it. "ts" and "time_window" are
// aliases the queries give to aggregates of the table's time index, whatever
// that column is called in the database.
type workerUsageRow struct {
	Namespace     string   `gorm:"column:namespace"`
	WorkloadName  string   `gorm:"column:workload"`
	WorkerName    string   `gorm:"column:worker"`
	ComputeTflops *float64 `gorm:"column:compute_tflops"`
	// ComputePercentage is the largest utilization (0-100) recorded in the
	// group; nil when no row carried one or the column was not selected.
	ComputePercentage *float64  `gorm:"column:compute_percentage"`
	VRAMBytes         uint64    `gorm:"column:memory_bytes"`
	Timestamp         time.Time `gorm:"column:ts"`
	TimeWindow        time.Time `gorm:"column:time_window"`
}

func (workerUsageRow) TableName() string {
	return metrics.HypervisorWorkerUsageMetrics{}.TableName()
}

// measuredTflops returns the compute figure of the group and whether it is a
// measurement. It is not one when
//
//   - no row of the group carried compute_tflops (the aggregate is NULL), or
//   - the figure is zero while a utilization above zero was recorded. Releases
//     before the hypervisor computed the figure wrote compute_tflops = 0 into
//     every row, next to the real compute_percentage; such a zero says nothing
//     about the usage.
//
// The rule is applied to the max() aggregates of both columns, and aggregating
// can not turn a row of the second kind into a measurement: a current
// hypervisor writes a zero figure only together with a zero utilization, so a
// group whose aggregates are (0, >0) holds no figure above zero and at least
// one row that is not a measurement, and is dropped as a whole; and a figure
// above zero is a real one and is the maximum whatever else the group holds.
// A zero figure with a zero or missing utilization can not be told apart from
// a measured idle interval here; WorkerUsageSampler ignores those until the
// worker has reported a figure above zero.
func (r *workerUsageRow) measuredTflops() (float64, bool) {
	if r.ComputeTflops == nil {
		return 0, false
	}
	if *r.ComputeTflops == 0 && r.ComputePercentage != nil && *r.ComputePercentage > 0 {
		return 0, false
	}
	return *r.ComputeTflops, true
}

func (r *workerUsageRow) toWorkerUsage(timestamp time.Time) *WorkerUsage {
	tflops, measured := r.measuredTflops()
	return &WorkerUsage{
		Namespace:     r.Namespace,
		WorkloadName:  r.WorkloadName,
		WorkerName:    r.WorkerName,
		TflopsUsage:   tflops,
		TflopsUnknown: !measured,
		VramUsage:     r.VRAMBytes,
		Timestamp:     timestamp,
	}
}

type Provider interface {
	// Deprecated, for test only
	GetWorkersMetrics(context.Context) ([]*WorkerUsage, error)

	// Per-workload metrics queries
	GetWorkloadHistoryMetrics(ctx context.Context, namespace, workloadName string, startTime, endTime time.Time) ([]*WorkerUsage, error)
	GetWorkloadRealtimeMetrics(ctx context.Context, namespace, workloadName string, startTime, endTime time.Time) ([]*WorkerUsage, error)
}

type greptimeDBProvider struct {
	db            *gorm.DB
	lastQueryTime time.Time
	// timeIndex tells what the time index of the usage table is called in this
	// database; nil means metrics.DefaultTimeIndexColumn.
	timeIndex *metrics.TimeIndexResolver
	// reducedColumns is the index in usageColumnSets of the column set the last
	// usage query was answered with, to report a change of it once.
	reducedColumns atomic.Int32
	// historyLength     time.Duration
	// historyResolution time.Duration
}

// NewProviderWithDB returns a Provider that queries the given connection.
func NewProviderWithDB(db *gorm.DB) Provider {
	return newGreptimeDBProvider(db)
}

func NewProvider() (Provider, error) {
	tsdb, err := setupTimeSeriesDB()
	if err != nil {
		return nil, err
	}
	return newGreptimeDBProvider(tsdb.DB), nil
}

func newGreptimeDBProvider(db *gorm.DB) *greptimeDBProvider {
	return &greptimeDBProvider{db: db, timeIndex: metrics.NewTimeIndexResolver(db)}
}

const (
	// compute_percentage is selected next to compute_tflops to recognize the
	// zero figures of earlier releases, see workerUsageRow.measuredTflops.
	usageColumns = "namespace, workload, worker, max(compute_tflops) as compute_tflops, max(compute_percentage) as compute_percentage, max(memory_bytes) as memory_bytes"
	// A time bound is passed as a timestamp string, see metrics.TimeFilterArg.
	usageTimeRange = "%[1]s > ? and %[1]s <= ?"
)

// usageColumnSets are the column lists a usage query is tried with, in order.
// The usage table is created by the database from the first rows written to
// it and has a column only once a row carried the field, so on a cluster whose
// utilization was never measured the compute columns do not exist and a query
// naming them fails. The query is then repeated without them: the compute
// usage is unknown, memory usage is still read.
var usageColumnSets = []string{
	usageColumns,
	"namespace, workload, worker, max(compute_tflops) as compute_tflops, max(memory_bytes) as memory_bytes",
	"namespace, workload, worker, max(memory_bytes) as memory_bytes",
}

// usageQueryParts are the parts of a usage query that name the time index.
type usageQueryParts struct {
	// latest selects, per worker, the aggregates and the time of the last row.
	latest string
	// perMinute selects the aggregates per worker and minute.
	perMinute string
	// timeRange filters on the half-open interval (start, end].
	timeRange string
}

func usageQueryPartsFor(timeColumn string) usageQueryParts {
	return usageQueryPartsWith(usageColumns, timeColumn)
}

func usageQueryPartsWith(columns, timeColumn string) usageQueryParts {
	return usageQueryParts{
		latest:    fmt.Sprintf("%s, max(%s) as ts", columns, timeColumn),
		perMinute: fmt.Sprintf("%s, date_bin('1 minute'::INTERVAL, %s) as time_window", columns, timeColumn),
		timeRange: fmt.Sprintf(usageTimeRange, timeColumn),
	}
}

// usageQuery describes one usage query: the aggregates per worker over the
// interval (start, end], or per worker and minute of it.
type usageQuery struct {
	perMinute  bool
	start, end time.Time
	// filter and filterArgs narrow the rows further; filter starts with " and ".
	filter     string
	filterArgs []any
}

// findUsageRows runs the query, with fewer columns when the table lacks some
// (see usageColumnSets). The error returned is the one of the full query.
func (g *greptimeDBProvider) findUsageRows(ctx context.Context, query usageQuery) ([]*workerUsageRow, error) {
	timeColumn := g.timeIndex.Column(ctx, workerUsageRow{}.TableName())
	args := append([]any{metrics.TimeFilterArg(query.start), metrics.TimeFilterArg(query.end)}, query.filterArgs...)

	var firstErr error
	for i, columns := range usageColumnSets {
		parts := usageQueryPartsWith(columns, timeColumn)
		selected, group, order := parts.latest, "namespace, workload, worker", "ts asc"
		if query.perMinute {
			selected, group, order = parts.perMinute, "namespace, workload, worker, time_window", "time_window asc"
		}
		data := []*workerUsageRow{}
		err := g.db.WithContext(ctx).
			Select(selected).
			Where(parts.timeRange+query.filter, args...).
			Group(group).
			Order(order).
			Find(&data).
			Error
		if err == nil {
			if previous := g.reducedColumns.Swap(int32(i)); int(previous) != i {
				log.FromContext(ctx).Info("worker usage is read with a different set of columns",
					"columns", columns, "errorOfFullQuery", firstErr)
			}
			return data, nil
		}
		if firstErr == nil {
			firstErr = err
		}
		// A cancelled or timed out query fails the same way with any columns.
		if ctx.Err() != nil {
			break
		}
	}
	return nil, firstErr
}

func (g *greptimeDBProvider) GetWorkersMetrics(ctx context.Context) ([]*WorkerUsage, error) {
	now := time.Now()

	timeoutCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	if g.lastQueryTime.IsZero() {
		g.lastQueryTime = now.Add(-time.Minute)
	}

	// actual meaning:  max(avg[10s])[1m]
	data, err := g.findUsageRows(timeoutCtx, usageQuery{start: g.lastQueryTime, end: now})
	if err != nil {
		return nil, err
	}

	g.lastQueryTime = now

	workersMetrics := make([]*WorkerUsage, 0, len(data))
	for _, row := range data {
		workersMetrics = append(workersMetrics, row.toWorkerUsage(row.Timestamp))
	}

	return workersMetrics, nil
}

type hypervisorWorkerUsageMetrics struct {
	metrics.HypervisorWorkerUsageMetrics
	TimeWindow time.Time `gorm:"column:time_window;index:,class:TIME"`
}

// Deprecated
func (g *greptimeDBProvider) GetHistoryMetrics(ctx context.Context) ([]*WorkerUsage, error) {
	now := time.Now()

	timeoutCtx, cancel := context.WithTimeout(ctx, defaultHistoryQueryTimeout)
	defer cancel()

	data, err := g.findUsageRows(timeoutCtx, usageQuery{perMinute: true, start: now.Add(-time.Hour * 24), end: now})
	if err != nil {
		return nil, err
	}

	g.lastQueryTime = now

	workersMetrics := make([]*WorkerUsage, 0, len(data))
	for _, row := range data {
		workersMetrics = append(workersMetrics, row.toWorkerUsage(row.TimeWindow))
	}

	return workersMetrics, nil
}

// Setup GreptimeDB connection
func setupTimeSeriesDB() (*metrics.TimeSeriesDB, error) {
	timeSeriesDB := &metrics.TimeSeriesDB{}
	connection := metrics.GreptimeDBConnection{
		Host:     utils.GetEnvOrDefault("TSDB_MYSQL_HOST", "127.0.0.1"),
		Port:     utils.GetEnvOrDefault("TSDB_MYSQL_PORT", "4002"),
		User:     utils.GetEnvOrDefault("TSDB_MYSQL_USER", "root"),
		Password: utils.GetEnvOrDefault("TSDB_MYSQL_PASSWORD", ""),
		Database: utils.GetEnvOrDefault("TSDB_MYSQL_DATABASE", "public"),
	}
	if err := timeSeriesDB.Setup(connection); err != nil {
		return nil, err
	}
	return timeSeriesDB, nil
}

const workloadFilter = " and namespace = ? and workload = ?"

func (g *greptimeDBProvider) GetWorkloadHistoryMetrics(ctx context.Context, namespace, workloadName string, startTime, endTime time.Time) ([]*WorkerUsage, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, defaultHistoryQueryTimeout)
	defer cancel()

	data, err := g.findUsageRows(timeoutCtx, usageQuery{
		perMinute: true, start: startTime, end: endTime,
		filter: workloadFilter, filterArgs: []any{namespace, workloadName},
	})
	if err != nil {
		return nil, err
	}

	workersMetrics := make([]*WorkerUsage, 0, len(data))
	for _, row := range data {
		workersMetrics = append(workersMetrics, row.toWorkerUsage(row.TimeWindow))
	}

	return workersMetrics, nil
}

func (g *greptimeDBProvider) GetWorkloadRealtimeMetrics(ctx context.Context, namespace, workloadName string, startTime, endTime time.Time) ([]*WorkerUsage, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	data, err := g.findUsageRows(timeoutCtx, usageQuery{
		start: startTime, end: endTime,
		filter: workloadFilter, filterArgs: []any{namespace, workloadName},
	})
	if err != nil {
		return nil, err
	}

	workersMetrics := make([]*WorkerUsage, 0, len(data))
	for _, row := range data {
		workersMetrics = append(workersMetrics, row.toWorkerUsage(row.Timestamp))
	}

	return workersMetrics, nil
}
