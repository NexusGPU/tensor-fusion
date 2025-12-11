package metrics

import (
	"context"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"gorm.io/gorm"
)

const (
	defaultQueryTimeout        time.Duration = 15 * time.Second
	defaultHistoryQueryTimeout time.Duration = 60 * time.Second
)

type WorkerUsage struct {
	Namespace    string
	WorkloadName string
	WorkerName   string
	TflopsUsage  float64
	VramUsage    uint64
	Timestamp    time.Time
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
	// historyLength     time.Duration
	// historyResolution time.Duration
}

func NewProvider() (Provider, error) {
	tsdb, err := setupTimeSeriesDB()
	if err != nil {
		return nil, err
	}
	return &greptimeDBProvider{db: tsdb.DB}, nil
}

func (g *greptimeDBProvider) GetWorkersMetrics(ctx context.Context) ([]*WorkerUsage, error) {
	now := time.Now()

	timeoutCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	if g.lastQueryTime.IsZero() {
		g.lastQueryTime = now.Add(-time.Minute)
	}

	data := []*metrics.HypervisorWorkerUsageMetrics{}
	// actual meaning:  max(avg[10s])[1m]
	err := g.db.WithContext(timeoutCtx).
		Select("namespace, workload, worker, max(compute_tflops) as compute_tflops, max(memory_bytes) as memory_bytes, max(ts) as ts").
		Where("ts > ? and ts <= ?", g.lastQueryTime.UnixNano(), now.UnixNano()).
		Group("namespace, workload, worker").
		Order("ts asc").
		Find(&data).
		Error

	if err != nil {
		return nil, err
	}

	g.lastQueryTime = now

	workersMetrics := make([]*WorkerUsage, 0, len(data))
	for _, row := range data {
		workersMetrics = append(workersMetrics, &WorkerUsage{
			Namespace:    row.Namespace,
			WorkloadName: row.WorkloadName,
			WorkerName:   row.WorkerName,
			TflopsUsage:  row.ComputeTflops,
			VramUsage:    row.VRAMBytes,
			Timestamp:    row.Timestamp,
		})
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

	data := []*hypervisorWorkerUsageMetrics{}
	err := g.db.WithContext(timeoutCtx).
		Select("namespace, workload, worker, max(compute_tflops) as compute_tflops, max(memory_bytes) as memory_bytes, date_bin('1 minute'::INTERVAL, ts) as time_window").
		Where("ts > ? and ts <= ?", now.Add(-time.Hour*24).UnixNano(), now.UnixNano()).
		Group("namespace, workload, worker, time_window").
		Order("time_window asc").
		Find(&data).
		Error

	if err != nil {
		return nil, err
	}

	g.lastQueryTime = now

	workersMetrics := make([]*WorkerUsage, 0, len(data))
	for _, row := range data {
		workersMetrics = append(workersMetrics, &WorkerUsage{
			Namespace:    row.Namespace,
			WorkloadName: row.WorkloadName,
			WorkerName:   row.WorkerName,
			TflopsUsage:  row.ComputeTflops,
			VramUsage:    row.VRAMBytes,
			Timestamp:    row.TimeWindow,
		})
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

func (g *greptimeDBProvider) GetWorkloadHistoryMetrics(ctx context.Context, namespace, workloadName string, startTime, endTime time.Time) ([]*WorkerUsage, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, defaultHistoryQueryTimeout)
	defer cancel()

	data := []*hypervisorWorkerUsageMetrics{}
	err := g.db.WithContext(timeoutCtx).
		Select("namespace, workload, worker, max(compute_tflops) as compute_tflops, max(memory_bytes) as memory_bytes, date_bin('1 minute'::INTERVAL, ts) as time_window").
		Where("ts > ? and ts <= ? and namespace = ? and workload = ?",
			startTime.UnixNano(), endTime.UnixNano(), namespace, workloadName).
		Group("namespace, workload, worker, time_window").
		Order("time_window asc").
		Find(&data).
		Error

	if err != nil {
		return nil, err
	}

	workersMetrics := make([]*WorkerUsage, 0, len(data))
	for _, row := range data {
		workersMetrics = append(workersMetrics, &WorkerUsage{
			Namespace:    row.Namespace,
			WorkloadName: row.WorkloadName,
			WorkerName:   row.WorkerName,
			TflopsUsage:  row.ComputeTflops,
			VramUsage:    row.VRAMBytes,
			Timestamp:    row.TimeWindow,
		})
	}

	return workersMetrics, nil
}

func (g *greptimeDBProvider) GetWorkloadRealtimeMetrics(ctx context.Context, namespace, workloadName string, startTime, endTime time.Time) ([]*WorkerUsage, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, defaultQueryTimeout)
	defer cancel()

	data := []*metrics.HypervisorWorkerUsageMetrics{}
	err := g.db.WithContext(timeoutCtx).
		Select("namespace, workload, worker, max(compute_tflops) as compute_tflops, max(memory_bytes) as memory_bytes, max(ts) as ts").
		Where("ts > ? and ts <= ? and namespace = ? and workload = ?",
			startTime.UnixNano(), endTime.UnixNano(), namespace, workloadName).
		Group("namespace, workload, worker").
		Order("ts asc").
		Find(&data).
		Error

	if err != nil {
		return nil, err
	}

	workersMetrics := make([]*WorkerUsage, 0, len(data))
	for _, row := range data {
		workersMetrics = append(workersMetrics, &WorkerUsage{
			Namespace:    row.Namespace,
			WorkloadName: row.WorkloadName,
			WorkerName:   row.WorkerName,
			TflopsUsage:  row.ComputeTflops,
			VramUsage:    row.VRAMBytes,
			Timestamp:    row.Timestamp,
		})
	}

	return workersMetrics, nil
}
