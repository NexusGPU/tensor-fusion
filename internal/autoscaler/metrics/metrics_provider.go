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
	GetWorkersMetrics(context.Context) ([]*WorkerUsage, error)
	GetHistoryMetrics(context.Context) ([]*WorkerUsage, error)
	LoadHistoryMetrics(context.Context, func(*WorkerUsage)) error
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

func (g *greptimeDBProvider) LoadHistoryMetrics(ctx context.Context, processMetricsFunc func(*WorkerUsage)) error {
	now := time.Now()

	timeoutCtx, cancel := context.WithTimeout(ctx, defaultHistoryQueryTimeout)
	defer cancel()

	rows, err := g.db.WithContext(timeoutCtx).
		Model(&hypervisorWorkerUsageMetrics{}).
		Select("namespace, workload, worker, max(compute_tflops) as compute_tflops, max(memory_bytes) as memory_bytes, date_bin('1 minute'::INTERVAL, ts) as time_window").
		Where("ts > ? and ts <= ?", now.Add(-time.Hour*24*7).UnixNano(), now.UnixNano()).
		Group("namespace, workload, worker, time_window").
		Order("time_window asc").
		Rows()
	if err != nil {
		return err
	}
	defer rows.Close()

	for rows.Next() {
		var usage hypervisorWorkerUsageMetrics
		if err := g.db.ScanRows(rows, &usage); err != nil {
			return err
		}
		processMetricsFunc(&WorkerUsage{
			Namespace:    usage.Namespace,
			WorkloadName: usage.WorkloadName,
			WorkerName:   usage.WorkerName,
			TflopsUsage:  usage.ComputeTflops,
			VramUsage:    usage.VRAMBytes,
			Timestamp:    usage.TimeWindow,
		})
	}

	g.lastQueryTime = now
	return nil
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
