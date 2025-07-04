package autoscaler

import (
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"gorm.io/gorm"
)

type WorkerMetrics struct {
	WorkloadName string
	WorkerName   string
	TflopsUsage  ResourceAmount
	VramUsage    ResourceAmount
	Timestamp    time.Time
}

type MetricsProvider interface {
	GetWorkersMetrics() ([]*WorkerMetrics, error)
	GetHistoryMetrics() ([]*WorkerMetrics, error)
}

func NewMetricsProvider(db *gorm.DB) MetricsProvider {
	return &greptimeDBProvider{db: db}
}

type greptimeDBProvider struct {
	db            *gorm.DB
	lastQueryTime time.Time
	// historyLength     time.Duration
	// historyResolution time.Duration
}

func (g *greptimeDBProvider) GetWorkersMetrics() ([]*WorkerMetrics, error) {
	data := []*metrics.HypervisorWorkerUsageMetrics{}
	now := time.Now()
	// actual meaning:  max(avg[10s])[1m]
	err := g.db.Select("workload, worker, max(compute_tflops) as compute_tflops, max(memory_bytes) as memory_bytes, max(ts) as ts").
		Where("ts > ? and ts <= ?", g.lastQueryTime.Nanosecond(), now.Nanosecond()).
		Group("workload, worker").
		Order("ts asc").
		Find(&data).
		Error

	if err != nil {
		return nil, err
	}

	g.lastQueryTime = now

	workersMetrics := make([]*WorkerMetrics, 0, len(data))
	for _, row := range data {
		workersMetrics = append(workersMetrics, &WorkerMetrics{
			WorkloadName: row.WorkloadName,
			WorkerName:   row.WorkerName,
			TflopsUsage:  resourceAmountFromFloat(row.ComputeTflops),
			VramUsage:    ResourceAmount(row.VRAMBytes),
			Timestamp:    row.Timestamp,
		})
	}

	return workersMetrics, nil
}

type hypervisorWorkerUsageMetrics struct {
	metrics.HypervisorWorkerUsageMetrics
	TimeWindow time.Time `gorm:"column:time_window;index:,class:TIME"`
}

func (g *greptimeDBProvider) GetHistoryMetrics() ([]*WorkerMetrics, error) {
	data := []*hypervisorWorkerUsageMetrics{}
	now := time.Now()
	// TODO: replace using iteration for handling large datasets efficiently
	// TODO: supply history resolution to config time window
	err := g.db.Select("workload, worker, max(compute_tflops) as compute_tflops, max(memory_bytes) as memory_bytes, date_bin('1 minute'::INTERVAL, ts) as time_window").
		Where("ts > ? and ts <= ?", now.Add(-time.Hour*24).Nanosecond(), now.Nanosecond()).
		Group("workload, worker, time_window").
		Order("time_window asc").
		Find(&data).
		Error

	if err != nil {
		return nil, err
	}

	g.lastQueryTime = now

	workersMetrics := make([]*WorkerMetrics, 0, len(data))
	for _, row := range data {
		workersMetrics = append(workersMetrics, &WorkerMetrics{
			WorkloadName: row.WorkloadName,
			WorkerName:   row.WorkerName,
			TflopsUsage:  resourceAmountFromFloat(row.ComputeTflops),
			VramUsage:    ResourceAmount(row.VRAMBytes),
			Timestamp:    row.TimeWindow,
		})
	}

	return workersMetrics, nil
}
