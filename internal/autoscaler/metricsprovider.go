package autoscaler

import "time"

type WorkerMetrics struct {
	Workload    string
	Worker      string
	TflopsUsage ResourceAmount
	VramUsage   ResourceAmount
	Timestamp   time.Time
}

type MetricsProvider interface {
	GetWorkersMetrics() []*WorkerMetrics
	GetHistoryMetrics() []*WorkerMetrics
}

func NewMetricsProvider() MetricsProvider {
	return &GreptimeDBProvider{}
}

type GreptimeDBProvider struct{}

func (*GreptimeDBProvider) GetWorkersMetrics() []*WorkerMetrics {
	panic("unimplemented")
}

func (*GreptimeDBProvider) GetHistoryMetrics() []*WorkerMetrics {
	panic("unimplemented")
}
