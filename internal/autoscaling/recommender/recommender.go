package recommender

import (
	"github.com/NexusGPU/tensor-fusion/internal/autoscaling"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaling/recommender/percentile"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaling/recommender/cron"
)

const (
	PercentileRecommender = "percentile"
	CronRecommender       = "cron"
)

type Interface interface {
	Name() string
	Recommend(*autoscaling.WorkloadState)
}

func New(name string) Interface {
	switch name {
	case PercentileRecommender:
		return percentile.NewRecommender()
	case CronRecommender:
	return cron.New()
	default:
		return nil
	}
}
