package recommender

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
)

type CronRecommender struct{}

func NewCronRecommender() *CronRecommender {
	return &CronRecommender{}
}

func (c *CronRecommender) Name() string {
	return "cron"
}

func (p *CronRecommender) Recommend(config *tfv1.AutoScalingConfig, w *metrics.WorkerUsageAggregator) RecommendedResources {
	return RecommendedResources{}
}

func (c *CronRecommender) getCronConfig(asc *tfv1.AutoScalingConfig) {

}
