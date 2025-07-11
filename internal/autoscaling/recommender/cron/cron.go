package cron

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaling"
)

type CronRecommender struct{}

func New() *CronRecommender {
	return &CronRecommender{}
}

func (c *CronRecommender) Name() string {
	return "cron"
}

func (c *CronRecommender) Recommend(w *autoscaling.WorkloadState) {
	c.getCronConfig(&w.AutoScalingConfig)
}

func (c *CronRecommender) getCronConfig(asc *tfv1.AutoScalingConfig) {
}
