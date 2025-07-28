package autoscaler

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

type RecommendationProcessor interface {
	Process()
}

type CronRecommendationProcessor struct{}

func (c *CronRecommendationProcessor) Process() {

}

func MergeRecommendations() *tfv1.RecommendedResources {
	return &tfv1.RecommendedResources{}
}
