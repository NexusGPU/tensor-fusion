package recommender

import (
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	RecommenderPercentile = "percentile"
	RecommenderCron       = "cron"
)

type RecommendedResources struct {
	LowerBoundTflops resource.Quantity
	TargetTflops     resource.Quantity
	UpperBoundTflops resource.Quantity
	LowerBoundVram   resource.Quantity
	TargetVram       resource.Quantity
	UpperBoundVram   resource.Quantity
}

// Interface defines the contract for resource recommendation strategies used by the autoscaler.
type Interface interface {
	Name() string
	Recommend(*tfv1.AutoScalingConfig, *metrics.WorkerUsageAggregator) RecommendedResources
}

func New(name string) (Interface, error) {
	switch name {
	case RecommenderPercentile:
		return NewPercentileRecommender(), nil
	case RecommenderCron:
		return NewCronRecommender(), nil
	default:
		return nil, fmt.Errorf("unknown recommender name: %s", name)
	}
}
