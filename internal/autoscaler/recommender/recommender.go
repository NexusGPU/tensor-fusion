package recommender

import (
	"context"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
)

const (
	Percentile = "percentile"
	Cron       = "cron"
)

// Interface defines the contract for resource recommendation strategies used by the autoscaler.
type Interface interface {
	Name() string
	Recommend(ctx context.Context, workload *workload.State) (*tfv1.Resources, error)
}

func New(name string) (Interface, error) {
	switch name {
	case Percentile:
		return NewPercentileRecommender(), nil
	case Cron:
		return NewCronRecommender(), nil
	default:
		return nil, fmt.Errorf("unknown recommender name: %s", name)
	}
}
