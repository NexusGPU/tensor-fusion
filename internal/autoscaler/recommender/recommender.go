package recommender

import (
	"context"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
)

// Interface defines the contract for resource recommendation strategies used by the autoscaler.
type Interface interface {
	Name() string
	Recommend(ctx context.Context, workload *workload.State) (*Recommendation, error)
}

type Recommendation struct {
	Resources        tfv1.Resources
	HasApplied       bool
	ScaleDownLocking bool
}

func GetResourcesFromRecommenders(ctx context.Context, workload *workload.State, recommenders []Interface) (*tfv1.Resources, error) {
	recommendations := map[string]*Recommendation{}
	for _, recommender := range recommenders {
		rec, err := recommender.Recommend(ctx, workload)
		if err != nil {
			return nil, fmt.Errorf("failed to get recommendation from %s: %v", recommender.Name(), err)
		}
		if rec != nil {
			recommendations[recommender.Name()] = rec
		}
	}

	if len(recommendations) <= 0 {
		return nil, nil
	}

	return getResourcesFromRecommendations(recommendations), nil
}

func getResourcesFromRecommendations(recommendations map[string]*Recommendation) *tfv1.Resources {
	result := &tfv1.Resources{}
	minRes := &tfv1.Resources{}
	for _, rec := range recommendations {
		if !rec.HasApplied {
			mergeResourcesByLargerRequests(result, &rec.Resources)
		}
		if rec.ScaleDownLocking {
			mergeResourcesByLargerRequests(minRes, &rec.Resources)
		}
	}

	if result.IsZero() ||
		(result.Requests.Tflops.Cmp(minRes.Requests.Tflops) < 0 &&
			result.Requests.Vram.Cmp(minRes.Requests.Vram) < 0) {
		return nil
	}

	return result
}

func mergeResourcesByLargerRequests(src *tfv1.Resources, target *tfv1.Resources) {
	if src.Requests.Tflops.Cmp(target.Requests.Tflops) < 0 {
		src.Requests.Tflops = target.Requests.Tflops
		src.Limits.Tflops = target.Limits.Tflops
	}
	if src.Requests.Vram.Cmp(target.Requests.Vram) < 0 {
		src.Requests.Vram = target.Requests.Vram
		src.Limits.Vram = target.Limits.Vram
	}
}
