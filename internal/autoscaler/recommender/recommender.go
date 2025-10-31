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
	Recommend(ctx context.Context, workload *workload.State) (*RecResult, error)
}

type RecResult struct {
	Resources        tfv1.Resources
	HasApplied       bool
	ScaleDownLocking bool
}

type recommenderToRecResult map[string]*RecResult

func (r recommenderToRecResult) generateRecommendation() *tfv1.Resources {
	if len(r) == 0 {
		return nil
	}

	recommendation := &tfv1.Resources{}
	minRes := &tfv1.Resources{}
	for _, result := range r {
		if !result.HasApplied {
			mergeResourcesByLargerRequests(recommendation, &result.Resources)
		}
		if result.ScaleDownLocking {
			mergeResourcesByLargerRequests(minRes, &result.Resources)
		}
	}

	if recommendation.IsZero() ||
		(recommendation.Requests.Tflops.Cmp(minRes.Requests.Tflops) < 0 &&
			recommendation.Requests.Vram.Cmp(minRes.Requests.Vram) < 0) {
		return nil
	}

	return recommendation
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

func GetRecommendation(ctx context.Context, workload *workload.State, recommenders []Interface) (*tfv1.Resources, error) {
	recResults := make(recommenderToRecResult)
	for _, recommender := range recommenders {
		result, err := recommender.Recommend(ctx, workload)
		if err != nil {
			return nil, fmt.Errorf("failed to get recommendation from %s: %v", recommender.Name(), err)
		}
		if result != nil {
			recResults[recommender.Name()] = result
		}
	}

	recommendation := recResults.generateRecommendation()
	if recommendation != nil {
		curRes := workload.GetCurrentResourcesSpec()
		// If a resource value is zero, replace it with current value
		if recommendation.Requests.Tflops.IsZero() || recommendation.Limits.Tflops.IsZero() {
			recommendation.Requests.Tflops = curRes.Requests.Tflops
			recommendation.Limits.Tflops = curRes.Limits.Tflops
		}

		if recommendation.Requests.Vram.IsZero() || recommendation.Limits.Vram.IsZero() {
			recommendation.Requests.Vram = curRes.Requests.Vram
			recommendation.Limits.Vram = curRes.Limits.Vram
		}
	}

	return recommendation, nil
}
