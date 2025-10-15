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

func GetRecommendation(ctx context.Context, workload *workload.State, recommenders []Interface) (*tfv1.Resources, error) {
	recResults := map[string]*RecResult{}
	for _, recommender := range recommenders {
		result, err := recommender.Recommend(ctx, workload)
		if err != nil {
			return nil, fmt.Errorf("failed to get recommendation from %s: %v", recommender.Name(), err)
		}
		if result != nil {
			recResults[recommender.Name()] = result
		}
	}

	if len(recResults) <= 0 {
		return nil, nil
	}

	resources := getResourcesFromRecResults(recResults)
	if resources != nil {
		curRes := workload.GetCurrentResourcesSpec()
		// If a resource value is zero, replace it with current value
		if resources.Requests.Tflops.IsZero() || resources.Limits.Tflops.IsZero() {
			resources.Requests.Tflops = curRes.Requests.Tflops
			resources.Limits.Tflops = curRes.Limits.Tflops
		}

		if resources.Requests.Vram.IsZero() || resources.Limits.Vram.IsZero() {
			resources.Requests.Vram = curRes.Requests.Vram
			resources.Limits.Vram = curRes.Limits.Vram
		}
	}

	return resources, nil
}

func getResourcesFromRecResults(recResults map[string]*RecResult) *tfv1.Resources {
	targetRes := &tfv1.Resources{}
	minRes := &tfv1.Resources{}
	for _, rec := range recResults {
		if !rec.HasApplied {
			mergeResourcesByLargerRequests(targetRes, &rec.Resources)
		}
		if rec.ScaleDownLocking {
			mergeResourcesByLargerRequests(minRes, &rec.Resources)
		}
	}

	if targetRes.IsZero() ||
		(targetRes.Requests.Tflops.Cmp(minRes.Requests.Tflops) < 0 &&
			targetRes.Requests.Vram.Cmp(minRes.Requests.Vram) < 0) {
		return nil
	}

	return targetRes
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
