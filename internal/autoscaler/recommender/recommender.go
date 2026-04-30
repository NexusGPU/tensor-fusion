package recommender

import (
	"context"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
)

// Interface defines the contract for resource recommendation strategies used by the autoscaler.
//
// Recommend operates on an immutable StateView and is therefore safe to call
// concurrently with State mutators. Any side-effect a recommender wants to
// record (status conditions, active cron rule changes) must be returned in
// RecResult.Intent so the caller can merge them under State.Mu.
type Interface interface {
	Name() string
	Recommend(ctx context.Context, view *workload.StateView) (*RecResult, error)
}

type RecResult struct {
	Resources        tfv1.Resources
	HasApplied       bool
	ScaleDownLocking bool
	// Intent describes the side-effect that should be merged back into
	// State.Status. The zero value means "no side-effect".
	Intent workload.Intent
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

// GetRecommendation runs every recommender against the given view and returns
// (mergedRecommendation, perRecommenderIntents). The caller is expected to
// pass intents to State.ApplyIntents to materialize them under State.Mu.
func GetRecommendation(ctx context.Context, view *workload.StateView, recommenders []Interface) (*tfv1.Resources, []workload.Intent, error) {
	recResults := make(recommenderToRecResult)
	intents := make([]workload.Intent, 0, len(recommenders))
	for _, recommender := range recommenders {
		result, err := recommender.Recommend(ctx, view)
		if err != nil {
			return nil, intents, fmt.Errorf("failed to get recommendation from %s: %v", recommender.Name(), err)
		}
		if result == nil {
			continue
		}
		recResults[recommender.Name()] = result
		if result.Intent != (workload.Intent{}) {
			intents = append(intents, result.Intent)
		}
	}

	recommendation := recResults.generateRecommendation()
	if recommendation != nil {
		curRes := view.GetCurrentResourcesSpec()
		if recommendation.Requests.Tflops.IsZero() || recommendation.Limits.Tflops.IsZero() {
			recommendation.Requests.Tflops = curRes.Requests.Tflops
			recommendation.Limits.Tflops = curRes.Limits.Tflops
		}

		if recommendation.Requests.Vram.IsZero() || recommendation.Limits.Vram.IsZero() {
			recommendation.Requests.Vram = curRes.Requests.Vram
			recommendation.Limits.Vram = curRes.Limits.Vram
		}
	}

	return recommendation, intents, nil
}
