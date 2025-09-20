package recommender

import (
	"context"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
)

type RecommendationProcessor interface {
	Apply(ctx context.Context, workload *workload.State, recommendation *tfv1.Resources) (tfv1.Resources, string, error)
}

func NewRecommendationProcessor(workloadHandler workload.Handler) RecommendationProcessor {
	return &recommendationProcessor{workloadHandler}
}

type recommendationProcessor struct {
	workloadHandler workload.Handler
}

func (r *recommendationProcessor) Apply(
	ctx context.Context,
	workload *workload.State,
	rec *tfv1.Resources) (tfv1.Resources, string, error) {
	result := *rec
	msg := ""
	allowedRes, err := r.workloadHandler.GetMaxAllowedResourcesSpec(workload)
	if err != nil || allowedRes == nil {
		return result, msg, err
	}

	if rec.Requests.Tflops.Cmp(allowedRes.Tflops) > 0 {
		maxTflopsLimit := getProportionalLimit(&rec.Limits.Tflops, &rec.Requests.Tflops, &allowedRes.Tflops)
		result.Requests.Tflops = allowedRes.Tflops
		result.Limits.Tflops = *maxTflopsLimit
		msg = fmt.Sprintf("TFLOPS reduced due to target (%s) exceed max allowed (%s)",
			rec.Requests.Tflops.String(), result.Requests.Tflops.String())
	}

	if rec.Requests.Vram.Cmp(allowedRes.Vram) > 0 {
		maxVramLimit := getProportionalLimit(&rec.Limits.Vram, &rec.Requests.Vram, &allowedRes.Vram)
		result.Requests.Vram = allowedRes.Vram
		result.Limits.Vram = *maxVramLimit
		if msg != "" {
			msg += ", "
		}
		msg += fmt.Sprintf("VRAM reduced due to target (%s) exceed max allowed (%s)",
			rec.Requests.Vram.String(), result.Requests.Vram.String())
	}

	return result, msg, nil
}
