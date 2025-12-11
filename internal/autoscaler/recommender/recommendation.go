package recommender

import (
	"context"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"sigs.k8s.io/controller-runtime/pkg/log"
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
	curRes := workload.GetCurrentResourcesSpec()

	isScaleUpTflops := curRes.Requests.Tflops.Cmp(rec.Requests.Tflops) < 0
	isScaleUpVram := curRes.Requests.Vram.Cmp(rec.Requests.Vram) < 0
	if !isScaleUpTflops && !isScaleUpVram {
		return result, msg, nil
	}

	// Get max allowed considering the node with min available resources
	allowedRes, err := r.workloadHandler.GetMaxAllowedResourcesSpec(workload)
	if err != nil || allowedRes == nil {
		return result, msg, err
	}
	log.FromContext(ctx).V(4).Info("fetched max allowed resources", "workload", workload.Name, "resources", allowedRes)

	if isScaleUpTflops && rec.Requests.Tflops.Cmp(allowedRes.Tflops) > 0 {
		result.Requests.Tflops = allowedRes.Tflops
		msg = fmt.Sprintf("TFlops request set to max allowed: (%s)", result.Requests.Tflops.String())
	}

	if isScaleUpVram && rec.Requests.Vram.Cmp(allowedRes.Vram) > 0 {
		result.Requests.Vram = allowedRes.Vram
		if msg != "" {
			msg += ", "
		}
		msg += fmt.Sprintf("VRAM request set to max allowed: (%s)", result.Requests.Vram.String())
	}

	return result, msg, nil
}
