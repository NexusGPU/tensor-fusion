package recommender

import (
	"context"

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
	Applied          bool
	ScaleDownLocking bool
}

func MergeRecommendations(recommendations map[string]*Recommendation) *tfv1.Resources {
	result := &tfv1.Resources{}
	for _, rec := range recommendations {
		if !rec.ScaleDownLocking && rec.Applied {
			continue
		}
		if result.Requests.Tflops.Cmp(rec.Resources.Requests.Tflops) < 0 {
			result.Requests.Tflops = rec.Resources.Requests.Tflops
			result.Limits.Tflops = rec.Resources.Limits.Tflops
		}
		if result.Requests.Vram.Cmp(rec.Resources.Requests.Vram) < 0 {
			result.Requests.Vram = rec.Resources.Requests.Vram
			result.Limits.Vram = rec.Resources.Limits.Vram
		}
	}
	return result
}
