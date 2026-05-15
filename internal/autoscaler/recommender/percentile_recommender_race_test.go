package recommender

import (
	"sync"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
)

// TestPercentileRecommender_ConcurrentDifferentConfigs guards against a
// shared-mutable-state regression in the percentile estimator. Previously
// resourcesEstimator stored the configured estimators on the embedded struct
// (`*r = resourcesEstimator{...}`), so two workloads with different
// AutoSetResources percentile settings would stomp on each other when running
// concurrently in their per-workload metrics-loader goroutines. The fix builds
// the percentileEstimators as a local value per call. Run with -race.
func TestPercentileRecommender_ConcurrentDifferentConfigs(t *testing.T) {
	rec := NewPercentileRecommender(nil)

	mkWorkload := func(percentile, margin string) *workload.State {
		ws := workload.NewWorkloadState()
		ws.Namespace = "ns"
		ws.Name = "wl-" + percentile
		ws.Spec.AutoScalingConfig.AutoSetResources = &tfv1.AutoSetResources{
			Enable:                      true,
			TargetResource:              tfv1.ScalingTargetResourceAll,
			TargetComputePercentile:     percentile,
			LowerBoundComputePercentile: "0.5",
			UpperBoundComputePercentile: "0.99",
			TargetVRAMPercentile:        percentile,
			LowerBoundVRAMPercentile:    "0.5",
			UpperBoundVRAMPercentile:    "0.99",
			MarginFraction:              margin,
		}
		// Seed a sample so the aggregator is non-empty; this forces the
		// estimator read path to actually execute.
		ws.AddSample(&metrics.WorkerUsage{
			WorkerName:  "w",
			TflopsUsage: 10,
			VramUsage:   1 << 20,
			Timestamp:   time.Now(),
		})
		return ws
	}

	workloads := []*workload.State{
		mkWorkload("0.95", "0.15"),
		mkWorkload("0.50", "0.30"),
		mkWorkload("0.99", "0.05"),
		mkWorkload("0.75", "0.20"),
	}

	// Drive the estimator path directly: skip the Recommend wrapper so the test
	// doesn't depend on InitialDelayPeriod / curRes plumbing. This is the
	// specific function that previously stomped on *r.
	const iterations = 200
	var wg sync.WaitGroup
	for _, w := range workloads {
		wg.Add(1)
		go func(w *workload.State) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				_ = rec.GetResourcesEstimation(w)
			}
		}(w)
	}
	wg.Wait()
}
