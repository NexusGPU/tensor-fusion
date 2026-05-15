package workload

import (
	"sync"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/stretchr/testify/assert"
)

// TestState_ConcurrentAddSampleAndSnapshots runs concurrent writers and readers
// on the same State to ensure Mu (now RWMutex) protects WorkerUsageSamplers
// against the data race that v1 PR #669 (workload.State locking) addressed.
//
// Run with `-race` for the real check; without -race we at least verify no
// panic and that the returned snapshots remain self-consistent.
func TestState_ConcurrentAddSampleAndSnapshots(t *testing.T) {
	ws := NewWorkloadState()
	ws.Namespace = "ns"
	ws.Name = "wl"

	const workers = 4
	const samplesPerWorker = 200

	var wg sync.WaitGroup

	// Writers: AddSample (Lock path).
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		workerName := workerNameFor(w)
		go func() {
			defer wg.Done()
			for i := 0; i < samplesPerWorker; i++ {
				ws.AddSample(&metrics.WorkerUsage{
					WorkerName:  workerName,
					TflopsUsage: float64(i),
					VramUsage:   uint64(i),
					Timestamp:   time.Now(),
				})
			}
		}()
	}

	// Readers: concurrent RLock paths.
	wg.Add(3)
	go func() {
		defer wg.Done()
		for i := 0; i < samplesPerWorker; i++ {
			ws.StatusSnapshot()
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < samplesPerWorker; i++ {
			_ = ws.IsAutoSetResourcesEnabled()
			_ = ws.ShouldScaleResource(tfv1.ResourceTflops)
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < samplesPerWorker; i++ {
			_ = ws.ActiveWorkersSnapshot()
		}
	}()

	wg.Wait()

	// Sanity: every worker we wrote ended up with a sampler.
	ws.Mu.RLock()
	defer ws.Mu.RUnlock()
	for w := 0; w < workers; w++ {
		_, ok := ws.WorkerUsageSamplers[workerNameFor(w)]
		assert.True(t, ok, "expected sampler for %s", workerNameFor(w))
	}
}

// TestState_SetAppliedRecommendedReplicas exercises the batched setter that
// replaced the per-iteration Reset+Inc pattern.
func TestState_SetAppliedRecommendedReplicas(t *testing.T) {
	ws := NewWorkloadState()
	ws.SetAppliedRecommendedReplicas(7)

	_, _, status := ws.StatusSnapshot()
	assert.Equal(t, int32(7), status.AppliedRecommendedReplicas)

	ws.SetAppliedRecommendedReplicas(0)
	_, _, status = ws.StatusSnapshot()
	assert.Equal(t, int32(0), status.AppliedRecommendedReplicas)
}

func workerNameFor(i int) string {
	return "worker-" + string(rune('a'+i))
}
