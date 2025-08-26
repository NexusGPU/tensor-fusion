package sched

import (
	"math/rand"
	"testing"
	"time"

	gpuResourceFitPlugin "github.com/NexusGPU/tensor-fusion/internal/scheduler/gpuresources"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// Benchmark individual components with optimized setup [Last benchmark result on Mac M4 Pro]
// BenchmarkGPUFitPlugin/PreFilter-14     7039        479846 ns/op     657177 B/op  2829 allocs/op
// BenchmarkGPUFitPlugin/Filter-14        23221830    155.0 ns/op      128 B/op     3 allocs/op
// BenchmarkGPUFitPlugin/Score-14         21526615    167.0 ns/op      96 B/op      2 allocs/op
// Estimated Performance: 2346 pods/second
func BenchmarkGPUFitPlugin(b *testing.B) {
	config := BenchmarkConfig{
		NumNodes:  500,
		NumGPUs:   3000,
		NumPods:   10000,
		BatchSize: 1,
		PoolName:  "test-pool",
		Namespace: "test-ns",
		Timeout:   5 * time.Minute,
	}
	fixture := NewBenchmarkFixture(b, config, nil, false)
	defer fixture.Close()

	testPod := fixture.pods[0]
	utils.SetProgressiveMigration(false)

	b.Run("PreFilter", func(b *testing.B) {
		state := framework.NewCycleState()
		b.ResetTimer()
		i := 0
		for b.Loop() {
			// for i < 4000 {
			if i >= len(fixture.pods) {
				b.Error("benchmark too long, pre-created Pods out of bounds, reduce time or increase NumPods: ", i)
				break
			}
			testPod := fixture.pods[i]
			fixture.plugin.PreFilter(fixture.ctx, state, testPod)
			filterResult, err := state.Read(gpuResourceFitPlugin.CycleStateGPUSchedulingResult)
			if err != nil {
				b.Fatal(err)
			}
			scheduledState := filterResult.(*gpuResourceFitPlugin.GPUSchedulingStateData)
			if len(scheduledState.ValidNodeGPUScore) == 0 {
				b.Fatal("no valid node found, gpu capacity not enough")
			}
			maxScore := -1
			maxScoreNode := ""
			for nodeName, scoreMap := range scheduledState.ValidNodeGPUScore {
				score := 0
				for _, gpuScore := range scoreMap {
					score += gpuScore
				}

				notMatchingGPUScoreMap, ok := scheduledState.ValidNodeNotMatchingGPUScore[nodeName]
				if ok {
					for _, otherGPUScore := range notMatchingGPUScoreMap {
						score += otherGPUScore
					}
				}

				if score > maxScore {
					maxScore = score
					maxScoreNode = nodeName
				}
			}
			fixture.plugin.Reserve(fixture.ctx, state, testPod, maxScoreNode)
			if i%500 == 0 {
				b.Logf("Reserve done, valid node count: %d, selected node: %s, pod index: %d",
					len(scheduledState.NodeGPUs), maxScoreNode, i)
			}
			i++
		}
	})

	b.Run("Filter", func(b *testing.B) {
		state := framework.NewCycleState()
		fixture.plugin.PreFilter(fixture.ctx, state, testPod)
		nodeInfo := &framework.NodeInfo{}

		b.ResetTimer()
		for b.Loop() {
			nodeInfo.SetNode(fixture.nodes[rand.Intn(len(fixture.nodes))])
			fixture.plugin.Filter(fixture.ctx, state, testPod, nodeInfo)
		}
	})

	b.Run("Score", func(b *testing.B) {
		state := framework.NewCycleState()
		fixture.plugin.PreFilter(fixture.ctx, state, testPod)
		nodeInfo := &framework.NodeInfo{}

		b.ResetTimer()
		for b.Loop() {
			nodeInfo.SetNode(fixture.nodes[rand.Intn(len(fixture.nodes))])
			fixture.plugin.Score(fixture.ctx, state, testPod, nodeInfo)
		}
	})
}
