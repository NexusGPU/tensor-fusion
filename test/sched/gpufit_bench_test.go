package sched

import (
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// Benchmark individual components with optimized setup
func BenchmarkGPUFitPlugin(b *testing.B) {
	config := BenchmarkConfig{
		NumNodes:  100,
		NumGPUs:   400,
		NumPods:   1,
		BatchSize: 1,
		PoolName:  "test-pool",
		Namespace: "test-ns",
		Timeout:   5 * time.Minute,
	}
	fixture := NewBenchmarkFixture(b, config)
	defer fixture.Close()

	testPod := fixture.pods[0]
	utils.SetProgressiveMigration(false)

	b.Run("PreFilter", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			state := framework.NewCycleState()
			state.Write(constants.SchedulerSimulationKey, &gpuallocator.SimulateSchedulingFilterDetail{})
			fixture.plugin.PreFilter(fixture.ctx, state, testPod)
		}
	})

	b.Run("Filter", func(b *testing.B) {
		state := framework.NewCycleState()
		state.Write(constants.SchedulerSimulationKey, &gpuallocator.SimulateSchedulingFilterDetail{})
		fixture.plugin.PreFilter(fixture.ctx, state, testPod)
		nodeInfo := &framework.NodeInfo{}
		nodeInfo.SetNode(fixture.nodes[0])

		b.ResetTimer()
		for b.Loop() {
			fixture.plugin.Filter(fixture.ctx, state, testPod, nodeInfo)
		}
	})

	b.Run("Score", func(b *testing.B) {
		state := framework.NewCycleState()
		state.Write(constants.SchedulerSimulationKey, &gpuallocator.SimulateSchedulingFilterDetail{})
		fixture.plugin.PreFilter(fixture.ctx, state, testPod)
		nodeInfo := &framework.NodeInfo{}
		nodeInfo.SetNode(fixture.nodes[0])

		b.ResetTimer()
		for b.Loop() {
			fixture.plugin.Score(fixture.ctx, state, testPod, nodeInfo)
		}
	})
}
