package sched

import (
	"context"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/cmd/sched"
	gpuResourceFitPlugin "github.com/NexusGPU/tensor-fusion/internal/scheduler/gpuresources"
	gpuTopoPlugin "github.com/NexusGPU/tensor-fusion/internal/scheduler/gputopo"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	configz "k8s.io/component-base/configz"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
)

// defaultBenchmarkConfig returns default benchmark configuration
func defaultBenchmarkConfig() BenchmarkConfig {
	return BenchmarkConfig{
		NumNodes:  1000,
		NumGPUs:   4000,
		NumPods:   10000,
		BatchSize: 100,
		PoolName:  "benchmark-pool",
		Namespace: "benchmark-ns",
		Timeout:   10 * time.Minute,
	}
}

func BenchmarkScheduler(b *testing.B) {
	config := defaultBenchmarkConfig()
	fixture := NewBenchmarkFixture(b, config)
	defer fixture.Close()

	utils.SetProgressiveMigration(false)

	gpuResourceFitOpt := app.WithPlugin(
		gpuResourceFitPlugin.Name,
		gpuResourceFitPlugin.NewWithDeps(fixture.allocator, fixture.client),
	)
	gpuTopoOpt := app.WithPlugin(
		gpuTopoPlugin.Name,
		gpuTopoPlugin.NewWithDeps(fixture.allocator, fixture.client),
	)

	cc, scheduler, err := sched.SetupScheduler(fixture.ctx, nil,
		"../../config/samples/scheduler-config.yaml", gpuResourceFitOpt, gpuTopoOpt)
	if err != nil {
		b.Error(err)
		return
	}

	// Config registration.
	if cz, err := configz.New("componentconfig"); err != nil {
		b.Error(err)
		return
	} else {
		cz.Set(cc.ComponentConfig)
	}

	cc.EventBroadcaster.StartRecordingToSink(fixture.ctx.Done())

	startInformersAndWaitForSync := func(ctx context.Context) {
		// Start all informers.
		cc.InformerFactory.Start(ctx.Done())
		// DynInformerFactory can be nil in tests.
		if cc.DynInformerFactory != nil {
			cc.DynInformerFactory.Start(ctx.Done())
		}

		// Wait for all caches to sync before scheduling.
		cc.InformerFactory.WaitForCacheSync(ctx.Done())
		// DynInformerFactory can be nil in tests.
		if cc.DynInformerFactory != nil {
			cc.DynInformerFactory.WaitForCacheSync(ctx.Done())
		}

		// Wait for all handlers to sync (all items in the initial list delivered) before scheduling.
		if err := scheduler.WaitForHandlersSync(fixture.ctx); err != nil {
			b.Error(err)
			return
		}
		b.Log("Handlers synced")
	}
	startInformersAndWaitForSync(fixture.ctx)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		scheduler.ScheduleOne(fixture.ctx)

		if i%100 == 0 {
			_, info := scheduler.SchedulingQueue.PendingPods()
			b.Log(info)
		}

		if i >= 9999 {
			b.StopTimer()
			return
		}
	}
}

// SchedulingMetrics holds scheduling performance metrics
type SchedulingMetrics struct {
	Scheduled int
	Failed    int
	LatencyNs float64
}
