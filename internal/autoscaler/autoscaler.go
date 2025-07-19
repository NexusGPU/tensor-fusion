package autoscaler

import (
	"context"
	"errors"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/recommender"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

var (
	_ manager.Runnable               = (*Autoscaler)(nil)
	_ manager.LeaderElectionRunnable = (*Autoscaler)(nil)
)

type Autoscaler struct {
	client.Client
	allocator       *gpuallocator.GpuAllocator
	metricsProvider metrics.Provider
	recommenders    []recommender.Interface
	workloadHandler *workload.Handler
	workloads       map[string]*workload.WorkloadState
}

func NewAutoscaler(c client.Client, allocator *gpuallocator.GpuAllocator) (*Autoscaler, error) {
	if c == nil {
		return nil, errors.New("must specify client")
	}

	if allocator == nil {
		return nil, errors.New("must specify allocator")
	}

	recommenders := []recommender.Interface{
		recommender.NewPercentileRecommender(),
		recommender.NewCronRecommender(),
	}

	return &Autoscaler{
		Client:          c,
		allocator:       allocator,
		metricsProvider: metrics.NewProvider(nil),
		recommenders:    recommenders,
		workloadHandler: workload.NewHandler(c, allocator),
		workloads:       map[string]*workload.WorkloadState{},
	}, nil
}

func (s *Autoscaler) Start(ctx context.Context) error {
	log := log.FromContext(ctx)
	log.Info("Starting autoscaler")

	// Handle timeout for loading historical metrics
	historyCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	s.loadHistoryMetrics(historyCtx)

	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			s.Run(ctx)
		case <-ctx.Done():
			log.Info("Stopping autoscaler")
			return nil
		}
	}
}

func (s *Autoscaler) NeedLeaderElection() bool {
	return true
}

func (s *Autoscaler) Run(ctx context.Context) {
	log := log.FromContext(ctx)

	log.Info("Autoscaler running")
	s.loadWorkloads(ctx)
	s.loadRealTimeMetrics(ctx)
	s.processWorkloads(ctx)
}

func (s *Autoscaler) loadWorkloads(ctx context.Context) {
	log := log.FromContext(ctx)

	workloadList := tfv1.TensorFusionWorkloadList{}
	if err := s.List(ctx, &workloadList); err != nil {
		log.Error(err, "failed to list workloads")
		return
	}

	observedWorkloads := map[string]bool{}
	for _, workload := range workloadList.Items {
		if !workload.DeletionTimestamp.IsZero() {
			continue
		}
		workloadState := s.findOrCreateWorkload(workload.Name)
		workloadState.Namespace = workload.Namespace
		workloadState.Spec = workload.Spec
		observedWorkloads[workload.Name] = true

		s.workloadHandler.UpdateWorkers(ctx, workloadState)
	}

	// remove non-existent workloads
	for key := range s.workloads {
		if !observedWorkloads[key] {
			delete(s.workloads, key)
		}
	}
}

func (s *Autoscaler) loadHistoryMetrics(ctx context.Context) {
	log := log.FromContext(ctx)
	log.Info("loading historical metrics")

	workersMetrics, err := s.metricsProvider.GetHistoryMetrics()
	if err != nil {
		log.Error(err, "failed to get history metrics")
		return
	}
	for _, sample := range workersMetrics {
		workload := s.findOrCreateWorkload(sample.WorkloadName)
		workload.AddSample(sample)
	}
}

func (s *Autoscaler) loadRealTimeMetrics(ctx context.Context) {
	log := log.FromContext(ctx)
	log.Info("loading realtime metrics")

	workersMetrics, err := s.metricsProvider.GetWorkersMetrics()
	if err != nil {
		log.Error(err, "failed to get workers metrics")
		return
	}

	for _, sample := range workersMetrics {
		if workload, exists := s.workloads[sample.WorkloadName]; exists {
			workload.AddSample(sample)
		}
	}
}

func (s *Autoscaler) processWorkloads(ctx context.Context) {
	log := log.FromContext(ctx)
	log.Info("processing workloads")

	for _, workload := range s.workloads {
		recommendations := map[string]recommender.RecommendedResources{}
		for _, recommender := range s.recommenders {
			name := recommender.Name()
			recommendations[name] = recommender.Recommend(&workload.Spec.AutoScalingConfig, workload.WorkerUsageAggregator)
			log.Info("recommendation", "recommender", name, "workload", workload.Name, "resources", recommendations[name])
		}

		// var finalRecommendation recommender.RecommendedResources
		// for _, recommendation := range recommendations {
		// 	if recommendation.TargetTflops.IsZero()
		// }

		// TODO: Implement updating the recommendation status of the workload CRD when the API is ready.
		workload.UpdateRecommendation(recommendations["percentile"])
		s.workloadHandler.ProcessWorkload(ctx, workload)
	}
}

func (s *Autoscaler) findOrCreateWorkload(name string) *workload.WorkloadState {
	w, ok := s.workloads[name]
	if !ok {
		w = workload.NewWorkloadState(name)
		s.workloads[name] = w
	}
	return w
}

// Start after manager started
func SetupWithManager(mgr ctrl.Manager, allocator *gpuallocator.GpuAllocator) error {
	autoScaler, err := NewAutoscaler(mgr.GetClient(), allocator)
	if err != nil {
		return err
	}
	return mgr.Add(autoScaler)
}
