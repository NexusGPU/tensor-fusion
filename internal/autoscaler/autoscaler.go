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
	workloadHandler workload.Handler
	workloads       map[string]*workload.State
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
		workloads:       map[string]*workload.State{},
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

		workloadState := s.findOrCreateWorkloadState(workload.Name)
		s.workloadHandler.UpdateWorkloadState(ctx, workloadState, &workload)
		observedWorkloads[workload.Name] = true
	}

	// remove non-existent workloads
	for name := range s.workloads {
		if !observedWorkloads[name] {
			delete(s.workloads, name)
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
		s.findOrCreateWorkloadState(sample.WorkloadName).AddSample(sample)
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
		recommendations := map[string]*tfv1.RecommendedResources{}
		for _, recommender := range s.recommenders {
			name := recommender.Name()
			recommendation, err := recommender.Recommend(workload)
			if err != nil {
				log.Error(err, "failed to recommend resources", "recommender", name)
				continue
			}
			if recommendation == nil {
				continue
			}

			recommendations[name] = recommendation
			log.Info("recommendation", "recommender", name, "workload", workload.Name, "resources", recommendations[name])
		}

		if len(recommendations) == 0 {
			continue
		}

		var finalRecommendation *tfv1.RecommendedResources
		// for _, recommendation := range recommendations {
		// 	finalRecommendation = recommendation
		// }
		// process cron recommendation
		if recommendation, ok := recommendations[recommender.Cron]; ok {
			finalRecommendation = recommendation
		}

		s.workloadHandler.ApplyRecommendationToWorkload(ctx, workload, finalRecommendation)
	}
}

func (s *Autoscaler) findOrCreateWorkloadState(name string) *workload.State {
	w, exists := s.workloads[name]
	if !exists {
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
