package autoscaler

import (
	"context"
	"errors"
	"fmt"
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

type WorkloadID struct {
	Namespace string
	Name      string
}

type Autoscaler struct {
	client.Client
	allocator       *gpuallocator.GpuAllocator
	metricsProvider metrics.Provider
	recommenders    []recommender.Interface
	workloadHandler workload.Handler
	workloads       map[WorkloadID]*workload.State
}

func NewAutoscaler(c client.Client,
	allocator *gpuallocator.GpuAllocator,
	metricsProvider metrics.Provider) (*Autoscaler, error) {
	if c == nil {
		return nil, errors.New("must specify client")
	}

	if allocator == nil {
		return nil, errors.New("must specify allocator")
	}

	if metricsProvider == nil {
		return nil, errors.New("must specify metricsProvider")
	}

	recommenders := []recommender.Interface{
		recommender.NewPercentileRecommender(),
		recommender.NewCronRecommender(),
	}

	return &Autoscaler{
		Client:          c,
		allocator:       allocator,
		metricsProvider: metricsProvider,
		recommenders:    recommenders,
		workloadHandler: workload.NewHandler(c, allocator),
		workloads:       map[WorkloadID]*workload.State{},
	}, nil
}

func (s *Autoscaler) Start(ctx context.Context) error {
	log := log.FromContext(ctx)
	log.Info("Starting autoscaler")

	if err := s.loadHistoryMetrics(ctx); err != nil {
		log.Error(err, "failed to load history metrics")
	}

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
	s.loadWorkloads(ctx)
	s.loadRealTimeMetrics(ctx)
	s.processWorkloads(ctx)
}

func (s *Autoscaler) loadWorkloads(ctx context.Context) {
	log := log.FromContext(ctx)
	log.Info("loading workloads")

	workloadList := tfv1.TensorFusionWorkloadList{}
	if err := s.List(ctx, &workloadList); err != nil {
		log.Error(err, "failed to list workloads")
		return
	}

	activeWorkloads := map[WorkloadID]bool{}
	for _, workload := range workloadList.Items {
		if !workload.DeletionTimestamp.IsZero() {
			continue
		}

		workloadID := WorkloadID{workload.Namespace, workload.Name}
		activeWorkloads[workloadID] = true
		workloadState := s.findOrCreateWorkloadState(workloadID.Namespace, workloadID.Name)
		if err := s.workloadHandler.UpdateWorkloadState(ctx, workloadState, &workload); err != nil {
			log.Error(err, "failed to update workload state", "workload", workloadID)
		}
	}

	// remove non-existent workloads
	for workloadID := range s.workloads {
		if !activeWorkloads[workloadID] {
			delete(s.workloads, workloadID)
		}
	}
}

func (s *Autoscaler) loadHistoryMetrics(ctx context.Context) error {
	workersMetrics, err := s.metricsProvider.GetHistoryMetrics(ctx)
	if err != nil {
		return fmt.Errorf("failed to get history metrics: %v", err)
	}
	for _, sample := range workersMetrics {
		s.findOrCreateWorkloadState(sample.Namespace, sample.WorkloadName).AddSample(sample)
	}

	if metricsCount := len(workersMetrics); metricsCount > 0 {
		log.FromContext(ctx).Info("historical metrics loaded", "from",
			workersMetrics[0].Timestamp, "to", workersMetrics[metricsCount-1].Timestamp, "metricsCount", metricsCount)
	}
	return nil
}

func (s *Autoscaler) loadRealTimeMetrics(ctx context.Context) {
	log := log.FromContext(ctx)
	log.Info("loading realtime metrics")

	workersMetrics, err := s.metricsProvider.GetWorkersMetrics(ctx)
	if err != nil {
		log.Error(err, "failed to get workers metrics")
		return
	}

	for _, sample := range workersMetrics {
		if workload, exists := s.findWorkloadState(sample.Namespace, sample.WorkloadName); exists {
			workload.AddSample(sample)
		}
	}
}

func (s *Autoscaler) processWorkloads(ctx context.Context) {
	log := log.FromContext(ctx)
	log.Info("processing workloads")

	for _, workload := range s.workloads {
		resources, err := recommender.GetResourcesFromRecommenders(ctx, workload, s.recommenders)
		if err != nil {
			log.Error(err, "failed to get resources from recommenders", "workload", workload.Name)
			continue
		}

		if workload.IsAutoSetResourcesEnabled() {
			if err := s.workloadHandler.ApplyResourcesToWorkload(ctx, workload, resources); err != nil {
				log.Error(err, "failed to apply resources to workload", "workload", workload.Name)
			}
		}

		if err := s.workloadHandler.UpdateWorkloadStatus(ctx, workload, resources); err != nil {
			log.Error(err, "failed to update workload status", "workload", workload.Name)
		}
	}
}

func (s *Autoscaler) findOrCreateWorkloadState(namespace, name string) *workload.State {
	w, exists := s.findWorkloadState(namespace, name)
	if !exists {
		w = workload.NewWorkloadState()
		s.workloads[WorkloadID{namespace, name}] = w
	}
	return w
}

func (s *Autoscaler) findWorkloadState(namespace, name string) (*workload.State, bool) {
	w, exists := s.workloads[WorkloadID{namespace, name}]
	return w, exists
}

// Start after manager started
func SetupWithManager(mgr ctrl.Manager, allocator *gpuallocator.GpuAllocator) error {
	metricsProvider, err := metrics.NewProvider()
	if err != nil {
		return fmt.Errorf("failed to create metrics provider: %v", err)
	}
	autoScaler, err := NewAutoscaler(mgr.GetClient(), allocator, metricsProvider)
	if err != nil {
		return fmt.Errorf("failed to create auto scaler: %v", err)
	}
	return mgr.Add(autoScaler)
}
