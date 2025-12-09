package autoscaler

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/recommender"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

var (
	_ manager.Runnable               = (*Autoscaler)(nil)
	_ manager.LeaderElectionRunnable = (*Autoscaler)(nil)

	DefaultAutoScalingInterval = "30s"
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
	metricsLoader   *workloadMetricsLoader
}

func NewAutoscaler(
	client client.Client,
	allocator *gpuallocator.GpuAllocator,
	metricsProvider metrics.Provider) (*Autoscaler, error) {
	if client == nil {
		return nil, errors.New("must specify client")
	}

	if allocator == nil {
		return nil, errors.New("must specify allocator")
	}

	if metricsProvider == nil {
		return nil, errors.New("must specify metricsProvider")
	}

	workloadHandler := workload.NewHandler(client, allocator)
	recommendationProcessor := recommender.NewRecommendationProcessor(workloadHandler)
	recommenders := []recommender.Interface{
		recommender.NewPercentileRecommender(recommendationProcessor),
		recommender.NewCronRecommender(recommendationProcessor),
		// ExternalRecommender will be added per-workload if configured
	}

	return &Autoscaler{
		Client:          client,
		allocator:       allocator,
		metricsProvider: metricsProvider,
		recommenders:    recommenders,
		workloadHandler: workloadHandler,
		workloads:       map[WorkloadID]*workload.State{},
		metricsLoader:   newWorkloadMetricsLoader(client, metricsProvider),
	}, nil
}

func (s *Autoscaler) Start(ctx context.Context) error {
	log := log.FromContext(ctx)
	log.Info("Starting autoscaler")

	// No longer load all history metrics at startup
	// Each workload will load its own history after InitialDelayPeriod

	autoScalingInterval := config.GetGlobalConfig().AutoScalingInterval
	if autoScalingInterval == "" {
		autoScalingInterval = DefaultAutoScalingInterval
	}
	interval, err := time.ParseDuration(autoScalingInterval)
	if err != nil {
		log.Error(err, "failed to parse auto scaling interval")
		return err
	}
	ticker := time.NewTicker(interval)
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
	// Metrics loading is now handled per-workload in goroutines
	s.processWorkloads(ctx)
}

func (s *Autoscaler) loadWorkloads(ctx context.Context) {
	log := log.FromContext(ctx)

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

		// Register workload with metrics loader for per-workload goroutine-based metrics fetching
		s.metricsLoader.addWorkload(ctx, workloadID, workloadState)
	}

	// remove non-existent workloads
	for workloadID := range s.workloads {
		if !activeWorkloads[workloadID] {
			s.metricsLoader.removeWorkload(workloadID)
			delete(s.workloads, workloadID)
		}
	}

	log.Info("workloads loaded", "workloadCount", len(s.workloads))
}

// loadHistoryMetrics and loadRealTimeMetrics are now handled per-workload
// in workloadMetricsLoader goroutines

func (s *Autoscaler) processWorkloads(ctx context.Context) {
	workloadList := make([]*workload.State, 0, len(s.workloads))
	for _, w := range s.workloads {
		workloadList = append(workloadList, w)
	}

	if len(workloadList) == 0 {
		return
	}

	maxWorkers := min(len(workloadList), constants.MaxConcurrentWorkloadProcessing)
	chunkSize := (len(workloadList) + maxWorkers - 1) / maxWorkers

	var wg sync.WaitGroup
	for i := 0; i < len(workloadList); i += chunkSize {
		end := min(i+chunkSize, len(workloadList))
		chunk := workloadList[i:end]
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, w := range chunk {
				s.processSingleWorkload(ctx, w)
			}
		}()
	}
	wg.Wait()
}

func (s *Autoscaler) processSingleWorkload(ctx context.Context, workload *workload.State) {
	log := log.FromContext(ctx)

	// Build recommenders list - add external recommender if configured
	recommenders := s.recommenders
	externalScalerConfig := workload.Spec.AutoScalingConfig.ExternalScaler
	if externalScalerConfig != nil && externalScalerConfig.Enable {
		recommendationProcessor := recommender.NewRecommendationProcessor(s.workloadHandler)
		externalRecommender := recommender.NewExternalRecommender(s.Client, externalScalerConfig, recommendationProcessor)
		recommenders = append(recommenders, externalRecommender)
	}

	recommendation, err := recommender.GetRecommendation(ctx, workload, recommenders)
	if err != nil {
		log.Error(err, "failed to get recommendation", "workload", workload.Name)
		return
	}

	if workload.IsAutoSetResourcesEnabled() {
		if err := s.workloadHandler.ApplyRecommendationToWorkload(ctx, workload, recommendation); err != nil {
			log.Error(err, "failed to apply recommendation to workload", "workload", workload.Name)
		}
	}

	if err := s.workloadHandler.UpdateWorkloadStatus(ctx, workload, recommendation); err != nil {
		log.Error(err, "failed to update workload status", "workload", workload.Name)
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
	// Update handler with event recorder
	recorder := mgr.GetEventRecorderFor("autoscaler")
	autoScaler.workloadHandler.SetEventRecorder(recorder, mgr.GetScheme())
	return mgr.Add(autoScaler)
}
