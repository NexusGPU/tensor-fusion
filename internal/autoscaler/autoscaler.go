package autoscaler

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/recommender"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

var (
	_ manager.Runnable               = (*Autoscaler)(nil)
	_ manager.LeaderElectionRunnable = (*Autoscaler)(nil)

	DefaultAutoScalingInterval      = "30s"
	MaxConcurrentWorkloadProcessing = 10
	FocusWorkloadName               = ""
)

func init() {
	if utils.IsDebugMode() {
		MaxConcurrentWorkloadProcessing = 1
	}
	focusWorkloadName := os.Getenv("AUTOSCALER_FOCUS_WORKLOAD_NAME")
	if focusWorkloadName != "" {
		FocusWorkloadName = focusWorkloadName
	}
}

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
		recommender.NewExternalRecommender(client, recommendationProcessor),
	}

	scaler := &Autoscaler{
		Client:          client,
		allocator:       allocator,
		metricsProvider: metricsProvider,
		recommenders:    recommenders,
		workloadHandler: workloadHandler,
		workloads:       map[WorkloadID]*workload.State{},
		metricsLoader:   newWorkloadMetricsLoader(client, metricsProvider),
	}
	scaler.metricsLoader.setProcessFunc(scaler.processSingleWorkload)
	return scaler, nil
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
		if workload.Status.WorkerCount == 0 {
			continue
		}

		// focus to certain name workload (for verification test or debug)
		if FocusWorkloadName != "" && workload.Name != FocusWorkloadName {
			continue
		}

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

func (s *Autoscaler) processSingleWorkload(ctx context.Context, workload *workload.State) {
	log := log.FromContext(ctx)
	recommendation, err := recommender.GetRecommendation(ctx, workload, s.recommenders)
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
	recorder := mgr.GetEventRecorder("autoscaler")
	autoScaler.workloadHandler.SetEventRecorder(recorder, mgr.GetScheme())
	return mgr.Add(autoScaler)
}
