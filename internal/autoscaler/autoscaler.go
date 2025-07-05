package autoscaler

import (
	"context"
	"errors"
	"fmt"
	"math/big"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
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
	allocator *gpuallocator.GpuAllocator
	ResourceRecommender
	MetricsProvider
	WorkloadStates map[string]*WorkloadState
	WorkerStates   map[string]*WorkerState
}

func NewAutoscaler(c client.Client, allocator *gpuallocator.GpuAllocator) (*Autoscaler, error) {
	if c == nil {
		return nil, errors.New("must specify client")
	}

	if allocator == nil {
		return nil, errors.New("must specify allocator")
	}

	return &Autoscaler{
		Client:              c,
		allocator:           allocator,
		ResourceRecommender: NewResourceRecommender(),
		MetricsProvider:     NewMetricsProvider(nil),
		WorkloadStates:      map[string]*WorkloadState{},
		WorkerStates:        map[string]*WorkerState{},
	}, nil
}

func (s *Autoscaler) Start(ctx context.Context) error {
	log := log.FromContext(ctx)
	log.Info("Starting autoscaler")

	s.LoadHistoryMetrics(ctx) // TODO: handle timeout

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
	s.LoadWorkloads(ctx)
	s.LoadRealTimeMetrics(ctx)
	s.ProcessWorkloads(ctx)
}

func (s *Autoscaler) LoadWorkloads(ctx context.Context) {
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

		workloadName := workload.Name
		workloadState, exists := s.WorkloadStates[workloadName]
		if !exists {
			workloadState = NewWorkloadState(workloadName)
		}
		workloadState.Namespace = workload.Namespace
		workloadState.Resources = workload.Spec.Resources
		workloadState.AutoScalingConfig = workload.Spec.AutoScalingConfig
		s.WorkloadStates[workloadName] = workloadState

		observedWorkloads[workloadName] = true

		podList := &corev1.PodList{}
		if err := s.List(ctx, podList,
			client.InNamespace(workload.Namespace),
			client.MatchingLabels{constants.WorkloadKey: workload.Name}); err != nil {
			log.Error(err, "failed to list workers")
			continue
		}

		observedWorkers := map[string]bool{}
		for _, worker := range podList.Items {
			if !worker.DeletionTimestamp.IsZero() {
				continue
			}
			if _, exists := s.WorkerStates[worker.Name]; !exists {
				s.WorkerStates[worker.Name] = NewWorkerState(worker.Name, workloadName)
			}
			observedWorkers[worker.Name] = true
		}

		s.WorkerStates = lo.OmitBy(s.WorkerStates, func(key string, state *WorkerState) bool {
			return state.Workload == workloadName && !observedWorkers[key]
		})
	}

	// remove unused workloadStates
	s.WorkloadStates = lo.OmitBy(s.WorkloadStates, func(key string, _ *WorkloadState) bool {
		return !observedWorkloads[key]
	})

	// remove unused workerStates
	s.WorkerStates = lo.OmitBy(s.WorkerStates, func(_ string, state *WorkerState) bool {
		return !observedWorkloads[state.Workload]
	})
}

func (s *Autoscaler) LoadHistoryMetrics(ctx context.Context) {
	log := log.FromContext(ctx)
	log.Info("loading historical metrics")

	workersMetrics, err := s.GetHistoryMetrics()
	if err != nil {
		log.Error(err, "failed to get history metrics")
		return
	}
	for _, metrics := range workersMetrics {
		workloadState, exists := s.WorkloadStates[metrics.WorkloadName]
		if !exists {
			workloadState = NewWorkloadState(metrics.WorkloadName)
			s.WorkloadStates[metrics.WorkloadName] = workloadState
		}
		workerState, exists := s.WorkerStates[metrics.WorkerName]
		if !exists {
			workerState = NewWorkerState(metrics.WorkerName, metrics.WorkloadName)
			s.WorkerStates[metrics.WorkerName] = workerState
		}

		s.addSamples(workloadState, workerState, metrics)
	}
}

func (s *Autoscaler) LoadRealTimeMetrics(ctx context.Context) {
	log := log.FromContext(ctx)
	log.Info("loading realtime metrics")

	workersMetrics, err := s.GetWorkersMetrics()
	if err != nil {
		log.Error(err, "failed to get workers metrics")
		return
	}

	for _, metrics := range workersMetrics {
		workloadState, workloadExists := s.WorkloadStates[metrics.WorkloadName]
		if !workloadExists {
			continue
		}
		workerState, workerExists := s.WorkerStates[metrics.WorkerName]
		if !workerExists {
			continue
		}

		s.addSamples(workloadState, workerState, metrics)
	}
}

func (s *Autoscaler) ProcessWorkloads(ctx context.Context) {
	log := log.FromContext(ctx)
	log.Info("processing workloads")

	for _, workloadState := range s.WorkloadStates {
		// TODO: continue if histogram is empty
		podList := &corev1.PodList{}
		if err := s.List(ctx, podList,
			client.InNamespace(workloadState.Namespace),
			client.MatchingLabels{constants.WorkloadKey: workloadState.Name}); err != nil {
			log.Error(err, "failed to list workers")
			continue
		}

		if len(podList.Items) <= 0 {
			continue
		}

		rr := s.GetRecommendedResources(workloadState)
		log.Info("recommend resources", "workload", workloadState.Name, "resources", rr)

		// TODO: update recommmendation status of workload

		if !workloadState.IsAutoScalingEnabled() {
			continue
		}

		for _, worker := range podList.Items {
			if !worker.DeletionTimestamp.IsZero() {
				continue
			}

			if err := s.updateWorkerResourcesIfNeeded(ctx, workloadState, &worker, rr); err != nil {
				log.Error(err, "failed to update worker")
			}
		}
	}
}

func (s *Autoscaler) updateWorkerResourcesIfNeeded(ctx context.Context, workloadState *WorkloadState, worker *corev1.Pod, rr *RecommendedResources) error {
	log := log.FromContext(ctx)

	adjustRequest, err := getInitialWorkerResourceRequest(worker)
	if err != nil {
		return fmt.Errorf("failed to get initial worker resource request, %v", err)
	}
	resourcesInfo := []struct {
		name       ResourceName
		requestKey string
		limitKey   string
		request    *resource.Quantity
		limit      *resource.Quantity
		lowerBound ResourceAmount
		upperBound ResourceAmount
		target     ResourceAmount
	}{
		{
			name:       ResourceTflops,
			requestKey: constants.TFLOPSRequestAnnotation,
			limitKey:   constants.TFLOPSLimitAnnotation,
			request:    &adjustRequest.NewRequest.Tflops,
			limit:      &adjustRequest.NewLimit.Tflops,
			lowerBound: rr.LowerBoundTflops,
			upperBound: rr.UpperBoundTflops,
			target:     rr.TargetTflops,
		},
		{
			name:       ResourceVram,
			requestKey: constants.VRAMRequestAnnotation,
			limitKey:   constants.VRAMLimitAnnotation,
			request:    &adjustRequest.NewRequest.Vram,
			limit:      &adjustRequest.NewLimit.Vram,
			lowerBound: rr.LowerBoundVram,
			upperBound: rr.UpperBoundVram,
			target:     rr.TargetVram,
		},
	}

	newAnnotations := map[string]string{}
	var upScaling, downScaling bool
	for _, resInfo := range resourcesInfo {
		if !workloadState.IsTargetResource(resInfo.name) {
			continue
		}
		upScaling = resInfo.request.Cmp(QuantityFromAmount(resInfo.lowerBound)) < 0
		downScaling = resInfo.request.Cmp(QuantityFromAmount(resInfo.upperBound)) > 0
		if upScaling || downScaling {
			targetRequest := QuantityFromAmount(resInfo.target)
			targetLimit := getProportionalLimit(resInfo.limit, resInfo.request, &targetRequest)
			if targetLimit == nil {
				return fmt.Errorf("failed to get limit for %s", resInfo.requestKey)
			}
			newAnnotations[resInfo.requestKey] = targetRequest.String()
			newAnnotations[resInfo.limitKey] = targetLimit.String()
			*resInfo.request = targetRequest
			*resInfo.limit = *targetLimit
		}
	}

	if len(newAnnotations) > 0 {
		adjustRequest.IsScaleUp = upScaling
		if _, err := s.allocator.AdjustAllocation(ctx, *adjustRequest, true); err != nil {
			return fmt.Errorf("failed to adjust allocation: %v", err)
		}
		log.Info("adjust allocation successfully", "adjustRequest", adjustRequest)
		// Patch the worker with updated annotations
		patch := client.MergeFrom(worker.DeepCopy())
		for key, value := range newAnnotations {
			worker.Annotations[key] = value
		}
		if err := s.Patch(ctx, worker, patch); err != nil {
			return fmt.Errorf("failed to patch worker: %v", err)
		}
	}

	return nil
}

func (*Autoscaler) addSamples(workloadState *WorkloadState, workerState *WorkerState, metrics *WorkerMetrics) {
	workerState.AddTflopsSample(workloadState, metrics)
	workerState.AddVramSample(workloadState, metrics)
	workloadState.UpdateSampleStats(metrics)
}

func getProportionalLimit(originalLimit, originalRequest, recommendedRequest *resource.Quantity) *resource.Quantity {
	if originalLimit == nil || originalLimit.IsZero() ||
		originalRequest == nil || originalRequest.IsZero() ||
		recommendedRequest == nil || recommendedRequest.IsZero() {
		return nil
	}

	originalValue := big.NewInt(originalLimit.Value())
	scaleBaseValue := big.NewInt(originalRequest.Value())
	scaleResultValue := big.NewInt(recommendedRequest.Value())
	var scaledOriginal big.Int
	scaledOriginal.Mul(originalValue, scaleResultValue)
	scaledOriginal.Div(&scaledOriginal, scaleBaseValue)
	if scaledOriginal.IsInt64() {
		return resource.NewQuantity(scaledOriginal.Int64(), originalLimit.Format)
	}

	return nil
}

func getInitialWorkerResourceRequest(worker *corev1.Pod) (*tfv1.AdjustRequest, error) {
	adjustRequest := tfv1.AdjustRequest{
		PodUID:     string(worker.UID),
		IsScaleUp:  false,
		NewRequest: tfv1.Resource{},
		NewLimit:   tfv1.Resource{},
	}
	annotations := worker.GetAnnotations()
	resInfo := []struct {
		key string
		dst *resource.Quantity
	}{
		{constants.TFLOPSRequestAnnotation, &adjustRequest.NewRequest.Tflops},
		{constants.TFLOPSLimitAnnotation, &adjustRequest.NewLimit.Tflops},
		{constants.VRAMRequestAnnotation, &adjustRequest.NewRequest.Vram},
		{constants.VRAMLimitAnnotation, &adjustRequest.NewLimit.Vram},
	}
	for _, info := range resInfo {
		q, err := resource.ParseQuantity(annotations[info.key])
		if err != nil {
			return nil, fmt.Errorf("failed to parse %s: %v", info.key, err)
		}
		*info.dst = q
	}

	return &adjustRequest, nil
}

// Start after manager started
func SetupWithManager(mgr ctrl.Manager, allocator *gpuallocator.GpuAllocator) error {
	autoScaler, err := NewAutoscaler(mgr.GetClient(), allocator)
	if err != nil {
		return err
	}
	return mgr.Add(autoScaler)
}
