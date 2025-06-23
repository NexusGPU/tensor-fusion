package autoscaler

import (
	"context"
	"errors"
	"math/big"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
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
	Recommender
	MetricsProvider
	WorkloadStates map[string]*WorkloadState
	WorkerStates   map[string]*WorkerState
}

func NewAutoscaler(c client.Client) (*Autoscaler, error) {
	if c == nil {
		return nil, errors.New("must specify client")
	}

	return &Autoscaler{
		Client:          c,
		Recommender:     NewRecommender(),
		MetricsProvider: NewMetricsProvider(nil),
		WorkloadStates:  map[string]*WorkloadState{},
		WorkerStates:    map[string]*WorkerState{},
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
		autoScalingConfig := workload.Spec.AutoScalingConfig
		// Currently only supports enabling both AutoSetLimits and AutoSetRequests simultaneously
		if !workload.DeletionTimestamp.IsZero() ||
			!autoScalingConfig.AutoSetLimits.Enable ||
			!autoScalingConfig.AutoSetRequests.Enable {
			continue
		}

		workloadName := workload.Name
		workloadState, exists := s.WorkloadStates[workloadName]
		if !exists {
			workloadState = NewWorkloadState(workloadName)
		}
		workloadState.Namespace = workload.Namespace
		workloadState.Resources = workload.Spec.Resources
		workloadState.AutoScalingConfig = autoScalingConfig
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
	s.WorkloadStates = lo.OmitBy(s.WorkloadStates, func(key string, value *WorkloadState) bool {
		return !observedWorkloads[key]
	})

	// remove unused workerStates
	s.WorkerStates = lo.OmitBy(s.WorkerStates, func(key string, state *WorkerState) bool {
		return !observedWorkloads[state.Workload]
	})
}

func (s *Autoscaler) LoadHistoryMetrics(ctx context.Context) {
	log := log.FromContext(ctx)
	log.Info("loading historical metrics")

	workersMetrics := s.MetricsProvider.GetHistoryMetrics()
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

	workersMetrics := s.MetricsProvider.GetWorkersMetrics()
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

		// TODO: apply config
		//  asConfig := workloadState.AutoScalingConfig
		rr := s.Recommender.GetRecommendedResources(workloadState)
		log.Info("Autoscaler processWorkloads", "recommended resources", rr)

		for _, worker := range podList.Items {
			if !worker.DeletionTimestamp.IsZero() {
				continue
			}

			annotations := worker.GetAnnotations()
			newAnnotations := map[string]string{}

			tflopsRequest, err := resource.ParseQuantity(annotations[constants.TFLOPSRequestAnnotation])
			if err != nil {
				log.Error(err, "failed to parse vram request")
				continue
			}
			if tflopsRequest.Cmp(QuantityFromAmount(rr.LowerBoundTflops)) < 0 ||
				tflopsRequest.Cmp(QuantityFromAmount(rr.UpperBoundTflops)) > 0 {
				targetTflopsRequest := QuantityFromAmount(rr.TargetTflops)
				newAnnotations[constants.TFLOPSRequestAnnotation] = targetTflopsRequest.String()
				tflopsLimit, err := resource.ParseQuantity(annotations[constants.TFLOPSLimitAnnotation])
				if err != nil {
					log.Error(err, "failed to parse tflops limit annotation")
					continue
				}
				targetTflopsLimit := getProportionalLimit(&tflopsLimit, &tflopsRequest, &targetTflopsRequest)
				if targetTflopsLimit == nil {
					log.Error(err, "failed to get limit for tflops")
					continue
				}
				newAnnotations[constants.TFLOPSLimitAnnotation] = targetTflopsLimit.String()
			}

			vramRequest, err := resource.ParseQuantity(annotations[constants.VRAMRequestAnnotation])
			if err != nil {
				log.Error(err, "failed to parse vram request")
				continue
			}
			if vramRequest.Cmp(QuantityFromAmount(rr.LowerBoundVram)) < 0 ||
				vramRequest.Cmp(QuantityFromAmount(rr.UpperBoundVram)) > 0 {
				targetVramRequest := QuantityFromAmount(rr.TargetVram)
				newAnnotations[constants.VRAMRequestAnnotation] = targetVramRequest.String()
				vramLimit, err := resource.ParseQuantity(annotations[constants.VRAMLimitAnnotation])
				if err != nil {
					log.Error(err, "failed to parse vram limit annotation")
					continue
				}
				targetVramLimit := getProportionalLimit(&vramLimit, &vramRequest, &targetVramRequest)
				if targetVramLimit == nil {
					log.Error(err, "failed to get limit for vram")
					continue
				}
				newAnnotations[constants.VRAMLimitAnnotation] = targetVramLimit.String()
			}

			if len(newAnnotations) > 0 {
				for key, value := range newAnnotations {
					worker.Annotations[key] = value
				}

				// TODO: replace using the patch method
				if err := s.Update(ctx, &worker); err != nil {
					log.Error(err, "failed to update worker")
				}
			}
		}
	}
}

func (*Autoscaler) addSamples(workloadState *WorkloadState, workerState *WorkerState, metrics *WorkerMetrics) {
	workerState.AddTflopsSample(workloadState, metrics)
	workerState.AddVramSample(workloadState, metrics)
	workloadState.UpdateSampleStats(metrics)
}

func getProportionalLimit(originalLimit, originalRequest, recommendedRequest *resource.Quantity) *resource.Quantity {
	if (originalLimit == nil || originalLimit.IsZero()) ||
		(recommendedRequest == nil || recommendedRequest.IsZero()) ||
		(originalRequest == nil || originalRequest.IsZero()) {
		return nil
	}

	originalValue := big.NewInt(originalLimit.Value())
	scaleBaseValue := big.NewInt(originalRequest.Value())
	scaleResultValue := big.NewInt(recommendedRequest.Value())
	var scaledOriginal big.Int
	scaledOriginal.Mul(originalValue, scaleResultValue)
	scaledOriginal.Div(&scaledOriginal, scaleBaseValue)
	if scaledOriginal.IsInt64() {
		result := resource.NewQuantity(scaledOriginal.Int64(), originalLimit.Format)
		return result
	}

	return nil
}

// Start after manager started
func SetupWithManager(mgr ctrl.Manager) error {
	autoScaler, err := NewAutoscaler(mgr.GetClient())
	if err != nil {
		return err
	}
	return mgr.Add(autoScaler)
}
