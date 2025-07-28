package workload

import (
	"context"
	"fmt"
	"math/big"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type Handler interface {
	UpdateWorkloadState(ctx context.Context, workloadState *State, workload *tfv1.TensorFusionWorkload)
	ApplyRecommendationToWorkload(ctx context.Context, state *State, recommendation *tfv1.RecommendedResources) error
	UpdateWorkerResourcesIfNeeded(ctx context.Context, workload *State, worker *corev1.Pod) error
}

type handler struct {
	client.Client
	allocator *gpuallocator.GpuAllocator
}

func NewHandler(client client.Client, allocator *gpuallocator.GpuAllocator) Handler {
	return &handler{
		Client:    client,
		allocator: allocator,
	}
}

func (h *handler) UpdateWorkloadState(ctx context.Context, workloadState *State, workload *tfv1.TensorFusionWorkload) {
	workloadState.Namespace = workload.Namespace
	workloadState.Spec = workload.Spec
	workloadState.Annotations = workload.Annotations

	workerList := &corev1.PodList{}
	if err := h.List(ctx, workerList,
		client.InNamespace(workloadState.Namespace),
		client.MatchingLabels{constants.WorkloadKey: workloadState.Name}); err != nil {
		log.FromContext(ctx).Error(err, "failed to list workers")
		return
	}
	workloadState.updateWorkers(workerList)
}

func (h *handler) ApplyRecommendationToWorkload(ctx context.Context, state *State, recommendation *tfv1.RecommendedResources) error {
	workload := &tfv1.TensorFusionWorkload{}
	if err := h.Get(ctx, client.ObjectKey{Namespace: state.Namespace, Name: state.Name}, workload); err != nil {
		return fmt.Errorf("failed to get workload: %v", err)
	}

	// record current and last resources by annotations
	patch := client.MergeFrom(workload.DeepCopy())
	if workload.Annotations == nil {
		workload.Annotations = map[string]string{}
	}
	if tflopsRequest, ok := workload.Annotations[constants.TFLOPSRequestAnnotation]; ok {
		workload.Annotations[constants.LastTFLOPSRequestAnnotation] = tflopsRequest
	} else {
		workload.Annotations[constants.LastTFLOPSRequestAnnotation] = workload.Spec.Resources.Requests.Tflops.String()
	}
	if vramRequest, ok := workload.Annotations[constants.VRAMRequestAnnotation]; ok {
		workload.Annotations[constants.LastVRAMRequestAnnotation] = vramRequest
	} else {
		workload.Annotations[constants.LastVRAMRequestAnnotation] = workload.Spec.Resources.Requests.Vram.String()
	}
	workload.Annotations[constants.TFLOPSRequestAnnotation] = recommendation.TargetTflops.String()
	workload.Annotations[constants.VRAMRequestAnnotation] = recommendation.TargetVram.String()

	if err := h.Patch(ctx, workload, patch); err != nil {
		return fmt.Errorf("failed to patch workload: %v", err)
	}

	state.Annotations = workload.Annotations
	state.Recommendation = *recommendation

	if err := h.ApplyRecommendationToWorkers(ctx, state); err != nil {
		return fmt.Errorf("failed to apply recommendation to workers: %v", err)
	}

	return nil
}

func (h *handler) ApplyRecommendationToWorkers(ctx context.Context, workload *State) error {
	log := log.FromContext(ctx)
	workerList := &corev1.PodList{}
	if err := h.List(ctx, workerList,
		client.InNamespace(workload.Namespace),
		client.MatchingLabels{constants.WorkloadKey: workload.Name}); err != nil {
		log.Error(err, "failed to list workers")
	}

	if !workload.IsAutoSetResourcesEnabled() {
		return nil
	}

	for _, worker := range workerList.Items {
		if !worker.DeletionTimestamp.IsZero() {
			continue
		}

		if err := h.UpdateWorkerResourcesIfNeeded(ctx, workload, &worker); err != nil {
			log.Error(err, "failed to update worker")
		}
	}

	return nil
}

func (h *handler) UpdateWorkerResourcesIfNeeded(ctx context.Context, workload *State, worker *corev1.Pod) error {
	log := log.FromContext(ctx)

	adjustRequest, err := getCurrentWorkerResourceRequest(worker)
	if err != nil {
		return fmt.Errorf("failed to get current worker resource request, %v", err)
	}

	recommendation := &workload.Recommendation
	resourcesInfo := []struct {
		name           tfv1.ResourceName
		requestKey     string
		limitKey       string
		lastRequestKey string
		lastLimitKey   string
		request        *resource.Quantity
		limit          *resource.Quantity
		lowerBound     resource.Quantity
		upperBound     resource.Quantity
		target         resource.Quantity
	}{
		{
			name:           tfv1.ResourceTflops,
			requestKey:     constants.TFLOPSRequestAnnotation,
			limitKey:       constants.TFLOPSLimitAnnotation,
			lastRequestKey: constants.LastTFLOPSRequestAnnotation,
			lastLimitKey:   constants.LastTFLOPSLimitAnnotation,
			request:        &adjustRequest.NewRequest.Tflops,
			limit:          &adjustRequest.NewLimit.Tflops,
			lowerBound:     recommendation.LowerBoundTflops,
			upperBound:     recommendation.UpperBoundTflops,
			target:         recommendation.TargetTflops,
		},
		{
			name:           tfv1.ResourceVram,
			requestKey:     constants.VRAMRequestAnnotation,
			limitKey:       constants.VRAMLimitAnnotation,
			lastRequestKey: constants.LastVRAMRequestAnnotation,
			lastLimitKey:   constants.LastVRAMLimitAnnotation,
			request:        &adjustRequest.NewRequest.Vram,
			limit:          &adjustRequest.NewLimit.Vram,
			lowerBound:     recommendation.LowerBoundVram,
			upperBound:     recommendation.UpperBoundVram,
			target:         recommendation.TargetVram,
		},
	}

	newAnnotations := map[string]string{}
	var upScaling, downScaling bool
	for _, resInfo := range resourcesInfo {
		if !workload.ShouldScaleResource(resInfo.name) {
			continue
		}
		upScaling = resInfo.request.Cmp(resInfo.lowerBound) < 0
		downScaling = resInfo.request.Cmp(resInfo.upperBound) > 0
		if upScaling || downScaling {
			targetRequest := resInfo.target
			targetLimit := getProportionalLimit(resInfo.limit, resInfo.request, &targetRequest)
			if targetLimit == nil {
				return fmt.Errorf("failed to get limit for %s", resInfo.requestKey)
			}
			newAnnotations[resInfo.lastRequestKey] = resInfo.request.String()
			newAnnotations[resInfo.lastLimitKey] = resInfo.limit.String()
			newAnnotations[resInfo.requestKey] = targetRequest.String()
			newAnnotations[resInfo.limitKey] = targetLimit.String()
			*resInfo.request = targetRequest
			*resInfo.limit = *targetLimit
		}
	}

	if len(newAnnotations) > 0 {
		adjustRequest.IsScaleUp = upScaling
		if _, err := h.allocator.AdjustAllocation(ctx, *adjustRequest, true); err != nil {
			return fmt.Errorf("failed to adjust allocation: %v", err)
		}
		log.Info("adjust allocation successfully", "adjustRequest", adjustRequest)
		// Patch the worker with updated annotations
		patch := client.MergeFrom(worker.DeepCopy())
		for key, value := range newAnnotations {
			worker.Annotations[key] = value
		}
		if err := h.Patch(ctx, worker, patch); err != nil {
			return fmt.Errorf("failed to patch worker: %v", err)
		}
	}

	return nil
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

func getCurrentWorkerResourceRequest(worker *corev1.Pod) (*tfv1.AdjustRequest, error) {
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
