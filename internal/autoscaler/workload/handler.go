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

type Handler struct {
	client.Client
	allocator *gpuallocator.GpuAllocator
}

func NewHandler(client client.Client, allocator *gpuallocator.GpuAllocator) *Handler {
	return &Handler{
		Client:    client,
		allocator: allocator,
	}
}

func (h *Handler) UpdateWorkers(ctx context.Context, workload *WorkloadState) {
	workerList := &corev1.PodList{}
	if err := h.List(ctx, workerList,
		client.InNamespace(workload.Namespace),
		client.MatchingLabels{constants.WorkloadKey: workload.Name}); err != nil {
		log.FromContext(ctx).Error(err, "failed to list workers")
		return
	}
	workload.UpdateWorkers(workerList)
}

func (h *Handler) ProcessWorkload(ctx context.Context, workload *WorkloadState) {
	log := log.FromContext(ctx)
	workerList := &corev1.PodList{}
	if err := h.List(ctx, workerList,
		client.InNamespace(workload.Namespace),
		client.MatchingLabels{constants.WorkloadKey: workload.Name}); err != nil {
		log.Error(err, "failed to list workers")
	}

	if !workload.IsAutoScalingEnabled() {
		return
	}

	for _, worker := range workerList.Items {
		if !worker.DeletionTimestamp.IsZero() {
			continue
		}

		if err := h.UpdateWorkerResourcesIfNeeded(ctx, workload, &worker); err != nil {
			log.Error(err, "failed to update worker")
		}
	}
}

func (h *Handler) UpdateWorkerResourcesIfNeeded(ctx context.Context, workload *WorkloadState, worker *corev1.Pod) error {
	log := log.FromContext(ctx)

	adjustRequest, err := getCurrentWorkerResourceRequest(worker)
	if err != nil {
		return fmt.Errorf("failed to get current worker resource request, %v", err)
	}

	rr := &workload.Recommendation
	resourcesInfo := []struct {
		name       tfv1.ResourceName
		requestKey string
		limitKey   string
		request    *resource.Quantity
		limit      *resource.Quantity
		lowerBound resource.Quantity
		upperBound resource.Quantity
		target     resource.Quantity
	}{
		{
			name:       tfv1.ResourceTflops,
			requestKey: constants.TFLOPSRequestAnnotation,
			limitKey:   constants.TFLOPSLimitAnnotation,
			request:    &adjustRequest.NewRequest.Tflops,
			limit:      &adjustRequest.NewLimit.Tflops,
			lowerBound: rr.LowerBoundTflops,
			upperBound: rr.UpperBoundTflops,
			target:     rr.TargetTflops,
		},
		{
			name:       tfv1.ResourceVram,
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
