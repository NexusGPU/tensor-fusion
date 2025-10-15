package workload

import (
	"context"
	"fmt"
	"maps"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type Handler interface {
	UpdateWorkloadState(ctx context.Context, workloadState *State, workload *tfv1.TensorFusionWorkload) error
	ApplyRecommendationToWorkload(ctx context.Context, workloadState *State, recommendation *tfv1.Resources) error
	UpdateWorkloadStatus(ctx context.Context, state *State, recommendation *tfv1.Resources) error
	GetMaxAllowedResourcesSpec(workload *State) (*tfv1.Resource, error)
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

func (h *handler) UpdateWorkloadState(ctx context.Context, workloadState *State, workload *tfv1.TensorFusionWorkload) error {
	workloadState.Namespace = workload.Namespace
	workloadState.Name = workload.Name
	workloadState.Spec = workload.Spec
	workloadState.Status = *workload.Status.DeepCopy()

	workerList := &corev1.PodList{}
	if err := h.List(ctx, workerList,
		client.InNamespace(workloadState.Namespace),
		client.MatchingLabels{constants.WorkloadKey: workloadState.Name}); err != nil {
		return err
	}
	workloadState.updateCurrentActiveWorkers(workerList)
	return nil
}

func (h *handler) ApplyRecommendationToWorkload(ctx context.Context, workload *State, recommendation *tfv1.Resources) error {
	// If the latest recommendation has not been applied to all workers,
	// we need to retry the update
	if recommendation == nil && !workload.IsRecommendationAppliedToAllWorkers() {
		recommendation = workload.Status.Recommendation
	}

	if recommendation != nil {
		workload.Status.AppliedRecommendedReplicas = 0
		for _, worker := range workload.CurrentActiveWorkers {
			if isWorkerHasDedicatedGPU(worker) {
				continue
			}

			if err := h.applyRecommendationToWorker(ctx, workload, worker, recommendation); err != nil {
				log.FromContext(ctx).Error(err, "failed to update worker resources", "worker", worker.Name)
				continue
			}
			workload.Status.AppliedRecommendedReplicas++
		}
	}

	return nil
}

func (h *handler) UpdateWorkloadStatus(ctx context.Context, state *State, recommendation *tfv1.Resources) error {
	workload := &tfv1.TensorFusionWorkload{}
	if err := h.Get(ctx, client.ObjectKey{Namespace: state.Namespace, Name: state.Name}, workload); err != nil {
		return fmt.Errorf("failed to get workload: %v", err)
	}

	if recommendation == nil &&
		!isAppliedRecommendedReplicasChanged(workload, state) {
		return nil
	}

	patch := client.MergeFrom(workload.DeepCopy())
	if isRecommendationChanged(&workload.Status, recommendation) {
		workload.Status.Recommendation = recommendation.DeepCopy()
		workload.Status.ActiveCronScalingRule = state.Status.ActiveCronScalingRule.DeepCopy()
		if condition := meta.FindStatusCondition(state.Status.Conditions,
			constants.ConditionStatusTypeRecommendationProvided); condition != nil {
			meta.SetStatusCondition(&workload.Status.Conditions, *condition)
		}
	}
	workload.Status.AppliedRecommendedReplicas = state.Status.AppliedRecommendedReplicas
	if err := h.Status().Patch(ctx, workload, patch); err != nil {
		return fmt.Errorf("failed to patch workload status %s: %v", workload.Name, err)
	}
	log.FromContext(ctx).Info("workload recommendation status updated successfully",
		"workload", workload.Name, "recommendation", recommendation)

	return nil
}

func isRecommendationChanged(status *tfv1.TensorFusionWorkloadStatus, recommendation *tfv1.Resources) bool {
	return recommendation != nil && (status.Recommendation == nil || !status.Recommendation.Equal(recommendation))
}

func isAppliedRecommendedReplicasChanged(workload *tfv1.TensorFusionWorkload, state *State) bool {
	return workload.Status.AppliedRecommendedReplicas != state.Status.AppliedRecommendedReplicas
}

func (h *handler) applyRecommendationToWorker(ctx context.Context, workload *State, worker *corev1.Pod, recommendation *tfv1.Resources) error {
	log := log.FromContext(ctx)

	curRes, err := utils.GPUResourcesFromAnnotations(worker.Annotations)
	if err != nil {
		log.Error(err, "invalid GPU resources annotations")
	}

	if recommendation.Equal(curRes) {
		return nil
	}

	annotationsToUpdate := utils.GPUResourcesToAnnotations(recommendation)
	if !workload.ShouldScaleResource(tfv1.ResourceTflops) {
		delete(annotationsToUpdate, constants.TFLOPSRequestAnnotation)
		delete(annotationsToUpdate, constants.TFLOPSLimitAnnotation)
	}
	if !workload.ShouldScaleResource(tfv1.ResourceVram) {
		delete(annotationsToUpdate, constants.VRAMRequestAnnotation)
		delete(annotationsToUpdate, constants.VRAMLimitAnnotation)
	}

	if len(annotationsToUpdate) <= 0 {
		return nil
	}

	isScaleUp := recommendation.Requests.Tflops.Cmp(curRes.Requests.Tflops) > 0 ||
		recommendation.Requests.Vram.Cmp(curRes.Requests.Vram) > 0

	if _, err := h.allocator.AdjustAllocation(ctx, tfv1.AdjustRequest{
		PodUID:     string(worker.UID),
		IsScaleUp:  isScaleUp,
		NewRequest: recommendation.Requests,
		NewLimit:   recommendation.Limits,
	}, true); err != nil {
		return fmt.Errorf("failed to adjust allocation: %v", err)
	}

	patch := client.MergeFrom(worker.DeepCopy())
	maps.Copy(worker.Annotations, annotationsToUpdate)
	if err := h.Patch(ctx, worker, patch); err != nil {
		return fmt.Errorf("failed to patch worker %s: %v", worker.Name, err)
	}

	log.Info("apply recommendation to worker successfully",
		"worker", worker.Name, "recommendation", recommendation, "currentResources", curRes)

	return nil
}

func (h *handler) GetMaxAllowedResourcesSpec(workload *State) (*tfv1.Resource, error) {
	if len(workload.CurrentActiveWorkers) <= 0 {
		return nil, nil
	}

	gpuStore, _, allocRequests := h.allocator.GetAllocationInfo()
	gpuToWorkers := map[*tfv1.GPU][]*corev1.Pod{}
	for _, worker := range workload.CurrentActiveWorkers {
		allocated, exists := allocRequests[string(worker.UID)]
		if !exists || allocated == nil {
			return nil, fmt.Errorf("worker %s has not allocated GPUs", worker.Name)
		}
		for _, gpuName := range allocated.GPUNames {
			gpuNameNs := types.NamespacedName{Name: gpuName}
			gpu, exists := gpuStore[gpuNameNs]
			if !exists {
				return nil, fmt.Errorf("GPU not found in allocator store %s", gpuName)
			}
			gpuToWorkers[gpu] = append(gpuToWorkers[gpu], worker)
		}
	}

	var (
		maxTflops int64 = -1
		maxVram   int64 = -1
	)
	for gpu, workers := range gpuToWorkers {
		if gpu.Status.Available == nil {
			return nil, fmt.Errorf("GPU available is nil")
		}
		avaiableTflops := gpu.Status.Available.Tflops.DeepCopy()
		avaiableVram := gpu.Status.Available.Vram.DeepCopy()
		for _, worker := range workers {
			avaiableTflops.Add(allocRequests[string(worker.UID)].Request.Tflops)
			avaiableVram.Add(allocRequests[string(worker.UID)].Request.Vram)
		}

		workerCount := int64(len(workers))
		tflopsPerWorker := int64(avaiableTflops.AsApproximateFloat64()) / workerCount
		vramPerWorker := avaiableVram.Value() / workerCount
		if maxTflops == -1 || tflopsPerWorker < maxTflops {
			maxTflops = tflopsPerWorker
		}
		if maxVram == -1 || vramPerWorker < maxVram {
			maxVram = vramPerWorker
		}
	}

	return &tfv1.Resource{
		Tflops: *resource.NewQuantity(maxTflops, resource.DecimalSI),
		Vram:   *resource.NewQuantity(maxVram, resource.BinarySI),
	}, nil
}
