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
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type Handler interface {
	UpdateWorkloadState(ctx context.Context, workloadState *State, workload *tfv1.TensorFusionWorkload)
	ApplyResourcesToWorkload(ctx context.Context, workloadState *State, targetRes *tfv1.Resources) error
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

func (h *handler) ApplyResourcesToWorkload(ctx context.Context, workload *State, targetRes *tfv1.Resources) error {
	if workload.IsAutoSetResourcesEnabled() {
		workerList := &corev1.PodList{}
		if err := h.List(ctx, workerList,
			client.InNamespace(workload.Namespace),
			client.MatchingLabels{constants.WorkloadKey: workload.Name}); err != nil {
			return fmt.Errorf("failed to list workers: %v", err)
		}

		for _, worker := range workerList.Items {
			if !worker.DeletionTimestamp.IsZero() {
				continue
			}
			if err := h.applyResourcesToWorker(ctx, workload, &worker, targetRes); err != nil {
				return fmt.Errorf("failed to update worker %s resources: %v", worker.Name, err)
			}
		}
	}

	if err := h.updateWorkload(ctx, workload, targetRes); err != nil {
		return fmt.Errorf("failed to update auto scaling annotations: %v", err)
	}

	return nil
}

func (h *handler) updateWorkload(
	ctx context.Context,
	state *State,
	targetRes *tfv1.Resources) error {
	workload := &tfv1.TensorFusionWorkload{}
	if err := h.Get(ctx, client.ObjectKey{Namespace: state.Namespace, Name: state.Name}, workload); err != nil {
		return fmt.Errorf("failed to get workload: %v", err)
	}

	if workload.Annotations == nil {
		workload.Annotations = map[string]string{}
	}
	patch := client.MergeFrom(workload.DeepCopy())
	maps.Copy(workload.Annotations, utils.GPUResourcesToAnnotations(targetRes))
	maps.Copy(workload.Annotations, state.ScalingAnnotations)
	if err := h.Patch(ctx, workload, patch); err != nil {
		return fmt.Errorf("failed to patch workload %s: %v", workload.Name, err)
	}

	workload.Status.Recommendation = *targetRes
	if err := h.Status().Patch(ctx, workload, patch); err != nil {
		return fmt.Errorf("failed to patch workload status %s: %v", workload.Name, err)
	}

	state.Annotations = workload.Annotations
	return nil
}

func (h *handler) applyResourcesToWorker(ctx context.Context, workload *State, worker *corev1.Pod, targetRes *tfv1.Resources) error {
	log := log.FromContext(ctx)

	curRes, err := utils.GPUResourcesFromAnnotations(worker.Annotations)
	if err != nil {
		return fmt.Errorf("failed to get current worker resources: %v", err)
	}
	if curRes != nil && curRes.Equal(targetRes) {
		return nil
	}

	annotationsToUpdate := utils.GPUResourcesToAnnotations(targetRes)
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

	isScaleUp := targetRes.Requests.Tflops.Cmp(curRes.Requests.Tflops) > 0 ||
		targetRes.Requests.Vram.Cmp(curRes.Requests.Vram) > 0

	adjustRequest := &tfv1.AdjustRequest{
		PodUID:     string(worker.UID),
		IsScaleUp:  isScaleUp,
		NewRequest: targetRes.Requests,
		NewLimit:   targetRes.Limits,
	}
	if _, err := h.allocator.AdjustAllocation(ctx, *adjustRequest, true); err != nil {
		return fmt.Errorf("failed to adjust allocation: %v", err)
	}
	log.Info("adjust allocation successfully", "worker", worker.Name, "currentResources", curRes, "adjustRequest", adjustRequest)

	patch := client.MergeFrom(worker.DeepCopy())
	maps.Copy(worker.Annotations, annotationsToUpdate)
	if err := h.Patch(ctx, worker, patch); err != nil {
		return fmt.Errorf("failed to patch worker %s: %v", worker.Name, err)
	}

	log.Info("apply resources successfully", "worker", worker.Name, "targetResources", targetRes, "currentResources", curRes)

	return nil
}
