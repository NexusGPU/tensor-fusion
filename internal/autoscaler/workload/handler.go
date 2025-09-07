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
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type Handler interface {
	UpdateWorkloadState(ctx context.Context, workloadState *State, workload *tfv1.TensorFusionWorkload) error
	ApplyResourcesToWorkload(ctx context.Context, workloadState *State, targetRes *tfv1.Resources) error
	UpdateWorkloadStatus(ctx context.Context, state *State, targetRes *tfv1.Resources) error
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
	workloadState.Annotations = workload.Annotations
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

func (h *handler) ApplyResourcesToWorkload(ctx context.Context, workload *State, targetRes *tfv1.Resources) error {
	// If the latest recommendation has not been applied to all workers,
	// we need to retry the update
	if targetRes == nil &&
		workload.Status.Recommendation != nil &&
		workload.CurrentReplicas != workload.Status.AppliedRecommendedReplicas {
		targetRes = workload.Status.Recommendation
	}

	if targetRes != nil {
		workload.Status.AppliedRecommendedReplicas = 0
		for _, worker := range workload.CurrentActiveWorkers {
			if err := h.applyResourcesToWorker(ctx, workload, worker, targetRes); err != nil {
				log.FromContext(ctx).Error(err, "failed to update worker resources: %v", "worker", worker.Name)
			}
			workload.Status.AppliedRecommendedReplicas++
		}
	}

	return nil
}

func (h *handler) UpdateWorkloadStatus(ctx context.Context, state *State, targetRes *tfv1.Resources) error {
	workload := &tfv1.TensorFusionWorkload{}
	if err := h.Get(ctx, client.ObjectKey{Namespace: state.Namespace, Name: state.Name}, workload); err != nil {
		return fmt.Errorf("failed to get workload: %v", err)
	}

	if targetRes == nil &&
		workload.Status.AppliedRecommendedReplicas == state.Status.AppliedRecommendedReplicas {
		return nil
	}

	patch := client.MergeFrom(workload.DeepCopy())
	if isResourcesChanged(&workload.Status, targetRes) {
		workload.Status.Recommendation = targetRes.DeepCopy()
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
		"workload", workload.Name, "recommendation", targetRes)

	return nil
}

func isResourcesChanged(status *tfv1.TensorFusionWorkloadStatus, targetRes *tfv1.Resources) bool {
	return targetRes != nil && (status.Recommendation == nil || !status.Recommendation.Equal(targetRes))
}

func (h *handler) applyResourcesToWorker(ctx context.Context,
	workload *State,
	worker *corev1.Pod,
	targetRes *tfv1.Resources) error {
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

	adjustRequest := tfv1.AdjustRequest{
		PodUID:     string(worker.UID),
		IsScaleUp:  isScaleUp,
		NewRequest: targetRes.Requests,
		NewLimit:   targetRes.Limits,
	}
	if _, err := h.allocator.AdjustAllocation(ctx, adjustRequest, true); err != nil {
		return fmt.Errorf("failed to adjust allocation: %v", err)
	}
	log.Info("adjust allocation successfully",
		"worker", worker.Name, "currentResources", curRes, "adjustRequest", adjustRequest)

	patch := client.MergeFrom(worker.DeepCopy())
	maps.Copy(worker.Annotations, annotationsToUpdate)
	if err := h.Patch(ctx, worker, patch); err != nil {
		return fmt.Errorf("failed to patch worker %s: %v", worker.Name, err)
	}

	log.Info("apply resources to worker successfully",
		"worker", worker.Name, "targetResources", targetRes, "currentResources", curRes)

	return nil
}
