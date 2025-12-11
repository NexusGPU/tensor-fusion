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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type Handler interface {
	UpdateWorkloadState(ctx context.Context, workloadState *State, workload *tfv1.TensorFusionWorkload) error
	ApplyRecommendationToWorkload(ctx context.Context, workloadState *State, recommendation *tfv1.Resources) error
	UpdateWorkloadStatus(ctx context.Context, state *State, recommendation *tfv1.Resources) error
	GetMaxAllowedResourcesSpec(workload *State) (*tfv1.Resource, error)
	SetEventRecorder(recorder record.EventRecorder, scheme *runtime.Scheme)
}

type handler struct {
	client.Client
	allocator     *gpuallocator.GpuAllocator
	eventRecorder record.EventRecorder
	scheme        *runtime.Scheme
}

func NewHandler(client client.Client, allocator *gpuallocator.GpuAllocator) Handler {
	return &handler{
		Client:    client,
		allocator: allocator,
	}
}

func NewHandlerWithRecorder(client client.Client, allocator *gpuallocator.GpuAllocator, recorder record.EventRecorder, scheme *runtime.Scheme) Handler {
	return &handler{
		Client:        client,
		allocator:     allocator,
		eventRecorder: recorder,
		scheme:        scheme,
	}
}

func (h *handler) SetEventRecorder(recorder record.EventRecorder, scheme *runtime.Scheme) {
	h.eventRecorder = recorder
	h.scheme = scheme
}

func (h *handler) UpdateWorkloadState(ctx context.Context, workloadState *State, workload *tfv1.TensorFusionWorkload) error {
	workloadState.Namespace = workload.Namespace
	workloadState.Name = workload.Name
	workloadState.Spec = workload.Spec
	workloadState.Status = *workload.Status.DeepCopy()
	workloadState.CreationTimestamp = workload.CreationTimestamp

	if workload.Spec.AutoScalingConfig.AutoSetResources != nil {
		workloadState.updateHistoryPeriod(workload.Spec.AutoScalingConfig.AutoSetResources.HistoryDataPeriod)
	}

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

	patch := client.MergeFrom(workload.DeepCopy())
	hasChanges := false

	if isRecommendationChanged(&workload.Status, recommendation) {
		workload.Status.Recommendation = recommendation
		workload.Status.ActiveCronScalingRule = state.Status.ActiveCronScalingRule.DeepCopy()
		hasChanges = true
	}

	if workload.Status.AppliedRecommendedReplicas != state.Status.AppliedRecommendedReplicas {
		workload.Status.AppliedRecommendedReplicas = state.Status.AppliedRecommendedReplicas
		hasChanges = true
	}

	// Update condition - check for both old and new condition types
	// Always check conditions even if recommendation is nil, as conditions may need to be updated
	if condition := meta.FindStatusCondition(state.Status.Conditions,
		constants.ConditionStatusTypeResourceUpdate); condition != nil {
		oldCondition := meta.FindStatusCondition(workload.Status.Conditions,
			constants.ConditionStatusTypeResourceUpdate)
		if oldCondition == nil || !isConditionEqual(oldCondition, condition) {
			meta.SetStatusCondition(&workload.Status.Conditions, *condition)
			hasChanges = true
		}
	} else if condition := meta.FindStatusCondition(state.Status.Conditions,
		constants.ConditionStatusTypeRecommendationProvided); condition != nil {
		// Migrate old condition to new type
		oldCondition := meta.FindStatusCondition(workload.Status.Conditions,
			constants.ConditionStatusTypeResourceUpdate)
		if oldCondition == nil || oldCondition.Status != condition.Status ||
			oldCondition.Reason != condition.Reason || oldCondition.Message != condition.Message {
			// Deep copy condition before modifying to avoid mutating state
			migratedCondition := condition.DeepCopy()
			migratedCondition.Type = constants.ConditionStatusTypeResourceUpdate
			meta.SetStatusCondition(&workload.Status.Conditions, *migratedCondition)
			hasChanges = true
		}
	}

	// Only return early if there are no changes and recommendation is nil and appliedRecommendedReplicas hasn't changed
	if !hasChanges && !isAppliedRecommendedReplicasChanged(workload, state) {
		return nil
	}

	if !hasChanges {
		return nil
	}

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

func isConditionEqual(c1, c2 *metav1.Condition) bool {
	if c1 == nil && c2 == nil {
		return true
	}
	if c1 == nil || c2 == nil {
		return false
	}
	return c1.Type == c2.Type &&
		c1.Status == c2.Status &&
		c1.Reason == c2.Reason &&
		c1.Message == c2.Message
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

	// Record event when scaling happens
	if h.eventRecorder != nil && h.scheme != nil {
		workloadObj := &tfv1.TensorFusionWorkload{}
		workloadObj.Namespace = workload.Namespace
		workloadObj.Name = workload.Name
		workloadObj.Kind = "TensorFusionWorkload"
		workloadObj.APIVersion = tfv1.GroupVersion.String()

		isScaleUp := recommendation.Requests.Tflops.Cmp(curRes.Requests.Tflops) > 0 ||
			recommendation.Requests.Vram.Cmp(curRes.Requests.Vram) > 0

		eventType := "Normal"
		reason := "ResourceScaledDown"
		message := fmt.Sprintf("Resources scaled down: Compute %s->%s, VRAM %s->%s",
			curRes.Requests.Tflops.String(), recommendation.Requests.Tflops.String(),
			curRes.Requests.Vram.String(), recommendation.Requests.Vram.String())

		if isScaleUp {
			reason = "ResourceScaledUp"
			message = fmt.Sprintf("Resources scaled up: Compute %s->%s, VRAM %s->%s",
				curRes.Requests.Tflops.String(), recommendation.Requests.Tflops.String(),
				curRes.Requests.Vram.String(), recommendation.Requests.Vram.String())
		}

		h.eventRecorder.Event(workloadObj, eventType, reason, message)
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

	_, deltaRes, err := h.allocator.AdjustAllocation(ctx, tfv1.AdjustRequest{
		PodUID:     string(worker.UID),
		IsScaleUp:  isScaleUp,
		NewRequest: recommendation.Requests,
		NewLimit:   recommendation.Limits,
	}, false)
	if err != nil {
		return fmt.Errorf("failed to adjust allocation: %v", err)
	}

	patch := client.MergeFrom(worker.DeepCopy())
	maps.Copy(worker.Annotations, annotationsToUpdate)
	if err := h.Patch(ctx, worker, patch); err != nil {
		// Rollback the allocation change by calculating original values from current state and delta
		// After AdjustAllocation, the allocator state is now recommendation, so we need to subtract deltaRes
		// to get back to the original curRes values
		originalRequest := tfv1.Resource{
			Tflops: recommendation.Requests.Tflops.DeepCopy(),
			Vram:   recommendation.Requests.Vram.DeepCopy(),
		}
		originalRequest.Tflops.Sub(deltaRes.Tflops)
		originalRequest.Vram.Sub(deltaRes.Vram)

		originalLimit := tfv1.Resource{
			Tflops: recommendation.Limits.Tflops.DeepCopy(),
			Vram:   recommendation.Limits.Vram.DeepCopy(),
		}
		originalLimit.Tflops.Sub(deltaRes.Tflops)
		originalLimit.Vram.Sub(deltaRes.Vram)

		if _, _, rollbackErr := h.allocator.AdjustAllocation(ctx, tfv1.AdjustRequest{
			PodUID:     string(worker.UID),
			IsScaleUp:  !isScaleUp,
			NewRequest: originalRequest,
			NewLimit:   originalLimit,
		}, false); rollbackErr != nil {
			log.Error(rollbackErr, "failed to rollback allocation after patch failure",
				"worker", worker.Name, "originalError", err)
		} else {
			log.Info("rolled back allocation after patch failure",
				"worker", worker.Name, "originalError", err)
		}
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
		allowedTflops int64 = -1
		allowedVram   int64 = -1
	)
	for gpu, workers := range gpuToWorkers {
		if gpu.Status.Available == nil {
			return nil, fmt.Errorf("GPU available is nil")
		}
		// gpu.Status.Available = Capacity - all allocated resources (including this workload and others)
		// To calculate this workload's max allowed resources, we need to add back this workload's
		// allocated resources, so: available = Capacity - other workloads' allocations
		availableTflops := gpu.Status.Available.Tflops.DeepCopy()
		availableVram := gpu.Status.Available.Vram.DeepCopy()
		for _, worker := range workers {
			// Add back this workload's allocated resources to get the total available for this workload
			availableTflops.Add(allocRequests[string(worker.UID)].Request.Tflops)
			availableVram.Add(allocRequests[string(worker.UID)].Request.Vram)
		}

		workerCount := int64(len(workers))
		tflopsPerWorker := int64(availableTflops.AsApproximateFloat64()) / workerCount
		vramPerWorker := availableVram.Value() / workerCount
		if allowedTflops == -1 || tflopsPerWorker < allowedTflops {
			allowedTflops = tflopsPerWorker
		}
		if allowedVram == -1 || vramPerWorker < allowedVram {
			allowedVram = vramPerWorker
		}
	}

	return &tfv1.Resource{
		Tflops: *resource.NewQuantity(allowedTflops, resource.DecimalSI),
		Vram:   *resource.NewQuantity(allowedVram, resource.BinarySI),
	}, nil
}
