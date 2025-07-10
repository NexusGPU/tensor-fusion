package router

import (
	"context"
	"io"
	"net/http"

	"time"

	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/scheduler/gpuresources"
	"github.com/gin-gonic/gin"
	"sigs.k8s.io/yaml"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type AllocatorInfoRouter struct {
	allocator *gpuallocator.GpuAllocator
	scheduler *scheduler.Scheduler
}

func NewAllocatorInfoRouter(
	ctx context.Context,
	allocator *gpuallocator.GpuAllocator,
	scheduler *scheduler.Scheduler,
) (*AllocatorInfoRouter, error) {
	return &AllocatorInfoRouter{allocator: allocator, scheduler: scheduler}, nil
}
func (r *AllocatorInfoRouter) Get(ctx *gin.Context) {
	gpuStore, nodeWorkerStore, uniqueAllocation := r.allocator.GetAllocationInfo()
	gpuStoreResp := make(map[string]tfv1.GPUStatus)
	for key, gpu := range gpuStore {
		gpuStoreResp[key.String()] = gpu.Status
	}
	nodeWorkerStoreResp := make(map[string]map[string]bool)
	for key, nodeWorker := range nodeWorkerStore {
		nodeWorkerStoreResp[key] = make(map[string]bool)
		for gpuKey := range nodeWorker {
			nodeWorkerStoreResp[key][gpuKey.String()] = true
		}
	}

	gpuToWorkerMap := make(map[string][]tfv1.GPUAllocationInfo, len(gpuStore))
	uniqueAllocationResp := make(map[string]*tfv1.AllocRequest)
	for key, allocRequest := range uniqueAllocation {
		uniqueAllocationResp[key] = allocRequest.DeepCopy()
		// remove managedFields and non useful fields
		uniqueAllocationResp[key].PodMeta = metav1.ObjectMeta{
			Name:            allocRequest.PodMeta.Name,
			Namespace:       allocRequest.PodMeta.Namespace,
			UID:             allocRequest.PodMeta.UID,
			ResourceVersion: allocRequest.PodMeta.ResourceVersion,
			Generation:      allocRequest.PodMeta.Generation,
			Labels:          allocRequest.PodMeta.Labels,
			Annotations:     allocRequest.PodMeta.Annotations,
			OwnerReferences: allocRequest.PodMeta.OwnerReferences,
		}
		for _, gpuName := range allocRequest.GPUNames {
			gpuToWorkerMap[gpuName] = append(gpuToWorkerMap[gpuName], tfv1.GPUAllocationInfo{
				Request:   allocRequest.Request,
				Limit:     allocRequest.Limit,
				PodUID:    string(allocRequest.PodMeta.UID),
				PodName:   allocRequest.PodMeta.Name,
				Namespace: allocRequest.PodMeta.Namespace,
			})
		}
	}
	ctx.JSON(http.StatusOK, gin.H{
		"gpuStore":        gpuStoreResp,
		"nodeWorkerStore": nodeWorkerStoreResp,
		"allocation":      uniqueAllocationResp,
		"gpuToWorkerMap":  gpuToWorkerMap,
	})
}

// Simulate the partial logic of schedulingCycle in ScheduleOne()
// make sure no side effect when simulate scheduling, only run PreFilter/Filter/Score plugins
// AssumePod, Reserve, bindingCycle has side effect on scheduler, thus not run them
// Permit plugin can not be run in simulate scheduling, because it's after AssumePod & Reserve stage
func (r *AllocatorInfoRouter) SimulateScheduleOnePod(ctx *gin.Context) {
	body, err := io.ReadAll(ctx.Request.Body)
	if err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	var pod = &v1.Pod{}
	if err := yaml.Unmarshal(body, pod); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// prepare exact the same context data as real scheduler
	log.FromContext(ctx).Info("Simulate schedule pod", "pod", pod.Name, "namespace", pod.Namespace)
	start := time.Now()
	state := framework.NewCycleState()
	state.SetRecordPluginMetrics(false)
	podsToActivate := framework.NewPodsToActivate()
	state.Write(framework.PodsToActivateKey, podsToActivate)

	// simulate schedulingCycle non side effect part
	fwk := r.scheduler.Profiles[pod.Spec.SchedulerName]
	if fwk == nil {
		log.FromContext(ctx).Error(nil, "scheduler framework not found", "pod", pod.Name, "namespace", pod.Namespace)
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "scheduler framework not found"})
		return
	}
	scheduleResult, err := r.scheduler.SchedulePod(ctx, fwk, state, pod)
	gpuCycleState, _ := state.Read(gpuresources.CycleStateGPUSchedulingResult)
	if err != nil {
		if fitError, ok := err.(*framework.FitError); ok {
			ctx.JSON(http.StatusOK, gin.H{
				"scheduleResult": scheduleResult,
				"error": gin.H{
					"numAllNodes": fitError.NumAllNodes,
					"diagnosis":   fitError.Diagnosis,
				},
				"cycleState":        state,
				"gpuSchedulerState": gpuCycleState,
			})
			return
		}
	}
	ctx.JSON(http.StatusOK, gin.H{
		"scheduleResult":    scheduleResult,
		"error":             err,
		"cycleState":        state,
		"gpuSchedulerState": gpuCycleState,
	})
	log.FromContext(ctx).Info("Simulate schedule pod completed", "pod", pod.Name, "namespace", pod.Namespace, "duration", time.Since(start))
}
