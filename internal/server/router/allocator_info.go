package router

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"

	"time"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/controller"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator/filter"
	"github.com/NexusGPU/tensor-fusion/internal/scheduler/gpuresources"
	"github.com/gin-gonic/gin"
	"sigs.k8s.io/yaml"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
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
	gpuStoreResp := make(map[string]tfv1.GPU)
	for key, gpu := range gpuStore {
		gpuStoreResp[key.String()] = *gpu
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
	state.Write(fwk.StateKey(constants.SchedulerSimulationKey), &gpuallocator.SimulateSchedulingFilterDetail{
		FilterStageDetails: []filter.FilterDetail{},
	})

	// simulate schedulingCycle non side effect part
	fwkInstance := r.scheduler.Profiles[pod.Spec.SchedulerName]
	if fwkInstance == nil {
		log.FromContext(ctx).Error(nil, "scheduler framework not found", "pod", pod.Name, "namespace", pod.Namespace)
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "scheduler framework not found"})
		return
	}
	scheduleResult, err := r.scheduler.SchedulePod(ctx, fwkInstance, state, pod)
	gpuCycleState, _ := state.Read(gpuresources.CycleStateGPUSchedulingResult)
	simulateSchedulingFilterDetail, _ := state.Read(fwk.StateKey(constants.SchedulerSimulationKey))
	progressiveNodes := readProgressiveNodeNames(state)

	// errorPayload collects every piece of diagnostic info from the scheduler error.
	// The old code used `"error": err` which serialises Go's error interface to {} —
	// callers were seeing empty JSON and no way to learn why scheduling failed.
	var errorPayload gin.H
	if err != nil {
		log.FromContext(ctx).Error(err, "Simulate schedule pod failed",
			"pod", pod.Name, "namespace", pod.Namespace,
			"errorType", fmt.Sprintf("%T", err))

		errorPayload = gin.H{
			"type":    fmt.Sprintf("%T", err),
			"message": err.Error(),
		}
		// Use errors.As so wrapped FitError (from fmt.Errorf("...: %w", fe)) still unwraps.
		var fitError *framework.FitError
		if errors.As(err, &fitError) {
			errorPayload["numAllNodes"] = fitError.NumAllNodes
			errorPayload["diagnosis"] = renderDiagnosis(&fitError.Diagnosis)
		}
	}

	ctx.JSON(http.StatusOK, gin.H{
		"scheduleResult":       scheduleResult,
		"filterDetail":         simulateSchedulingFilterDetail,
		"error":                errorPayload,
		"cycleState":           state,
		"gpuSchedulerState":    gpuCycleState,
		"progressiveNodeNames": progressiveNodes,
	})
	log.FromContext(ctx).Info("Simulate schedule pod completed",
		"pod", pod.Name, "namespace", pod.Namespace,
		"duration", time.Since(start), "scheduleError", err != nil)
}

// renderDiagnosis flattens framework.Diagnosis into JSON-serialisable form.
// Diagnosis.NodeToStatus has unexported map fields and sets.Set[string] serialises
// to `{"pluginName": {}}` — both lose their information through encoding/json.
func renderDiagnosis(d *framework.Diagnosis) gin.H {
	if d == nil {
		return nil
	}
	out := gin.H{
		"preFilterMsg":         d.PreFilterMsg,
		"postFilterMsg":        d.PostFilterMsg,
		"unschedulablePlugins": d.UnschedulablePlugins.UnsortedList(),
		"pendingPlugins":       d.PendingPlugins.UnsortedList(),
	}
	if d.NodeToStatus != nil {
		nodeStatuses := gin.H{}
		d.NodeToStatus.ForEachExplicitNode(func(name string, s *fwk.Status) {
			nodeStatuses[name] = renderStatus(s)
		})
		out["nodeToStatus"] = gin.H{
			"nodes":             nodeStatuses,
			"absentNodesStatus": renderStatus(d.NodeToStatus.AbsentNodesStatus()),
		}
	}
	return out
}

func renderStatus(s *fwk.Status) gin.H {
	if s == nil {
		return gin.H{"code": "Success"}
	}
	return gin.H{
		"code":    s.Code().String(),
		"plugin":  s.Plugin(),
		"reasons": s.Reasons(),
		"message": s.Message(),
	}
}

// readProgressiveNodeNames extracts the candidate node set written by the
// progressive migration PreFilter branch. Returns nil when the branch was not
// taken (e.g. TF worker pod path, or progressive migration disabled).
func readProgressiveNodeNames(state *framework.CycleState) gin.H {
	raw, err := state.Read(gpuresources.CycleStateProgressiveNodeNames)
	if err != nil || raw == nil {
		return nil
	}
	s, ok := raw.(*gpuresources.ProgressiveNodeNamesState)
	if !ok || s.NodeNames == nil {
		return nil
	}
	names := s.NodeNames.UnsortedList()
	sort.Strings(names)
	return gin.H{
		"count": len(names),
		"nodes": names,
	}
}

// DefragStatus returns the last in-memory defrag stats per pool.
func (r *AllocatorInfoRouter) DefragStatus(ctx *gin.Context) {
	ctx.JSON(http.StatusOK, gin.H{
		"lastRuns": controller.GetDefragLastRunStats(),
	})
}
