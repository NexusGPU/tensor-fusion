package recommender

import (
	"context"
	"strings"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// The reported case, from usage samples to the worker's annotations and the
// allocator: a worker with a limit of 20 TFLOPS on a GPU of 71.2 TFLOPS uses
// the whole GPU. The usage calls for a limit of 82 TFLOPS (usage plus margin),
// which was applied at once. It has to stay within the GPU and get there in
// bounded steps, with the minimum interval between them, and the workload has
// to tell what was recommended and what was applied instead.
func TestInPlaceScalingOfABusyWorkerIsBounded(t *testing.T) {
	const (
		ns           = "bounds-ns"
		workloadName = "bounds-workload"
		poolName     = "bounds-pool"
		podName      = "bounds-worker-0"
		podUID       = "bounds-worker-0-uid"
		gpuName      = "bounds-gpu"
		gpuTflops    = 71.2
	)
	ctx := context.Background()
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	capacity := tfv1.Resource{Tflops: resource.MustParse("71.2"), Vram: resource.MustParse("24Gi")}
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name:   gpuName,
			Labels: map[string]string{constants.LabelKeyOwner: "bounds-node", constants.GpuPoolKey: poolName},
		},
		Status: tfv1.GPUStatus{
			Phase:        tfv1.TensorFusionGPUPhaseRunning,
			Capacity:     capacity.DeepCopy(),
			Available:    capacity.DeepCopy(),
			NodeSelector: map[string]string{constants.KubernetesHostNameLabel: "bounds-host"},
		},
	}
	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: poolName},
		Status:     tfv1.GPUPoolStatus{Phase: tfv1.TensorFusionPoolPhaseRunning},
	}
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(gpu, pool).Build()

	allocator := gpuallocator.NewGpuAllocator(ctx, nil, fakeClient, 0)
	require.NoError(t, allocator.InitGPUAndQuotaStore())
	allocator.ReconcileAllocationState()
	allocator.SetAllocatorReady()

	original := tfv1.Resources{
		Requests: tfv1.Resource{Tflops: resource.MustParse("10"), Vram: resource.MustParse("4Gi")},
		Limits:   tfv1.Resource{Tflops: resource.MustParse("20"), Vram: resource.MustParse("8Gi")},
	}
	allocated, err := allocator.Alloc(&tfv1.AllocRequest{
		PoolName:              poolName,
		WorkloadNameNamespace: tfv1.NameNamespace{Namespace: ns, Name: workloadName},
		Request:               original.Requests,
		Limit:                 original.Limits,
		Count:                 1,
		PodMeta:               metav1.ObjectMeta{Namespace: ns, Name: podName, UID: podUID},
	})
	require.NoError(t, err)
	require.Len(t, allocated, 1)

	annotations := utils.GPUResourcesToAnnotations(&original)
	annotations[constants.GpuCountAnnotation] = "1"
	annotations[constants.GpuPoolKey] = poolName
	worker := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Namespace: ns, Name: podName, UID: podUID, Annotations: annotations,
	}}
	require.NoError(t, fakeClient.Create(ctx, worker.DeepCopy()))

	recorder := events.NewFakeRecorder(100)
	handler := workload.NewHandlerWithRecorder(fakeClient, allocator, recorder, scheme)

	state := workload.NewWorkloadState()
	state.Namespace, state.Name = ns, workloadName
	state.CreationTimestamp = metav1.NewTime(time.Now().Add(-3 * time.Hour))
	state.Spec.Resources = original
	state.Spec.AutoScalingConfig.AutoSetResources = &tfv1.AutoSetResources{
		Enable: true, TargetResource: tfv1.ScalingTargetResourceAll,
	}
	state.CurrentActiveWorkers = map[string]*corev1.Pod{podName: worker.DeepCopy()}

	// Two hours at 100% of the GPU, holding 6Gi.
	start := time.Now()
	for minute := 119; minute >= 0; minute-- {
		state.AddSample(&metrics.WorkerUsage{
			Namespace: ns, WorkloadName: workloadName, WorkerName: podUID,
			TflopsUsage: gpuTflops, VramUsage: 6 << 30,
			Timestamp: start.Add(-time.Duration(minute) * time.Minute),
		})
	}
	// What the usage calls for is more than the GPU has: usage plus the margin.
	estimations := (&resourcesEstimator{}).GetResourcesEstimation(state)
	require.NotNil(t, estimations)
	require.InDelta(t, 82, estimations.UpperBoundTflops.AsApproximateFloat64(), 5)
	require.Greater(t, estimations.UpperBoundTflops.AsApproximateFloat64(), gpuTflops)

	recommenders := []Interface{NewPercentileRecommender(NewRecommendationProcessor(handler))}
	clock := start
	timeNow = func() time.Time { return clock }
	t.Cleanup(func() { timeNow = time.Now })

	// evaluate is one round of the autoscaler for the workload; it returns
	// what the worker holds afterwards, according to its annotations.
	evaluate := func() *tfv1.Resources {
		t.Helper()
		recommendation, err := GetRecommendation(ctx, state, recommenders)
		require.NoError(t, err)
		if recommendation != nil {
			state.SetRecommendation(recommendation)
		}
		require.NoError(t, handler.ApplyRecommendationToWorkload(ctx, state, recommendation))

		pod := &corev1.Pod{}
		require.NoError(t, fakeClient.Get(ctx, client.ObjectKey{Namespace: ns, Name: podName}, pod))
		held, err := utils.GPUResourcesFromAnnotations(pod.Annotations)
		require.NoError(t, err)
		// The allocator's record of the worker is the same.
		_, _, allocations := allocator.GetAllocationInfo()
		require.True(t, allocations[podUID].Limit.Tflops.Equal(held.Limits.Tflops))
		require.True(t, allocations[podUID].Request.Tflops.Equal(held.Requests.Tflops))
		return held
	}
	conditionMessage := func() string { return resourceUpdateCondition(t, state).Message }

	// The first change: half of the current value up, not the 82 TFLOPS.
	held := evaluate()
	require.Equal(t, "30", held.Limits.Tflops.String())
	require.Equal(t, "15", held.Requests.Tflops.String())
	require.Contains(t, conditionMessage(), "Compute scaled: request 10 -> 15, limit 20 -> 30")
	require.Contains(t, conditionMessage(), "Compute limit: recommended ")
	require.Contains(t, conditionMessage(), ", applied 30 (step limit: at most +50% per change)")

	// Evaluations within the minimum interval change nothing.
	for _, sinceChange := range []time.Duration{30 * time.Second, 2 * time.Minute, minChangeInterval - time.Second} {
		clock = time.Now().Add(sinceChange)
		held = evaluate()
		require.Equal(t, "30", held.Limits.Tflops.String(), "%s after the change", sinceChange)
		require.Equal(t, "15", held.Requests.Tflops.String(), "%s after the change", sinceChange)
		require.Equal(t, "MinChangeInterval", resourceUpdateCondition(t, state).Reason)
	}

	// From then on one change per interval.
	limits, requests := []float64{20, 30}, []float64{10, 15}
	capacityNamed := false
	for round := 0; round < 20; round++ {
		clock = time.Now().Add(minChangeInterval + time.Second)
		held = evaluate()
		limit, request := held.Limits.Tflops.AsApproximateFloat64(), held.Requests.Tflops.AsApproximateFloat64()
		if limit == limits[len(limits)-1] && request == requests[len(requests)-1] {
			break
		}
		limits, requests = append(limits, limit), append(requests, request)
		capacityNamed = capacityNamed || strings.Contains(conditionMessage(), "(device capacity: 71200m per GPU held)")
	}
	t.Logf("limits %v, requests %v", limits, requests)

	// The limit ends at the capacity of the GPU, in whole TFLOPS: 20, 30, 45, 67, 71.
	require.Equal(t, []float64{20, 30, 45, 67, 71}, limits[:5])
	for i := 1; i < len(limits); i++ {
		require.LessOrEqual(t, limits[i], gpuTflops, "limit above the capacity of the GPU")
		require.LessOrEqual(t, requests[i], gpuTflops, "request above the capacity of the GPU")
		require.LessOrEqual(t, limits[i], limits[i-1]*(1+maxStepUpFraction), "limit step %d", i)
		require.LessOrEqual(t, requests[i], requests[i-1]*(1+maxStepUpFraction), "request step %d", i)
		require.LessOrEqual(t, requests[i], limits[i])
	}
	require.Equal(t, 71.0, limits[len(limits)-1])
	require.Equal(t, 71.0, requests[len(requests)-1])
	require.GreaterOrEqual(t, len(requests), 6, "the request takes several changes: 10, 15, 22, 33, 49, 71")
	require.True(t, capacityNamed, "the condition names the capacity when it is what bounds the change")

	// Every change was recorded as an event that carries the bounds.
	close(recorder.Events)
	recorded := make([]string, 0, len(recorder.Events))
	for event := range recorder.Events {
		recorded = append(recorded, event)
	}
	require.Len(t, recorded, len(requests)-1)
	require.Contains(t, recorded[0], "Normal ResourceScaledUp Resources scaled up: Compute 10->15, VRAM 4Gi->")
	require.Contains(t, recorded[0], "limits: Compute 20->30")
	require.Contains(t, recorded[0], "bounds: Compute request: recommended ")
	require.Contains(t, recorded[0], "(step limit: at most +50% per change)")
	require.Contains(t, strings.Join(recorded, "\n"), "(device capacity: 71200m per GPU held)")

	// The workload's record of the applied recommendation matches the worker.
	latest := state.LatestRecommendation()
	require.NotNil(t, latest)
	require.True(t, latest.Limits.Tflops.Equal(held.Limits.Tflops))
	require.True(t, state.IsRecommendationAppliedToAllWorkers())
}
