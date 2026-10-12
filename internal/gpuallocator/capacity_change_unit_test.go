package gpuallocator

import (
	"maps"
	"strings"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/quota"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// The tests in this file follow one GPU through a change of its capacity while
// workloads are allocated on it. The node agent publishes one TFLOPS capacity
// per device: the vendor-published FP16 figure when the hardware table lists
// the model (71.2 for the card used here), its own estimate otherwise (44.08).
// Upgrading the agent or correcting the table therefore moves
// status.capacity.tflops of a GPU in use, in either direction.

const (
	capTestGPUName  = "gpu-capacity"
	capTestNodeName = "node-capacity"
	capTestPoolName = "pool-capacity"
	capTestModel    = "capacity-test-model"
	// A model with one partition template, used by the partitioned case only.
	capTestPartitionModel    = "capacity-test-partition-model"
	capTestPartitionTemplate = "half"
)

var capTestGPUKey = types.NamespacedName{Name: capTestGPUName}

// capTestGPU is the GPU resource as it is on the API server. available may be
// nil for a status the allocator has never written.
func capTestGPU(capacity tfv1.Resource, available *tfv1.Resource) *tfv1.GPU {
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: capTestGPUName,
			Labels: map[string]string{
				constants.GpuPoolKey:    capTestPoolName,
				constants.LabelKeyOwner: capTestNodeName,
			},
		},
	}
	gpu.Status.Phase = tfv1.TensorFusionGPUPhaseRunning
	gpu.Status.GPUModel = capTestModel
	gpu.Status.NodeSelector = map[string]string{constants.KubernetesHostNameLabel: capTestNodeName}
	gpu.Status.Capacity = capacity.DeepCopy()
	if available != nil {
		gpu.Status.Available = available.DeepCopy()
	}
	return gpu
}

func capRes(tflops, vram string) tfv1.Resource {
	return tfv1.Resource{Tflops: qty(tflops), Vram: qty(vram)}
}

// capTestWorker is a running worker Pod bound to the test GPU. These Pods are
// what a starting operator rebuilds its allocations from.
func capTestWorker(name string, annotations map[string]string) *corev1.Pod {
	all := map[string]string{
		constants.GPUDeviceIDsAnnotation: capTestGPUName,
		constants.GpuPoolKey:             capTestPoolName,
	}
	maps.Copy(all, annotations)
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: "default", UID: types.UID(name + "-uid"),
			Labels: map[string]string{
				constants.LabelComponent: constants.ComponentWorker,
				constants.WorkloadKey:    name,
			},
			Annotations: all,
		},
		Spec:   corev1.PodSpec{NodeName: capTestNodeName},
		Status: corev1.PodStatus{Phase: corev1.PodRunning},
	}
}

func capTestFixedWorker(name, tflops, vram string) *corev1.Pod {
	return capTestWorker(name, map[string]string{
		constants.TFLOPSRequestAnnotation: tflops, constants.TFLOPSLimitAnnotation: tflops,
		constants.VRAMRequestAnnotation: vram, constants.VRAMLimitAnnotation: vram,
	})
}

func capTestPercentWorker(name, percent, vram string) *corev1.Pod {
	return capTestWorker(name, map[string]string{
		constants.ComputeRequestAnnotation: percent, constants.ComputeLimitAnnotation: percent,
		constants.VRAMRequestAnnotation: vram, constants.VRAMLimitAnnotation: vram,
	})
}

func capTestClient(t *testing.T, gpu *tfv1.GPU, workers ...*corev1.Pod) client.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, tfv1.AddToScheme(scheme))
	builder := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPU{}).
		WithObjects(gpu.DeepCopy(), &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{Name: capTestNodeName}})
	for _, worker := range workers {
		builder = builder.WithObjects(worker.DeepCopy())
	}
	return builder.Build()
}

// startCapTestAllocator is an operator start: the GPU store is listed from the
// API server and the allocations are rebuilt from the worker Pods.
func startCapTestAllocator(t *testing.T, k8sClient client.Client) (*GpuAllocator, *events.FakeRecorder) {
	t.Helper()
	s := newTestAllocator()
	s.Client = k8sClient
	s.quotaStore = quota.NewQuotaStore(k8sClient, s.ctx)
	s.nodeGpuStore = map[string]map[string]*tfv1.GPU{}
	s.poolGpuStore = map[string]map[string]*tfv1.GPU{}
	recorder := events.NewFakeRecorder(32)
	s.SetEventRecorder(recorder)

	require.NoError(t, s.initGPUAndQuotaStore())
	s.reconcileAllocationState()
	return s, recorder
}

func capTestPublished(t *testing.T, k8sClient client.Client) *tfv1.GPU {
	t.Helper()
	gpu := &tfv1.GPU{}
	require.NoError(t, k8sClient.Get(t.Context(), capTestGPUKey, gpu))
	return gpu
}

// nodeAgentPublishes writes a GPU status the way the node agent does and
// delivers the resulting watch event to the allocator. mutate changes the
// status; the allocator's own fields are whatever mutate leaves there.
func nodeAgentPublishes(t *testing.T, k8sClient client.Client, s *GpuAllocator, mutate func(status *tfv1.GPUStatus)) {
	t.Helper()
	gpu := capTestPublished(t, k8sClient)
	mutate(&gpu.Status)
	require.NoError(t, k8sClient.Status().Update(t.Context(), gpu))
	s.handleGPUUpdate(t.Context(), capTestPublished(t, k8sClient))
}

func setCapacity(capacity tfv1.Resource) func(status *tfv1.GPUStatus) {
	return func(status *tfv1.GPUStatus) {
		status.Capacity = capacity.DeepCopy()
	}
}

// syncCapTest runs one status sync and checks that nothing negative left the
// allocator: not on the API server and not in the reporting snapshot.
func syncCapTest(t *testing.T, k8sClient client.Client, s *GpuAllocator) *tfv1.GPU {
	t.Helper()
	s.SyncGPUsToK8s()

	published := capTestPublished(t, k8sClient)
	require.NotNil(t, published.Status.Available)
	require.GreaterOrEqual(t, published.Status.Available.Tflops.Sign(), 0,
		"published available tflops %s", published.Status.Available.Tflops.String())
	require.GreaterOrEqual(t, published.Status.Available.Vram.Sign(), 0,
		"published available vram %s", published.Status.Available.Vram.String())

	reported, _, _ := s.GetAllocationInfo()
	for _, gpu := range reported {
		require.GreaterOrEqual(t, gpu.Status.Available.Tflops.Sign(), 0)
		require.GreaterOrEqual(t, gpu.Status.Available.Vram.Sign(), 0)
	}
	return published
}

func requireQty(t *testing.T, want string, got resource.Quantity, what string) {
	t.Helper()
	require.True(t, qty(want).Equal(got), "%s: got %s, want %s", what, got.String(), want)
}

func requireSameQty(t *testing.T, want, got resource.Quantity, what string) {
	t.Helper()
	require.True(t, want.Equal(got), "%s: got %s, want %s", what, got.String(), want.String())
}

func drainCapTestEvents(recorder *events.FakeRecorder) []string {
	var got []string
	for {
		select {
		case event := <-recorder.Events:
			got = append(got, event)
		default:
			return got
		}
	}
}

func queuedForSync(s *GpuAllocator) bool {
	s.dirtyQueueLock.Lock()
	defer s.dirtyQueueLock.Unlock()
	_, queued := s.dirtyQueue[capTestGPUKey]
	return queued
}

// A capacity that changes while the operator is running must leave exactly the
// accounting an operator started after the change computes from the worker
// Pods, for every kind of allocation, and the two must stay equal while the
// workloads leave one by one until the GPU is empty again.
func TestCapacityChangeWhileRunningMatchesRestart(t *testing.T) {
	MutatePartitionConfigForTesting(func(cfg *partitionConfig) {
		cfg.Templates[capTestPartitionModel] = map[string]config.PartitionTemplateInfo{
			capTestPartitionTemplate: {
				TemplateID: capTestPartitionTemplate, Name: capTestPartitionTemplate,
				MemoryGigabytes: 8, ComputePercent: 50,
			},
		}
	})
	t.Cleanup(func() {
		MutatePartitionConfigForTesting(func(cfg *partitionConfig) {
			delete(cfg.Templates, capTestPartitionModel)
		})
	})

	fixed := func(specs ...[3]string) func(t *testing.T, gpu *tfv1.GPU) []*corev1.Pod {
		return func(*testing.T, *tfv1.GPU) []*corev1.Pod {
			workers := make([]*corev1.Pod, 0, len(specs))
			for _, spec := range specs {
				workers = append(workers, capTestFixedWorker(spec[0], spec[1], spec[2]))
			}
			return workers
		}
	}

	tests := []struct {
		name     string
		model    string
		from, to tfv1.Resource
		workers  func(t *testing.T, gpu *tfv1.GPU) []*corev1.Pod
		// The allocator's own figure, capacity - allocations, which is negative
		// while the GPU is over-committed, and the one it publishes.
		wantHeadroom  tfv1.Resource
		wantPublished tfv1.Resource
	}{
		{
			name: "increase with a fixed TFLOPS allocation",
			from: capRes("44.08", "24Gi"), to: capRes("71.2", "24Gi"),
			workers:      fixed([3]string{"a", "30", "8Gi"}),
			wantHeadroom: capRes("41.2", "16Gi"), wantPublished: capRes("41.2", "16Gi"),
		},
		{
			name: "decrease, allocations still fit",
			from: capRes("71.2", "24Gi"), to: capRes("44.08", "24Gi"),
			workers:      fixed([3]string{"a", "30", "8Gi"}),
			wantHeadroom: capRes("14.08", "16Gi"), wantPublished: capRes("14.08", "16Gi"),
		},
		{
			name: "decrease into TFLOPS over-commit",
			from: capRes("71.2", "24Gi"), to: capRes("44.08", "24Gi"),
			workers:      fixed([3]string{"a", "30", "8Gi"}, [3]string{"b", "30", "8Gi"}),
			wantHeadroom: capRes("-15.92", "8Gi"), wantPublished: capRes("0", "8Gi"),
		},
		{
			name: "decrease into VRAM over-commit",
			from: capRes("71.2", "24Gi"), to: capRes("71.2", "16Gi"),
			workers:      fixed([3]string{"a", "10", "16Gi"}, [3]string{"b", "10", "6Gi"}),
			wantHeadroom: capRes("51.2", "-6Gi"), wantPublished: capRes("51.2", "0"),
		},
		{
			name: "increase out of over-commit",
			from: capRes("44.08", "24Gi"), to: capRes("71.2", "24Gi"),
			workers:      fixed([3]string{"a", "30", "8Gi"}, [3]string{"b", "30", "8Gi"}),
			wantHeadroom: capRes("11.2", "8Gi"), wantPublished: capRes("11.2", "8Gi"),
		},
		{
			// 50% of 44.08 is 22 TFLOPS, 50% of 71.2 is 35: the allocation
			// grows with the capacity, so adding the capacity difference to
			// the old figure (22.08 + 27.12 = 49.2) would hand out 13 TFLOPS
			// that the workload holds.
			name: "compute-percent allocation follows the capacity",
			from: capRes("44.08", "24Gi"), to: capRes("71.2", "24Gi"),
			workers: func(*testing.T, *tfv1.GPU) []*corev1.Pod {
				return []*corev1.Pod{capTestPercentWorker("a", "50", "8Gi")}
			},
			wantHeadroom: capRes("36.2", "16Gi"), wantPublished: capRes("36.2", "16Gi"),
		},
		{
			name: "shared whole-GPU allocation keeps the GPU exclusive",
			from: capRes("44.08", "24Gi"), to: capRes("71.2", "24Gi"),
			workers: func(t *testing.T, gpu *tfv1.GPU) []*corev1.Pod {
				worker := capTestWorker("a", map[string]string{
					constants.IsolationModeAnnotation: tfv1.IsolationModeShared,
					constants.TFLOPSRequestAnnotation: "10", constants.TFLOPSLimitAnnotation: "10",
					constants.VRAMRequestAnnotation: "2Gi", constants.VRAMLimitAnnotation: "2Gi",
				})
				require.NoError(t, utils.ApplySharedLegacyResources(worker, []*tfv1.GPU{gpu}))
				return []*corev1.Pod{worker}
			},
			wantHeadroom: capRes("0", "0"), wantPublished: capRes("0", "0"),
		},
		{
			// The template takes half of the card's compute, whatever that is.
			name:  "partition template follows the capacity",
			model: capTestPartitionModel,
			from:  capRes("100", "24Gi"), to: capRes("60", "24Gi"),
			workers: func(*testing.T, *tfv1.GPU) []*corev1.Pod {
				return []*corev1.Pod{capTestWorker("a", map[string]string{
					constants.IsolationModeAnnotation:       tfv1.IsolationModePartitioned,
					constants.PartitionTemplateIDAnnotation: capTestPartitionTemplate,
				})}
			},
			wantHeadroom: capRes("30", "16Gi"), wantPublished: capRes("30", "16Gi"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := capTestGPU(tt.from, &tt.from)
			if tt.model != "" {
				before.Status.GPUModel = tt.model
			}
			workers := tt.workers(t, before)

			// The operator that is running when the capacity changes.
			runningClient := capTestClient(t, before, workers...)
			running, _ := startCapTestAllocator(t, runningClient)
			leftByRunning := syncCapTest(t, runningClient, running)
			nodeAgentPublishes(t, runningClient, running, setCapacity(tt.to))

			// The operator that starts after it changed. It finds the new
			// capacity next to the figures published for the old one.
			after := leftByRunning.DeepCopy()
			after.ResourceVersion = ""
			after.Status.Capacity = tt.to.DeepCopy()
			restartedClient := capTestClient(t, after, workers...)
			restarted, _ := startCapTestAllocator(t, restartedClient)

			compare := func(step string) {
				t.Helper()
				runningGPU := running.gpuStore[capTestGPUKey]
				restartedGPU := restarted.gpuStore[capTestGPUKey]
				requireSameQty(t, restartedGPU.Status.Available.Tflops, runningGPU.Status.Available.Tflops, step+": tflops headroom")
				requireSameQty(t, restartedGPU.Status.Available.Vram, runningGPU.Status.Available.Vram, step+": vram headroom")
				requireSameQty(t, restartedGPU.Status.Capacity.Tflops, runningGPU.Status.Capacity.Tflops, step+": tflops capacity")
				require.Len(t, running.uniqueAllocation, len(restarted.uniqueAllocation), step)

				publishedByRunning := syncCapTest(t, runningClient, running)
				publishedByRestarted := syncCapTest(t, restartedClient, restarted)
				requireSameQty(t, publishedByRestarted.Status.Available.Tflops, publishedByRunning.Status.Available.Tflops, step+": published tflops")
				requireSameQty(t, publishedByRestarted.Status.Available.Vram, publishedByRunning.Status.Available.Vram, step+": published vram")
				require.Len(t, publishedByRunning.Status.RunningApps, len(publishedByRestarted.Status.RunningApps), step)
			}

			compare("after the capacity change")
			headroom := running.gpuStore[capTestGPUKey].Status.Available
			requireQty(t, tt.wantHeadroom.Tflops.String(), headroom.Tflops, "tflops headroom")
			requireQty(t, tt.wantHeadroom.Vram.String(), headroom.Vram, "vram headroom")
			published := capTestPublished(t, runningClient)
			requireQty(t, tt.wantPublished.Tflops.String(), published.Status.Available.Tflops, "published tflops")
			requireQty(t, tt.wantPublished.Vram.String(), published.Status.Available.Vram, "published vram")
			require.Len(t, running.uniqueAllocation, len(workers), "no allocation is dropped by a capacity change")

			// Nothing was counted twice and nothing was lost: releasing every
			// workload ends with the whole (new) capacity available.
			for _, worker := range workers {
				podKey := types.NamespacedName{Namespace: worker.Namespace, Name: worker.Name}
				running.DeallocByPodIdentifier(t.Context(), podKey)
				restarted.DeallocByPodIdentifier(t.Context(), podKey)
				compare("after " + worker.Name + " left")
			}
			requireQty(t, tt.to.Tflops.String(), running.gpuStore[capTestGPUKey].Status.Available.Tflops, "tflops of the empty GPU")
			requireQty(t, tt.to.Vram.String(), running.gpuStore[capTestGPUKey].Status.Available.Vram, "vram of the empty GPU")
			require.Empty(t, running.uniqueAllocation)
		})
	}
}

func capTestRequest(name string, request tfv1.Resource) *tfv1.AllocRequest {
	return &tfv1.AllocRequest{
		PoolName:              capTestPoolName,
		PodMeta:               metav1.ObjectMeta{Name: name, Namespace: "default", UID: types.UID(name + "-uid")},
		WorkloadNameNamespace: tfv1.NameNamespace{Name: name, Namespace: "default"},
		Request:               request,
		Limit:                 request,
		Count:                 1,
		Isolation:             tfv1.IsolationModeSoft,
	}
}

// A capacity lowered below what is allocated: the running workloads keep their
// allocations, the GPU publishes zero, every new placement on it is refused,
// the state is reported once, and the GPU returns to normal as soon as enough
// is released.
func TestOvercommittedGPUKeepsWorkloadsAndRefusesPlacements(t *testing.T) {
	k8sClient := capTestClient(t, capTestGPU(capRes("71.2", "24Gi"), ptrRes(capRes("71.2", "24Gi"))))
	s, recorder := startCapTestAllocator(t, k8sClient)

	podA := capTestRequest("a", capRes("30", "8Gi"))
	podB := capTestRequest("b", capRes("30", "8Gi"))
	for _, req := range []*tfv1.AllocRequest{podA, podB} {
		_, err := s.Bind([]string{capTestGPUName}, req)
		require.NoError(t, err)
	}
	published := syncCapTest(t, k8sClient, s)
	requireQty(t, "11.2", published.Status.Available.Tflops, "published tflops before the change")
	require.Empty(t, drainCapTestEvents(recorder))

	// The hardware table is corrected: 71.2 -> 44.08, 60 are allocated.
	nodeAgentPublishes(t, k8sClient, s, setCapacity(capRes("44.08", "24Gi")))

	published = syncCapTest(t, k8sClient, s)
	requireQty(t, "44.08", published.Status.Capacity.Tflops, "published capacity")
	requireQty(t, "0", published.Status.Available.Tflops, "published tflops while over-committed")
	requireQty(t, "8Gi", published.Status.Available.Vram, "published vram while over-committed")

	// Existing allocations are untouched.
	require.Len(t, s.uniqueAllocation, 2)
	requireQty(t, "30", s.uniqueAllocation["a-uid"].Request.Tflops, "allocation of a")
	requireQty(t, "30", s.uniqueAllocation["b-uid"].Request.Tflops, "allocation of b")
	require.Len(t, published.Status.RunningApps, 2)

	// Reported once, at the transition.
	reported := drainCapTestEvents(recorder)
	require.Len(t, reported, 1)
	require.Contains(t, reported[0], "GPUOvercommitted")
	require.True(t, strings.HasPrefix(reported[0], corev1.EventTypeWarning))
	for range 3 {
		s.markGPUDirty(capTestGPUKey)
		syncCapTest(t, k8sClient, s)
		nodeAgentPublishes(t, k8sClient, s, func(*tfv1.GPUStatus) {})
	}
	require.Empty(t, drainCapTestEvents(recorder), "an over-committed GPU is not reported again on later syncs")

	// Every new placement is refused, also one that asks for nothing of the
	// over-committed resource: only TFLOPS is short here, 8Gi of VRAM is free.
	refused := map[string]tfv1.Resource{
		"tflops and vram": capRes("1", "1Gi"),
		"vram only":       capRes("0", "1Gi"),
		"nothing":         {},
		"compute percent": {ComputePercent: qty("10"), Vram: qty("1Gi")},
	}
	for name, request := range refused {
		req := capTestRequest("new-"+strings.ReplaceAll(name, " ", "-"), request)
		_, _, err := s.CheckQuotaAndFilter(t.Context(), req, false)
		require.Error(t, err, "filter must not offer the GPU for %s", name)
		_, err = s.Bind([]string{capTestGPUName}, req)
		require.Error(t, err, "bind must refuse %s", name)
	}
	sharedReq := capTestRequest("new-shared", capRes("1", "1Gi"))
	sharedReq.Isolation = tfv1.IsolationModeShared
	_, err := s.Bind([]string{capTestGPUName}, sharedReq)
	require.Error(t, err)
	require.Len(t, s.uniqueAllocation, 2, "a refused placement leaves no allocation behind")
	requireQty(t, "-15.92", s.gpuStore[capTestGPUKey].Status.Available.Tflops, "headroom is unchanged by refused placements")

	// b leaves: 44.08 - 30 fits again.
	s.DeallocByPodIdentifier(t.Context(), types.NamespacedName{Namespace: "default", Name: "b"})
	published = syncCapTest(t, k8sClient, s)
	requireQty(t, "14.08", published.Status.Available.Tflops, "published tflops after b left")
	requireQty(t, "16Gi", published.Status.Available.Vram, "published vram after b left")
	reported = drainCapTestEvents(recorder)
	require.Len(t, reported, 1)
	require.Contains(t, reported[0], "GPUOvercommitResolved")

	// And placements are accepted again, up to what is really free.
	_, err = s.Bind([]string{capTestGPUName}, capTestRequest("too-big", capRes("20", "1Gi")))
	require.Error(t, err)
	_, _, err = s.CheckQuotaAndFilter(t.Context(), capTestRequest("c", capRes("10", "1Gi")), false)
	require.NoError(t, err)
	_, err = s.Bind([]string{capTestGPUName}, capTestRequest("c", capRes("10", "1Gi")))
	require.NoError(t, err)
	published = syncCapTest(t, k8sClient, s)
	requireQty(t, "4.08", published.Status.Available.Tflops, "published tflops after c was placed")
	require.Empty(t, drainCapTestEvents(recorder))
}

func ptrRes(res tfv1.Resource) *tfv1.Resource {
	return &res
}

// Preemption is planned on the real headroom of an over-committed GPU, not on
// the published zero: evicting one 30-TFLOPS workload from a GPU that is 15.92
// over frees 14.08, so a 20-TFLOPS request must not count on it.
func TestPreemptionOnOvercommittedGPUCountsTheDeficit(t *testing.T) {
	k8sClient := capTestClient(t, capTestGPU(capRes("44.08", "24Gi"), ptrRes(capRes("44.08", "24Gi"))),
		capTestFixedWorker("a", "30", "8Gi"), capTestFixedWorker("b", "30", "8Gi"))
	s, _ := startCapTestAllocator(t, k8sClient)
	victim := s.uniqueAllocation["b-uid"]
	require.NotNil(t, victim)

	candidates, _, err := s.FilterWithPreempt(capTestRequest("big", capRes("20", "1Gi")),
		[]*tfv1.AllocRequest{victim}, capTestNodeName)
	require.NoError(t, err)
	require.Empty(t, candidates)

	candidates, _, err = s.FilterWithPreempt(capTestRequest("small", capRes("10", "1Gi")),
		[]*tfv1.AllocRequest{victim}, capTestNodeName)
	require.NoError(t, err)
	require.Len(t, candidates, 1)
	requireQty(t, "14.08", candidates[0].Status.Available.Tflops, "tflops after the simulated release")
}

// Resizing a workload in place on an over-committed GPU: shrinking is how the
// GPU fits again and is accepted, growing is refused until it does.
func TestAdjustAllocationOnOvercommittedGPU(t *testing.T) {
	k8sClient := capTestClient(t, capTestGPU(capRes("44.08", "24Gi"), ptrRes(capRes("44.08", "24Gi"))),
		capTestFixedWorker("a", "30", "8Gi"), capTestFixedWorker("b", "30", "8Gi"))
	s, _ := startCapTestAllocator(t, k8sClient)
	adjust := func(podUID, tflops string) error {
		_, _, _, err := s.AdjustAllocation(t.Context(), tfv1.AdjustRequest{
			PodUID: podUID, NewRequest: capRes(tflops, "8Gi"), NewLimit: capRes(tflops, "8Gi"),
		}, false)
		return err
	}
	headroom := func() resource.Quantity {
		return s.gpuStore[capTestGPUKey].Status.Available.Tflops
	}
	requireQty(t, "-15.92", headroom(), "headroom at start")

	require.ErrorIs(t, adjust("a-uid", "31"), ScalingQuotaExceededError)
	require.NoError(t, adjust("a-uid", "20"))
	requireQty(t, "-5.92", headroom(), "headroom after a shrank")
	require.ErrorIs(t, adjust("a-uid", "21"), ScalingQuotaExceededError, "still over-committed")
	require.NoError(t, adjust("b-uid", "20"))
	requireQty(t, "4.08", headroom(), "headroom after b shrank")
	require.NoError(t, adjust("a-uid", "24"))
	requireQty(t, "0.08", headroom(), "headroom after a grew into the free part")
	require.ErrorIs(t, adjust("b-uid", "21"), ScalingQuotaExceededError)

	published := syncCapTest(t, k8sClient, s)
	requireQty(t, "0.08", published.Status.Available.Tflops, "published tflops")
}

// Upgrade order 1: the operator is upgraded first, the node agents later. An
// agent that predates the capacity rework writes back the whole GPU status it
// read earlier, the allocator's fields included. The allocator publishes its
// figures again and does not adopt the agent's; once the agent is upgraded and
// publishes the new capacity with the same rule, nothing is left to correct.
func TestUpgradeOperatorFirstThenNodeAgent(t *testing.T) {
	k8sClient := capTestClient(t, capTestGPU(capRes("44.08", "24Gi"), ptrRes(capRes("44.08", "24Gi"))),
		capTestFixedWorker("a", "30", "8Gi"))
	s, recorder := startCapTestAllocator(t, k8sClient)
	published := syncCapTest(t, k8sClient, s)
	requireQty(t, "14.08", published.Status.Available.Tflops, "published tflops")
	require.Len(t, published.Status.RunningApps, 1)

	// The watch event of the allocator's own write queues nothing.
	s.handleGPUUpdate(t.Context(), published)
	require.False(t, queuedForSync(s))

	// The old agent syncs the device: old capacity, and the allocation fields
	// as they were before a was placed.
	nodeAgentPublishes(t, k8sClient, s, func(status *tfv1.GPUStatus) {
		status.Available = ptrRes(capRes("44.08", "24Gi"))
		status.RunningApps = nil
	})
	requireQty(t, "14.08", s.gpuStore[capTestGPUKey].Status.Available.Tflops, "the allocator does not adopt the agent's figure")
	require.True(t, queuedForSync(s), "the stale figure is published again")
	published = syncCapTest(t, k8sClient, s)
	requireQty(t, "14.08", published.Status.Available.Tflops, "published tflops after the old agent's write")
	requireQty(t, "16Gi", published.Status.Available.Vram, "published vram after the old agent's write")
	require.Len(t, published.Status.RunningApps, 1)

	// Only the allocation fields are compared: an agent write that leaves
	// them alone, as every sync of a current agent does, queues nothing.
	nodeAgentPublishes(t, k8sClient, s, func(status *tfv1.GPUStatus) { status.Message = "managed" })
	require.False(t, queuedForSync(s))

	// The agent is upgraded and the model is listed: 71.2, and the agent
	// settles available itself (71.2 - 30).
	nodeAgentPublishes(t, k8sClient, s, func(status *tfv1.GPUStatus) {
		status.Capacity = ptrRes(capRes("71.2", "24Gi"))
		status.Available = ptrRes(capRes("41.2", "16Gi"))
	})
	requireQty(t, "41.2", s.gpuStore[capTestGPUKey].Status.Available.Tflops, "headroom at the new capacity")
	require.False(t, queuedForSync(s), "both sides computed the same figure, there is nothing to write")
	published = syncCapTest(t, k8sClient, s)
	requireQty(t, "41.2", published.Status.Available.Tflops, "published tflops at the new capacity")
	require.Empty(t, drainCapTestEvents(recorder))
}

// Upgrade order 2: the node agents are upgraded while the previous operator
// is still running, then the operator. Whatever the previous operator left in
// status.available for the changed capacity, the new one rebuilds the
// accounting from the worker Pods, publishes it, and reports an over-committed
// GPU once.
func TestUpgradeNodeAgentFirstThenOperator(t *testing.T) {
	tests := []struct {
		name string
		// What is in status.available when the new operator starts. The
		// capacity is already the new, lower one: 44.08 with 60 allocated.
		leftAvailable *tfv1.Resource
		// Whether runningApps were lost too (a status written back by a node
		// agent from an older read).
		runningAppsLost bool
	}{
		{"negative headroom published as is", ptrRes(capRes("-15.92", "8Gi")), false},
		{"figure for the previous capacity", ptrRes(capRes("11.2", "8Gi")), false},
		{"figure above the capacity", ptrRes(capRes("71.2", "24Gi")), true},
		{"already floored by the upgraded agent", ptrRes(capRes("0", "8Gi")), false},
		{"never published", nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			workers := []*corev1.Pod{capTestFixedWorker("a", "30", "8Gi"), capTestFixedWorker("b", "30", "8Gi")}

			// The status the previous operator left: runningApps exactly as an
			// allocator writes them for these Pods, so that available is the
			// only thing the new operator can find wrong.
			previousClient := capTestClient(t, capTestGPU(capRes("71.2", "24Gi"), ptrRes(capRes("71.2", "24Gi"))), workers...)
			previous, _ := startCapTestAllocator(t, previousClient)
			left := syncCapTest(t, previousClient, previous)
			require.Len(t, left.Status.RunningApps, 2)
			left.ResourceVersion = ""
			left.Status.Capacity = ptrRes(capRes("44.08", "24Gi"))
			left.Status.Available = tt.leftAvailable
			if tt.runningAppsLost {
				left.Status.RunningApps = nil
			}
			k8sClient := capTestClient(t, left, workers...)

			s, recorder := startCapTestAllocator(t, k8sClient)

			requireQty(t, "-15.92", s.gpuStore[capTestGPUKey].Status.Available.Tflops, "headroom rebuilt from the worker Pods")
			requireQty(t, "8Gi", s.gpuStore[capTestGPUKey].Status.Available.Vram, "vram rebuilt from the worker Pods")
			require.Len(t, s.uniqueAllocation, 2)

			published := syncCapTest(t, k8sClient, s)
			requireQty(t, "0", published.Status.Available.Tflops, "published tflops")
			requireQty(t, "8Gi", published.Status.Available.Vram, "published vram")
			require.Len(t, published.Status.RunningApps, 2)
			reported := drainCapTestEvents(recorder)
			require.Len(t, reported, 1)
			require.Contains(t, reported[0], "GPUOvercommitted")

			// The informer's first event for the GPU changes nothing.
			s.handleGPUCreate(t.Context(), published)
			require.False(t, queuedForSync(s))
			_, err := s.Bind([]string{capTestGPUName}, capTestRequest("c", capRes("1", "1Gi")))
			require.Error(t, err)
		})
	}
}

// The store is listed before the informer starts. A capacity that changes in
// between arrives with the informer's first event for the GPU and is applied.
func TestCapacityChangeBetweenListAndInformerStart(t *testing.T) {
	k8sClient := capTestClient(t, capTestGPU(capRes("44.08", "24Gi"), ptrRes(capRes("44.08", "24Gi"))),
		capTestFixedWorker("a", "30", "8Gi"))
	s, _ := startCapTestAllocator(t, k8sClient)

	gpu := capTestPublished(t, k8sClient)
	gpu.Status.Capacity = ptrRes(capRes("71.2", "24Gi"))
	require.NoError(t, k8sClient.Status().Update(t.Context(), gpu))
	s.handleGPUCreate(t.Context(), capTestPublished(t, k8sClient))

	requireQty(t, "41.2", s.gpuStore[capTestGPUKey].Status.Available.Tflops, "headroom at the new capacity")
	published := syncCapTest(t, k8sClient, s)
	requireQty(t, "41.2", published.Status.Available.Tflops, "published tflops")
}
