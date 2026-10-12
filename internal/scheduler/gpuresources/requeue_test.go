package gpuresources

import (
	"context"
	"sync"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/indexallocator"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	framework "k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	schedmetrics "k8s.io/kubernetes/pkg/scheduler/metrics"
	tffwk "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	clientfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	requeueSchedulerName = "tensor-fusion-scheduler"
	requeuePool          = "pool-a"
	requeueNewNode       = "node-new"
	requeueNewGPU        = "gpu-new"
)

var requeueGPUEvent = fwk.ClusterEvent{
	Resource:   fwk.EventResource("gpus.v1.tensor-fusion.ai"),
	ActionType: fwk.Update,
}

// lateActivator lets the framework be built before the queue it activates.
type lateActivator struct {
	mu    sync.Mutex
	queue internalqueue.SchedulingQueue
}

func (a *lateActivator) Activate(logger klog.Logger, pods map[string]*corev1.Pod) {
	a.mu.Lock()
	queue := a.queue
	a.mu.Unlock()
	if queue != nil {
		queue.Activate(logger, pods)
	}
}

// requeueHarness is the plugin behind a real scheduling queue that is wired
// like the scheduler wires it: the plugin's registered events and queueing
// hints decide what leaves the unschedulable pool, and the framework handle
// activates through the queue. The queue's periodic flush is never started.
type requeueHarness struct {
	t         *testing.T
	ctx       context.Context
	logger    klog.Logger
	allocator *gpuallocator.GpuAllocator
	plugin    *GPUFit
	queue     internalqueue.SchedulingQueue
}

func requeueTestGPU(phase tfv1.TensorFusionGPUPhase) *tfv1.GPU {
	return &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name:   requeueNewGPU,
			Labels: map[string]string{constants.GpuPoolKey: requeuePool, constants.LabelKeyOwner: requeueNewNode},
		},
		Status: tfv1.GPUStatus{
			Phase:        phase,
			NodeSelector: map[string]string{constants.KubernetesHostNameLabel: requeueNewNode},
			UsedBy:       tfv1.UsedByTensorFusion,
			Capacity:     &tfv1.Resource{Tflops: resource.MustParse("1000"), Vram: resource.MustParse("20Gi")},
			Available:    &tfv1.Resource{Tflops: resource.MustParse("1000"), Vram: resource.MustParse("20Gi")},
		},
	}
}

func requeueTestPod(name, pool string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "ns1",
			UID:       types.UID(name),
			Labels: map[string]string{
				constants.LabelComponent: constants.ComponentWorker,
				constants.WorkloadKey:    "workload-1",
			},
			Annotations: map[string]string{
				constants.GpuPoolKey:              pool,
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "1Gi",
				constants.TFLOPSLimitAnnotation:   "100",
				constants.VRAMLimitAnnotation:     "1Gi",
				constants.GpuCountAnnotation:      "1",
			},
		},
		Spec: corev1.PodSpec{
			SchedulerName: requeueSchedulerName,
			Containers:    []corev1.Container{{Name: "main", Image: "busybox"}},
		},
	}
}

// newRequeueHarness starts with one GPU of the pool on a node that has just
// joined: the GPU is registered but still Pending, in the API and in the
// allocator's store.
func newRequeueHarness(t *testing.T) *requeueHarness {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	k8sClient := clientfake.NewClientBuilder().WithScheme(scheme).
		WithRuntimeObjects(
			&tfv1.TensorFusionWorkload{ObjectMeta: metav1.ObjectMeta{Name: "workload-1", Namespace: "ns1"}},
		).
		WithStatusSubresource(&tfv1.GPU{}, &tfv1.GPUResourceQuota{}, &tfv1.TensorFusionWorkload{}).
		Build()
	require.NoError(t, k8sClient.Create(ctx, requeueTestGPU(tfv1.TensorFusionGPUPhasePending)))

	allocator := gpuallocator.NewGpuAllocator(ctx, nil, k8sClient, time.Second)
	require.NoError(t, allocator.InitGPUAndQuotaStore())
	allocator.ReconcileAllocationState()
	allocator.SetAllocatorReady()
	// The fake client drops the status on Create; put the GPU into the store
	// the way the informer would deliver it.
	allocator.UpsertGPUForTesting(ctx, requeueTestGPU(tfv1.TensorFusionGPUPhasePending))

	indexAlloc, err := indexallocator.NewIndexAllocator(ctx, k8sClient)
	require.NoError(t, err)

	activator := &lateActivator{}
	fwkInstance, err := tffwk.NewFramework(
		ctx,
		[]tffwk.RegisterPluginFunc{
			tffwk.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			tffwk.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		},
		requeueSchedulerName,
		frameworkruntime.WithSnapshotSharedLister(internalcache.NewEmptySnapshot()),
		frameworkruntime.WithEventRecorder(&events.FakeRecorder{}),
		frameworkruntime.WithPodActivator(activator),
	)
	require.NoError(t, err)

	plug, err := NewWithDeps(allocator, indexAlloc, nil, k8sClient)(
		ctx, &runtime.Unknown{Raw: []byte(`{"maxWorkerPerNode": 8}`)}, fwkInstance)
	require.NoError(t, err)
	plugin := plug.(*GPUFit)

	registered, err := plugin.EventsToRegister(ctx)
	require.NoError(t, err)
	hints := internalqueue.QueueingHintMap{}
	for _, event := range registered {
		hints[event.Event] = append(hints[event.Event], &internalqueue.QueueingHintFunction{
			PluginName:     plugin.Name(),
			QueueingHintFn: event.QueueingHintFn,
		})
	}
	schedmetrics.Register()
	queue := internalqueue.NewSchedulingQueue(
		fwkInstance.QueueSortFunc(),
		informers.NewSharedInformerFactory(clientsetfake.NewClientset(), 0),
		internalqueue.WithMetricsRecorder(schedmetrics.NewMetricsAsyncRecorder(100, time.Second, ctx.Done())),
		internalqueue.WithQueueingHintMapPerProfile(internalqueue.QueueingHintMapPerProfile{requeueSchedulerName: hints}),
		// Keeps the attempts of this test out of the backoff queue.
		internalqueue.WithPodInitialBackoffDuration(time.Millisecond),
		internalqueue.WithPodMaxBackoffDuration(2*time.Millisecond),
	)
	t.Cleanup(queue.Close)
	activator.mu.Lock()
	activator.queue = queue
	activator.mu.Unlock()

	return &requeueHarness{
		t: t, ctx: ctx, logger: klog.FromContext(ctx),
		allocator: allocator, plugin: plugin, queue: queue,
	}
}

// attempt pops the next pod and runs the plugin's PreFilter for it, as one
// scheduling cycle does. A rejected pod goes back to the queue with the plugin
// recorded as the one that rejected it. It fails the test when no pod becomes
// ready within the wait.
func (h *requeueHarness) attempt(wait time.Duration) (*corev1.Pod, *fwk.Status) {
	h.t.Helper()
	popped := make(chan *framework.QueuedPodInfo, 1)
	go func() {
		pInfo, err := h.queue.Pop(h.logger)
		if err == nil {
			popped <- pInfo
		}
	}()
	var pInfo *framework.QueuedPodInfo
	select {
	case pInfo = <-popped:
	case <-time.After(wait):
		require.FailNowf(h.t, "no pod became ready for scheduling",
			"waited %s; unschedulable pods: %v", wait, podNames(h.queue.UnschedulablePods()))
	}
	_, status := h.plugin.PreFilter(h.ctx, framework.NewCycleState(), pInfo.Pod, nil)
	if status.IsSuccess() {
		h.queue.Done(pInfo.Pod.UID)
		return pInfo.Pod, status
	}
	pInfo.UnschedulablePlugins = sets.New(h.plugin.Name())
	require.NoError(h.t, h.queue.AddUnschedulableIfNotPresent(h.logger, pInfo, h.queue.SchedulingCycle()))
	return pInfo.Pod, status
}

// gpuTurnsRunningInScheduler delivers the GPU's Pending -> Running update to
// the scheduling queue, the way the scheduler's informer does.
func (h *requeueHarness) gpuTurnsRunningInScheduler() *tfv1.GPU {
	pending := requeueTestGPU(tfv1.TensorFusionGPUPhasePending)
	pending.ResourceVersion = "1"
	running := requeueTestGPU(tfv1.TensorFusionGPUPhaseRunning)
	running.ResourceVersion = "2"
	// Past the backoff of the earlier attempt, as a node that takes a minute
	// to come up is.
	time.Sleep(20 * time.Millisecond)
	h.queue.MoveAllToActiveOrBackoffQueue(h.logger, requeueGPUEvent, pending, running, nil)
	return running
}

func podNames(pods []*corev1.Pod) []string {
	names := make([]string, 0, len(pods))
	for _, pod := range pods {
		names = append(names, pod.Name)
	}
	return names
}

// A pod that is unschedulable because the pool's only free GPU is still
// Pending must be scheduled as soon as the allocator knows the GPU is Running,
// also when the scheduler's informer reported the change first and the retry
// it caused ran against the allocator's older store. No further cluster event
// and no periodic flush is involved: the queue's flush loop is not running.
func TestPendingPodIsRetriedWhenAllocatorSeesGPURunning(t *testing.T) {
	h := newRequeueHarness(t)
	waiting := requeueTestPod("waiting", requeuePool)
	otherPool := requeueTestPod("other-pool", "pool-b")
	h.queue.Add(h.ctx, waiting)
	h.queue.Add(h.ctx, otherPool)
	for range 2 {
		pod, status := h.attempt(2 * time.Second)
		require.Equal(t, fwk.Unschedulable, status.Code(), "pod %s: %s", pod.Name, status.Message())
	}
	require.ElementsMatch(t, []string{"waiting", "other-pool"}, podNames(h.queue.UnschedulablePods()))

	// The scheduler hears about the GPU first.
	running := h.gpuTurnsRunningInScheduler()
	require.Equal(t, []string{"other-pool"}, podNames(h.queue.UnschedulablePods()),
		"only the pod of the GPU's pool is re-queued")
	pod, status := h.attempt(2 * time.Second)
	require.Equal(t, "waiting", pod.Name)
	require.Equal(t, fwk.Unschedulable, status.Code(),
		"the allocator's store still has the GPU Pending, so this attempt cannot succeed")
	require.ElementsMatch(t, []string{"waiting", "other-pool"}, podNames(h.queue.UnschedulablePods()))

	// The allocator's informer delivers the same update.
	h.allocator.UpsertGPUForTesting(h.ctx, running)

	pod, status = h.attempt(5 * time.Second)
	require.Equal(t, "waiting", pod.Name)
	require.True(t, status.IsSuccess(), "PreFilter after the store caught up: %s", status.Message())
	require.Equal(t, []string{"other-pool"}, podNames(h.queue.UnschedulablePods()),
		"a pod of another pool is not activated")
}

// When the allocator's store already shows the GPU Running, the retry the hint
// causes is enough and nothing is kept waiting.
func TestGPURunningHintDoesNotWaitWhenAllocatorIsCurrent(t *testing.T) {
	h := newRequeueHarness(t)
	activated := make(chan map[string]*corev1.Pod, 1)
	h.plugin.allocatorSync.activate = func(pods map[string]*corev1.Pod) { activated <- pods }

	waiting := requeueTestPod("waiting", requeuePool)
	h.queue.Add(h.ctx, waiting)
	_, status := h.attempt(2 * time.Second)
	require.Equal(t, fwk.Unschedulable, status.Code())

	running := requeueTestGPU(tfv1.TensorFusionGPUPhaseRunning)
	running.ResourceVersion = "2"
	h.allocator.UpsertGPUForTesting(h.ctx, running)
	h.gpuTurnsRunningInScheduler()

	pod, status := h.attempt(2 * time.Second)
	require.Equal(t, "waiting", pod.Name)
	require.True(t, status.IsSuccess(), status.Message())

	h.plugin.allocatorSync.mu.Lock()
	waiters := len(h.plugin.allocatorSync.waiting)
	h.plugin.allocatorSync.mu.Unlock()
	require.Zero(t, waiters)
	select {
	case pods := <-activated:
		require.Failf(t, "unexpected activation", "%v", pods)
	case <-time.After(100 * time.Millisecond):
	}
}

// A store that never shows the GPU Running ends the wait without activating
// anything and without leaving the waiter behind.
func TestGPURunningWakeupGivesUpAfterTimeout(t *testing.T) {
	previous := allocatorSyncTimeout
	allocatorSyncTimeout = 100 * time.Millisecond
	t.Cleanup(func() { allocatorSyncTimeout = previous })

	h := newRequeueHarness(t)
	activated := make(chan map[string]*corev1.Pod, 1)
	h.plugin.allocatorSync.activate = func(pods map[string]*corev1.Pod) { activated <- pods }

	pending := requeueTestGPU(tfv1.TensorFusionGPUPhasePending)
	running := requeueTestGPU(tfv1.TensorFusionGPUPhaseRunning)
	running.ResourceVersion = "2"
	hint, err := h.plugin.queueingHint(h.logger, requeueTestPod("waiting", requeuePool), pending, running)
	require.NoError(t, err)
	require.Equal(t, fwk.Queue, hint)

	require.Eventually(t, func() bool {
		h.plugin.allocatorSync.mu.Lock()
		defer h.plugin.allocatorSync.mu.Unlock()
		return len(h.plugin.allocatorSync.waiting) == 0
	}, 3*time.Second, 20*time.Millisecond)
	select {
	case pods := <-activated:
		require.Failf(t, "unexpected activation", "%v", pods)
	default:
	}
}

func TestNodeQueueingHint(t *testing.T) {
	h := newRequeueHarness(t)
	idx0 := corev1.ResourceName(constants.PodIndexAnnotation + constants.PodIndexDelimiter + "0")
	node := func(pool string, index *int64) *corev1.Node {
		n := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: requeueNewNode, Labels: map[string]string{}}}
		if pool != "" {
			n.Labels[constants.GPUNodePoolIdentifierLabelPrefix+pool] = constants.TrueStringValue
		}
		if index != nil {
			n.Status.Allocatable = corev1.ResourceList{idx0: *resource.NewQuantity(*index, resource.DecimalSI)}
		}
		return n
	}
	zero, some := int64(0), int64(36)
	worker := requeueTestPod("waiting", requeuePool)
	plain := requeueTestPod("plain", requeuePool)
	delete(plain.Labels, constants.LabelComponent)

	tests := []struct {
		name     string
		pod      *corev1.Pod
		old, new any
		want     fwk.QueueingHint
	}{
		{"index resources allocatable again", worker, node(requeuePool, &zero), node(requeuePool, &some), fwk.Queue},
		{"node without pool labels", worker, node("", &zero), node("", &some), fwk.Queue},
		{"node of another pool", worker, node("pool-b", &zero), node("pool-b", &some), fwk.QueueSkip},
		{"first registration was never rejected", worker, node(requeuePool, nil), node(requeuePool, &some), fwk.QueueSkip},
		{"still nothing allocatable", worker, node(requeuePool, &zero), node(requeuePool, &zero), fwk.QueueSkip},
		{"unchanged", worker, node(requeuePool, &some), node(requeuePool, &some), fwk.QueueSkip},
		{"node added", worker, nil, node(requeuePool, &some), fwk.QueueSkip},
		{"not a TensorFusion worker", plain, node(requeuePool, &zero), node(requeuePool, &some), fwk.QueueSkip},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := h.plugin.nodeQueueingHint(h.logger, tt.pod, tt.old, tt.new)
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}
}
