package sched

import (
	"context"
	"strconv"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/indexallocator"
	gpuResourceFitPlugin "github.com/NexusGPU/tensor-fusion/internal/scheduler/gpuresources"
	gpuTopoPlugin "github.com/NexusGPU/tensor-fusion/internal/scheduler/gputopo"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	informers "k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func topoInt32Ptr(v int32) *int32 { return &v }

type topoFixture struct {
	ctx       context.Context
	cancel    context.CancelFunc
	gpuFit    *gpuResourceFitPlugin.GPUFit
	gpuTopo   fwk.Plugin
	nodes     []*v1.Node
	allocator *gpuallocator.GpuAllocator
}

func newTopoFixture(t *testing.T, nodes []*v1.Node, gpus []*tfv1.GPU, topoConfigJSON string) *topoFixture {
	t.Helper()
	require.NoError(t, tfv1.AddToScheme(scheme.Scheme))

	ns := "test-ns"
	cl := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithRuntimeObjects(&tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{Name: "test-workload", Namespace: ns},
		}).
		WithStatusSubresource(&tfv1.GPU{}, &tfv1.GPUNode{}, &tfv1.TensorFusionWorkload{}, &v1.Pod{}, &v1.Node{}).
		Build()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	ctx = log.IntoContext(ctx, klog.NewKlogr())

	require.NoError(t, cl.Create(ctx, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}))

	k8sObjs := make([]k8sruntime.Object, 0, len(nodes))
	for _, node := range nodes {
		cp := node.DeepCopy()
		require.NoError(t, cl.Create(ctx, cp))
		k8sObjs = append(k8sObjs, cp)
	}
	for _, gpu := range gpus {
		require.NoError(t, cl.Create(ctx, gpu.DeepCopy()))
	}

	allocator := gpuallocator.NewGpuAllocator(ctx, nil, cl, time.Second)
	require.NoError(t, allocator.InitGPUAndQuotaStore())
	allocator.ReconcileAllocationState()
	allocator.SetAllocatorReady()

	indexAlloc, err := indexallocator.NewIndexAllocator(ctx, cl)
	require.NoError(t, err)

	fakeClientSet := clientsetfake.NewClientset(k8sObjs...)
	informerFactory := informers.NewSharedInformerFactory(fakeClientSet, 0)
	metrics.Register()
	metricsRecorder := metrics.NewMetricsAsyncRecorder(1000, time.Second, ctx.Done())
	fw, err := tf.NewFramework(
		ctx,
		[]tf.RegisterPluginFunc{
			tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		}, "",
		frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
		frameworkruntime.WithSnapshotSharedLister(internalcache.NewEmptySnapshot()),
		frameworkruntime.WithEventRecorder(&events.FakeRecorder{}),
		frameworkruntime.WithMetricsRecorder(metricsRecorder),
	)
	require.NoError(t, err)

	fitPlugin, err := gpuResourceFitPlugin.NewWithDeps(allocator, indexAlloc, nil, cl)(
		ctx, &k8sruntime.Unknown{Raw: []byte(`{"maxWorkerPerNode": 256, "vramWeight": 0.7, "tflopsWeight": 0.3}`)}, fw,
	)
	require.NoError(t, err)

	topoPlugin, err := gpuTopoPlugin.New()(
		ctx, &k8sruntime.Unknown{Raw: []byte(topoConfigJSON)}, fw,
	)
	require.NoError(t, err)

	return &topoFixture{
		ctx:       ctx,
		cancel:    cancel,
		gpuFit:    fitPlugin.(*gpuResourceFitPlugin.GPUFit),
		gpuTopo:   topoPlugin,
		nodes:     nodes,
		allocator: allocator,
	}
}

func (f *topoFixture) close() { f.cancel() }

func topoMakeGPU(name, nodeName string, numaNode *int32, poolName string) *tfv1.GPU {
	return &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				constants.GpuPoolKey:    poolName,
				constants.LabelKeyOwner: nodeName,
			},
		},
		Status: tfv1.GPUStatus{
			Phase:        tfv1.TensorFusionGPUPhaseRunning,
			NodeSelector: map[string]string{constants.KubernetesHostNameLabel: nodeName},
			UsedBy:       tfv1.UsedByTensorFusion,
			NUMANode:     numaNode,
			Capacity: &tfv1.Resource{
				Tflops: resource.MustParse("500"),
				Vram:   resource.MustParse("40Gi"),
			},
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("500"),
				Vram:   resource.MustParse("40Gi"),
			},
		},
	}
}

func topoMakeNode(name string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Labels:      map[string]string{"test": "value", constants.KubernetesHostNameLabel: name},
			Annotations: map[string]string{"test": "value2"},
		},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("128"),
				v1.ResourceMemory: resource.MustParse("512Gi"),
				"pods":            resource.MustParse("110"),
			},
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("128"),
				v1.ResourceMemory: resource.MustParse("512Gi"),
				"pods":            resource.MustParse("110"),
			},
			Phase:      v1.NodeRunning,
			Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
		},
	}
}

func topoMakePod(name, poolName string, gpuCount int) *v1.Pod {
	pod := st.MakePod().
		Namespace("test-ns").Name(name).UID(name).
		SchedulerName("tensor-fusion-scheduler").
		Res(map[v1.ResourceName]string{v1.ResourceCPU: "10m", v1.ResourceMemory: "128Mi"}).
		NodeAffinityIn("test", []string{"value", "value2"}, st.NodeSelectorTypeMatchExpressions).
		Toleration("node.kubernetes.io/not-ready").
		ZeroTerminationGracePeriod().Obj()

	pod.Labels = map[string]string{
		constants.LabelComponent: constants.ComponentWorker,
		constants.WorkloadKey:    "test-workload",
	}
	pod.Annotations = map[string]string{
		constants.GpuPoolKey:              poolName,
		constants.TFLOPSRequestAnnotation: "100",
		constants.VRAMRequestAnnotation:   "8Gi",
		constants.TFLOPSLimitAnnotation:   "100",
		constants.VRAMLimitAnnotation:     "8Gi",
		constants.GpuCountAnnotation:      strconv.Itoa(gpuCount),
	}
	indexKey := v1.ResourceName(constants.PodIndexAnnotation + constants.PodIndexDelimiter + "0")
	if pod.Spec.Containers[0].Resources.Limits == nil {
		pod.Spec.Containers[0].Resources.Limits = make(v1.ResourceList)
	}
	pod.Spec.Containers[0].Resources.Limits[indexKey] = *resource.NewQuantity(1, resource.DecimalSI)
	return pod
}

// TestIntegration_MultiGPU_SameNUMA_Preferred verifies same-NUMA nodes score higher.
func TestIntegration_MultiGPU_SameNUMA_Preferred(t *testing.T) {
	utils.SetProgressiveMigration(false)
	pool := "topo-test-pool"
	nodes := []*v1.Node{topoMakeNode("node-same"), topoMakeNode("node-cross")}
	gpus := []*tfv1.GPU{
		topoMakeGPU("gpu-0", "node-same", topoInt32Ptr(0), pool),
		topoMakeGPU("gpu-1", "node-same", topoInt32Ptr(0), pool),
		topoMakeGPU("gpu-2", "node-cross", topoInt32Ptr(0), pool),
		topoMakeGPU("gpu-3", "node-cross", topoInt32Ptr(1), pool),
	}

	f := newTopoFixture(
		t,
		nodes,
		gpus,
		`{"mode":"soft","topologySource":"auto","maxAllowedTier":1,"preferLeastDamage":true}`,
	)
	defer f.close()

	pod := topoMakePod("multi-gpu", pool, 2)
	state := framework.NewCycleState()

	// GPUResourcesFit PreFilter
	f.gpuFit.PreFilter(f.ctx, state, pod, nil)

	// GPUNetworkTopologyAware PreFilter
	topoPreFilter := f.gpuTopo.(fwk.PreFilterPlugin)
	_, topoStatus := topoPreFilter.PreFilter(f.ctx, state, pod, nil)
	t.Logf("topo PreFilter status: %v", topoStatus)

	// Read topology state
	topoStateRaw, err := state.Read(gpuTopoPlugin.CycleStateGPUTopologyResult)
	require.NoError(t, err, "topology state should be written")
	topoState := topoStateRaw.(*gpuTopoPlugin.GPUTopologyStateData)

	samePlan := topoState.Plans["node-same"]
	crossPlan := topoState.Plans["node-cross"]

	if samePlan != nil && crossPlan != nil {
		t.Logf("node-same: tier=%d score=%d bestGPUs=%v", samePlan.Tier, samePlan.Score, samePlan.BestGPUIds)
		t.Logf("node-cross: tier=%d score=%d bestGPUs=%v", crossPlan.Tier, crossPlan.Score, crossPlan.BestGPUIds)

		assert.Equal(t, gpuTopoPlugin.TierSameNUMA, samePlan.Tier, "same-NUMA node should have TierSameNUMA")
		assert.Equal(t, gpuTopoPlugin.TierCrossNUMA, crossPlan.Tier, "cross-NUMA node should have TierCrossNUMA")
		assert.Greater(t, samePlan.Score, crossPlan.Score, "same-NUMA should score higher")
	}

	// Filter: soft mode should pass both
	topoFilter := f.gpuTopo.(fwk.FilterPlugin)
	for _, node := range nodes {
		ni := &framework.NodeInfo{}
		ni.SetNode(node)
		status := topoFilter.Filter(f.ctx, state, pod, ni)
		assert.True(t, status.IsSuccess(), "soft mode should pass node %s: %v", node.Name, status)
	}
}

// TestIntegration_HardMode_RejectsCrossNUMA verifies hard mode rejects cross-NUMA.
func TestIntegration_HardMode_RejectsCrossNUMA(t *testing.T) {
	utils.SetProgressiveMigration(false)
	pool := "topo-hard-pool"
	nodes := []*v1.Node{topoMakeNode("node-cross")}
	gpus := []*tfv1.GPU{
		topoMakeGPU("gpu-0", "node-cross", topoInt32Ptr(0), pool),
		topoMakeGPU("gpu-1", "node-cross", topoInt32Ptr(1), pool),
	}

	f := newTopoFixture(t, nodes, gpus, `{"mode":"hard","maxAllowedTier":1}`)
	defer f.close()

	pod := topoMakePod("hard-pod", pool, 2)
	state := framework.NewCycleState()

	f.gpuFit.PreFilter(f.ctx, state, pod, nil)
	topoPreFilter := f.gpuTopo.(fwk.PreFilterPlugin)
	topoPreFilter.PreFilter(f.ctx, state, pod, nil)

	ni := &framework.NodeInfo{}
	ni.SetNode(nodes[0])
	topoFilter := f.gpuTopo.(fwk.FilterPlugin)
	filterStatus := topoFilter.Filter(f.ctx, state, pod, ni)

	t.Logf("hard mode filter status: code=%d msg=%s", filterStatus.Code(), filterStatus.Message())
	assert.False(t, filterStatus.IsSuccess(), "hard mode should reject cross-NUMA node")
}

// TestIntegration_SingleGPU_LeastDamage verifies orphan GPU is preferred.
func TestIntegration_SingleGPU_LeastDamage(t *testing.T) {
	utils.SetProgressiveMigration(false)
	pool := "topo-ld-pool"
	nodes := []*v1.Node{topoMakeNode("node-0")}
	gpus := []*tfv1.GPU{
		topoMakeGPU("gpu-0", "node-0", topoInt32Ptr(0), pool),
		topoMakeGPU("gpu-1", "node-0", topoInt32Ptr(0), pool),
		topoMakeGPU("gpu-2", "node-0", topoInt32Ptr(0), pool),
		topoMakeGPU("gpu-3", "node-0", topoInt32Ptr(0), pool),
		topoMakeGPU("gpu-4", "node-0", topoInt32Ptr(1), pool), // orphan
	}

	f := newTopoFixture(t, nodes, gpus, `{"mode":"soft","maxAllowedTier":1,"preferLeastDamage":true}`)
	defer f.close()

	pod := topoMakePod("single-gpu", pool, 1)
	state := framework.NewCycleState()

	f.gpuFit.PreFilter(f.ctx, state, pod, nil)
	topoPreFilter := f.gpuTopo.(fwk.PreFilterPlugin)
	topoPreFilter.PreFilter(f.ctx, state, pod, nil)

	topoStateRaw, err := state.Read(gpuTopoPlugin.CycleStateGPUTopologyResult)
	require.NoError(t, err)
	topoState := topoStateRaw.(*gpuTopoPlugin.GPUTopologyStateData)
	plan := topoState.Plans["node-0"]

	require.NotNil(t, plan, "should have plan for node-0")
	t.Logf("BestGPUIds: %v, Reason: %s", plan.BestGPUIds, plan.Reason)
	assert.Equal(t, 1, len(plan.BestGPUIds))
	assert.Equal(t, "gpu-4", plan.BestGPUIds[0], "should select orphan GPU from smaller NUMA domain")
}

// TestIntegration_PluginDisabled_ZeroImpact verifies GPUResourcesFit works alone.
func TestIntegration_PluginDisabled_ZeroImpact(t *testing.T) {
	utils.SetProgressiveMigration(false)
	pool := "topo-disabled-pool"
	nodes := []*v1.Node{topoMakeNode("node-0")}
	gpus := []*tfv1.GPU{
		topoMakeGPU("gpu-0", "node-0", topoInt32Ptr(0), pool),
		topoMakeGPU("gpu-1", "node-0", topoInt32Ptr(1), pool),
	}

	require.NoError(t, tfv1.AddToScheme(scheme.Scheme))
	cl := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithRuntimeObjects(&tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{Name: "test-workload", Namespace: "test-ns"},
		}).
		WithStatusSubresource(&tfv1.GPU{}, &tfv1.GPUNode{}, &tfv1.TensorFusionWorkload{}, &v1.Pod{}, &v1.Node{}).
		Build()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	ctx = log.IntoContext(ctx, klog.NewKlogr())

	require.NoError(t, cl.Create(ctx, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "test-ns"}}))
	for _, n := range nodes {
		require.NoError(t, cl.Create(ctx, n.DeepCopy()))
	}
	for _, g := range gpus {
		require.NoError(t, cl.Create(ctx, g.DeepCopy()))
	}

	allocator := gpuallocator.NewGpuAllocator(ctx, nil, cl, time.Second)
	require.NoError(t, allocator.InitGPUAndQuotaStore())
	allocator.ReconcileAllocationState()
	allocator.SetAllocatorReady()

	indexAlloc, err := indexallocator.NewIndexAllocator(ctx, cl)
	require.NoError(t, err)

	k8sObjs := []k8sruntime.Object{nodes[0]}
	fakeClientSet := clientsetfake.NewClientset(k8sObjs...)
	informerFactory := informers.NewSharedInformerFactory(fakeClientSet, 0)
	metrics.Register()
	metricsRecorder := metrics.NewMetricsAsyncRecorder(1000, time.Second, ctx.Done())
	fw, err := tf.NewFramework(ctx,
		[]tf.RegisterPluginFunc{
			tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		}, "",
		frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
		frameworkruntime.WithSnapshotSharedLister(internalcache.NewEmptySnapshot()),
		frameworkruntime.WithEventRecorder(&events.FakeRecorder{}),
		frameworkruntime.WithMetricsRecorder(metricsRecorder),
	)
	require.NoError(t, err)

	fitPlugin, err := gpuResourceFitPlugin.NewWithDeps(allocator, indexAlloc, nil, cl)(
		ctx, &k8sruntime.Unknown{Raw: []byte(`{"maxWorkerPerNode": 256, "vramWeight": 0.7, "tflopsWeight": 0.3}`)}, fw,
	)
	require.NoError(t, err)
	gpuFit := fitPlugin.(*gpuResourceFitPlugin.GPUFit)

	pod := topoMakePod("no-topo-pod", pool, 1)
	state := framework.NewCycleState()

	// Only GPUResourcesFit — no topology plugin
	_, preStatus := gpuFit.PreFilter(ctx, state, pod, nil)
	assert.True(t, preStatus.IsSuccess(), "PreFilter without topo: %v", preStatus)

	// Reserve should use fallback path (no topology state)
	reserveStatus := gpuFit.Reserve(ctx, state, pod, "node-0")
	assert.True(t, reserveStatus.IsSuccess(), "Reserve without topo should succeed: %v", reserveStatus)

	result, err := state.Read(gpuResourceFitPlugin.CycleStateGPUSchedulingResult)
	require.NoError(t, err)
	schedulingResult := result.(*gpuResourceFitPlugin.GPUSchedulingStateData)
	assert.NotEmpty(t, schedulingResult.FinalGPUs, "FinalGPUs should be set via fallback")
	t.Logf("FinalGPUs (no topo plugin): %v", schedulingResult.FinalGPUs)

	// Confirm no topology state exists
	_, topoErr := state.Read(gpuTopoPlugin.CycleStateGPUTopologyResult)
	assert.Error(t, topoErr, "topology state should NOT exist when plugin is disabled")
}
