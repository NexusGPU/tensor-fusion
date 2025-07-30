package gpuresources

import (
	"context"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/log"
	testutil "sigs.k8s.io/scheduler-plugins/test/util"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

type GPUResourcesSuite struct {
	suite.Suite
	client    client.Client
	fwk       framework.Framework
	allocator *gpuallocator.GpuAllocator
	plugin    *GPUFit
	ctx       context.Context
	cancel    context.CancelFunc
}

func (s *GPUResourcesSuite) SetupTest() {
	utils.SetProgressiveMigration(true)
	s.ctx, s.cancel = context.WithCancel(context.Background())
	log.FromContext(s.ctx).Info("Setting up test")
	_ = tfv1.AddToScheme(scheme.Scheme)
	// Initial objects for the fake client
	pods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "worker-1",
				Namespace: "ns1",
				Labels: map[string]string{
					constants.LabelComponent: constants.ComponentWorker,
					constants.WorkloadKey:    "workload-1",
				},
				Annotations: map[string]string{
					constants.TFLOPSRequestAnnotation: "100",
					constants.VRAMRequestAnnotation:   "2Gi",
					constants.TFLOPSLimitAnnotation:   "100",
					constants.VRAMLimitAnnotation:     "4Gi",
					constants.GpuCountAnnotation:      "1",
					constants.GPUDeviceIDsAnnotation:  "gpu-1",
				},
			},
			Spec: v1.PodSpec{
				NodeName: "node-a",
			},
		},
	}
	nodes := []*v1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "node-a",
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "node-b",
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "node-c",
			},
		},
	}
	gpus := []*tfv1.GPU{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "gpu-1",
				Labels: map[string]string{
					constants.GpuPoolKey:    "pool-a",
					constants.LabelKeyOwner: "node-a",
				},
			},
			Status: tfv1.GPUStatus{
				Phase:        tfv1.TensorFusionGPUPhaseRunning,
				NodeSelector: map[string]string{constants.KubernetesHostNameLabel: "node-a"},
				UsedBy:       tfv1.UsedByTensorFusion,
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("1000"),
					Vram:   resource.MustParse("20Gi"),
				},
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("1000"),
					Vram:   resource.MustParse("20Gi"),
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "gpu-2",
				Labels: map[string]string{
					constants.GpuPoolKey:    "pool-a",
					constants.LabelKeyOwner: "node-b",
				},
			},
			Status: tfv1.GPUStatus{
				Phase:        tfv1.TensorFusionGPUPhaseRunning,
				NodeSelector: map[string]string{constants.KubernetesHostNameLabel: "node-b"},
				UsedBy:       tfv1.UsedByTensorFusion,
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("1000"),
					Vram:   resource.MustParse("20Gi"),
				},
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("1000"),
					Vram:   resource.MustParse("20Gi"),
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "gpu-3",
				Labels: map[string]string{
					constants.GpuPoolKey:    "pool-a",
					constants.LabelKeyOwner: "node-b",
				},
			},
			Status: tfv1.GPUStatus{
				Phase:        tfv1.TensorFusionGPUPhaseRunning,
				NodeSelector: map[string]string{constants.KubernetesHostNameLabel: "node-b"},
				UsedBy:       tfv1.UsedByTensorFusion,
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("2000"),
					Vram:   resource.MustParse("40Gi"),
				},
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("2000"),
					Vram:   resource.MustParse("40Gi"),
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "gpu-4",
				Labels: map[string]string{
					constants.GpuPoolKey:    "pool-a",
					constants.LabelKeyOwner: "node-c",
				},
			},
			Status: tfv1.GPUStatus{
				Phase:        tfv1.TensorFusionGPUPhaseRunning,
				NodeSelector: map[string]string{constants.KubernetesHostNameLabel: "node-c"},
				UsedBy:       tfv1.UsedByNvidiaDevicePlugin,
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("2000"),
					Vram:   resource.MustParse("40Gi"),
				},
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("2000"),
					Vram:   resource.MustParse("40Gi"),
				},
			},
		},
	}
	objList := []runtime.Object{
		&tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "workload-1",
				Namespace: "ns1",
			},
		},
		&tfv1.GPUResourceQuota{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "quota-ns1",
				Namespace: "ns1",
			},
			Spec: tfv1.GPUResourceQuotaSpec{
				Total: tfv1.GPUResourceQuotaTotal{
					Requests: &tfv1.Resource{
						Tflops: resource.MustParse("50000"),
						Vram:   resource.MustParse("80Gi"),
					},
					Limits: &tfv1.Resource{
						Tflops: resource.MustParse("50000"),
						Vram:   resource.MustParse("80Gi"),
					},
				},
			},
		},
	}
	s.client = fake.NewClientBuilder().WithScheme(scheme.Scheme).
		WithRuntimeObjects(objList...).
		WithStatusSubresource(
			&tfv1.GPU{},
			&tfv1.GPUNode{},
			&tfv1.GPUResourceQuota{},
			&tfv1.TensorFusionWorkload{},
			&v1.Pod{},
			&v1.Node{},
		).
		Build()

	for _, pod := range pods {
		err := s.client.Create(s.ctx, pod)
		s.NoError(err)
	}
	for _, gpu := range gpus {
		err := s.client.Create(s.ctx, gpu)
		s.NoError(err)
	}
	for _, node := range nodes {
		err := s.client.Create(s.ctx, node)
		s.NoError(err)
	}

	var registerPlugins []tf.RegisterPluginFunc
	registeredPlugins := append(
		registerPlugins,
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
	)

	fwk, err := tf.NewFramework(
		s.ctx, registeredPlugins, "",
		frameworkruntime.WithPodNominator(testutil.NewPodNominator(nil)),
		frameworkruntime.WithSnapshotSharedLister(testutil.NewFakeSharedLister(pods, nodes)),
		frameworkruntime.WithEventRecorder(&events.FakeRecorder{}),
	)
	s.NoError(err)
	s.fwk = fwk

	s.allocator = gpuallocator.NewGpuAllocator(s.ctx, s.client, time.Second)
	err = s.allocator.InitGPUAndQuotaStore()
	s.NoError(err)
	s.allocator.ReconcileAllocationState()
	s.allocator.SetAllocatorReady()

	pluginFactory := NewWithDeps(s.allocator, s.client)
	pluginConfig := &runtime.Unknown{
		Raw: []byte(`{
			"maxWorkerPerNode": 3,
			"vramWeight": 0.7,
			"tflopsWeight": 0.3
		}`),
	}
	p, err := pluginFactory(s.ctx, pluginConfig, s.fwk)
	s.NoError(err)
	s.plugin = p.(*GPUFit)
}

func (s *GPUResourcesSuite) TearDownTest() {
	log.FromContext(s.ctx).Info("Tearing down test")
	s.cancel()
}

func (s *GPUResourcesSuite) TestPreFilter() {
	log.FromContext(s.ctx).Info("Running TestPreFilter")
	tests := []struct {
		name           string
		pod            *v1.Pod
		expectedStatus framework.Code
		expectedNodes  string
	}{
		{
			name: "pod requires 1 GPU, enough capacity",
			pod: s.makePod("p1",
				map[string]string{
					constants.GpuCountAnnotation:      "1",
					constants.TFLOPSRequestAnnotation: "100",
					constants.VRAMRequestAnnotation:   "10Gi",
				}),
			expectedStatus: framework.Success,
			expectedNodes:  "node-a node-b",
		},
		{
			name: "pod requires 1 GPU, enough tflops",
			pod: s.makePod("p2",
				map[string]string{
					constants.GpuCountAnnotation:      "1",
					constants.TFLOPSRequestAnnotation: "2000",
					constants.VRAMRequestAnnotation:   "10Gi",
				}),
			expectedStatus: framework.Success,
			expectedNodes:  "node-b",
		},
		{
			name: "pod requires 2 GPUs should be scheduled on node-b",
			pod: s.makePod("p3",
				map[string]string{
					constants.GpuCountAnnotation:      "2",
					constants.TFLOPSRequestAnnotation: "100",
					constants.VRAMRequestAnnotation:   "10Gi",
				}),
			expectedStatus: framework.Success,
			expectedNodes:  "node-b",
		},
		{
			name: "pod requires 1 GPU, not enough vram",
			pod: s.makePod("p2",
				map[string]string{
					constants.GpuCountAnnotation:      "1",
					constants.TFLOPSRequestAnnotation: "2000",
					constants.VRAMRequestAnnotation:   "80Gi",
				}),
			expectedStatus: framework.Unschedulable,
			expectedNodes:  "",
		},
		{
			name: "pod requires 3 GPUs, but at most 2 on existing nodes",
			pod: s.makePod("p3",
				map[string]string{
					constants.GpuCountAnnotation:      "3",
					constants.TFLOPSRequestAnnotation: "100",
					constants.VRAMRequestAnnotation:   "10Gi",
				}),
			expectedStatus: framework.Unschedulable,
			expectedNodes:  "",
		},
	}

	for _, tt := range tests {
		s.Run(tt.name, func() {
			state := framework.NewCycleState()
			res, status := s.plugin.PreFilter(s.ctx, state, tt.pod)
			s.Equal(tt.expectedStatus, status.Code(), status.Message())
			if tt.expectedStatus == framework.Success {
				s.Require().NotNil(res)
				nodes := sort.StringSlice(res.NodeNames.UnsortedList())
				nodes.Sort()
				s.Equal(tt.expectedNodes, strings.Join(nodes, " "))
			}
		})
	}
}

func (s *GPUResourcesSuite) TestPreFilterForNonTensorFusionPod() {
	log.FromContext(s.ctx).Info("Running TestPreFilterForNonTensorFusionPod")
	tests := []struct {
		name           string
		pod            *v1.Pod
		expectedStatus framework.Code
		expectedNodes  string
	}{
		{
			name:           "pod requires 1 GPU, enough capacity",
			pod:            s.makeNonTensorFusionPod("p1", 1),
			expectedStatus: framework.Success,
			expectedNodes:  "node-b node-c",
		},
		{
			name:           "pod requires 2 GPU, enough capacity",
			pod:            s.makeNonTensorFusionPod("p1", 2),
			expectedStatus: framework.Success,
			expectedNodes:  "node-b node-c",
		},
	}

	for _, tt := range tests {
		s.Run(tt.name, func() {
			state := framework.NewCycleState()
			res, status := s.plugin.PreFilter(s.ctx, state, tt.pod)
			s.Equal(tt.expectedStatus, status.Code(), status.Message())
			if tt.expectedStatus == framework.Success {
				s.Require().NotNil(res)
				nodes := sort.StringSlice(res.NodeNames.UnsortedList())
				nodes.Sort()
				s.Equal(tt.expectedNodes, strings.Join(nodes, " "))
			}
		})
	}
}

func (s *GPUResourcesSuite) TestFilter() {
	log.FromContext(s.ctx).Info("Running TestFilter")
	state := framework.NewCycleState()
	pod := s.makePod("p1",
		map[string]string{
			constants.GpuCountAnnotation:      "1",
			constants.TFLOPSRequestAnnotation: "100",
			constants.VRAMRequestAnnotation:   "10Gi",
			constants.TFLOPSLimitAnnotation:   "100",
			constants.VRAMLimitAnnotation:     "40Gi",
		})
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod)
	s.Require().True(preFilterStatus.IsSuccess())

	tests := []struct {
		name           string
		nodeName       string
		expectedStatus framework.Code
	}{
		{
			name:           "node with available GPU",
			nodeName:       "node-a",
			expectedStatus: framework.Success,
		},
		{
			name:           "node without available GPU",
			nodeName:       "node-c",
			expectedStatus: framework.Unschedulable,
		},
	}

	for _, tt := range tests {
		s.Run(tt.name, func() {
			nodeInfo := &framework.NodeInfo{}
			nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: tt.nodeName}})
			status := s.plugin.Filter(s.ctx, state, pod, nodeInfo)
			s.Equal(tt.expectedStatus, status.Code())
		})
	}
}

func (s *GPUResourcesSuite) TestScore() {
	log.FromContext(s.ctx).Info("Running TestScore")
	state := framework.NewCycleState()
	pod := s.makePod("p1",
		map[string]string{
			constants.GpuCountAnnotation:      "1",
			constants.TFLOPSRequestAnnotation: "100",
			constants.VRAMRequestAnnotation:   "10Gi",
			constants.TFLOPSLimitAnnotation:   "100",
			constants.VRAMLimitAnnotation:     "40Gi",
		})
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod)
	s.Require().True(preFilterStatus.IsSuccess())

	// node a as one worker consumed 10% GPU resources
	// the score should be 100 - 90 = 10
	nodeInfo := &framework.NodeInfo{}
	nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}})
	score, status := s.plugin.Score(s.ctx, state, pod, nodeInfo)
	s.True(status.IsSuccess())
	s.Equal(int64(10), score)

	// node-b has no worker, in compact first mode,
	// it's available resources is 100%, thus score is 100-100 = 0
	nodeInfo = &framework.NodeInfo{}
	nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-b"}})
	score, status = s.plugin.Score(s.ctx, state, pod, nodeInfo)
	s.True(status.IsSuccess())
	s.Zero(score)
}

func (s *GPUResourcesSuite) TestReserveAndUnreserve() {
	log.FromContext(s.ctx).Info("Running TestReserveAndUnreserve")
	state := framework.NewCycleState()
	pod := s.makePod("p1",
		map[string]string{
			constants.GpuCountAnnotation:      "1",
			constants.TFLOPSRequestAnnotation: "100",
			constants.VRAMRequestAnnotation:   "10Gi",
			constants.TFLOPSLimitAnnotation:   "100",
			constants.VRAMLimitAnnotation:     "40Gi",
		})
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod)
	s.Require().True(preFilterStatus.IsSuccess())

	// Reserve on node-a
	reserveStatus := s.plugin.Reserve(s.ctx, state, pod, "node-a")
	s.True(reserveStatus.IsSuccess())

	// Manual trigger a sync loop for dirty GPUs
	s.plugin.allocator.SyncGPUsToK8s()

	// Check allocator state
	gpu := &tfv1.GPU{}
	s.NoError(s.client.Get(s.ctx, types.NamespacedName{Name: "gpu-1"}, gpu))
	// considering existing worker already consumed 100 TFlops, 2GiB VRAM
	s.Equal("800", gpu.Status.Available.Tflops.String())
	s.Equal("8Gi", gpu.Status.Available.Vram.String())
	s.Len(gpu.Status.RunningApps, 1)
	s.Equal("workload-1", gpu.Status.RunningApps[0].Name)
	s.Equal(2, gpu.Status.RunningApps[0].Count)

	s.plugin.Unreserve(s.ctx, state, pod, "node-a")
	s.plugin.allocator.SyncGPUsToK8s()

	// Check allocator state again
	s.NoError(s.client.Get(s.ctx, types.NamespacedName{Name: "gpu-1"}, gpu))
	s.Equal("900", gpu.Status.Available.Tflops.String())
	s.Equal("18Gi", gpu.Status.Available.Vram.String())
	s.Len(gpu.Status.RunningApps, 1)
}

func (s *GPUResourcesSuite) TestPostBind() {
	log.FromContext(s.ctx).Info("Running TestPostBind")
	state := framework.NewCycleState()
	pod := s.makePod("p1",
		map[string]string{
			constants.GpuCountAnnotation:      "1",
			constants.TFLOPSRequestAnnotation: "100",
			constants.VRAMRequestAnnotation:   "10Gi",
			constants.TFLOPSLimitAnnotation:   "100",
			constants.VRAMLimitAnnotation:     "40Gi",
		})
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod)
	s.Require().True(preFilterStatus.IsSuccess())

	reserveStatus := s.plugin.Reserve(s.ctx, state, pod, "node-a")
	s.Require().True(reserveStatus.IsSuccess())

	s.plugin.PostBind(s.ctx, state, pod, "node-a")

	updatedPod := &v1.Pod{}
	s.NoError(s.client.Get(s.ctx, types.NamespacedName{Name: "p1", Namespace: "ns1"}, updatedPod))
	s.Equal("gpu-1", updatedPod.Annotations[constants.GPUDeviceIDsAnnotation])
}

func TestGPUResourcesSuite(t *testing.T) {
	log.FromContext(context.Background()).Info("Running GPUResourcesSuite")
	suite.Run(t, new(GPUResourcesSuite))
}

func (s *GPUResourcesSuite) makeNonTensorFusionPod(name string, gpuCount int) *v1.Pod {
	log.FromContext(s.ctx).Info("Making pod", "name", name)
	pod := st.MakePod().
		Namespace("ns1").
		Name(name).
		UID(name).
		ZeroTerminationGracePeriod().Obj()
	pod.Spec.Containers = []v1.Container{
		{
			Name: "container-1",
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName("nvidia.com/gpu"): *resource.NewQuantity(int64(gpuCount), resource.DecimalSI),
				},
			},
		},
	}
	return pod
}

func (s *GPUResourcesSuite) makePod(name string, annotations map[string]string) *v1.Pod {
	log.FromContext(s.ctx).Info("Making pod", "name", name)
	pod := st.MakePod().
		Namespace("ns1").
		Name(name).
		UID(name).
		ZeroTerminationGracePeriod().Obj()
	pod.Labels = map[string]string{
		constants.LabelComponent: constants.ComponentWorker,
		constants.WorkloadKey:    "workload-1",
	}
	pod.Annotations = annotations
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[constants.GpuPoolKey] = "pool-a"
	if annotations[constants.TFLOPSLimitAnnotation] == "" {
		pod.Annotations[constants.TFLOPSLimitAnnotation] = pod.Annotations[constants.TFLOPSRequestAnnotation]
	}
	if annotations[constants.VRAMLimitAnnotation] == "" {
		pod.Annotations[constants.VRAMLimitAnnotation] = pod.Annotations[constants.VRAMRequestAnnotation]
	}
	if annotations[constants.GpuCountAnnotation] == "" {
		pod.Annotations[constants.GpuCountAnnotation] = "1"
	}

	existingPod := &v1.Pod{}
	if err := s.client.Get(s.ctx, client.ObjectKey{Name: name, Namespace: "ns1"}, existingPod); err != nil {
		if errors.IsNotFound(err) {
			s.NoError(s.client.Create(s.ctx, pod))
		}
	}
	return pod
}

func (s *GPUResourcesSuite) TestNewWithDeps() {
	log.FromContext(s.ctx).Info("Running TestNewWithDeps")
	pluginFactory := NewWithDeps(s.allocator, s.client)
	s.NotNil(pluginFactory)

	// Test with valid config
	pluginConfig := &runtime.Unknown{
		Raw: []byte(`{"maxWorkerPerNode": 10}`),
	}
	p, err := pluginFactory(s.ctx, pluginConfig, s.fwk)
	s.NoError(err)
	s.NotNil(p)
	s.Equal(Name, p.Name())

	// Test with invalid config
	invalidPluginConfig := &runtime.Unknown{
		Raw: []byte(`{"maxWorkerPerNode": "invalid"}`),
	}
	_, err = pluginFactory(s.ctx, invalidPluginConfig, s.fwk)
	s.Error(err)
}

func (s *GPUResourcesSuite) TestGPUSchedulingStateData_Clone() {
	log.FromContext(s.ctx).Info("Running TestGPUSchedulingStateData_Clone")
	state := &GPUSchedulingStateData{
		NodeGPUs: map[string][]tfv1.GPU{
			"node-a": {{ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"}}},
		},
		ValidNodeGPUScore: map[string]map[string]int{
			"node-a": {"gpu-1": 100},
		},
		FinalGPUs: []string{"gpu-1"},
	}
	cloned := state.Clone()
	s.Equal(state, cloned)
	// Should be a shallow copy
	s.Equal(&state.NodeGPUs, &cloned.(*GPUSchedulingStateData).NodeGPUs)
}

func (s *GPUResourcesSuite) TestScoreExtensions() {
	log.FromContext(s.ctx).Info("Running TestScoreExtensions")
	s.Nil(s.plugin.ScoreExtensions())
}

func (s *GPUResourcesSuite) TestPreFilterExtensions() {
	log.FromContext(s.ctx).Info("Running TestPreFilterExtensions")
	s.Nil(s.plugin.PreFilterExtensions())
}

func (s *GPUResourcesSuite) TestName() {
	log.FromContext(s.ctx).Info("Running TestName")
	s.Equal(Name, s.plugin.Name())
}

func (s *GPUResourcesSuite) TestReserve_ErrorHandling() {
	state := framework.NewCycleState()
	pod := s.makePod("p1",
		map[string]string{
			constants.GpuCountAnnotation:      "1",
			constants.TFLOPSRequestAnnotation: "100",
			constants.VRAMRequestAnnotation:   "10Gi",
		})

	// No pre-filter call, so state is empty
	status := s.plugin.Reserve(s.ctx, state, pod, "node-a")
	s.Error(status.AsError())
	s.Equal(framework.Error, status.Code())

	// Pre-filter, but for a different node
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod)
	s.Require().True(preFilterStatus.IsSuccess())
	status = s.plugin.Reserve(s.ctx, state, pod, "node-c-non-existent")
	s.Equal(framework.Unschedulable, status.Code())
}

func (s *GPUResourcesSuite) TestUnreserve_ErrorHandling() {
	log.FromContext(s.ctx).Info("Running TestUnreserve_ErrorHandling")
	state := framework.NewCycleState()
	pod := s.makePod("p1",
		map[string]string{
			constants.GpuCountAnnotation:      "1",
			constants.TFLOPSRequestAnnotation: "100",
			constants.VRAMRequestAnnotation:   "10Gi",
		})

	// No pre-filter call, so state is empty. Unreserve should not panic.
	s.NotPanics(func() {
		s.plugin.Unreserve(s.ctx, state, pod, "node-a")
	})
}

func (s *GPUResourcesSuite) TestPostBind_ErrorHandling() {
	log.FromContext(s.ctx).Info("Running TestPostBind_ErrorHandling")
	state := framework.NewCycleState()
	pod := s.makePod("p1",
		map[string]string{
			constants.GpuCountAnnotation:      "1",
			constants.TFLOPSRequestAnnotation: "100",
			constants.VRAMRequestAnnotation:   "10Gi",
		})

	// No pre-filter call, so state is empty
	s.plugin.PostBind(s.ctx, state, pod, "node-a")

	// Test with a pod that doesn't exist in the client
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod)
	s.Require().True(preFilterStatus.IsSuccess())
	reserveStatus := s.plugin.Reserve(s.ctx, state, pod, "node-a")
	s.Require().True(reserveStatus.IsSuccess())

	nonExistentPod := pod.DeepCopy()
	nonExistentPod.Name = "p-non-existent"
	s.plugin.PostBind(s.ctx, state, nonExistentPod, "node-a")
}

func (s *GPUResourcesSuite) TestFilter_ErrorHandling() {
	log.FromContext(s.ctx).Info("Running TestFilter_ErrorHandling")
	state := framework.NewCycleState()
	pod := s.makePod("p1", nil)
	nodeInfo := &framework.NodeInfo{}
	nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}})

	// No pre-filter call, so state is empty
	status := s.plugin.Filter(s.ctx, state, pod, nodeInfo)
	s.Error(status.AsError())
	s.Equal(framework.Error, status.Code())
}

func (s *GPUResourcesSuite) TestScore_ErrorHandling() {
	log.FromContext(s.ctx).Info("Running TestScore_ErrorHandling")
	state := framework.NewCycleState()
	pod := s.makePod("p1", map[string]string{
		constants.TFLOPSRequestAnnotation: "100",
		constants.VRAMRequestAnnotation:   "10Gi",
	})

	// No pre-filter call, so state is empty
	nodeInfo := &framework.NodeInfo{}
	nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}})
	_, status := s.plugin.Score(s.ctx, state, pod, nodeInfo)
	s.Error(status.AsError())
	s.Equal(framework.Error, status.Code())

	// Pre-filter, but for a different node
	nodeInfo = &framework.NodeInfo{}
	nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-c-non-existent"}})
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod)
	s.Require().True(preFilterStatus.IsSuccess())
	_, status = s.plugin.Score(s.ctx, state, pod, nodeInfo)
	s.Equal(framework.Unschedulable, status.Code())
}
