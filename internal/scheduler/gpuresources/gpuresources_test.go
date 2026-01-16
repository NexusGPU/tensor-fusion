package gpuresources

import (
	"context"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/suite"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	fwk "k8s.io/kube-scheduler/framework"
	framework "k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/log"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/indexallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
)

type GPUResourcesSuite struct {
	suite.Suite
	client         client.Client
	fwk            framework.Framework
	allocator      *gpuallocator.GpuAllocator
	indexAllocator *indexallocator.IndexAllocator
	plugin         *GPUFit
	ctx            context.Context
	cancel         context.CancelFunc
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
				UsedBy:       "nvidia-device-plugin",
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

	k8sObjs := make([]runtime.Object, 0, len(pods)+len(nodes))
	for _, pod := range pods {
		err := s.client.Create(s.ctx, pod)
		s.NoError(err)
		k8sObjs = append(k8sObjs, pod)
	}
	for _, gpu := range gpus {
		err := s.client.Create(s.ctx, gpu)
		s.NoError(err)
	}
	for _, node := range nodes {
		err := s.client.Create(s.ctx, node)
		s.NoError(err)
		k8sObjs = append(k8sObjs, node)
	}

	var registerPlugins []tf.RegisterPluginFunc
	registeredPlugins := append(
		registerPlugins,
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
	)

	fakeClientSet := clientsetfake.NewSimpleClientset(k8sObjs...)
	informerFactory := informers.NewSharedInformerFactory(fakeClientSet, 0)
	metrics.Register()
	metricsRecorder := metrics.NewMetricsAsyncRecorder(1000, time.Second, s.ctx.Done())
	fwk, err := tf.NewFramework(
		s.ctx, registeredPlugins, "",
		frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
		frameworkruntime.WithSnapshotSharedLister(internalcache.NewEmptySnapshot()),
		frameworkruntime.WithEventRecorder(&events.FakeRecorder{}),
		frameworkruntime.WithMetricsRecorder(metricsRecorder),
	)
	s.NoError(err)
	s.fwk = fwk

	s.allocator = gpuallocator.NewGpuAllocator(s.ctx, nil, s.client, time.Second)
	err = s.allocator.InitGPUAndQuotaStore()
	s.NoError(err)
	s.allocator.ReconcileAllocationState()
	s.allocator.SetAllocatorReady()

	s.indexAllocator, err = indexallocator.NewIndexAllocator(s.ctx, s.client)
	s.NoError(err)
	s.indexAllocator.IsLeader = true
	s.indexAllocator.SetReady()

	pluginFactory := NewWithDeps(s.allocator, s.indexAllocator, s.client)
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
		expectedStatus fwk.Code
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
			expectedStatus: fwk.Success,
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
			expectedStatus: fwk.Success,
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
			expectedStatus: fwk.Success,
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
			expectedStatus: fwk.Unschedulable,
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
			expectedStatus: fwk.Unschedulable,
			expectedNodes:  "",
		},
	}

	for _, tt := range tests {
		s.Run(tt.name, func() {
			state := framework.NewCycleState()
			res, status := s.plugin.PreFilter(s.ctx, state, tt.pod, []fwk.NodeInfo{})
			s.Equal(tt.expectedStatus, status.Code(), status.Message())
			if tt.expectedStatus == fwk.Success {
				s.Require().NotNil(res)
				nodes := sort.StringSlice(getPreFilterResult(state))
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
		expectedStatus fwk.Code
		expectedNodes  string
	}{
		{
			name:           "pod requires 1 GPU, enough capacity",
			pod:            s.makeNonTensorFusionPod("p1", 1),
			expectedStatus: fwk.Success,
			expectedNodes:  "node-b node-c",
		},
		{
			name:           "pod requires 2 GPU, enough capacity",
			pod:            s.makeNonTensorFusionPod("p1", 2),
			expectedStatus: fwk.Success,
			expectedNodes:  "node-b node-c",
		},
	}

	for _, tt := range tests {
		s.Run(tt.name, func() {
			state := framework.NewCycleState()
			res, status := s.plugin.PreFilter(s.ctx, state, tt.pod, []fwk.NodeInfo{})
			s.Equal(tt.expectedStatus, status.Code(), status.Message())
			if tt.expectedStatus == fwk.Success {
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
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod, []fwk.NodeInfo{})
	s.Require().True(preFilterStatus.IsSuccess())

	tests := []struct {
		name           string
		nodeName       string
		expectedStatus fwk.Code
	}{
		{
			name:           "node with available GPU",
			nodeName:       "node-a",
			expectedStatus: fwk.Success,
		},
		{
			name:           "node without available GPU",
			nodeName:       "node-c",
			expectedStatus: fwk.Unschedulable,
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
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod, []fwk.NodeInfo{})
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
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod, []fwk.NodeInfo{})
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
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod, []fwk.NodeInfo{})
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
	// Add container with index resource limits for ParsePodIndexResourceClaim
	// The index resource key format is: tensor-fusion.ai/index_<hex-key> with value <mod-value>
	// This simulates what the mutating webhook sets up
	indexResourceKey := v1.ResourceName(constants.PodIndexAnnotation + constants.PodIndexDelimiter + "0")
	pod.Spec.Containers = []v1.Container{
		{
			Name:  "worker",
			Image: "test-image",
			Resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					indexResourceKey: *resource.NewQuantity(1, resource.DecimalSI), // index 1
				},
			},
		},
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
	pluginFactory := NewWithDeps(s.allocator, s.indexAllocator, s.client)
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

func (s *GPUResourcesSuite) TestScoreExtensions() {
	log.FromContext(s.ctx).Info("Running TestScoreExtensions")
	s.Nil(s.plugin.ScoreExtensions())
}

func (s *GPUResourcesSuite) TestPreFilterExtensions() {
	log.FromContext(s.ctx).Info("Running TestPreFilterExtensions")
	s.NotNil(s.plugin.PreFilterExtensions())
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
	s.Equal(fwk.Error, status.Code())

	// Pre-filter, but for a different node
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod, []fwk.NodeInfo{})
	s.Require().True(preFilterStatus.IsSuccess())
	status = s.plugin.Reserve(s.ctx, state, pod, "node-c-non-existent")
	s.Equal(fwk.Unschedulable, status.Code())
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
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod, []fwk.NodeInfo{})
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
	s.Equal(fwk.Error, status.Code())
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
	s.Equal(fwk.Error, status.Code())

	// Pre-filter, but for a different node
	nodeInfo = &framework.NodeInfo{}
	nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-c-non-existent"}})
	_, preFilterStatus := s.plugin.PreFilter(s.ctx, state, pod, []fwk.NodeInfo{})
	s.Require().True(preFilterStatus.IsSuccess())
	_, status = s.plugin.Score(s.ctx, state, pod, nodeInfo)
	s.Equal(fwk.Unschedulable, status.Code())
}

func getPreFilterResult(state *framework.CycleState) []string {
	data, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		return nil
	}
	return lo.Keys(data.(*GPUSchedulingStateData).NodeGPUs)
}

func (s *GPUResourcesSuite) TestEventsToRegister() {
	log.FromContext(s.ctx).Info("Running TestEventsToRegister")
	events, err := s.plugin.EventsToRegister(s.ctx)
	s.NoError(err)
	s.Len(events, 1)
	s.Equal(fwk.EventResource("gpus.v1.tensor-fusion.ai"), events[0].Event.Resource)
	s.Equal(fwk.Add|fwk.Update, events[0].Event.ActionType)
	s.NotNil(events[0].QueueingHintFn)
}

func (s *GPUResourcesSuite) TestQueueingHint_ResourceDecrease() {
	log.FromContext(s.ctx).Info("Running TestQueueingHint_ResourceDecrease")
	pod := s.makePod("test-pod", map[string]string{
		constants.TFLOPSRequestAnnotation: "100",
		constants.VRAMRequestAnnotation:   "10Gi",
		constants.GpuCountAnnotation:      "1",
	})

	oldGPU := &tfv1.GPU{
		Status: tfv1.GPUStatus{
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("500"),
				Vram:   resource.MustParse("10Gi"),
			},
		},
	}
	newGPU := &tfv1.GPU{
		Status: tfv1.GPUStatus{
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("400"),
				Vram:   resource.MustParse("8Gi"),
			},
		},
	}

	hint, err := s.plugin.queueingHint(*s.plugin.logger, pod, oldGPU, newGPU)
	s.NoError(err)
	s.Equal(fwk.QueueSkip, hint)
}

func (s *GPUResourcesSuite) TestQueueingHint_ResourceIncrease_Sufficient() {
	log.FromContext(s.ctx).Info("Running TestQueueingHint_ResourceIncrease_Sufficient")

	// Create a pending pod
	pendingPod := s.makePod("pending-pod", map[string]string{
		constants.TFLOPSRequestAnnotation: "100",
		constants.VRAMRequestAnnotation:   "10Gi",
		constants.GpuCountAnnotation:      "1",
		constants.GpuPoolKey:              "pool-a",
	})
	pendingPod.Spec.NodeName = "" // Make it pending

	oldGPU := &tfv1.GPU{
		Status: tfv1.GPUStatus{
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("50"),
				Vram:   resource.MustParse("5Gi"),
			},
		},
	}
	newGPU := &tfv1.GPU{
		Status: tfv1.GPUStatus{
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("150"),  // Increase by 100
				Vram:   resource.MustParse("15Gi"), // Increase by 10Gi
			},
		},
	}

	hint, err := s.plugin.queueingHint(*s.plugin.logger, pendingPod, oldGPU, newGPU)
	s.NoError(err)
	s.Equal(fwk.Queue, hint)
}

func (s *GPUResourcesSuite) TestQueueingHint_NonTensorFusionPod() {
	log.FromContext(s.ctx).Info("Running TestQueueingHint_NonTensorFusionPod")
	nonTFPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "non-tf-pod",
			Namespace: "default",
		},
	}

	oldGPU := &tfv1.GPU{
		Status: tfv1.GPUStatus{
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("50"),
				Vram:   resource.MustParse("5Gi"),
			},
		},
	}
	newGPU := &tfv1.GPU{
		Status: tfv1.GPUStatus{
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("150"),
				Vram:   resource.MustParse("15Gi"),
			},
		},
	}

	hint, err := s.plugin.queueingHint(*s.plugin.logger, nonTFPod, oldGPU, newGPU)
	s.NoError(err)
	s.Equal(fwk.QueueSkip, hint)
}

func (s *GPUResourcesSuite) TestQueueingHint_GPUAdd() {
	log.FromContext(s.ctx).Info("Running TestQueueingHint_GPUAdd")

	// Create a pending pod
	pendingPod := s.makePod("pending-pod-2", map[string]string{
		constants.TFLOPSRequestAnnotation: "50",
		constants.VRAMRequestAnnotation:   "5Gi",
		constants.GpuCountAnnotation:      "1",
		constants.GpuPoolKey:              "pool-a",
	})
	pendingPod.Spec.NodeName = ""

	// New GPU added with available resources
	newGPU := &tfv1.GPU{
		Status: tfv1.GPUStatus{
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("100"),  // More than pod request (50)
				Vram:   resource.MustParse("10Gi"), // More than pod request (5Gi)
			},
		},
	}

	hint, err := s.plugin.queueingHint(*s.plugin.logger, pendingPod, nil, newGPU)
	s.NoError(err)
	s.Equal(fwk.Queue, hint)
}

func (s *GPUResourcesSuite) TestReconcileAllocationState_ComputePercent() {
	log.FromContext(s.ctx).Info("Running TestReconcileAllocationState_ComputePercent")

	// Create a GPU with capacity 1000 TFLOPs
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-test-percent",
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
	}
	s.NoError(s.client.Create(s.ctx, gpu))

	// Create a worker pod using compute-percent-request (10% = 100 TFLOPs)
	workerPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-percent",
			Namespace: "ns1",
			UID:       "worker-percent-uid",
			Labels: map[string]string{
				constants.LabelComponent: constants.ComponentWorker,
				constants.WorkloadKey:    "workload-percent",
			},
			Annotations: map[string]string{
				constants.ComputeRequestAnnotation: "10", // 10% of 1000 = 100 TFLOPs
				constants.VRAMRequestAnnotation:    "2Gi",
				constants.ComputeLimitAnnotation:   "10",
				constants.VRAMLimitAnnotation:      "4Gi",
				constants.GpuCountAnnotation:       "1",
				constants.GPUDeviceIDsAnnotation:   "gpu-test-percent",
				constants.GpuPoolKey:               "pool-a",
			},
		},
		Spec: v1.PodSpec{
			NodeName: "node-a",
		},
	}
	s.NoError(s.client.Create(s.ctx, workerPod))

	// Manually add GPU to allocator store
	// Since InitGPUAndQuotaStore uses sync.Once, we need to manually add the GPU to the store
	gpuStore, _, _ := s.allocator.GetAllocationInfo()
	key := types.NamespacedName{Name: "gpu-test-percent"}
	gpuCopy := gpu.DeepCopy()
	if gpuCopy.Status.Available == nil {
		gpuCopy.Status.Available = gpuCopy.Status.Capacity.DeepCopy()
	}
	gpuStore[key] = gpuCopy

	// Reconcile allocation state
	s.allocator.ReconcileAllocationStateForTesting()
	s.allocator.SyncGPUsToK8s()

	// Verify GPU available resources are correctly deducted
	updatedGPU := &tfv1.GPU{}
	s.NoError(s.client.Get(s.ctx, types.NamespacedName{Name: "gpu-test-percent"}, updatedGPU))

	// Expected: 1000 - 100 (10% of 1000) = 900 TFLOPs
	expectedTflops := resource.MustParse("900")
	s.True(updatedGPU.Status.Available.Tflops.Equal(expectedTflops),
		"Expected TFLOPs: %s, Got: %s", expectedTflops.String(), updatedGPU.Status.Available.Tflops.String())

	// Expected: 20Gi - 2Gi = 18Gi VRAM
	expectedVram := resource.MustParse("18Gi")
	s.True(updatedGPU.Status.Available.Vram.Equal(expectedVram),
		"Expected VRAM: %s, Got: %s", expectedVram.String(), updatedGPU.Status.Available.Vram.String())

	// Verify running apps
	s.Len(updatedGPU.Status.RunningApps, 1)
	s.Equal("workload-percent", updatedGPU.Status.RunningApps[0].Name)
}

func (s *GPUResourcesSuite) TestReconcileAllocationState_MixedTflopsAndComputePercent() {
	log.FromContext(s.ctx).Info("Running TestReconcileAllocationState_MixedTflopsAndComputePercent")

	// Create a GPU with capacity 2000 TFLOPs
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-test-mixed",
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
				Tflops: resource.MustParse("2000"),
				Vram:   resource.MustParse("40Gi"),
			},
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("2000"),
				Vram:   resource.MustParse("40Gi"),
			},
		},
	}
	s.NoError(s.client.Create(s.ctx, gpu))

	// Create first worker pod using TFLOPs directly (300 TFLOPs)
	workerPod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-tflops",
			Namespace: "ns1",
			UID:       "worker-tflops-uid",
			Labels: map[string]string{
				constants.LabelComponent: constants.ComponentWorker,
				constants.WorkloadKey:    "workload-tflops",
			},
			Annotations: map[string]string{
				constants.TFLOPSRequestAnnotation: "300",
				constants.VRAMRequestAnnotation:   "5Gi",
				constants.TFLOPSLimitAnnotation:   "300",
				constants.VRAMLimitAnnotation:     "10Gi",
				constants.GpuCountAnnotation:      "1",
				constants.GPUDeviceIDsAnnotation:  "gpu-test-mixed",
				constants.GpuPoolKey:              "pool-a",
			},
		},
		Spec: v1.PodSpec{
			NodeName: "node-a",
		},
	}
	s.NoError(s.client.Create(s.ctx, workerPod1))

	// Create second worker pod using compute-percent-request (15% = 300 TFLOPs)
	workerPod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-percent-mixed",
			Namespace: "ns1",
			UID:       "worker-percent-mixed-uid",
			Labels: map[string]string{
				constants.LabelComponent: constants.ComponentWorker,
				constants.WorkloadKey:    "workload-percent-mixed",
			},
			Annotations: map[string]string{
				constants.ComputeRequestAnnotation: "15", // 15% of 2000 = 300 TFLOPs
				constants.VRAMRequestAnnotation:    "3Gi",
				constants.ComputeLimitAnnotation:   "15",
				constants.VRAMLimitAnnotation:      "6Gi",
				constants.GpuCountAnnotation:       "1",
				constants.GPUDeviceIDsAnnotation:   "gpu-test-mixed",
				constants.GpuPoolKey:               "pool-a",
			},
		},
		Spec: v1.PodSpec{
			NodeName: "node-a",
		},
	}
	s.NoError(s.client.Create(s.ctx, workerPod2))

	// Manually add GPU to allocator store
	gpuStore, _, _ := s.allocator.GetAllocationInfo()
	key := types.NamespacedName{Name: "gpu-test-mixed"}
	gpuCopy := gpu.DeepCopy()
	if gpuCopy.Status.Available == nil {
		gpuCopy.Status.Available = gpuCopy.Status.Capacity.DeepCopy()
	}
	gpuStore[key] = gpuCopy

	// Reconcile allocation state
	s.allocator.ReconcileAllocationStateForTesting()
	s.allocator.SyncGPUsToK8s()

	// Verify GPU available resources are correctly deducted
	updatedGPU := &tfv1.GPU{}
	s.NoError(s.client.Get(s.ctx, types.NamespacedName{Name: "gpu-test-mixed"}, updatedGPU))

	// Expected: 2000 - 300 (direct TFLOPs) - 300 (15% of 2000) = 1400 TFLOPs
	expectedTflops := resource.MustParse("1400")
	s.True(updatedGPU.Status.Available.Tflops.Equal(expectedTflops),
		"Expected TFLOPs: %s, Got: %s", expectedTflops.String(), updatedGPU.Status.Available.Tflops.String())

	// Expected: 40Gi - 5Gi - 3Gi = 32Gi VRAM
	expectedVram := resource.MustParse("32Gi")
	s.True(updatedGPU.Status.Available.Vram.Equal(expectedVram),
		"Expected VRAM: %s, Got: %s", expectedVram.String(), updatedGPU.Status.Available.Vram.String())

	// Verify running apps - should have both workloads
	s.Len(updatedGPU.Status.RunningApps, 2)

	// Check both workloads are present
	workloadNames := []string{
		updatedGPU.Status.RunningApps[0].Name,
		updatedGPU.Status.RunningApps[1].Name,
	}
	s.Contains(workloadNames, "workload-tflops")
	s.Contains(workloadNames, "workload-percent-mixed")
}

func (s *GPUResourcesSuite) TestReconcileAllocationState_MultipleGPUsWithComputePercent() {
	log.FromContext(s.ctx).Info("Running TestReconcileAllocationState_MultipleGPUsWithComputePercent")

	// Create two GPUs with different capacities
	gpu1 := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-mixed-1",
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
	}
	s.NoError(s.client.Create(s.ctx, gpu1))

	gpu2 := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-mixed-2",
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
				Tflops: resource.MustParse("2000"),
				Vram:   resource.MustParse("40Gi"),
			},
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("2000"),
				Vram:   resource.MustParse("40Gi"),
			},
		},
	}
	s.NoError(s.client.Create(s.ctx, gpu2))

	// Create worker pod 1 using compute-percent on GPU 1 (20% of 1000 = 200 TFLOPs)
	workerPod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-mixed-1",
			Namespace: "ns1",
			UID:       "worker-mixed-1-uid",
			Labels: map[string]string{
				constants.LabelComponent: constants.ComponentWorker,
				constants.WorkloadKey:    "workload-mixed-1",
			},
			Annotations: map[string]string{
				constants.ComputeRequestAnnotation: "20", // 20% of 1000 = 200 TFLOPs
				constants.VRAMRequestAnnotation:    "4Gi",
				constants.ComputeLimitAnnotation:   "20",
				constants.VRAMLimitAnnotation:      "8Gi",
				constants.GpuCountAnnotation:       "1",
				constants.GPUDeviceIDsAnnotation:   "gpu-mixed-1",
				constants.GpuPoolKey:               "pool-a",
			},
		},
		Spec: v1.PodSpec{
			NodeName: "node-a",
		},
	}
	s.NoError(s.client.Create(s.ctx, workerPod1))

	// Create worker pod 2 using compute-percent on GPU 2 (10% of 2000 = 200 TFLOPs)
	workerPod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-mixed-2",
			Namespace: "ns1",
			UID:       "worker-mixed-2-uid",
			Labels: map[string]string{
				constants.LabelComponent: constants.ComponentWorker,
				constants.WorkloadKey:    "workload-mixed-2",
			},
			Annotations: map[string]string{
				constants.ComputeRequestAnnotation: "10", // 10% of 2000 = 200 TFLOPs
				constants.VRAMRequestAnnotation:    "6Gi",
				constants.ComputeLimitAnnotation:   "10",
				constants.VRAMLimitAnnotation:      "12Gi",
				constants.GpuCountAnnotation:       "1",
				constants.GPUDeviceIDsAnnotation:   "gpu-mixed-2",
				constants.GpuPoolKey:               "pool-a",
			},
		},
		Spec: v1.PodSpec{
			NodeName: "node-a",
		},
	}
	s.NoError(s.client.Create(s.ctx, workerPod2))

	// Manually add GPUs to allocator store
	gpuStore, _, _ := s.allocator.GetAllocationInfo()
	key1 := types.NamespacedName{Name: "gpu-mixed-1"}
	gpuCopy1 := gpu1.DeepCopy()
	if gpuCopy1.Status.Available == nil {
		gpuCopy1.Status.Available = gpuCopy1.Status.Capacity.DeepCopy()
	}
	gpuStore[key1] = gpuCopy1

	key2 := types.NamespacedName{Name: "gpu-mixed-2"}
	gpuCopy2 := gpu2.DeepCopy()
	if gpuCopy2.Status.Available == nil {
		gpuCopy2.Status.Available = gpuCopy2.Status.Capacity.DeepCopy()
	}
	gpuStore[key2] = gpuCopy2

	// Reconcile allocation state
	s.allocator.ReconcileAllocationStateForTesting()
	s.allocator.SyncGPUsToK8s()

	// Verify GPU 1 available resources
	updatedGPU1 := &tfv1.GPU{}
	s.NoError(s.client.Get(s.ctx, types.NamespacedName{Name: "gpu-mixed-1"}, updatedGPU1))

	// Expected: 1000 - 200 (20% of 1000) = 800 TFLOPs
	expectedTflops1 := resource.MustParse("800")
	s.True(updatedGPU1.Status.Available.Tflops.Equal(expectedTflops1),
		"GPU1 Expected TFLOPs: %s, Got: %s", expectedTflops1.String(), updatedGPU1.Status.Available.Tflops.String())

	// Expected: 20Gi - 4Gi = 16Gi VRAM
	expectedVram1 := resource.MustParse("16Gi")
	s.True(updatedGPU1.Status.Available.Vram.Equal(expectedVram1),
		"GPU1 Expected VRAM: %s, Got: %s", expectedVram1.String(), updatedGPU1.Status.Available.Vram.String())

	// Verify GPU 2 available resources
	updatedGPU2 := &tfv1.GPU{}
	s.NoError(s.client.Get(s.ctx, types.NamespacedName{Name: "gpu-mixed-2"}, updatedGPU2))

	// Expected: 2000 - 200 (10% of 2000) = 1800 TFLOPs
	expectedTflops2 := resource.MustParse("1800")
	s.True(updatedGPU2.Status.Available.Tflops.Equal(expectedTflops2),
		"GPU2 Expected TFLOPs: %s, Got: %s", expectedTflops2.String(), updatedGPU2.Status.Available.Tflops.String())

	// Expected: 40Gi - 6Gi = 34Gi VRAM
	expectedVram2 := resource.MustParse("34Gi")
	s.True(updatedGPU2.Status.Available.Vram.Equal(expectedVram2),
		"GPU2 Expected VRAM: %s, Got: %s", expectedVram2.String(), updatedGPU2.Status.Available.Vram.String())
}
