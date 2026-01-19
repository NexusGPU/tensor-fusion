package gpuresources

import (
	"context"
	"sort"
	"strings"
	"time"

	"github.com/samber/lo"
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

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/indexallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
)

var _ = Describe("GPUFit Plugin", func() {
	var (
		k8sClient      client.Client
		fwkInstance    framework.Framework
		allocator      *gpuallocator.GpuAllocator
		indexAllocator *indexallocator.IndexAllocator
		plugin         *GPUFit
		ctx            context.Context
		cancel         context.CancelFunc
	)

	// Helper functions
	makeNonTensorFusionPod := func(name string, gpuCount int) *v1.Pod {
		log.FromContext(ctx).Info("Making pod", "name", name)
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

	makePod := func(name string, annotations map[string]string) *v1.Pod {
		log.FromContext(ctx).Info("Making pod", "name", name)
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
		indexResourceKey := v1.ResourceName(constants.PodIndexAnnotation + constants.PodIndexDelimiter + "0")
		pod.Spec.Containers = []v1.Container{
			{
				Name:  "worker",
				Image: "test-image",
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						indexResourceKey: *resource.NewQuantity(1, resource.DecimalSI),
					},
				},
			},
		}

		existingPod := &v1.Pod{}
		if err := k8sClient.Get(ctx, client.ObjectKey{Name: name, Namespace: "ns1"}, existingPod); err != nil {
			if errors.IsNotFound(err) {
				Expect(k8sClient.Create(ctx, pod)).To(Succeed())
			}
		}
		return pod
	}

	BeforeEach(func() {
		utils.SetProgressiveMigration(true)
		ctx, cancel = context.WithCancel(context.Background())
		log.FromContext(ctx).Info("Setting up test")
		Expect(tfv1.AddToScheme(scheme.Scheme)).To(Succeed())

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
			{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "node-b"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "node-c"}},
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
					Capacity:     &tfv1.Resource{Tflops: resource.MustParse("1000"), Vram: resource.MustParse("20Gi")},
					Available:    &tfv1.Resource{Tflops: resource.MustParse("1000"), Vram: resource.MustParse("20Gi")},
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
					Capacity:     &tfv1.Resource{Tflops: resource.MustParse("1000"), Vram: resource.MustParse("20Gi")},
					Available:    &tfv1.Resource{Tflops: resource.MustParse("1000"), Vram: resource.MustParse("20Gi")},
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
					Capacity:     &tfv1.Resource{Tflops: resource.MustParse("2000"), Vram: resource.MustParse("40Gi")},
					Available:    &tfv1.Resource{Tflops: resource.MustParse("2000"), Vram: resource.MustParse("40Gi")},
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
					Capacity:     &tfv1.Resource{Tflops: resource.MustParse("2000"), Vram: resource.MustParse("40Gi")},
					Available:    &tfv1.Resource{Tflops: resource.MustParse("2000"), Vram: resource.MustParse("40Gi")},
				},
			},
		}
		objList := []runtime.Object{
			&tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{Name: "workload-1", Namespace: "ns1"},
			},
			&tfv1.GPUResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota-ns1", Namespace: "ns1"},
				Spec: tfv1.GPUResourceQuotaSpec{
					Total: tfv1.GPUResourceQuotaTotal{
						Requests: &tfv1.Resource{Tflops: resource.MustParse("50000"), Vram: resource.MustParse("80Gi")},
						Limits:   &tfv1.Resource{Tflops: resource.MustParse("50000"), Vram: resource.MustParse("80Gi")},
					},
				},
			},
		}

		k8sClient = fake.NewClientBuilder().WithScheme(scheme.Scheme).
			WithRuntimeObjects(objList...).
			WithStatusSubresource(&tfv1.GPU{}, &tfv1.GPUNode{}, &tfv1.GPUResourceQuota{}, &tfv1.TensorFusionWorkload{}, &v1.Pod{}, &v1.Node{}).
			Build()

		k8sObjs := make([]runtime.Object, 0, len(pods)+len(nodes))
		for _, pod := range pods {
			Expect(k8sClient.Create(ctx, pod)).To(Succeed())
			k8sObjs = append(k8sObjs, pod)
		}
		for _, gpu := range gpus {
			Expect(k8sClient.Create(ctx, gpu)).To(Succeed())
		}
		for _, node := range nodes {
			Expect(k8sClient.Create(ctx, node)).To(Succeed())
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
		metricsRecorder := metrics.NewMetricsAsyncRecorder(1000, time.Second, ctx.Done())
		var err error
		fwkInstance, err = tf.NewFramework(
			ctx, registeredPlugins, "",
			frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
			frameworkruntime.WithSnapshotSharedLister(internalcache.NewEmptySnapshot()),
			frameworkruntime.WithEventRecorder(&events.FakeRecorder{}),
			frameworkruntime.WithMetricsRecorder(metricsRecorder),
		)
		Expect(err).NotTo(HaveOccurred())

		allocator = gpuallocator.NewGpuAllocator(ctx, nil, k8sClient, time.Second)
		Expect(allocator.InitGPUAndQuotaStore()).To(Succeed())
		allocator.ReconcileAllocationState()
		allocator.SetAllocatorReady()

		indexAllocator, err = indexallocator.NewIndexAllocator(ctx, k8sClient)
		Expect(err).NotTo(HaveOccurred())
		indexAllocator.IsLeader = true
		indexAllocator.SetReady()

		pluginFactory := NewWithDeps(allocator, indexAllocator, nil, k8sClient)
		pluginConfig := &runtime.Unknown{
			Raw: []byte(`{"maxWorkerPerNode": 3, "vramWeight": 0.7, "tflopsWeight": 0.3}`),
		}
		p, err := pluginFactory(ctx, pluginConfig, fwkInstance)
		Expect(err).NotTo(HaveOccurred())
		plugin = p.(*GPUFit)
	})

	AfterEach(func() {
		log.FromContext(ctx).Info("Tearing down test")
		cancel()
	})

	Describe("PreFilter", func() {
		DescribeTable("filters nodes based on GPU requirements",
			func(annotations map[string]string, expectedStatus fwk.Code, expectedNodes string) {
				state := framework.NewCycleState()
				pod := makePod("p-test", annotations)
				res, status := plugin.PreFilter(ctx, state, pod, []fwk.NodeInfo{})
				Expect(status.Code()).To(Equal(expectedStatus), status.Message())
				if expectedStatus == fwk.Success {
					Expect(res).NotTo(BeNil())
					nodes := sort.StringSlice(getPreFilterResult(state))
					nodes.Sort()
					Expect(strings.Join(nodes, " ")).To(Equal(expectedNodes))
				}
			},
			Entry("pod requires 1 GPU, enough capacity", map[string]string{
				constants.GpuCountAnnotation:      "1",
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
			}, fwk.Success, "node-a node-b"),
			Entry("pod requires 1 GPU, enough tflops", map[string]string{
				constants.GpuCountAnnotation:      "1",
				constants.TFLOPSRequestAnnotation: "2000",
				constants.VRAMRequestAnnotation:   "10Gi",
			}, fwk.Success, "node-b"),
			Entry("pod requires 2 GPUs should be scheduled on node-b", map[string]string{
				constants.GpuCountAnnotation:      "2",
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
			}, fwk.Success, "node-b"),
			Entry("pod requires 1 GPU, not enough vram", map[string]string{
				constants.GpuCountAnnotation:      "1",
				constants.TFLOPSRequestAnnotation: "2000",
				constants.VRAMRequestAnnotation:   "80Gi",
			}, fwk.Unschedulable, ""),
			Entry("pod requires 3 GPUs, but at most 2 on existing nodes", map[string]string{
				constants.GpuCountAnnotation:      "3",
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
			}, fwk.Unschedulable, ""),
		)
	})

	Describe("PreFilter for non-TensorFusion pods", func() {
		DescribeTable("filters nodes for non-TF pods",
			func(gpuCount int, expectedStatus fwk.Code, expectedNodes string) {
				state := framework.NewCycleState()
				pod := makeNonTensorFusionPod("p-nontf", gpuCount)
				res, status := plugin.PreFilter(ctx, state, pod, []fwk.NodeInfo{})
				Expect(status.Code()).To(Equal(expectedStatus), status.Message())
				if expectedStatus == fwk.Success {
					Expect(res).NotTo(BeNil())
					nodes := sort.StringSlice(res.NodeNames.UnsortedList())
					nodes.Sort()
					Expect(strings.Join(nodes, " ")).To(Equal(expectedNodes))
				}
			},
			Entry("pod requires 1 GPU", 1, fwk.Success, "node-b node-c"),
			Entry("pod requires 2 GPU", 2, fwk.Success, "node-b node-c"),
		)
	})

	Describe("Filter", func() {
		It("should filter nodes correctly", func() {
			state := framework.NewCycleState()
			pod := makePod("p-filter", map[string]string{
				constants.GpuCountAnnotation:      "1",
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
				constants.TFLOPSLimitAnnotation:   "100",
				constants.VRAMLimitAnnotation:     "40Gi",
			})
			_, preFilterStatus := plugin.PreFilter(ctx, state, pod, []fwk.NodeInfo{})
			Expect(preFilterStatus.IsSuccess()).To(BeTrue())

			// Node with available GPU
			nodeInfo := &framework.NodeInfo{}
			nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}})
			status := plugin.Filter(ctx, state, pod, nodeInfo)
			Expect(status.Code()).To(Equal(fwk.Success))

			// Node without available GPU
			nodeInfo = &framework.NodeInfo{}
			nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-c"}})
			status = plugin.Filter(ctx, state, pod, nodeInfo)
			Expect(status.Code()).To(Equal(fwk.Unschedulable))
		})
	})

	Describe("Score", func() {
		It("should score nodes correctly", func() {
			state := framework.NewCycleState()
			pod := makePod("p-score", map[string]string{
				constants.GpuCountAnnotation:      "1",
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
				constants.TFLOPSLimitAnnotation:   "100",
				constants.VRAMLimitAnnotation:     "40Gi",
			})
			_, preFilterStatus := plugin.PreFilter(ctx, state, pod, []fwk.NodeInfo{})
			Expect(preFilterStatus.IsSuccess()).To(BeTrue())

			// node-a has one worker consumed 10% GPU resources, score = 100 - 90 = 10
			nodeInfo := &framework.NodeInfo{}
			nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}})
			score, status := plugin.Score(ctx, state, pod, nodeInfo)
			Expect(status.IsSuccess()).To(BeTrue())
			Expect(score).To(Equal(int64(10)))

			// node-b has no worker, score = 100 - 100 = 0
			nodeInfo = &framework.NodeInfo{}
			nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-b"}})
			score, status = plugin.Score(ctx, state, pod, nodeInfo)
			Expect(status.IsSuccess()).To(BeTrue())
			Expect(score).To(BeZero())
		})
	})

	Describe("Reserve and Unreserve", func() {
		It("should reserve and unreserve correctly", func() {
			state := framework.NewCycleState()
			pod := makePod("p-reserve", map[string]string{
				constants.GpuCountAnnotation:      "1",
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
				constants.TFLOPSLimitAnnotation:   "100",
				constants.VRAMLimitAnnotation:     "40Gi",
			})
			_, preFilterStatus := plugin.PreFilter(ctx, state, pod, []fwk.NodeInfo{})
			Expect(preFilterStatus.IsSuccess()).To(BeTrue())

			reserveStatus := plugin.Reserve(ctx, state, pod, "node-a")
			Expect(reserveStatus.IsSuccess()).To(BeTrue())

			plugin.allocator.SyncGPUsToK8s()

			gpu := &tfv1.GPU{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "gpu-1"}, gpu)).To(Succeed())
			Expect(gpu.Status.Available.Tflops.String()).To(Equal("800"))
			Expect(gpu.Status.Available.Vram.String()).To(Equal("8Gi"))
			Expect(gpu.Status.RunningApps).To(HaveLen(1))
			Expect(gpu.Status.RunningApps[0].Name).To(Equal("workload-1"))
			Expect(gpu.Status.RunningApps[0].Count).To(Equal(2))

			plugin.Unreserve(ctx, state, pod, "node-a")
			plugin.allocator.SyncGPUsToK8s()

			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "gpu-1"}, gpu)).To(Succeed())
			Expect(gpu.Status.Available.Tflops.String()).To(Equal("900"))
			Expect(gpu.Status.Available.Vram.String()).To(Equal("18Gi"))
			Expect(gpu.Status.RunningApps).To(HaveLen(1))
		})
	})

	Describe("PostBind", func() {
		It("should update pod annotations after bind", func() {
			state := framework.NewCycleState()
			pod := makePod("p-postbind", map[string]string{
				constants.GpuCountAnnotation:      "1",
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
				constants.TFLOPSLimitAnnotation:   "100",
				constants.VRAMLimitAnnotation:     "40Gi",
			})
			_, preFilterStatus := plugin.PreFilter(ctx, state, pod, []fwk.NodeInfo{})
			Expect(preFilterStatus.IsSuccess()).To(BeTrue())

			reserveStatus := plugin.Reserve(ctx, state, pod, "node-a")
			Expect(reserveStatus.IsSuccess()).To(BeTrue())

			plugin.PostBind(ctx, state, pod, "node-a")

			updatedPod := &v1.Pod{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "p-postbind", Namespace: "ns1"}, updatedPod)).To(Succeed())
			Expect(updatedPod.Annotations[constants.GPUDeviceIDsAnnotation]).To(Equal("gpu-1"))
		})
	})

	Describe("NewWithDeps", func() {
		It("should create plugin with valid config", func() {
			pluginFactory := NewWithDeps(allocator, indexAllocator, nil, k8sClient)
			Expect(pluginFactory).NotTo(BeNil())

			pluginConfig := &runtime.Unknown{Raw: []byte(`{"maxWorkerPerNode": 10}`)}
			p, err := pluginFactory(ctx, pluginConfig, fwkInstance)
			Expect(err).NotTo(HaveOccurred())
			Expect(p).NotTo(BeNil())
			Expect(p.Name()).To(Equal(Name))
		})

		It("should return error with invalid config", func() {
			pluginFactory := NewWithDeps(allocator, indexAllocator, nil, k8sClient)
			invalidPluginConfig := &runtime.Unknown{Raw: []byte(`{"maxWorkerPerNode": "invalid"}`)}
			_, err := pluginFactory(ctx, invalidPluginConfig, fwkInstance)
			Expect(err).To(HaveOccurred())
		})
	})

	Describe("Plugin extensions", func() {
		It("should return nil for ScoreExtensions", func() {
			Expect(plugin.ScoreExtensions()).To(BeNil())
		})

		It("should return non-nil for PreFilterExtensions", func() {
			Expect(plugin.PreFilterExtensions()).NotTo(BeNil())
		})

		It("should return correct name", func() {
			Expect(plugin.Name()).To(Equal(Name))
		})
	})

	Describe("Error handling", func() {
		It("should handle Reserve error when state is empty", func() {
			state := framework.NewCycleState()
			pod := makePod("p-err1", map[string]string{
				constants.GpuCountAnnotation:      "1",
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
			})

			status := plugin.Reserve(ctx, state, pod, "node-a")
			Expect(status.AsError()).To(HaveOccurred())
			Expect(status.Code()).To(Equal(fwk.Error))
		})

		It("should handle Reserve error for non-existent node", func() {
			state := framework.NewCycleState()
			pod := makePod("p-err2", map[string]string{
				constants.GpuCountAnnotation:      "1",
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
			})
			_, preFilterStatus := plugin.PreFilter(ctx, state, pod, []fwk.NodeInfo{})
			Expect(preFilterStatus.IsSuccess()).To(BeTrue())

			status := plugin.Reserve(ctx, state, pod, "node-c-non-existent")
			Expect(status.Code()).To(Equal(fwk.Unschedulable))
		})

		It("should not panic on Unreserve with empty state", func() {
			state := framework.NewCycleState()
			pod := makePod("p-err3", map[string]string{
				constants.GpuCountAnnotation:      "1",
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
			})

			Expect(func() {
				plugin.Unreserve(ctx, state, pod, "node-a")
			}).NotTo(Panic())
		})

		It("should handle Filter error when state is empty", func() {
			state := framework.NewCycleState()
			pod := makePod("p-err4", nil)
			nodeInfo := &framework.NodeInfo{}
			nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}})

			status := plugin.Filter(ctx, state, pod, nodeInfo)
			Expect(status.AsError()).To(HaveOccurred())
			Expect(status.Code()).To(Equal(fwk.Error))
		})

		It("should handle Score error when state is empty", func() {
			state := framework.NewCycleState()
			pod := makePod("p-err5", map[string]string{
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
			})

			nodeInfo := &framework.NodeInfo{}
			nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}})
			_, status := plugin.Score(ctx, state, pod, nodeInfo)
			Expect(status.AsError()).To(HaveOccurred())
			Expect(status.Code()).To(Equal(fwk.Error))
		})
	})

	Describe("EventsToRegister", func() {
		It("should return correct events", func() {
			events, err := plugin.EventsToRegister(ctx)
			Expect(err).NotTo(HaveOccurred())
			Expect(events).To(HaveLen(1))
			Expect(events[0].Event.Resource).To(Equal(fwk.EventResource("gpus.v1.tensor-fusion.ai")))
			Expect(events[0].Event.ActionType).To(Equal(fwk.Add | fwk.Update))
			Expect(events[0].QueueingHintFn).NotTo(BeNil())
		})
	})

	Describe("QueueingHint", func() {
		It("should skip when resources decrease", func() {
			pod := makePod("test-pod-hint", map[string]string{
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
				constants.GpuCountAnnotation:      "1",
			})

			oldGPU := &tfv1.GPU{
				Status: tfv1.GPUStatus{
					Available: &tfv1.Resource{Tflops: resource.MustParse("500"), Vram: resource.MustParse("10Gi")},
				},
			}
			newGPU := &tfv1.GPU{
				Status: tfv1.GPUStatus{
					Available: &tfv1.Resource{Tflops: resource.MustParse("400"), Vram: resource.MustParse("8Gi")},
				},
			}

			hint, err := plugin.queueingHint(*plugin.logger, pod, oldGPU, newGPU)
			Expect(err).NotTo(HaveOccurred())
			Expect(hint).To(Equal(fwk.QueueSkip))
		})

		It("should queue when resources increase sufficiently", func() {
			pendingPod := makePod("pending-pod-hint", map[string]string{
				constants.TFLOPSRequestAnnotation: "100",
				constants.VRAMRequestAnnotation:   "10Gi",
				constants.GpuCountAnnotation:      "1",
				constants.GpuPoolKey:              "pool-a",
			})
			pendingPod.Spec.NodeName = ""

			oldGPU := &tfv1.GPU{
				Status: tfv1.GPUStatus{
					Available: &tfv1.Resource{Tflops: resource.MustParse("50"), Vram: resource.MustParse("5Gi")},
				},
			}
			newGPU := &tfv1.GPU{
				Status: tfv1.GPUStatus{
					Available: &tfv1.Resource{Tflops: resource.MustParse("150"), Vram: resource.MustParse("15Gi")},
				},
			}

			hint, err := plugin.queueingHint(*plugin.logger, pendingPod, oldGPU, newGPU)
			Expect(err).NotTo(HaveOccurred())
			Expect(hint).To(Equal(fwk.Queue))
		})

		It("should skip for non-TensorFusion pod", func() {
			nonTFPod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "non-tf-pod", Namespace: "default"},
			}

			oldGPU := &tfv1.GPU{
				Status: tfv1.GPUStatus{
					Available: &tfv1.Resource{Tflops: resource.MustParse("50"), Vram: resource.MustParse("5Gi")},
				},
			}
			newGPU := &tfv1.GPU{
				Status: tfv1.GPUStatus{
					Available: &tfv1.Resource{Tflops: resource.MustParse("150"), Vram: resource.MustParse("15Gi")},
				},
			}

			hint, err := plugin.queueingHint(*plugin.logger, nonTFPod, oldGPU, newGPU)
			Expect(err).NotTo(HaveOccurred())
			Expect(hint).To(Equal(fwk.QueueSkip))
		})

		It("should queue on GPU add", func() {
			pendingPod := makePod("pending-pod-add", map[string]string{
				constants.TFLOPSRequestAnnotation: "50",
				constants.VRAMRequestAnnotation:   "5Gi",
				constants.GpuCountAnnotation:      "1",
				constants.GpuPoolKey:              "pool-a",
			})
			pendingPod.Spec.NodeName = ""

			newGPU := &tfv1.GPU{
				Status: tfv1.GPUStatus{
					Available: &tfv1.Resource{Tflops: resource.MustParse("100"), Vram: resource.MustParse("10Gi")},
				},
			}

			hint, err := plugin.queueingHint(*plugin.logger, pendingPod, nil, newGPU)
			Expect(err).NotTo(HaveOccurred())
			Expect(hint).To(Equal(fwk.Queue))
		})
	})

	Describe("ReconcileAllocationState", func() {
		It("should correctly handle compute-percent-request", func() {
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
			Expect(k8sClient.Create(ctx, gpu)).To(Succeed())

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
			Expect(k8sClient.Create(ctx, workerPod)).To(Succeed())

			// Manually add GPU to allocator store
			gpuStore, _, _ := allocator.GetAllocationInfo()
			key := types.NamespacedName{Name: "gpu-test-percent"}
			gpuCopy := gpu.DeepCopy()
			if gpuCopy.Status.Available == nil {
				gpuCopy.Status.Available = gpuCopy.Status.Capacity.DeepCopy()
			}
			gpuStore[key] = gpuCopy

			// Reconcile allocation state
			allocator.ReconcileAllocationStateForTesting()
			allocator.SyncGPUsToK8s()

			// Verify GPU available resources are correctly deducted
			updatedGPU := &tfv1.GPU{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "gpu-test-percent"}, updatedGPU)).To(Succeed())

			// Expected: 1000 - 100 (10% of 1000) = 900 TFLOPs
			expectedTflops := resource.MustParse("900")
			Expect(updatedGPU.Status.Available.Tflops.Equal(expectedTflops)).To(BeTrue(),
				"Expected TFLOPs: %s, Got: %s", expectedTflops.String(), updatedGPU.Status.Available.Tflops.String())

			// Expected: 20Gi - 2Gi = 18Gi VRAM
			expectedVram := resource.MustParse("18Gi")
			Expect(updatedGPU.Status.Available.Vram.Equal(expectedVram)).To(BeTrue(),
				"Expected VRAM: %s, Got: %s", expectedVram.String(), updatedGPU.Status.Available.Vram.String())

			// Verify running apps
			Expect(updatedGPU.Status.RunningApps).To(HaveLen(1))
			Expect(updatedGPU.Status.RunningApps[0].Name).To(Equal("workload-percent"))
		})

		It("should correctly handle mixed TFLOPs and compute-percent requests", func() {
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
			Expect(k8sClient.Create(ctx, gpu)).To(Succeed())

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
			Expect(k8sClient.Create(ctx, workerPod1)).To(Succeed())

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
			Expect(k8sClient.Create(ctx, workerPod2)).To(Succeed())

			// Manually add GPU to allocator store
			gpuStore, _, _ := allocator.GetAllocationInfo()
			key := types.NamespacedName{Name: "gpu-test-mixed"}
			gpuCopy := gpu.DeepCopy()
			if gpuCopy.Status.Available == nil {
				gpuCopy.Status.Available = gpuCopy.Status.Capacity.DeepCopy()
			}
			gpuStore[key] = gpuCopy

			// Reconcile allocation state
			allocator.ReconcileAllocationStateForTesting()
			allocator.SyncGPUsToK8s()

			// Verify GPU available resources are correctly deducted
			updatedGPU := &tfv1.GPU{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "gpu-test-mixed"}, updatedGPU)).To(Succeed())

			// Expected: 2000 - 300 (direct TFLOPs) - 300 (15% of 2000) = 1400 TFLOPs
			expectedTflops := resource.MustParse("1400")
			Expect(updatedGPU.Status.Available.Tflops.Equal(expectedTflops)).To(BeTrue(),
				"Expected TFLOPs: %s, Got: %s", expectedTflops.String(), updatedGPU.Status.Available.Tflops.String())

			// Expected: 40Gi - 5Gi - 3Gi = 32Gi VRAM
			expectedVram := resource.MustParse("32Gi")
			Expect(updatedGPU.Status.Available.Vram.Equal(expectedVram)).To(BeTrue(),
				"Expected VRAM: %s, Got: %s", expectedVram.String(), updatedGPU.Status.Available.Vram.String())

			// Verify running apps - should have both workloads
			Expect(updatedGPU.Status.RunningApps).To(HaveLen(2))

			// Check both workloads are present
			workloadNames := []string{
				updatedGPU.Status.RunningApps[0].Name,
				updatedGPU.Status.RunningApps[1].Name,
			}
			Expect(workloadNames).To(ContainElement("workload-tflops"))
			Expect(workloadNames).To(ContainElement("workload-percent-mixed"))
		})

		It("should correctly handle multiple GPUs with compute-percent requests", func() {
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
			Expect(k8sClient.Create(ctx, gpu1)).To(Succeed())

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
			Expect(k8sClient.Create(ctx, gpu2)).To(Succeed())

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
			Expect(k8sClient.Create(ctx, workerPod1)).To(Succeed())

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
			Expect(k8sClient.Create(ctx, workerPod2)).To(Succeed())

			// Manually add GPUs to allocator store
			gpuStore, _, _ := allocator.GetAllocationInfo()
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
			allocator.ReconcileAllocationStateForTesting()
			allocator.SyncGPUsToK8s()

			// Verify GPU 1 available resources
			updatedGPU1 := &tfv1.GPU{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "gpu-mixed-1"}, updatedGPU1)).To(Succeed())

			// Expected: 1000 - 200 (20% of 1000) = 800 TFLOPs
			expectedTflops1 := resource.MustParse("800")
			Expect(updatedGPU1.Status.Available.Tflops.Equal(expectedTflops1)).To(BeTrue(),
				"GPU1 Expected TFLOPs: %s, Got: %s", expectedTflops1.String(), updatedGPU1.Status.Available.Tflops.String())

			// Expected: 20Gi - 4Gi = 16Gi VRAM
			expectedVram1 := resource.MustParse("16Gi")
			Expect(updatedGPU1.Status.Available.Vram.Equal(expectedVram1)).To(BeTrue(),
				"GPU1 Expected VRAM: %s, Got: %s", expectedVram1.String(), updatedGPU1.Status.Available.Vram.String())

			// Verify GPU 2 available resources
			updatedGPU2 := &tfv1.GPU{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "gpu-mixed-2"}, updatedGPU2)).To(Succeed())

			// Expected: 2000 - 200 (10% of 2000) = 1800 TFLOPs
			expectedTflops2 := resource.MustParse("1800")
			Expect(updatedGPU2.Status.Available.Tflops.Equal(expectedTflops2)).To(BeTrue(),
				"GPU2 Expected TFLOPs: %s, Got: %s", expectedTflops2.String(), updatedGPU2.Status.Available.Tflops.String())

			// Expected: 40Gi - 6Gi = 34Gi VRAM
			expectedVram2 := resource.MustParse("34Gi")
			Expect(updatedGPU2.Status.Available.Vram.Equal(expectedVram2)).To(BeTrue(),
				"GPU2 Expected VRAM: %s, Got: %s", expectedVram2.String(), updatedGPU2.Status.Available.Vram.String())
		})
	})
})

func getPreFilterResult(state *framework.CycleState) []string {
	data, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		return nil
	}
	return lo.Keys(data.(*GPUSchedulingStateData).NodeGPUs)
}
