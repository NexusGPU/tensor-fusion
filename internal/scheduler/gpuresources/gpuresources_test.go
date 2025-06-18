package gpuresources

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	plfeature "k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/scheduler-plugins/apis/scheduling/v1alpha1"
	testutil "sigs.k8s.io/scheduler-plugins/test/util"
)

const ResourceGPU v1.ResourceName = "nvidia.com/gpu"

var (
	midPriority, highPriority = int32(100), int32(1000)
)

func TestPreFilter(t *testing.T) {
	type podInfo struct {
		podName      string
		podNamespace string
		memReq       int64
	}

	tests := []struct {
		name     string
		podInfos []podInfo
		gpus     []tfv1.GPU
		expected []framework.Code
	}{
		{
			name: "find gpu",
			podInfos: []podInfo{
				{podName: "ns1-p1", podNamespace: "ns1", memReq: 500},
				{podName: "ns1-p2", podNamespace: "ns1", memReq: 1800},
			},
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "gpu-1",
					},
					Status: tfv1.GPUStatus{
						Capacity: &tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("20Gi"),
						},
					},
				},
			},
			expected: []framework.Code{
				framework.Success,
				framework.Unschedulable,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			var registerPlugins []tf.RegisterPluginFunc
			registeredPlugins := append(
				registerPlugins,
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)

			fwk, err := tf.NewFramework(
				ctx, registeredPlugins, "",
				frameworkruntime.WithPodNominator(testutil.NewPodNominator(nil)),
				frameworkruntime.WithSnapshotSharedLister(testutil.NewFakeSharedLister(make([]*v1.Pod, 0), make([]*v1.Node, 0))),
			)

 			if err != nil {
				t.Fatal(err)
			}

			// TODO add gpu store
			cs := &GPUFit{
				fh: fwk,
			}

			pods := make([]*v1.Pod, 0)

			// for _, _ := range tt.gpus {
			// 	// TODO make gpu
			// }
			for _, podInfo := range tt.podInfos {
				pod := makePod(podInfo.podName, podInfo.podNamespace, podInfo.memReq, 0, 0, 0, podInfo.podName, "")
				pods = append(pods, pod)
			}

			state := framework.NewCycleState()
			for i := range pods {
				if got := cs.Filter(context.TODO(), state, pods[i], nil); got.Code() != tt.expected[i] {
					t.Errorf("expected %v, got %v : %v", tt.expected[i], got.Code(), got.Message())
				}
			}
		})
	}
}

func TestReserve(t *testing.T) {
	tests := []struct {
		name          string
		pods          []*v1.Pod
		expectedCodes []framework.Code
		expected      []tfv1.GPUStatus
	}{
		{
			name: "Reserve pods",
			pods: []*v1.Pod{
				makePod("t1-p1", "ns1", 50, 0, 0, midPriority, "t1-p1", "node-a"),
				makePod("t1-p2", "ns2", 50, 0, 0, midPriority, "t1-p2", "node-a"),
			},
			expectedCodes: []framework.Code{
				framework.Success,
				framework.Success,
			},
			expected: []tfv1.GPUStatus{
				{
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("1000"),
						Vram:   resource.MustParse("20Gi"),
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			var registerPlugins []tf.RegisterPluginFunc
			registeredPlugins := append(
				registerPlugins,
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)

			fwk, err := tf.NewFramework(
				ctx, registeredPlugins, "",
				frameworkruntime.WithPodNominator(testutil.NewPodNominator(nil)),
				frameworkruntime.WithSnapshotSharedLister(testutil.NewFakeSharedLister(make([]*v1.Pod, 0), make([]*v1.Node, 0))),
			)

			if err != nil {
				t.Fatal(err)
			}

			cs := &GPUFit{
				fh: fwk,
			}

			state := framework.NewCycleState()
			for i, pod := range tt.pods {
				got := cs.Reserve(context.TODO(), state, pod, "node-a")
				if got.Code() != tt.expectedCodes[i] {
					t.Errorf("expected %v, got %v : %v", tt.expected[i], got.Code(), got.Message())
				}
				// TODO assert
			}
		})
	}
}

func TestUnreserve(t *testing.T) {
	tests := []struct {
		name     string
		pods     []*v1.Pod
		expected []tfv1.GPUStatus
	}{
		{
			name: "Unreserve pods",
			pods: []*v1.Pod{
				makePod("t1-p1", "ns1", 50, 0, 0, midPriority, "t1-p1", "node-a"),
				makePod("t1-p2", "ns2", 50, 0, 0, midPriority, "t1-p2", "node-a"),
				makePod("t1-p3", "ns1", 50, 0, 0, midPriority, "t1-p3", "node-a"),
			},
			expected: []tfv1.GPUStatus{
				{
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("100"),
						Vram:   resource.MustParse("20Gi"),
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			var registerPlugins []tf.RegisterPluginFunc
			registeredPlugins := append(
				registerPlugins,
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)

			fwk, err := tf.NewFramework(
				ctx, registeredPlugins, "",
				frameworkruntime.WithPodNominator(testutil.NewPodNominator(nil)),
				frameworkruntime.WithSnapshotSharedLister(testutil.NewFakeSharedLister(make([]*v1.Pod, 0), make([]*v1.Node, 0))),
			)

			if err != nil {
				t.Fatal(err)
			}

			cs := &GPUFit{
				fh: fwk,
			}

			state := framework.NewCycleState()
			for _, pod := range tt.pods {
				cs.Unreserve(context.TODO(), state, pod, "node-a")
				// TODO asserts
			}
		})
	}
}

func makeUnschedulableNodeStatusReader() *framework.NodeToStatus {
	nodeStatusReader := framework.NewDefaultNodeToStatus()
	nodeStatusReader.Set("node-a", framework.NewStatus(framework.Unschedulable))
	return nodeStatusReader
}

func makePod(podName string, namespace string, memReq int64, cpuReq int64, gpuReq int64, priority int32, uid string, nodeName string) *v1.Pod {
	pause := imageutils.GetPauseImageName()
	pod := st.MakePod().Namespace(namespace).Name(podName).Container(pause).
		Priority(priority).Node(nodeName).UID(uid).ZeroTerminationGracePeriod().Obj()
	pod.Spec.Containers[0].Resources = v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceMemory: *resource.NewQuantity(memReq, resource.DecimalSI),
			v1.ResourceCPU:    *resource.NewMilliQuantity(cpuReq, resource.DecimalSI),
			ResourceGPU:       *resource.NewQuantity(gpuReq, resource.DecimalSI),
		},
	}
	return pod
}

func makePodWithStatus(pod *v1.Pod, podPhase v1.PodPhase) *v1.Pod {
	pod.Status.Phase = podPhase
	return pod
}

func makeEQ(namespace, name string, max, min v1.ResourceList) *v1alpha1.ElasticQuota {
	eq := &v1alpha1.ElasticQuota{
		TypeMeta: metav1.TypeMeta{Kind: "ElasticQuota", APIVersion: "scheduling.sigs.k8s.io/v1alpha1"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}
	eq.Spec.Max = max
	eq.Spec.Min = min
	return eq
}

func makeResourceList(cpu, mem int64) v1.ResourceList {
	return v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(cpu, resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(mem, resource.BinarySI),
	}
}

func makeRegisteredPlugin() []tf.RegisterPluginFunc {
	registeredPlugins := []tf.RegisterPluginFunc{
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		tf.RegisterPluginAsExtensions(noderesources.Name, func(ctx context.Context, plArgs apiruntime.Object, fh framework.Handle) (framework.Plugin, error) {
			return noderesources.NewFit(ctx, plArgs, fh, plfeature.Features{})
		}, "Filter", "PreFilter"),
	}
	return registeredPlugins
}
