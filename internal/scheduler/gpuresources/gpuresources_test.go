package gpuresources

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	gpuallocator "github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	testutil "sigs.k8s.io/scheduler-plugins/test/util"
)

const ResourceGPU v1.ResourceName = "nvidia.com/gpu"

type PodInfo struct {
	podName      string
	podNamespace string
	tflopsReq    int64
	vramReqGiB   int64
	tflopsLimit  int64
	vramLimitGiB int64
}

func TestPreFilter(t *testing.T) {

	tests := []struct {
		name     string
		podInfos []PodInfo
		gpus     []tfv1.GPU
		expected []framework.Code
	}{
		{
			name: "find gpu",
			podInfos: []PodInfo{
				// wrong, all the test pod should test the case that make sense
				{podName: "ns1-p1", podNamespace: "ns1", tflopsReq: 500, vramReqGiB: 0, tflopsLimit: 0, vramLimitGiB: 0},
				{podName: "ns1-p2", podNamespace: "ns1", tflopsReq: 1800, vramReqGiB: 0, tflopsLimit: 0, vramLimitGiB: 0},
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
			// "TODO, this is wrong, how to make a k8s client that share the same data with scheduler test framework, which use fake data, but allocator need data from kubebuilder's client?"
			cs := makeScheduler(ctx, nil, fwk)

			// wrong, should use test case data
			pods := make([]*v1.Pod, 0)

			for _, podInfo := range tt.podInfos {
				pod := makePod(podInfo)
				pods = append(pods, pod)
			}

			state := framework.NewCycleState()
			for i := range pods {
				// wrong, should judge first return result, the scheduling actually done at PreFilter stage
				if _, got := cs.PreFilter(context.TODO(), state, pods[i]); got.Code() != tt.expected[i] {
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
				makePod(PodInfo{
					podName:      "t1-p1",
					podNamespace: "ns1",
					tflopsReq:    50,
					vramReqGiB:   0,
					tflopsLimit:  0,
					vramLimitGiB: 0,
				}),
				makePod(PodInfo{
					podName:      "t1-p2",
					podNamespace: "ns2",
					tflopsReq:    50,
					vramReqGiB:   0,
					tflopsLimit:  0,
					vramLimitGiB: 0,
				}),
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
				makePod(PodInfo{
					podName:      "t1-p1",
					podNamespace: "ns1",
					tflopsReq:    50,
					vramReqGiB:   0,
					tflopsLimit:  100,
					vramLimitGiB: 0,
				}),
				makePod(PodInfo{
					podName:      "t1-p2",
					podNamespace: "ns2",
					tflopsReq:    50,
					vramReqGiB:   0,
					tflopsLimit:  0,
					vramLimitGiB: 0,
				}),
				makePod(PodInfo{
					podName:      "t1-p3",
					podNamespace: "ns1",
					tflopsReq:    50,
					vramReqGiB:   100,
					tflopsLimit:  0,
					vramLimitGiB: 0,
				}),
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

func makePod(podInfo PodInfo) *v1.Pod {
	pause := imageutils.GetPauseImageName()
	pod := st.MakePod().
		Namespace(podInfo.podNamespace).
		Name(podInfo.podName).
		Container(pause).
		UID(podInfo.podName).
		ZeroTerminationGracePeriod().Obj()
	return pod
}

func makeScheduler(ctx context.Context, client client.Client, fwk framework.Framework) *GPUFit {
	return &GPUFit{
		allocator: gpuallocator.NewGpuAllocator(
			ctx,
			client,
			time.Second,
		),
		fh: fwk,
	}
}
