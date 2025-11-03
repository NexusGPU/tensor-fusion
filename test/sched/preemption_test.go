//go:build !nobench

package sched

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/cmd/sched"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	gpuResourceFitPlugin "github.com/NexusGPU/tensor-fusion/internal/scheduler/gpuresources"
	gpuTopoPlugin "github.com/NexusGPU/tensor-fusion/internal/scheduler/gputopo"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
)

// PreemptionTestSuite holds common test setup for preemption tests
type PreemptionTestSuite struct {
	ctx            context.Context
	cancel         context.CancelFunc
	k8sClient      client.Client
	scheduler      *scheduler.Scheduler
	fixture        *BenchmarkFixture
	testEnv        *envtest.Environment
	kubeconfigPath string
}

// SetupSuite initializes the test environment for preemption tests
func (pts *PreemptionTestSuite) SetupSuite() {
	// Setup test environment
	ver, cfg, err := setupKubernetes()
	Expect(err).To(Succeed())
	pts.testEnv = testEnv

	kubeconfigPath, err := writeKubeconfigToTempFileAndSetEnv(cfg)
	Expect(err).To(Succeed())
	pts.kubeconfigPath = kubeconfigPath

	k8sClient, err := client.New(cfg, client.Options{Scheme: scheme.Scheme})
	Expect(err).To(Succeed())
	pts.k8sClient = k8sClient

	// Configure test with limited resources for preemption scenarios
	benchConfig := BenchmarkConfig{
		NumNodes:  2,
		NumGPUs:   4,
		PoolName:  "preemption-test-pool",
		Namespace: "preemption-test-ns",
		Timeout:   1 * time.Minute,
	}

	mockBench := &testing.B{}
	fixture := NewBenchmarkFixture(mockBench, benchConfig, k8sClient, true)
	pts.fixture = fixture

	utils.SetProgressiveMigration(false)

	gpuResourceFitOpt := app.WithPlugin(
		gpuResourceFitPlugin.Name,
		gpuResourceFitPlugin.NewWithDeps(fixture.allocator, fixture.client),
	)
	gpuTopoOpt := app.WithPlugin(
		gpuTopoPlugin.Name,
		gpuTopoPlugin.NewWithDeps(fixture.allocator, fixture.client),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	pts.ctx = ctx
	pts.cancel = cancel

	cc, scheduler, _, err := sched.SetupScheduler(ctx, nil,
		"../../config/samples/scheduler-config.yaml", true, ver, fixture.allocator, false, false, gpuResourceFitOpt, gpuTopoOpt)
	Expect(err).To(Succeed())
	pts.scheduler = scheduler
	scheduler.SchedulingQueue.Run(klog.FromContext(ctx))
	if scheduler.APIDispatcher != nil {
		scheduler.APIDispatcher.Run(klog.FromContext(ctx))
	}

	// Start scheduler components
	cc.EventBroadcaster.StartRecordingToSink(ctx.Done())
	cc.InformerFactory.Start(ctx.Done())
	cc.InformerFactory.WaitForCacheSync(ctx.Done())
	Expect(scheduler.WaitForHandlersSync(ctx)).To(Succeed())
}

// TearDownSuite cleans up the test environment
func (pts *PreemptionTestSuite) TearDownSuite() {
	time.Sleep(300 * time.Millisecond)
	if pts.cancel != nil {
		pts.cancel()
	}
	if pts.fixture != nil {
		pts.fixture.Close()
	}
	if pts.kubeconfigPath != "" {
		Expect(cleanupKubeconfigTempFile(pts.kubeconfigPath)).To(Succeed())
	}
	if pts.testEnv != nil {
		Expect(pts.testEnv.Stop()).To(Succeed())
	}
}

// TestPreemption tests comprehensive preemption scenarios
func TestPreemption(t *testing.T) {
	suiteConfig, reporterConfig := GinkgoConfiguration()
	suiteConfig.Timeout = 2 * time.Minute
	RegisterFailHandler(Fail)
	RunSpecs(t, "Preemption Test Suite", suiteConfig, reporterConfig)
}

var _ = Describe("GPU Resource Preemption", func() {
	var suite *PreemptionTestSuite

	BeforeEach(func() {
		suite = &PreemptionTestSuite{}
		suite.SetupSuite()
	})

	AfterEach(func() {
		suite.TearDownSuite()
	})

	PIt("should preempt lower priority pods for higher priority ones", func() {
		testGPUResourcePreemption(suite)
	})

	It("should respect eviction protection periods", func() {
		testGPUResourceEvictProtection(suite)
	})
})

// testGPUResourcePreemption tests GPU shortage detection logic
func testGPUResourcePreemption(suite *PreemptionTestSuite) {
	// Mock cluster resources
	// {"2250", "141Gi"}, // Simulate B200
	// {"989", "80Gi"},   // Simulate H100
	// {"450", "48Gi"},   // Simulate L40s
	// {"312", "40Gi"},   // Simulate A100

	// Create pods that will exhaust resources
	toBeVictimPods := createPreemptionTestPodsWithQoS("victim", constants.QoSLevelMedium, 7+3+1+1, "300", "1Gi")

	for _, pod := range toBeVictimPods {
		Expect(suite.k8sClient.Create(suite.ctx, pod)).To(Succeed())
		defer func(p *v1.Pod) {
			_ = suite.k8sClient.Delete(suite.ctx, p)
		}(pod)
	}

	// Try scheduling all pending pods
	for range 12 {
		suite.scheduler.ScheduleOne(suite.ctx)
	}

	// schedule high priority pod
	highPriorityPod := createPreemptionTestPodsWithQoS("high-priority", constants.QoSLevelHigh, 1, "300", "1Gi")[0]
	Expect(suite.k8sClient.Create(suite.ctx, highPriorityPod)).To(Succeed())
	defer func() {
		_ = suite.k8sClient.Delete(suite.ctx, highPriorityPod)
	}()

	suite.scheduler.ScheduleOne(suite.ctx)

	// schedule critical priority pod
	criticalPriorityPod := createPreemptionTestPodsWithQoS(
		"critical-priority", constants.QoSLevelCritical, 1, "300", "1Gi")[0]
	Expect(suite.k8sClient.Create(suite.ctx, criticalPriorityPod)).To(Succeed())
	defer func() {
		_ = suite.k8sClient.Delete(suite.ctx, criticalPriorityPod)
	}()
	time.Sleep(10 * time.Millisecond)
	suite.scheduler.SchedulingQueue.Add(klog.FromContext(suite.ctx), criticalPriorityPod)
	suite.scheduler.ScheduleOne(suite.ctx)

	// Preemption should be triggered and victims deleted, wait informer sync
	Eventually(func() int {
		podList := &v1.PodList{}
		err := suite.k8sClient.List(suite.ctx, podList, &client.ListOptions{Namespace: "preemption-test-ns"})
		Expect(err).To(Succeed())
		return len(podList.Items)
	}, 5*time.Second, 200*time.Millisecond).Should(Equal(12)) // 2 Pods deleted, 14 - 2 = 12

	podList := &v1.PodList{}
	err := suite.k8sClient.List(suite.ctx, podList, &client.ListOptions{Namespace: "preemption-test-ns"})
	Expect(err).To(Succeed())
	scheduledNodeMap := make(map[string]string)
	for _, pod := range podList.Items {
		scheduledNodeMap[pod.Name] = pod.Spec.NodeName
	}

	// without Pod Controller, directly reconcile all state to simulate the Pod deletion
	suite.fixture.allocator.ReconcileAllocationStateForTesting()

	// Trigger next 2 scheduling cycle, make sure the two higher priority pods are scheduled
	suite.scheduler.SchedulingQueue.Activate(klog.FromContext(suite.ctx), map[string]*v1.Pod{
		highPriorityPod.Name:     highPriorityPod,
		criticalPriorityPod.Name: criticalPriorityPod,
	})
	suite.scheduler.ScheduleOne(suite.ctx)
	suite.scheduler.ScheduleOne(suite.ctx)
	time.Sleep(10 * time.Millisecond)

	// Wait for high priority pods to be scheduled
	Eventually(func() bool {
		podList := &v1.PodList{}
		err := suite.k8sClient.List(suite.ctx, podList, &client.ListOptions{Namespace: "preemption-test-ns"})
		Expect(err).To(Succeed())

		scheduledNodeMap := make(map[string]string)
		for _, pod := range podList.Items {
			if strings.Contains(pod.Name, "victim") {
				continue
			}
			scheduledNodeMap[pod.Name] = pod.Spec.NodeName
		}

		// Check if both high priority pods are scheduled
		return scheduledNodeMap["high-priority-0"] != "" && scheduledNodeMap["critical-priority-0"] != ""
	}, 5*time.Second, 100*time.Millisecond).Should(BeTrue())
}

func testGPUResourceEvictProtection(suite *PreemptionTestSuite) {
	toBeVictimPods := createPreemptionTestPodsWithQoS("victim", constants.QoSLevelMedium, 1, "2000", "2Gi")
	toBeVictimPods[0].Annotations[constants.EvictionProtectionAnnotation] = "3s"
	Expect(suite.k8sClient.Create(suite.ctx, toBeVictimPods[0])).To(Succeed())
	defer func() {
		_ = suite.k8sClient.Delete(suite.ctx, toBeVictimPods[0])
	}()

	suite.scheduler.ScheduleOne(suite.ctx)

	preemptionPod := createPreemptionTestPodsWithQoS("high-priority", constants.QoSLevelHigh, 1, "2000", "2Gi")
	Expect(suite.k8sClient.Create(suite.ctx, preemptionPod[0])).To(Succeed())
	defer func() {
		_ = suite.k8sClient.Delete(suite.ctx, preemptionPod[0])
	}()

	// should not evict since it's inside protection period
	suite.scheduler.ScheduleOne(suite.ctx)

	// Verify that both pods still exist (no eviction during protection period)
	Consistently(func() int {
		podList := &v1.PodList{}
		err := suite.k8sClient.List(suite.ctx, podList, &client.ListOptions{Namespace: "preemption-test-ns"})
		Expect(err).To(Succeed())
		return len(podList.Items)
	}, 3*time.Second, 500*time.Millisecond).Should(Equal(2))

	// Trigger eviction after protection period
	suite.scheduler.SchedulingQueue.Activate(klog.FromContext(suite.ctx), map[string]*v1.Pod{
		preemptionPod[0].Name: preemptionPod[0],
	})
	suite.scheduler.ScheduleOne(suite.ctx)

	time.Sleep(500 * time.Millisecond)
	suite.fixture.allocator.ReconcileAllocationStateForTesting()

	// Should schedule the new high priority pod
	suite.scheduler.ScheduleOne(suite.ctx)

	// Wait for eviction and new pod scheduling to complete
	Eventually(func() bool {
		podList := &v1.PodList{}
		err := suite.k8sClient.List(suite.ctx, podList, &client.ListOptions{Namespace: "preemption-test-ns"})
		Expect(err).To(Succeed())

		if len(podList.Items) != 1 {
			return false
		}
		return podList.Items[0].Name == "high-priority-0" && podList.Items[0].Spec.NodeName == "node-0"
	}, 5*time.Second, 100*time.Millisecond).Should(BeTrue())
}

// Helper functions
func createPreemptionTestPodsWithQoS(baseName, qosLevel string, count int, tflops, vram string) []*v1.Pod {
	pods := make([]*v1.Pod, count)
	for i := 0; i < count; i++ {
		pod := st.MakePod().
			Namespace("preemption-test-ns").
			Name(fmt.Sprintf("%s-%d", baseName, i)).
			UID(fmt.Sprintf("%s-%d", baseName, i)).
			SchedulerName("tensor-fusion-scheduler").
			Res(map[v1.ResourceName]string{
				v1.ResourceCPU:    "100m",
				v1.ResourceMemory: "256Mi",
			}).
			Toleration("node.kubernetes.io/not-ready").
			ZeroTerminationGracePeriod().Obj()

		pod.Labels = map[string]string{
			constants.LabelComponent: constants.ComponentWorker,
			constants.WorkloadKey:    "test-workload",
		}

		pod.Annotations = map[string]string{
			constants.GpuPoolKey:              "preemption-test-pool",
			constants.QoSLevelAnnotation:      qosLevel,
			constants.TFLOPSRequestAnnotation: tflops,
			constants.VRAMRequestAnnotation:   vram,
			constants.TFLOPSLimitAnnotation:   tflops,
			constants.VRAMLimitAnnotation:     vram,
			constants.GpuCountAnnotation:      "1",
		}
		pod.Spec.PriorityClassName = "tensor-fusion-" + qosLevel

		pods[i] = pod
	}
	return pods
}
