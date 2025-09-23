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
	"go.uber.org/zap/zapcore"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
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
	klog.SetLogger(zap.New(zap.WriteTo(discardWriter{}), zap.UseDevMode(false), zap.Level(zapcore.InfoLevel)))

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

	ctx, cancel := context.WithCancel(context.Background())
	pts.ctx = ctx
	pts.cancel = cancel

	cc, scheduler, _, err := sched.SetupScheduler(ctx, nil,
		"../../config/samples/scheduler-config.yaml", true, ver, fixture.allocator, false, gpuResourceFitOpt, gpuTopoOpt)
	Expect(err).To(Succeed())
	pts.scheduler = scheduler
	scheduler.SchedulingQueue.Run(klog.FromContext(ctx))

	// Start scheduler components
	cc.EventBroadcaster.StartRecordingToSink(ctx.Done())
	cc.InformerFactory.Start(ctx.Done())
	cc.InformerFactory.WaitForCacheSync(ctx.Done())
	Expect(scheduler.WaitForHandlersSync(ctx)).To(Succeed())
}

// TearDownSuite cleans up the test environment
func (pts *PreemptionTestSuite) TearDownSuite() {
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

// discardWriter implements io.Writer to discard log output during tests
type discardWriter struct{}

func (discardWriter) Write(p []byte) (n int, err error) {
	return len(p), nil
}

// TestPreemption tests comprehensive preemption scenarios
func TestPreemption(t *testing.T) {
	t.Skip("Skipping preemption test")
	suite := &PreemptionTestSuite{}
	suite.SetupSuite(t)
	defer suite.TearDownSuite(t)
	testGPUResourcePreemption(t, suite)
}

// TestPreemptionEvictProtection tests comprehensive preemption scenarios
func TestPreemptionEvictProtection(t *testing.T) {
	t.Skip("Skipping preemption test")
	suite := &PreemptionTestSuite{}
	suite.SetupSuite(t)
	defer suite.TearDownSuite(t)
	testGPUResourceEvictProtection(t, suite)
}

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
	suite.scheduler.ScheduleOne(suite.ctx)

	// Preemption should be triggered and victims deleted, wait informer sync
	Eventually(func() int {
		podList := &v1.PodList{}
		err := suite.k8sClient.List(suite.ctx, podList, &client.ListOptions{Namespace: "preemption-test-ns"})
		Expect(err).To(Succeed())
		return len(podList.Items)
	}, 5*time.Second, 100*time.Millisecond).Should(Equal(12)) // 2 Pods deleted, 14 - 2 = 12

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
	suite.scheduler.ScheduleOne(suite.ctx)
	suite.scheduler.ScheduleOne(suite.ctx)

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

	toBeVictimPods = createPreemptionTestPodsWithQoS("high-priority", constants.QoSLevelHigh, 1, "2000", "2Gi")
	Expect(suite.k8sClient.Create(suite.ctx, toBeVictimPods[0])).To(Succeed())
	defer func() {
		_ = suite.k8sClient.Delete(suite.ctx, toBeVictimPods[0])
	}()

	// should not evict since it's inside protection period
	suite.scheduler.ScheduleOne(suite.ctx)

	// Verify that both pods still exist (no eviction during protection period)
	Consistently(func() int {
		podList := &v1.PodList{}
		err := suite.k8sClient.List(suite.ctx, podList, &client.ListOptions{Namespace: "preemption-test-ns"})
		Expect(err).To(Succeed())
		return len(podList.Items)
	}, 3*time.Second, 100*time.Millisecond).Should(Equal(2))

	// Trigger eviction after protection period
	suite.scheduler.ScheduleOne(suite.ctx)

	time.Sleep(1 * time.Second)
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
