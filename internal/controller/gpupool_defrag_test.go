// Unit tests for GPUPool defrag helpers. These tests intentionally avoid
// envtest / ginkgo so they can run in isolation and fast (<1s). The full
// simulate-schedule loop is covered indirectly via the allocator's own
// suite and via the gpuresources Filter test in ../scheduler/gpuresources.
package controller

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	clientgofake "k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	fwk "k8s.io/kube-scheduler/framework"
	framework "k8s.io/kubernetes/pkg/scheduler/framework"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// ---- parseDefragMaxDuration / cron base guard --------------------------

type infoLogger struct{ messages []string }

func (l *infoLogger) Info(msg string, _ ...any) { l.messages = append(l.messages, msg) }

func TestParseDefragMaxDuration(t *testing.T) {
	cases := []struct {
		name  string
		raw   string
		want  time.Duration
		hints []string
	}{
		{"empty falls back to default", "", defaultDefragMaxDuration, nil},
		{"invalid falls back to default", "not-a-duration", defaultDefragMaxDuration, []string{"invalid defrag maxDuration"}},
		{"zero falls back to default", "0s", defaultDefragMaxDuration, []string{"invalid defrag maxDuration"}},
		{"valid is honored", "45m", 45 * time.Minute, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			lg := &infoLogger{}
			got := parseDefragMaxDuration(tc.raw, lg)
			if got != tc.want {
				t.Fatalf("got %v want %v", got, tc.want)
			}
			if len(tc.hints) > 0 && len(lg.messages) == 0 {
				t.Fatalf("expected fallback log; got none")
			}
		})
	}
}

// ---- countPoolGPUUsage -------------------------------------------------

func gpuWithUsage(name, pool string, avTflops, avVram string, usedBy tfv1.UsedBySystem) *tfv1.GPU {
	return &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				constants.GpuPoolKey: pool,
			},
		},
		Status: tfv1.GPUStatus{
			UsedBy:   usedBy,
			Capacity: &tfv1.Resource{Tflops: resource.MustParse("100"), Vram: resource.MustParse("10Gi")},
			Available: &tfv1.Resource{
				Tflops: resource.MustParse(avTflops),
				Vram:   resource.MustParse(avVram),
			},
		},
	}
}

func TestCountPoolGPUUsage(t *testing.T) {
	const pool = "pool-a"
	// 5 GPUs on the same node, different states.
	gpus := map[string]*tfv1.GPU{
		// Fully free -- counted in total but not in used.
		"free": gpuWithUsage("free", pool, "100", "10Gi", tfv1.UsedByTensorFusion),
		// Partially used by Tflops.
		"partial-tf": gpuWithUsage("partial-tf", pool, "50", "10Gi", tfv1.UsedByTensorFusion),
		// Partially used by VRAM.
		"partial-vram": gpuWithUsage("partial-vram", pool, "100", "5Gi", tfv1.UsedByTensorFusion),
		// Non-TF tenant GPU; excluded from both total and used.
		"non-tf": gpuWithUsage("non-tf", pool, "10", "1Gi", tfv1.UsedBySystem("external")),
		// Different pool; excluded.
		"other-pool": gpuWithUsage("other-pool", "pool-b", "50", "5Gi", tfv1.UsedByTensorFusion),
	}

	total, used := countPoolGPUUsage(gpus, pool)
	if total != 3 {
		t.Fatalf("total=%d want 3", total)
	}
	if used != 2 {
		t.Fatalf("used=%d want 2", used)
	}

	// Empty input returns zero without panic.
	t0, u0 := countPoolGPUUsage(nil, pool)
	if t0 != 0 || u0 != 0 {
		t.Fatalf("nil map expected 0/0 got %d/%d", t0, u0)
	}
}

// ---- subtractGPURequest ------------------------------------------------

func TestSubtractGPURequest_RawTflops(t *testing.T) {
	gpu := gpuWithUsage("g", "p", "100", "10Gi", tfv1.UsedByTensorFusion)
	req := &tfv1.AllocRequest{
		Request: tfv1.Resource{
			Tflops: resource.MustParse("30"),
			Vram:   resource.MustParse("4Gi"),
		},
	}
	subtractGPURequest(gpu, req)
	if got := gpu.Status.Available.Tflops.String(); got != "70" {
		t.Fatalf("tflops after sub=%s want 70", got)
	}
	if got := gpu.Status.Available.Vram.String(); got != "6Gi" {
		t.Fatalf("vram after sub=%s want 6Gi", got)
	}
}

func TestSubtractGPURequest_ComputePercent(t *testing.T) {
	gpu := gpuWithUsage("g", "p", "100", "10Gi", tfv1.UsedByTensorFusion)
	// ComputePercent=40 on a 100-tflops capacity means 40 tflops consumed.
	cp := resource.MustParse("40")
	req := &tfv1.AllocRequest{
		Request: tfv1.Resource{
			ComputePercent: cp,
			Vram:           resource.MustParse("2Gi"),
		},
	}
	subtractGPURequest(gpu, req)
	// Quantity stringification preserves the original format, so compare via float.
	if v := gpu.Status.Available.Tflops.AsApproximateFloat64(); v != 60 {
		t.Fatalf("tflops after sub=%v want 60", v)
	}
	if got := gpu.Status.Available.Vram.String(); got != "8Gi" {
		t.Fatalf("vram after sub=%s want 8Gi", got)
	}
}

func TestSubtractGPURequest_SafelyIgnoresBadInput(t *testing.T) {
	// No panic on nil GPU / nil capacity / nil available.
	subtractGPURequest(nil, &tfv1.AllocRequest{})
	g := &tfv1.GPU{}
	subtractGPURequest(g, &tfv1.AllocRequest{})
}

// ---- evictWorkerPods with fake clientset ------------------------------

type errLogger struct{ infoCount, errCount int }

func (l *errLogger) Info(_ string, _ ...any)           { l.infoCount++ }
func (l *errLogger) Error(_ error, _ string, _ ...any) { l.errCount++ }

func newPod(name string) *corev1.Pod {
	return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: name}}
}

func newReconcilerWithFake(client *clientgofake.Clientset) *GPUPoolCompactionReconciler {
	return &GPUPoolCompactionReconciler{
		KubeClient: client,
		// FakeRecorder so Eventf doesn't panic -- 16 is well over what a
		// single test will produce, otherwise Recorder.Eventf would block.
		Recorder: record.NewFakeRecorder(16),
	}
}

func TestEvictWorkerPods_AllSucceed(t *testing.T) {
	// Seed two pods so the fake can delete them through EvictV1.
	fakeClient := clientgofake.NewSimpleClientset(
		newPod("p1"),
		newPod("p2"),
	)
	r := newReconcilerWithFake(fakeClient)
	cand := &defragCandidate{
		nodeName:   "node-a",
		workerPods: []*corev1.Pod{newPod("p1"), newPod("p2")},
	}
	stats := &defragRunStats{}
	ok := r.evictWorkerPods(context.Background(), &tfv1.GPUPool{}, cand, stats, &errLogger{})
	if !ok {
		t.Fatalf("expected success")
	}
	if stats.EvictedPods != 2 {
		t.Fatalf("evictedPods=%d want 2", stats.EvictedPods)
	}
	if stats.EvictionFailures != 0 {
		t.Fatalf("evictionFailures=%d want 0", stats.EvictionFailures)
	}
}

func TestEvictWorkerPods_AbortsOnFirstFailure(t *testing.T) {
	fakeClient := clientgofake.NewSimpleClientset(
		newPod("p1"),
		newPod("p2"),
	)
	// Inject a 429 on the first eviction, before the clientset's default
	// handler runs. Subsequent evictions must not be attempted because
	// evictWorkerPods is supposed to abort the entire node.
	attempts := 0
	fakeClient.PrependReactor("create", "pods", func(a k8stesting.Action) (bool, runtime.Object, error) {
		if a.GetSubresource() != "eviction" {
			return false, nil, nil
		}
		attempts++
		return true, nil, apierrors.NewTooManyRequestsError("simulated PDB contention")
	})
	r := newReconcilerWithFake(fakeClient)
	cand := &defragCandidate{
		nodeName:   "node-a",
		workerPods: []*corev1.Pod{newPod("p1"), newPod("p2")},
	}
	stats := &defragRunStats{}
	el := &errLogger{}
	ok := r.evictWorkerPods(context.Background(), &tfv1.GPUPool{}, cand, stats, el)
	if ok {
		t.Fatalf("expected abort=false")
	}
	if stats.EvictionFailures != 1 {
		t.Fatalf("evictionFailures=%d want 1", stats.EvictionFailures)
	}
	if stats.EvictedPods != 0 {
		t.Fatalf("evictedPods=%d want 0 (abort before any success)", stats.EvictedPods)
	}
	if attempts != 1 {
		t.Fatalf("EvictV1 called %d times; expected abort after first failure", attempts)
	}
	if el.errCount == 0 {
		t.Fatalf("expected error log entry on failure")
	}
}

func TestEvictWorkerPods_NotFoundCountsAsSuccess(t *testing.T) {
	fakeClient := clientgofake.NewSimpleClientset()
	// No seeded pods, so every EvictV1 resolves to NotFound, which the
	// fake clientset synthesizes by default. But the eviction subresource
	// short-circuits in the fake to a plain action Invoke; inject NotFound
	// explicitly to match the path we actually hit in production.
	fakeClient.PrependReactor("create", "pods", func(a k8stesting.Action) (bool, runtime.Object, error) {
		if a.GetSubresource() != "eviction" {
			return false, nil, nil
		}
		return true, nil, apierrors.NewNotFound(schema.GroupResource{Resource: "pods"}, "gone")
	})
	r := newReconcilerWithFake(fakeClient)
	cand := &defragCandidate{
		nodeName:   "node-a",
		workerPods: []*corev1.Pod{newPod("gone")},
	}
	stats := &defragRunStats{}
	ok := r.evictWorkerPods(context.Background(), &tfv1.GPUPool{}, cand, stats, &errLogger{})
	if !ok {
		t.Fatalf("NotFound should not abort; got abort")
	}
	if stats.EvictedPods != 1 {
		t.Fatalf("evictedPods=%d want 1 (NotFound counts as success)", stats.EvictedPods)
	}
}

func TestEvictWorkerPods_HonorsCtxDeadline(t *testing.T) {
	fakeClient := clientgofake.NewSimpleClientset(newPod("p1"))
	r := newReconcilerWithFake(fakeClient)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	cand := &defragCandidate{
		nodeName:   "node-a",
		workerPods: []*corev1.Pod{newPod("p1")},
	}
	stats := &defragRunStats{}
	if r.evictWorkerPods(ctx, &tfv1.GPUPool{}, cand, stats, &errLogger{}) {
		t.Fatalf("expected ctx-done abort")
	}
	if !stats.DeadlineExceeded {
		t.Fatalf("stats.DeadlineExceeded not set")
	}
}

// ---- getDefragConfig guard ---------------------------------------------

func TestGetDefragConfig(t *testing.T) {
	// nil NodeManagerConfig
	if getDefragConfig(&tfv1.GPUPool{}) != nil {
		t.Fatalf("nil node manager config should return nil")
	}
	// nil NodeCompaction
	p := &tfv1.GPUPool{}
	p.Spec.NodeManagerConfig = &tfv1.NodeManagerConfig{}
	if getDefragConfig(p) != nil {
		t.Fatalf("nil NodeCompaction should return nil")
	}
	// NodeCompaction with nil Defrag
	p.Spec.NodeManagerConfig.NodeCompaction = &tfv1.NodeCompaction{}
	if getDefragConfig(p) != nil {
		t.Fatalf("nil Defrag should return nil")
	}
	// Wired Defrag is returned as-is.
	cfg := &tfv1.NodeDefragConfig{Enabled: true, Schedule: "0 3 * * *"}
	p.Spec.NodeManagerConfig.NodeCompaction.Defrag = cfg
	got := getDefragConfig(p)
	if got != cfg {
		t.Fatalf("expected same pointer back, got %+v", got)
	}
}

func TestPrepareDefragSimulationPod_ClearsAssignedGPUState(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns1",
			Name:      "worker-1",
			Annotations: map[string]string{
				constants.GPUDeviceIDsAnnotation:  "gpu-source",
				constants.ContainerGPUsAnnotation: `{"tensorfusion-worker":["gpu-source"]}`,
				constants.GpuCountAnnotation:      "1",
				constants.GpuPoolKey:              "pool-a",
			},
		},
		Spec: corev1.PodSpec{
			NodeName: "source-node",
		},
		Status: corev1.PodStatus{
			NominatedNodeName: "source-node",
		},
	}

	got := prepareDefragSimulationPod(pod)

	if got == pod {
		t.Fatalf("expected a deep copy, got original pointer")
	}
	if got.Spec.NodeName != "" {
		t.Fatalf("Spec.NodeName=%q, want empty for simulation", got.Spec.NodeName)
	}
	if got.Status.NominatedNodeName != "" {
		t.Fatalf("Status.NominatedNodeName=%q, want empty for simulation", got.Status.NominatedNodeName)
	}
	if _, exists := got.Annotations[constants.GPUDeviceIDsAnnotation]; exists {
		t.Fatalf("assigned GPU ids annotation should be cleared: %+v", got.Annotations)
	}
	if _, exists := got.Annotations[constants.ContainerGPUsAnnotation]; exists {
		t.Fatalf("container GPU assignment annotation should be cleared: %+v", got.Annotations)
	}
	if got.Annotations[constants.GpuCountAnnotation] != "1" {
		t.Fatalf("gpu count request annotation was not preserved: %+v", got.Annotations)
	}
	if pod.Annotations[constants.GPUDeviceIDsAnnotation] == "" {
		t.Fatalf("original pod annotations were mutated")
	}
}

func TestCanFitVirtualNodeResources_RejectsOversubscribedCPU(t *testing.T) {
	nodeInfo := newFrameworkNodeInfo(
		"node-b",
		nil,
		corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("2"),
			corev1.ResourceMemory: resource.MustParse("8Gi"),
			corev1.ResourcePods:   resource.MustParse("4"),
		},
	)
	firstPod := newSchedulerTestPod("ns1", "worker-1", "1500m", "1Gi")
	firstPodInfo, _ := framework.NewPodInfo(firstPod)
	nodeInfo.AddPodInfo(firstPodInfo)

	secondPod := newSchedulerTestPod("ns1", "worker-2", "1500m", "1Gi")
	if canFitVirtualNodeResources(nodeInfo, secondPod) {
		t.Fatal("expected second pod to be rejected once virtual CPU budget is exhausted")
	}
}

func TestBuildDefragNodeBudgets_SkipsDeletionMarkedNodes(t *testing.T) {
	budgets, err := buildDefragNodeBudgets(
		"pool-a",
		"",
		map[string]map[string]*tfv1.GPU{
			"node-keep": {
				"gpu-1": gpuWithUsage("gpu-1", "pool-a", "100", "10Gi", tfv1.UsedByTensorFusion),
			},
			"node-delete": {
				"gpu-2": gpuWithUsage("gpu-2", "pool-a", "100", "10Gi", tfv1.UsedByTensorFusion),
			},
		},
		map[string]map[types.NamespacedName]struct{}{},
		&fakeNodeInfoLister{
			infos: map[string]fwk.NodeInfo{
				"node-keep": newFrameworkNodeInfo(
					"node-keep",
					nil,
					corev1.ResourceList{
						corev1.ResourceCPU:    resource.MustParse("4"),
						corev1.ResourceMemory: resource.MustParse("16Gi"),
						corev1.ResourcePods:   resource.MustParse("32"),
					},
				),
				"node-delete": newFrameworkNodeInfo(
					"node-delete",
					map[string]string{constants.NodeDeletionMark: constants.TrueStringValue},
					corev1.ResourceList{
						corev1.ResourceCPU:    resource.MustParse("4"),
						corev1.ResourceMemory: resource.MustParse("16Gi"),
						corev1.ResourcePods:   resource.MustParse("32"),
					},
				),
			},
		},
		map[string]struct{}{},
	)
	if err != nil {
		t.Fatalf("build budgets: %v", err)
	}
	if _, ok := budgets["node-delete"]; ok {
		t.Fatalf("expected deletion-marked node to be excluded, got budgets=%v", budgets)
	}
	if _, ok := budgets["node-keep"]; !ok {
		t.Fatalf("expected healthy node to remain in budget, got budgets=%v", budgets)
	}
}

func TestRunDefrag_PersistsLastDefragTimeWhenRunContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	pool := newDefragTestPool()
	r, kubeClient := newDefragControllerTestReconciler(t, pool)
	runStart := time.Now().Add(-time.Minute).Round(time.Second)

	r.runDefrag(ctx, pool.DeepCopy(), runStart)

	updated := &tfv1.GPUPool{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: pool.Name}, updated); err != nil {
		t.Fatalf("get updated pool: %v", err)
	}
	if updated.Status.LastDefragTime == nil {
		t.Fatal("expected LastDefragTime to be persisted")
	}
	if !updated.Status.LastDefragTime.Time.Equal(runStart) {
		t.Fatalf("lastDefragTime=%v want %v", updated.Status.LastDefragTime.Time, runStart)
	}
}

func TestDefragStaleLabelCleanupTick_ScopedToPool(t *testing.T) {
	pool := newDefragTestPool()
	staleSince := time.Now().Add(-2 * time.Hour)

	nodeA := newDefragDrainingNode("node-a", "pool-a", staleSince)
	nodeB := newDefragDrainingNode("node-b", "pool-b", staleSince)

	r, kubeClient := newDefragControllerTestReconciler(t, pool, nodeA, nodeB)
	r.defragStaleLabelCleanupTick(context.Background(), pool)

	updatedA := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: nodeA.Name}, updatedA); err != nil {
		t.Fatalf("get updated node-a: %v", err)
	}
	if _, exists := updatedA.Labels[constants.DefragDrainingLabel]; exists {
		t.Fatalf("expected node-a drain label to be cleared, labels=%v", updatedA.Labels)
	}

	updatedB := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: nodeB.Name}, updatedB); err != nil {
		t.Fatalf("get updated node-b: %v", err)
	}
	if updatedB.Labels[constants.DefragDrainingLabel] != constants.TrueStringValue {
		t.Fatalf("expected node-b drain label to remain, labels=%v", updatedB.Labels)
	}
}

func TestDefragStaleLabelCleanupTick_UsesGPUNodePoolWhenNodePoolLabelMissing(t *testing.T) {
	pool := newDefragTestPool()
	staleSince := time.Now().Add(-2 * time.Hour)

	node := newDefragDrainingNodeWithoutPoolLabel("node-a", staleSince)
	gpuNode := newDefragGPUNode("node-a", "pool-a")

	r, kubeClient := newDefragControllerTestReconciler(t, pool, node, gpuNode)
	r.defragStaleLabelCleanupTick(context.Background(), pool)

	updated := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: node.Name}, updated); err != nil {
		t.Fatalf("get updated node: %v", err)
	}
	if _, exists := updated.Labels[constants.DefragDrainingLabel]; exists {
		t.Fatalf("expected drain label to be cleared via GPUNode pool ownership, labels=%v", updated.Labels)
	}
}

func TestDefragDrainWatcherTick_ScopedToPool(t *testing.T) {
	pool := newDefragTestPool()
	nodeB := newDefragDrainingNode("node-b", "pool-b", time.Now().Add(-10*time.Minute))

	r, kubeClient := newDefragControllerTestReconciler(t, pool, nodeB)
	r.defragDrainingNodes.Store(nodeB.Name, time.Now().Add(-time.Minute))

	r.defragDrainWatcherTick(context.Background(), pool)

	updated := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: nodeB.Name}, updated); err != nil {
		t.Fatalf("get updated node: %v", err)
	}
	if updated.Labels[constants.DefragDrainingLabel] != constants.TrueStringValue {
		t.Fatalf("expected foreign-pool drain label to remain, labels=%v", updated.Labels)
	}
}

func TestDefragDrainWatcherTick_UsesGPUNodePoolWhenNodePoolLabelMissing(t *testing.T) {
	pool := newDefragTestPool()
	node := newDefragDrainingNodeWithoutPoolLabel("node-a", time.Now().Add(-10*time.Minute))
	gpuNode := newDefragGPUNode("node-a", "pool-a")

	r, kubeClient := newDefragControllerTestReconciler(t, pool, node, gpuNode)
	r.defragDrainingNodes.Store(node.Name, time.Now().Add(-time.Minute))

	r.defragDrainWatcherTick(context.Background(), pool)

	updated := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: node.Name}, updated); err != nil {
		t.Fatalf("get updated node: %v", err)
	}
	if _, exists := updated.Labels[constants.DefragDrainingLabel]; exists {
		t.Fatalf("expected drain label to be cleared via GPUNode pool ownership, labels=%v", updated.Labels)
	}
}

func TestDefragDrainingLabelPatch_IncludesAndClearsPoolOwner(t *testing.T) {
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}}
	r, kubeClient := newDefragControllerTestReconciler(t, node)

	if err := r.applyDefragDrainingLabel(context.Background(), node.Name, "pool-a"); err != nil {
		t.Fatalf("apply drain label: %v", err)
	}

	updated := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: node.Name}, updated); err != nil {
		t.Fatalf("get updated node: %v", err)
	}
	if updated.Labels[constants.DefragDrainingLabel] != constants.TrueStringValue {
		t.Fatalf("expected drain label, labels=%v", updated.Labels)
	}
	if updated.Annotations[constants.DefragDrainingPoolAnnotation] != "pool-a" {
		t.Fatalf("expected drain pool annotation, annotations=%v", updated.Annotations)
	}
	if updated.Annotations[constants.DefragDrainingSinceAnnotation] == "" {
		t.Fatalf("expected drain since annotation, annotations=%v", updated.Annotations)
	}

	if err := r.clearDefragDrainingLabel(context.Background(), node.Name); err != nil {
		t.Fatalf("clear drain label: %v", err)
	}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: node.Name}, updated); err != nil {
		t.Fatalf("get cleared node: %v", err)
	}
	if _, exists := updated.Labels[constants.DefragDrainingLabel]; exists {
		t.Fatalf("expected drain label cleared, labels=%v", updated.Labels)
	}
	if _, exists := updated.Annotations[constants.DefragDrainingPoolAnnotation]; exists {
		t.Fatalf("expected drain pool annotation cleared, annotations=%v", updated.Annotations)
	}
	if _, exists := updated.Annotations[constants.DefragDrainingSinceAnnotation]; exists {
		t.Fatalf("expected drain since annotation cleared, annotations=%v", updated.Annotations)
	}
}

func TestWaitForSchedulerNodeDefragLabel(t *testing.T) {
	lister := &fakeNodeInfoLister{
		infos: map[string]fwk.NodeInfo{
			"node-a": newFrameworkNodeInfo(
				"node-a",
				map[string]string{constants.DefragDrainingLabel: constants.TrueStringValue},
				corev1.ResourceList{},
			),
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := waitForSchedulerNodeDefragLabel(ctx, nil, lister, "node-a", time.Millisecond); err != nil {
		t.Fatalf("waitForSchedulerNodeDefragLabel returned error: %v", err)
	}
}

func TestWaitForSchedulerNodeDefragLabel_RefreshesUntilLabelVisible(t *testing.T) {
	lister := &fakeNodeInfoLister{
		infos: map[string]fwk.NodeInfo{
			"node-a": newFrameworkNodeInfo("node-a", map[string]string{}, corev1.ResourceList{}),
		},
	}
	refreshCalls := 0
	refresh := func(context.Context) error {
		refreshCalls++
		if refreshCalls == 2 {
			lister.infos["node-a"] = newFrameworkNodeInfo(
				"node-a",
				map[string]string{constants.DefragDrainingLabel: constants.TrueStringValue},
				corev1.ResourceList{},
			)
		}
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := waitForSchedulerNodeDefragLabel(ctx, refresh, lister, "node-a", time.Millisecond); err != nil {
		t.Fatalf("waitForSchedulerNodeDefragLabel returned error: %v", err)
	}
	if refreshCalls < 2 {
		t.Fatalf("refresh calls=%d want at least 2", refreshCalls)
	}
}

func TestWaitForSchedulerNodeDefragLabel_ReturnsRefreshErrorAfterTimeout(t *testing.T) {
	lister := &fakeNodeInfoLister{
		infos: map[string]fwk.NodeInfo{
			"node-a": newFrameworkNodeInfo("node-a", map[string]string{}, corev1.ResourceList{}),
		},
	}
	refreshErr := errors.New("refresh failed")
	refreshCalls := 0
	refresh := func(context.Context) error {
		refreshCalls++
		return refreshErr
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	err := waitForSchedulerNodeDefragLabel(ctx, refresh, lister, "node-a", time.Millisecond)
	if err == nil {
		t.Fatal("expected refresh timeout error")
	}
	if refreshCalls == 0 {
		t.Fatal("expected refresh to be called")
	}
	got := err.Error()
	if !strings.Contains(got, "refresh scheduler node info snapshot") || !strings.Contains(got, refreshErr.Error()) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestFindNewOrFreshDefragWorker(t *testing.T) {
	runStart := time.Now()
	original := []*corev1.Pod{{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:         "ns",
			Name:              "worker-a",
			CreationTimestamp: metav1.NewTime(runStart.Add(-time.Minute)),
		},
	}}
	current := append([]*corev1.Pod{}, original...)
	current = append(current, &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:         "ns",
			Name:              "worker-b",
			CreationTimestamp: metav1.NewTime(runStart.Add(time.Second)),
		},
	})

	pod, ok := findNewOrFreshDefragWorker(runStart, original, current)
	if !ok {
		t.Fatal("expected new worker to be detected")
	}
	if pod.Namespace != "ns" || pod.Name != "worker-b" {
		t.Fatalf("unexpected worker detected: %s/%s", pod.Namespace, pod.Name)
	}

	if pod, ok := findNewOrFreshDefragWorker(runStart, original, original); ok {
		t.Fatalf("unchanged worker set should not be considered fresh, got %s/%s", pod.Namespace, pod.Name)
	}
}

func newDefragControllerTestReconciler(
	t *testing.T,
	objects ...ctrlclient.Object,
) (*GPUPoolCompactionReconciler, ctrlclient.Client) {
	t.Helper()

	scheme := runtime.NewScheme()
	if err := tfv1.AddToScheme(scheme); err != nil {
		t.Fatalf("add TensorFusion scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core/v1 scheme: %v", err)
	}

	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPUPool{}).
		WithObjects(objects...).
		Build()

	allocator := gpuallocator.NewGpuAllocator(context.Background(), kubeClient, time.Minute)
	allocator.SetAllocatorReady()

	return &GPUPoolCompactionReconciler{
		Client:    kubeClient,
		Scheme:    scheme,
		Recorder:  record.NewFakeRecorder(16),
		Allocator: allocator,
	}, kubeClient
}

func newDefragTestPool() *tfv1.GPUPool {
	return &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: "pool-a"},
		Spec: tfv1.GPUPoolSpec{
			NodeManagerConfig: &tfv1.NodeManagerConfig{
				NodeCompaction: &tfv1.NodeCompaction{
					Defrag: &tfv1.NodeDefragConfig{
						Enabled:     true,
						Schedule:    "0 3 * * *",
						MaxDuration: "30m",
					},
				},
			},
		},
	}
}

func newDefragDrainingNode(name, poolName string, since time.Time) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				constants.DefragDrainingLabel:                                     constants.TrueStringValue,
				fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, poolName): constants.TrueStringValue,
			},
			Annotations: map[string]string{
				constants.DefragDrainingSinceAnnotation: since.Format(time.RFC3339),
			},
		},
	}
}

func newDefragDrainingNodeWithoutPoolLabel(name string, since time.Time) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				constants.DefragDrainingLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				constants.DefragDrainingSinceAnnotation: since.Format(time.RFC3339),
			},
		},
	}
}

func newDefragGPUNode(name, poolName string) *tfv1.GPUNode {
	return &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, poolName): constants.TrueStringValue,
			},
		},
	}
}

type fakeNodeInfoLister struct {
	infos map[string]fwk.NodeInfo
}

func (l *fakeNodeInfoLister) List() ([]fwk.NodeInfo, error) {
	out := make([]fwk.NodeInfo, 0, len(l.infos))
	for _, info := range l.infos {
		out = append(out, info)
	}
	return out, nil
}

func (l *fakeNodeInfoLister) HavePodsWithAffinityList() ([]fwk.NodeInfo, error) {
	return l.List()
}

func (l *fakeNodeInfoLister) HavePodsWithRequiredAntiAffinityList() ([]fwk.NodeInfo, error) {
	return l.List()
}

func (l *fakeNodeInfoLister) Get(nodeName string) (fwk.NodeInfo, error) {
	info, ok := l.infos[nodeName]
	if !ok {
		return nil, fmt.Errorf("node %s not found", nodeName)
	}
	return info, nil
}

func newFrameworkNodeInfo(name string, labels map[string]string, allocatable corev1.ResourceList) fwk.NodeInfo {
	info := framework.NewNodeInfo()
	info.SetNode(&corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Status: corev1.NodeStatus{
			Allocatable: allocatable,
		},
	})
	return info
}

func newSchedulerTestPod(namespace, name, cpu, memory string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name: "main",
				Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						corev1.ResourceCPU:    resource.MustParse(cpu),
						corev1.ResourceMemory: resource.MustParse(memory),
					},
				},
			}},
		},
	}
}

// ---- compile-time sanity check ----------------------------------------
var _ = errors.New
var _ = fmt.Errorf
var _ ctrl.Result
