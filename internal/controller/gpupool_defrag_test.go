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
	policyv1 "k8s.io/api/policy/v1"
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

func TestEvictWorkerPods_MarksEvictedPodLabel(t *testing.T) {
	p1 := newPod("p1")
	fakeClient := clientgofake.NewSimpleClientset(p1)
	fakeClient.PrependReactor("create", "pods", func(a k8stesting.Action) (bool, runtime.Object, error) {
		if a.GetSubresource() != "eviction" {
			return false, nil, nil
		}
		// Keep the pod object around so the controller must patch the
		// defrag marker after a successful eviction request.
		return true, nil, nil
	})
	r := newReconcilerWithFake(fakeClient)
	cand := &defragCandidate{
		nodeName:   "node-a",
		workerPods: []*corev1.Pod{p1},
	}
	stats := &defragRunStats{}

	if !r.evictWorkerPods(context.Background(), &tfv1.GPUPool{}, cand, stats, &errLogger{}) {
		t.Fatalf("expected eviction to succeed")
	}

	updated, err := fakeClient.CoreV1().Pods(p1.Namespace).Get(context.Background(), p1.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get patched pod: %v", err)
	}
	if updated.Labels[constants.DefragEvictedPodLabel] != constants.TrueStringValue {
		t.Fatalf("expected defrag evicted label, labels=%v", updated.Labels)
	}
}

func TestEvictWorkerPods_LabelPatchFailureAborts(t *testing.T) {
	p1 := newPod("p1")
	fakeClient := clientgofake.NewSimpleClientset(p1)
	fakeClient.PrependReactor("create", "pods", func(a k8stesting.Action) (bool, runtime.Object, error) {
		if a.GetSubresource() != "eviction" {
			return false, nil, nil
		}
		return true, nil, nil
	})
	fakeClient.PrependReactor("patch", "pods", func(a k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("patch failed")
	})
	r := newReconcilerWithFake(fakeClient)
	cand := &defragCandidate{
		nodeName:   "node-a",
		workerPods: []*corev1.Pod{p1},
	}
	stats := &defragRunStats{}

	if r.evictWorkerPods(context.Background(), &tfv1.GPUPool{}, cand, stats, &errLogger{}) {
		t.Fatalf("expected label patch failure to abort")
	}
	if stats.EvictionFailures != 1 {
		t.Fatalf("evictionFailures=%d want 1", stats.EvictionFailures)
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
	if len(fakeClient.Actions()) == 0 {
		t.Fatalf("expected eviction attempt to be recorded")
	}
	if el.errCount == 0 {
		t.Fatalf("expected error log entry on failure")
	}
}

func TestCheckDefragPDBPreflight_BlockedPDBSkipsWholeNode(t *testing.T) {
	p1 := newPod("p1")
	p1.Labels = map[string]string{"app": "blocked"}
	p2 := newPod("p2")
	p2.Labels = map[string]string{"app": "free"}

	pdb := &policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns1",
			Name:      "pdb-blocked",
		},
		Spec: policyv1.PodDisruptionBudgetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "blocked"},
			},
		},
		Status: policyv1.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 0,
		},
	}
	fakeClient := clientgofake.NewSimpleClientset(p1, p2, pdb)
	r := newReconcilerWithFake(fakeClient)
	cand := &defragCandidate{
		nodeName:   "node-a",
		workerPods: []*corev1.Pod{p1, p2},
	}

	blocked, reason, err := r.checkDefragPDBPreflight(context.Background(), cand)
	if err != nil {
		t.Fatalf("check PDB preflight: %v", err)
	}
	if !blocked {
		t.Fatalf("expected PDB to block node")
	}
	if !strings.Contains(reason, "pdb-blocked") || !strings.Contains(reason, "p1") {
		t.Fatalf("reason should identify blocking PDB and pod, got %q", reason)
	}
}

func TestCheckDefragPDBPreflight_AllowsNodeWhenDisruptionsAvailable(t *testing.T) {
	p1 := newPod("p1")
	p1.Labels = map[string]string{"app": "allowed"}

	pdb := &policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns1",
			Name:      "pdb-allowed",
		},
		Spec: policyv1.PodDisruptionBudgetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "allowed"},
			},
		},
		Status: policyv1.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 1,
		},
	}
	fakeClient := clientgofake.NewSimpleClientset(p1, pdb)
	r := newReconcilerWithFake(fakeClient)
	cand := &defragCandidate{
		nodeName:   "node-a",
		workerPods: []*corev1.Pod{p1},
	}

	blocked, reason, err := r.checkDefragPDBPreflight(context.Background(), cand)
	if err != nil {
		t.Fatalf("check PDB preflight: %v", err)
	}
	if blocked {
		t.Fatalf("expected PDB to allow node, reason=%q", reason)
	}
}

func TestCheckDefragPDBPreflight_ListsPDBsOncePerNamespace(t *testing.T) {
	p1 := newPod("p1")
	p1.Labels = map[string]string{"app": "allowed"}
	p2 := newPod("p2")
	p2.Labels = map[string]string{"app": "allowed"}

	pdb := &policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns1",
			Name:      "pdb-allowed",
		},
		Spec: policyv1.PodDisruptionBudgetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "allowed"},
			},
		},
		Status: policyv1.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 2,
		},
	}
	directClient := clientgofake.NewSimpleClientset(p1, p2, pdb)
	r := newReconcilerWithFake(directClient)
	cand := &defragCandidate{
		nodeName:   "node-a",
		workerPods: []*corev1.Pod{p1, p2},
	}

	blocked, reason, err := r.checkDefragPDBPreflight(context.Background(), cand)
	if err != nil {
		t.Fatalf("check PDB preflight: %v", err)
	}
	if blocked {
		t.Fatalf("expected PDB to allow node, reason=%q", reason)
	}
	listActions := 0
	for _, action := range directClient.Actions() {
		if action.GetVerb() == "list" && action.GetResource().Resource == "poddisruptionbudgets" {
			listActions++
		}
	}
	if listActions != 1 {
		t.Fatalf("expected one PDB list for namespace, got %d actions=%v", listActions, directClient.Actions())
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

func TestCollectDefragCandidates_SkipsFreshWorkerPods(t *testing.T) {
	now := time.Now()
	pool := newDefragTestPool()
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}}
	gpuNode := newDefragGPUNode("node-a", "pool-a")
	freshWorker := newDefragWorkerPod("worker-fresh", "node-a", now.Add(-10*time.Minute))
	objects := []ctrlclient.Object{pool, node, gpuNode, freshWorker}
	objects = append(objects, newDefragNodeGPUs("node-a", "pool-a")...)

	r, _ := newDefragControllerTestReconciler(t, objects...)
	if err := r.Allocator.InitGPUAndQuotaStore(); err != nil {
		t.Fatalf("init GPU store: %v", err)
	}
	r.Allocator.ReconcileAllocationStateForTesting()

	candidates, err := r.collectDefragCandidates(context.Background(), pool)
	if err != nil {
		t.Fatalf("collect defrag candidates: %v", err)
	}
	if len(candidates) != 0 {
		t.Fatalf("fresh worker pod should make node ineligible, got %d candidates", len(candidates))
	}
}

func TestCollectDefragCandidates_IncludesOldWorkerPods(t *testing.T) {
	now := time.Now()
	pool := newDefragTestPool()
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}}
	gpuNode := newDefragGPUNode("node-a", "pool-a")
	oldWorker := newDefragWorkerPod("worker-old", "node-a", now.Add(-31*time.Minute))
	objects := []ctrlclient.Object{pool, node, gpuNode, oldWorker}
	objects = append(objects, newDefragNodeGPUs("node-a", "pool-a")...)

	r, _ := newDefragControllerTestReconciler(t, objects...)
	if err := r.Allocator.InitGPUAndQuotaStore(); err != nil {
		t.Fatalf("init GPU store: %v", err)
	}
	r.Allocator.ReconcileAllocationStateForTesting()

	candidates, err := r.collectDefragCandidates(context.Background(), pool)
	if err != nil {
		t.Fatalf("collect defrag candidates: %v", err)
	}
	if len(candidates) != 1 {
		t.Fatalf("old worker pod should be eligible, got %d candidates", len(candidates))
	}
	if candidates[0].nodeName != "node-a" {
		t.Fatalf("candidate node=%q want node-a", candidates[0].nodeName)
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
	budgets := buildDefragNodeBudgets(
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
	)
	if _, ok := budgets["node-delete"]; ok {
		t.Fatalf("expected deletion-marked node to be excluded, got budgets=%v", budgets)
	}
	if _, ok := budgets["node-keep"]; !ok {
		t.Fatalf("expected healthy node to remain in budget, got budgets=%v", budgets)
	}
}

func TestBuildDefragNodeBudgets_SkipsMissingSchedulerNodeInfo(t *testing.T) {
	budgets := buildDefragNodeBudgets(
		"pool-a",
		"source-node",
		map[string]map[string]*tfv1.GPU{
			"source-node": {
				"gpu-source": gpuWithUsage("gpu-source", "pool-a", "50", "5Gi", tfv1.UsedByTensorFusion),
			},
			"node-keep": {
				"gpu-keep": gpuWithUsage("gpu-keep", "pool-a", "100", "10Gi", tfv1.UsedByTensorFusion),
			},
			"node-stale": {
				"gpu-stale": gpuWithUsage("gpu-stale", "pool-a", "100", "10Gi", tfv1.UsedByTensorFusion),
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
			},
		},
	)
	if _, ok := budgets["node-stale"]; ok {
		t.Fatalf("stale node should be excluded, got budgets=%v", budgets)
	}
	if _, ok := budgets["node-keep"]; !ok {
		t.Fatalf("healthy target node should remain, got budgets=%v", budgets)
	}
}

func TestRunDefragStep_NoCandidatesFinishesCampaign(t *testing.T) {
	pool := newDefragTestPool()
	r, _ := newDefragControllerTestReconciler(t, pool)

	result := r.runDefragStep(context.Background(), pool.DeepCopy(), time.Now().Add(-time.Minute))

	if !result.finishCampaign {
		t.Fatalf("expected empty defrag step to finish campaign")
	}
	if result.evictedNode {
		t.Fatalf("empty defrag step should not evict a node")
	}
}

func TestRunDefragStep_BlockedByDefragEvictedPod(t *testing.T) {
	pool := newDefragTestPool()
	blocker := newPod("evicted-worker")
	blocker.Labels = map[string]string{
		constants.DefragEvictedPodLabel: constants.TrueStringValue,
	}
	blocker.Annotations = map[string]string{
		constants.DefragEvictedPodPoolAnnotation:  pool.Name,
		constants.DefragEvictedPodSinceAnnotation: time.Now().Format(time.RFC3339),
	}
	r, _ := newDefragControllerTestReconciler(t, pool, blocker)

	result := r.runDefragStep(context.Background(), pool.DeepCopy(), time.Now().Add(-time.Minute))

	if result.finishCampaign {
		t.Fatalf("defrag-evicted pod should pause, not finish campaign")
	}
	if result.evictedNode {
		t.Fatalf("blocked defrag step should not evict a node")
	}
}

func TestHasDefragEvictedPods_ScopedToPool(t *testing.T) {
	pool := newDefragTestPool()
	otherPoolPod := newPod("other-pool-evicted-worker")
	otherPoolPod.Labels = map[string]string{
		constants.DefragEvictedPodLabel: constants.TrueStringValue,
	}
	otherPoolPod.Annotations = map[string]string{
		constants.DefragEvictedPodPoolAnnotation:  "pool-b",
		constants.DefragEvictedPodSinceAnnotation: time.Now().Format(time.RFC3339),
	}
	samePoolPod := newPod("same-pool-evicted-worker")
	samePoolPod.Labels = map[string]string{
		constants.DefragEvictedPodLabel: constants.TrueStringValue,
	}
	samePoolPod.Annotations = map[string]string{
		constants.DefragEvictedPodPoolAnnotation:  pool.Name,
		constants.DefragEvictedPodSinceAnnotation: time.Now().Format(time.RFC3339),
	}

	rOther, _ := newDefragControllerTestReconciler(t, pool, otherPoolPod)
	blocked, err := rOther.hasDefragEvictedPods(context.Background(), pool)
	if err != nil {
		t.Fatalf("check evicted pods for other pool: %v", err)
	}
	if blocked {
		t.Fatalf("evicted pod from another pool must not block pool %s", pool.Name)
	}

	rSame, _ := newDefragControllerTestReconciler(t, pool, samePoolPod)
	blocked, err = rSame.hasDefragEvictedPods(context.Background(), pool)
	if err != nil {
		t.Fatalf("check evicted pods for same pool: %v", err)
	}
	if !blocked {
		t.Fatalf("evicted pod from same pool should block the next defrag step")
	}
}

func TestHasDefragEvictedPods_ClearsStaleSamePoolMarker(t *testing.T) {
	pool := newDefragTestPool()
	stalePod := newPod("stale-evicted-worker")
	stalePod.Labels = map[string]string{
		constants.DefragEvictedPodLabel: constants.TrueStringValue,
	}
	stalePod.Annotations = map[string]string{
		constants.DefragEvictedPodPoolAnnotation:  pool.Name,
		constants.DefragEvictedPodSinceAnnotation: time.Now().Add(-2 * time.Hour).Format(time.RFC3339),
	}

	r, kubeClient := newDefragControllerTestReconciler(t, pool, stalePod)
	blocked, err := r.hasDefragEvictedPods(context.Background(), pool)
	if err != nil {
		t.Fatalf("check stale evicted pod marker: %v", err)
	}
	if blocked {
		t.Fatalf("stale evicted pod marker should not block defrag")
	}

	updated := &corev1.Pod{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Namespace: stalePod.Namespace, Name: stalePod.Name}, updated); err != nil {
		t.Fatalf("get stale pod after cleanup: %v", err)
	}
	if updated.Labels[constants.DefragEvictedPodLabel] == constants.TrueStringValue {
		t.Fatalf("stale defrag-evicted label should be cleared, labels=%v", updated.Labels)
	}
}

func TestDefragRequeueAfter_ContinuesActiveCampaignQuickly(t *testing.T) {
	result := defragStepResult{evictedNode: true}
	got := defragRequeueAfter(result, 30*time.Minute)
	if got <= 0 || got >= 30*time.Minute {
		t.Fatalf("active defrag campaign should requeue quickly, got %s", got)
	}

	if got := defragRequeueAfter(defragStepResult{finishCampaign: true}, 30*time.Minute); got != 30*time.Minute {
		t.Fatalf("finished campaign should use normal compaction period, got %s", got)
	}
}

func TestRunDefragCandidateLoop_StopsAfterFirstEvictedNode(t *testing.T) {
	candidates := []*defragCandidate{
		{nodeName: "node-a"},
		{nodeName: "node-b"},
	}
	stats := &defragRunStats{}
	visited := []string{}

	result := runDefragCandidateLoop(context.Background(), candidates, stats, func(cand *defragCandidate) defragCandidateOutcome {
		visited = append(visited, cand.nodeName)
		return defragCandidateEvicted
	})

	if !result.evictedNode {
		t.Fatalf("expected one node to be evicted")
	}
	if result.finishCampaign {
		t.Fatalf("evicting one node should not finish the campaign")
	}
	if stats.ProcessedNodes != 1 {
		t.Fatalf("processedNodes=%d want 1", stats.ProcessedNodes)
	}
	if len(visited) != 1 || visited[0] != "node-a" {
		t.Fatalf("visited=%v want only node-a", visited)
	}
}

func TestRunDefragCandidateLoop_SkipsAndContinuesToNextCandidate(t *testing.T) {
	candidates := []*defragCandidate{
		{nodeName: "node-a"},
		{nodeName: "node-b"},
	}
	stats := &defragRunStats{}
	visited := []string{}

	result := runDefragCandidateLoop(context.Background(), candidates, stats, func(cand *defragCandidate) defragCandidateOutcome {
		visited = append(visited, cand.nodeName)
		if cand.nodeName == "node-a" {
			return defragCandidateSkipped
		}
		return defragCandidateEvicted
	})

	if !result.evictedNode {
		t.Fatalf("expected second candidate to be evicted")
	}
	if stats.ProcessedNodes != 2 {
		t.Fatalf("processedNodes=%d want 2", stats.ProcessedNodes)
	}
	if fmt.Sprint(visited) != "[node-a node-b]" {
		t.Fatalf("visited=%v", visited)
	}
}

func TestFindFreshDefragWorker(t *testing.T) {
	now := time.Now()
	fresh := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Namespace:         "ns",
		Name:              "worker-fresh",
		CreationTimestamp: metav1.NewTime(now.Add(-time.Minute)),
	}}
	old := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Namespace:         "ns",
		Name:              "worker-old",
		CreationTimestamp: metav1.NewTime(now.Add(-time.Hour)),
	}}

	got, ok := findFreshDefragWorker(now, 30*time.Minute, []*corev1.Pod{old, fresh})
	if !ok {
		t.Fatal("expected fresh worker to be detected")
	}
	if got.Name != fresh.Name {
		t.Fatalf("got %s want %s", got.Name, fresh.Name)
	}

	if pod, ok := findFreshDefragWorker(now, 30*time.Minute, []*corev1.Pod{old}); ok {
		t.Fatalf("old worker should not be fresh, got %s/%s", pod.Namespace, pod.Name)
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
						Enabled:                     true,
						Schedule:                    "0 3 * * *",
						MaxDuration:                 "30m",
						UtilizationThresholdPercent: 40,
					},
				},
			},
		},
	}
}

func newDefragWorkerPod(name, nodeName string, createdAt time.Time) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:         "ns1",
			Name:              name,
			UID:               types.UID(name + "-uid"),
			CreationTimestamp: metav1.NewTime(createdAt),
			Labels: map[string]string{
				constants.LabelComponent: constants.ComponentWorker,
				constants.WorkloadKey:    "workload-a",
			},
			Annotations: map[string]string{
				constants.GpuPoolKey:              "pool-a",
				constants.GpuCountAnnotation:      "1",
				constants.TFLOPSRequestAnnotation: "10",
				constants.VRAMRequestAnnotation:   "1Gi",
				constants.TFLOPSLimitAnnotation:   "10",
				constants.VRAMLimitAnnotation:     "1Gi",
				constants.GPUDeviceIDsAnnotation:  fmt.Sprintf("%s-gpu-0", nodeName),
			},
		},
		Spec: corev1.PodSpec{
			NodeName: nodeName,
			Containers: []corev1.Container{{
				Name:  "main",
				Image: "worker:test",
			}},
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning},
	}
}

func newDefragNodeGPUs(nodeName, poolName string) []ctrlclient.Object {
	out := make([]ctrlclient.Object, 0, 4)
	for i := 0; i < 4; i++ {
		availableTflops := "100"
		if i == 0 {
			availableTflops = "50"
		}
		out = append(out, gpuWithUsageOnNode(
			fmt.Sprintf("%s-gpu-%d", nodeName, i),
			poolName,
			nodeName,
			availableTflops,
			"10Gi",
		))
	}
	return out
}

func gpuWithUsageOnNode(name, poolName, nodeName, avTflops, avVram string) *tfv1.GPU {
	g := gpuWithUsage(name, poolName, avTflops, avVram, tfv1.UsedByTensorFusion)
	g.Status.NodeSelector = map[string]string{
		constants.KubernetesHostNameLabel: nodeName,
	}
	return g
}

func newDefragGPUNode(name, poolName string) *tfv1.GPUNode {
	return &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, poolName): constants.TrueStringValue,
			},
		},
		Status: tfv1.GPUNodeStatus{
			Phase: tfv1.TensorFusionGPUNodePhaseRunning,
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
