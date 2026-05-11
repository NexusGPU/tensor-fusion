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
	"github.com/robfig/cron/v3"
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

func TestSelectDefragCampaign_StaleAnchorUsesCurrentWindow(t *testing.T) {
	parser := cron.NewParser(cron.Minute | cron.Hour | cron.Dom | cron.Month | cron.Dow)
	schedule, err := parser.Parse("* * * * *")
	if err != nil {
		t.Fatalf("parse schedule: %v", err)
	}
	now := time.Now()
	staleAnchor := now.Add(-24 * time.Hour)

	decision := selectDefragCampaign(schedule, staleAnchor, now, 10*time.Minute)
	if !decision.due {
		t.Fatalf("expected a due current-window campaign, got %+v", decision)
	}
	if decision.skipMissed {
		t.Fatalf("expected current-window campaign, got missed-window skip %+v", decision)
	}
	if decision.start.Before(now.Add(-2 * time.Minute)) {
		t.Fatalf("campaign advanced only one stale tick, got %s from stale %s", decision.start, staleAnchor)
	}
	if decision.start.After(time.Now()) {
		t.Fatalf("campaign must not advance into the future, got %s", decision.start)
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

const (
	testDefragNodeA         = "node-a"
	testEvictionSubresource = "eviction"

	testDefragSourceNodeSinceAnnotation = constants.Domain + "/defrag-source-since"
	testDefragSourceNodePoolAnnotation  = constants.Domain + "/defrag-source-pool"
)

type errLogger struct{ infoCount, errCount int }

func (l *errLogger) Info(_ string, _ ...any)           { l.infoCount++ }
func (l *errLogger) Error(_ error, _ string, _ ...any) { l.errCount++ }

func newPod(name string) *corev1.Pod {
	return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: name}}
}

func newReconcilerWithFake(client *clientgofake.Clientset) *GPUPoolCompactionReconciler {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(&corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: testDefragNodeA}}).
		Build()
	return &GPUPoolCompactionReconciler{
		Client:     kubeClient,
		Scheme:     scheme,
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
		nodeName:   testDefragNodeA,
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
		if a.GetSubresource() != testEvictionSubresource {
			return false, nil, nil
		}
		// Keep the pod object around so the controller must patch the
		// defrag marker after a successful eviction request.
		return true, nil, nil
	})
	r := newReconcilerWithFake(fakeClient)
	cand := &defragCandidate{
		nodeName:   testDefragNodeA,
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

func TestEvictWorkerPods_MarksSourceNodeBeforeEviction(t *testing.T) {
	p1 := newPod("p1")
	fakeClient := clientgofake.NewSimpleClientset(p1)
	fakeClient.PrependReactor("create", "pods", func(a k8stesting.Action) (bool, runtime.Object, error) {
		if a.GetSubresource() != testEvictionSubresource {
			return false, nil, nil
		}
		return true, nil, nil
	})
	pool := newDefragTestPool()
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: testDefragNodeA}}
	r, kubeClient := newDefragControllerTestReconciler(t, pool, node)
	r.KubeClient = fakeClient
	cand := &defragCandidate{
		nodeName:   testDefragNodeA,
		workerPods: []*corev1.Pod{p1},
	}
	stats := &defragRunStats{}

	if !r.evictWorkerPods(context.Background(), pool, cand, stats, &errLogger{}) {
		t.Fatalf("expected eviction to succeed")
	}

	updated := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: testDefragNodeA}, updated); err != nil {
		t.Fatalf("get source node: %v", err)
	}
	if updated.Labels[constants.DefragSourceNodeLabel] != constants.TrueStringValue {
		t.Fatalf("expected source node defrag label, labels=%v", updated.Labels)
	}
	if updated.Annotations[testDefragSourceNodePoolAnnotation] != pool.Name {
		t.Fatalf("expected source node pool annotation %q=%q, annotations=%v",
			testDefragSourceNodePoolAnnotation, pool.Name, updated.Annotations)
	}
	rawSince := updated.Annotations[testDefragSourceNodeSinceAnnotation]
	if rawSince == "" {
		t.Fatalf("expected source node since annotation %q, annotations=%v",
			testDefragSourceNodeSinceAnnotation, updated.Annotations)
	}
	if _, err := time.Parse(time.RFC3339, rawSince); err != nil {
		t.Fatalf("source node since annotation should be RFC3339, got %q: %v", rawSince, err)
	}
}

func TestEvictWorkerPods_LabelPatchFailureAborts(t *testing.T) {
	p1 := newPod("p1")
	fakeClient := clientgofake.NewSimpleClientset(p1)
	fakeClient.PrependReactor("create", "pods", func(a k8stesting.Action) (bool, runtime.Object, error) {
		if a.GetSubresource() != testEvictionSubresource {
			return false, nil, nil
		}
		return true, nil, nil
	})
	fakeClient.PrependReactor("patch", "pods", func(a k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("patch failed")
	})
	r := newReconcilerWithFake(fakeClient)
	cand := &defragCandidate{
		nodeName:   testDefragNodeA,
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
		if a.GetSubresource() != testEvictionSubresource {
			return false, nil, nil
		}
		attempts++
		return true, nil, apierrors.NewTooManyRequestsError("simulated PDB contention")
	})
	r := newReconcilerWithFake(fakeClient)
	cand := &defragCandidate{
		nodeName:   testDefragNodeA,
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
		nodeName:   testDefragNodeA,
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
		nodeName:   testDefragNodeA,
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
		nodeName:   testDefragNodeA,
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
		if a.GetSubresource() != testEvictionSubresource {
			return false, nil, nil
		}
		return true, nil, apierrors.NewNotFound(schema.GroupResource{Resource: "pods"}, "gone")
	})
	r := newReconcilerWithFake(fakeClient)
	cand := &defragCandidate{
		nodeName:   testDefragNodeA,
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
		nodeName:   testDefragNodeA,
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
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: testDefragNodeA}}
	gpuNode := newDefragGPUNode(testDefragNodeA)
	freshWorker := newDefragWorkerPod("worker-fresh", testDefragNodeA, now.Add(-10*time.Minute))
	objects := []ctrlclient.Object{pool, node, gpuNode, freshWorker}
	objects = append(objects, newDefragNodeGPUs(testDefragNodeA)...)

	r, _ := newDefragControllerTestReconciler(t, objects...)
	if err := r.Allocator.InitGPUAndQuotaStore(); err != nil {
		t.Fatalf("init GPU store: %v", err)
	}
	r.Allocator.ReconcileAllocationStateForTesting()

	candidates, err := r.collectDefragCandidates(context.Background(), pool, &defragRunStats{})
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
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: testDefragNodeA}}
	gpuNode := newDefragGPUNode(testDefragNodeA)
	oldWorker := newDefragWorkerPod("worker-old", testDefragNodeA, now.Add(-31*time.Minute))
	objects := []ctrlclient.Object{pool, node, gpuNode, oldWorker}
	objects = append(objects, newDefragNodeGPUs(testDefragNodeA)...)

	r, _ := newDefragControllerTestReconciler(t, objects...)
	r.KubeClient = clientgofake.NewSimpleClientset(newDefragWorkerPDB("pdb-worker", "ns1"))
	if err := r.Allocator.InitGPUAndQuotaStore(); err != nil {
		t.Fatalf("init GPU store: %v", err)
	}
	r.Allocator.ReconcileAllocationStateForTesting()

	candidates, err := r.collectDefragCandidates(context.Background(), pool, &defragRunStats{})
	if err != nil {
		t.Fatalf("collect defrag candidates: %v", err)
	}
	if len(candidates) != 1 {
		t.Fatalf("old worker pod should be eligible, got %d candidates", len(candidates))
	}
	if candidates[0].nodeName != testDefragNodeA {
		t.Fatalf("candidate node=%q want node-a", candidates[0].nodeName)
	}
}

func TestCollectDefragCandidates_SkipsNodeWithMissingPDB(t *testing.T) {
	now := time.Now()
	pool := newDefragTestPool()
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: testDefragNodeA}}
	gpuNode := newDefragGPUNode(testDefragNodeA)
	oldWorker := newDefragWorkerPod("worker-old", testDefragNodeA, now.Add(-2*time.Hour))
	objects := []ctrlclient.Object{pool, node, gpuNode, oldWorker}
	objects = append(objects, newDefragNodeGPUs(testDefragNodeA)...)

	r, _ := newDefragControllerTestReconciler(t, objects...)
	// Default helper already injects an empty fake clientset. With no PDB
	// in ns1, the worker is uncovered so the node must be filtered out.
	if err := r.Allocator.InitGPUAndQuotaStore(); err != nil {
		t.Fatalf("init GPU store: %v", err)
	}
	r.Allocator.ReconcileAllocationStateForTesting()

	stats := &defragRunStats{}
	candidates, err := r.collectDefragCandidates(context.Background(), pool, stats)
	if err != nil {
		t.Fatalf("collect defrag candidates: %v", err)
	}
	if len(candidates) != 0 {
		t.Fatalf("node with uncovered worker pod must not be a candidate, got %d", len(candidates))
	}
	if stats.MissingPDBNodes != 1 {
		t.Fatalf("MissingPDBNodes=%d want 1", stats.MissingPDBNodes)
	}
	assertWarningEvent(t, r.Recorder, defragEventSkipMissingPDB, "no PodDisruptionBudget")
}

func TestCollectDefragCandidates_SkipsNodeWhenPDBSelectorIsEmpty(t *testing.T) {
	now := time.Now()
	pool := newDefragTestPool()
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: testDefragNodeA}}
	gpuNode := newDefragGPUNode(testDefragNodeA)
	oldWorker := newDefragWorkerPod("worker-old", testDefragNodeA, now.Add(-2*time.Hour))
	emptySelectorPDB := &policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns1",
			Name:      "pdb-empty",
		},
		Spec: policyv1.PodDisruptionBudgetSpec{
			// Empty selector intentionally: must NOT count as coverage.
			Selector: &metav1.LabelSelector{},
		},
		Status: policyv1.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 1,
		},
	}
	objects := []ctrlclient.Object{pool, node, gpuNode, oldWorker}
	objects = append(objects, newDefragNodeGPUs(testDefragNodeA)...)

	r, _ := newDefragControllerTestReconciler(t, objects...)
	r.KubeClient = clientgofake.NewSimpleClientset(emptySelectorPDB)
	if err := r.Allocator.InitGPUAndQuotaStore(); err != nil {
		t.Fatalf("init GPU store: %v", err)
	}
	r.Allocator.ReconcileAllocationStateForTesting()

	stats := &defragRunStats{}
	candidates, err := r.collectDefragCandidates(context.Background(), pool, stats)
	if err != nil {
		t.Fatalf("collect defrag candidates: %v", err)
	}
	if len(candidates) != 0 {
		t.Fatalf("empty-selector PDB must not count as coverage, got %d candidates", len(candidates))
	}
	if stats.MissingPDBNodes != 1 {
		t.Fatalf("MissingPDBNodes=%d want 1", stats.MissingPDBNodes)
	}
	assertWarningEvent(t, r.Recorder, defragEventSkipMissingPDB, "no PodDisruptionBudget")
}

func TestCollectDefragCandidates_AllowsNodeCoveredByPDB(t *testing.T) {
	now := time.Now()
	pool := newDefragTestPool()
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: testDefragNodeA}}
	gpuNode := newDefragGPUNode(testDefragNodeA)
	oldWorker := newDefragWorkerPod("worker-old", testDefragNodeA, now.Add(-2*time.Hour))
	objects := []ctrlclient.Object{pool, node, gpuNode, oldWorker}
	objects = append(objects, newDefragNodeGPUs(testDefragNodeA)...)

	r, _ := newDefragControllerTestReconciler(t, objects...)
	r.KubeClient = clientgofake.NewSimpleClientset(newDefragWorkerPDB("pdb-worker", "ns1"))
	if err := r.Allocator.InitGPUAndQuotaStore(); err != nil {
		t.Fatalf("init GPU store: %v", err)
	}
	r.Allocator.ReconcileAllocationStateForTesting()

	stats := &defragRunStats{}
	candidates, err := r.collectDefragCandidates(context.Background(), pool, stats)
	if err != nil {
		t.Fatalf("collect defrag candidates: %v", err)
	}
	if len(candidates) != 1 || candidates[0].nodeName != testDefragNodeA {
		t.Fatalf("node covered by PDB should be a candidate, got %+v", candidates)
	}
	if stats.MissingPDBNodes != 0 {
		t.Fatalf("MissingPDBNodes=%d want 0", stats.MissingPDBNodes)
	}
	assertNoEvent(t, r.Recorder, defragEventSkipMissingPDB)
}

func TestCloneAsUnscheduledWorker_ClearsAssignedGPUState(t *testing.T) {
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

	got := cloneAsUnscheduledWorker(pod)

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

func TestFitsVirtualNodeAllocatable_RejectsOversubscribedCPU(t *testing.T) {
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
	secondPodInfo, _ := framework.NewPodInfo(secondPod)
	virtual := cloneNodeInfoWithVirtualPod(nodeInfo, secondPodInfo)
	if fitsVirtualNodeAllocatable(virtual) {
		t.Fatal("expected second pod to be rejected once virtual CPU budget is exhausted")
	}
}

func TestBuildDefragNodeBudgets_SkipsDeletionMarkedNodes(t *testing.T) {
	budgets := buildDefragNodeBudgets(
		"pool-a",
		"",
		map[string]map[string]*tfv1.GPU{
			// Partially-used GPUs so the empty-target gate keeps the node;
			// this test is about node-label filtering, not the empty gate.
			"node-keep": {
				"gpu-1": gpuWithUsage("gpu-1", "pool-a", "60", "6Gi", tfv1.UsedByTensorFusion),
			},
			"node-delete": {
				"gpu-2": gpuWithUsage("gpu-2", "pool-a", "60", "6Gi", tfv1.UsedByTensorFusion),
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
			// Partially-used GPUs so the empty-target gate keeps these
			// nodes; this test is about missing scheduler-cache nodes.
			"node-keep": {
				"gpu-keep": gpuWithUsage("gpu-keep", "pool-a", "70", "7Gi", tfv1.UsedByTensorFusion),
			},
			"node-stale": {
				"gpu-stale": gpuWithUsage("gpu-stale", "pool-a", "70", "7Gi", tfv1.UsedByTensorFusion),
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

func TestEvaluateDefragGuards_BlockedByActiveEvictedPod(t *testing.T) {
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

	guard := r.evaluateDefragGuards(context.Background(), pool)
	if guard.err != nil {
		t.Fatalf("guard returned error: %v", guard.err)
	}
	if !guard.blockedByEvictedPod {
		t.Fatalf("expected blockedByEvictedPod=true, got %+v", guard)
	}
	if guard.blockedBySourceNode {
		t.Fatalf("evicted-pod block should be reported on its own, got %+v", guard)
	}
}

func TestHasActiveDefragEvictedPods_ScopedToPool(t *testing.T) {
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
	blocked, err := rOther.hasActiveDefragEvictedPods(context.Background(), pool)
	if err != nil {
		t.Fatalf("check evicted pods for other pool: %v", err)
	}
	if blocked {
		t.Fatalf("evicted pod from another pool must not block pool %s", pool.Name)
	}

	rSame, _ := newDefragControllerTestReconciler(t, pool, samePoolPod)
	blocked, err = rSame.hasActiveDefragEvictedPods(context.Background(), pool)
	if err != nil {
		t.Fatalf("check evicted pods for same pool: %v", err)
	}
	if !blocked {
		t.Fatalf("evicted pod from same pool should block the next defrag step")
	}
}

func TestSweepStaleDefragEvictedPodMarkers_ClearsStaleSamePoolMarker(t *testing.T) {
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
	if err := r.sweepStaleDefragEvictedPodMarkers(context.Background(), pool); err != nil {
		t.Fatalf("sweep stale evicted pod markers: %v", err)
	}

	updated := &corev1.Pod{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Namespace: stalePod.Namespace, Name: stalePod.Name}, updated); err != nil {
		t.Fatalf("get stale pod after cleanup: %v", err)
	}
	if updated.Labels[constants.DefragEvictedPodLabel] == constants.TrueStringValue {
		t.Fatalf("stale defrag-evicted label should be cleared, labels=%v", updated.Labels)
	}

	// Guard now has no reason to block: stale marker was cleaned.
	guard := r.evaluateDefragGuards(context.Background(), pool)
	if guard.err != nil {
		t.Fatalf("guard error after cleanup: %v", guard.err)
	}
	if guard.blockedByEvictedPod || guard.blockedBySourceNode {
		t.Fatalf("guard should be clear after sweeping stale markers, got %+v", guard)
	}
}

func TestSafetySweep_ClearsStaleDefragSourceNodeMarker(t *testing.T) {
	pool := newDefragTestPool()
	staleNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testDefragNodeA,
			Labels: map[string]string{
				constants.DefragSourceNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				testDefragSourceNodePoolAnnotation:  pool.Name,
				testDefragSourceNodeSinceAnnotation: time.Now().Add(-2 * time.Hour).Format(time.RFC3339),
			},
		},
	}
	r, kubeClient := newDefragControllerTestReconciler(t, pool, staleNode)

	r.runDefragSafetySweep(context.Background(), pool)

	updated := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: testDefragNodeA}, updated); err != nil {
		t.Fatalf("get stale source node after cleanup: %v", err)
	}
	if updated.Labels[constants.DefragSourceNodeLabel] == constants.TrueStringValue {
		t.Fatalf("stale defrag-source label should be cleared, labels=%v", updated.Labels)
	}
	if updated.Annotations[testDefragSourceNodePoolAnnotation] != "" ||
		updated.Annotations[testDefragSourceNodeSinceAnnotation] != "" {
		t.Fatalf("stale defrag-source annotations should be cleared, annotations=%v", updated.Annotations)
	}

	guard := r.evaluateDefragGuards(context.Background(), pool)
	if guard.err != nil {
		t.Fatalf("guard error after cleanup: %v", guard.err)
	}
	if guard.blockedBySourceNode {
		t.Fatalf("guard must not stay blocked once stale source marker is gone: %+v", guard)
	}
}

func TestSafetySweep_ClearsDefragSourceNodeMarkerOnceDrained(t *testing.T) {
	// Happy path: source node still has the since-annotation well within the
	// safety TTL, but the workers are already gone. The marker should clear
	// on the next safety sweep rather than wait out the TTL.
	pool := newDefragTestPool()
	freshSourceNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testDefragNodeA,
			Labels: map[string]string{
				constants.DefragSourceNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				testDefragSourceNodePoolAnnotation:  pool.Name,
				testDefragSourceNodeSinceAnnotation: time.Now().Add(-5 * time.Minute).Format(time.RFC3339),
			},
		},
	}
	r, kubeClient := newDefragControllerTestReconciler(t, pool, freshSourceNode)

	r.runDefragSafetySweep(context.Background(), pool)

	updated := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: testDefragNodeA}, updated); err != nil {
		t.Fatalf("get source node after cleanup: %v", err)
	}
	if updated.Labels[constants.DefragSourceNodeLabel] == constants.TrueStringValue {
		t.Fatalf("drained defrag-source label should be cleared, labels=%v", updated.Labels)
	}
	if updated.Annotations[testDefragSourceNodePoolAnnotation] != "" ||
		updated.Annotations[testDefragSourceNodeSinceAnnotation] != "" {
		t.Fatalf("drained defrag-source annotations should be cleared, annotations=%v", updated.Annotations)
	}
}

func TestSafetySweepAndGuards_KeepsDefragSourceNodeMarkerWithActiveWorker(t *testing.T) {
	// Workers still on the node -- the safety sweep must keep the marker so
	// we don't release the node back as a target mid-campaign, and the
	// guard must report blockedBySourceNode for the same pool.
	pool := newDefragTestPool()
	freshSourceNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testDefragNodeA,
			Labels: map[string]string{
				constants.DefragSourceNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				testDefragSourceNodePoolAnnotation:  pool.Name,
				testDefragSourceNodeSinceAnnotation: time.Now().Add(-5 * time.Minute).Format(time.RFC3339),
			},
		},
	}
	activeWorker := newDefragWorkerPod("active-worker", testDefragNodeA, time.Now().Add(-time.Hour))
	r, kubeClient := newDefragControllerTestReconciler(t, pool, freshSourceNode, activeWorker)
	primeAllocatorWorkerStore(t, r)

	r.runDefragSafetySweep(context.Background(), pool)

	updated := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: testDefragNodeA}, updated); err != nil {
		t.Fatalf("get source node after cleanup: %v", err)
	}
	if updated.Labels[constants.DefragSourceNodeLabel] != constants.TrueStringValue {
		t.Fatalf("defrag-source label with active worker should be kept, labels=%v", updated.Labels)
	}

	guard := r.evaluateDefragGuards(context.Background(), pool)
	if guard.err != nil {
		t.Fatalf("guard error: %v", guard.err)
	}
	if !guard.blockedBySourceNode {
		t.Fatalf("expected guard to report blockedBySourceNode=true, got %+v", guard)
	}
	if guard.blockedByEvictedPod {
		t.Fatalf("source-node block should be reported on its own, got %+v", guard)
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

	if got := defragRequeueAfter(defragStepResult{blockedBySourceNode: true}, 30*time.Minute); got != defragActiveStepRequeue {
		t.Fatalf("source-node block should requeue quickly, got %s", got)
	}
}

func TestSafetySweepAndGuards_BlocksOnLingeringSourceNode(t *testing.T) {
	// Source-node marker on node-A still has an active TF worker, so the
	// safety sweep keeps it. node-B is a perfectly valid low-util candidate
	// in the same pool. The guard must report blockedBySourceNode so the
	// caller never advances to runDefragStep and we don't accumulate
	// multiple source nodes for the same pool.
	const otherNode = "node-b"
	now := time.Now()
	pool := newDefragTestPool()

	sourceNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testDefragNodeA,
			Labels: map[string]string{
				constants.DefragSourceNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				testDefragSourceNodePoolAnnotation:  pool.Name,
				testDefragSourceNodeSinceAnnotation: now.Add(-5 * time.Minute).Format(time.RFC3339),
			},
		},
	}
	activeWorker := newDefragWorkerPod("worker-source", testDefragNodeA, now.Add(-time.Hour))

	candidateNode := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: otherNode}}
	candidateGPUNode := newDefragGPUNode(otherNode)
	candidateWorker := newDefragWorkerPod("worker-candidate", otherNode, now.Add(-2*time.Hour))

	objects := []ctrlclient.Object{
		pool, sourceNode, activeWorker, candidateNode, candidateGPUNode, candidateWorker,
	}
	objects = append(objects, newDefragNodeGPUs(otherNode)...)

	r, _ := newDefragControllerTestReconciler(t, objects...)
	if err := r.Allocator.InitGPUAndQuotaStore(); err != nil {
		t.Fatalf("init GPU store: %v", err)
	}
	r.Allocator.ReconcileAllocationStateForTesting()

	r.runDefragSafetySweep(context.Background(), pool)
	guard := r.evaluateDefragGuards(context.Background(), pool)
	if guard.err != nil {
		t.Fatalf("guard error: %v", guard.err)
	}
	if !guard.blockedBySourceNode {
		t.Fatalf("expected guard to report blockedBySourceNode=true, got %+v", guard)
	}
	if guard.blockedByEvictedPod {
		t.Fatalf("source-node block should be reported on its own, got %+v", guard)
	}
}

func TestHasActiveDefragSourceNodes_ScopedToPool(t *testing.T) {
	pool := newDefragTestPool()

	otherPoolSource := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "other-pool-source-node",
			Labels: map[string]string{
				constants.DefragSourceNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				testDefragSourceNodePoolAnnotation:  "pool-b",
				testDefragSourceNodeSinceAnnotation: time.Now().Add(-time.Minute).Format(time.RFC3339),
			},
		},
	}
	rOther, _ := newDefragControllerTestReconciler(t, pool, otherPoolSource)
	blocked, err := rOther.hasActiveDefragSourceNodes(context.Background(), pool)
	if err != nil {
		t.Fatalf("check source nodes for other pool: %v", err)
	}
	if blocked {
		t.Fatalf("source node from another pool must not block pool %s", pool.Name)
	}

	samePoolSource := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testDefragNodeA,
			Labels: map[string]string{
				constants.DefragSourceNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				testDefragSourceNodePoolAnnotation:  pool.Name,
				testDefragSourceNodeSinceAnnotation: time.Now().Add(-time.Minute).Format(time.RFC3339),
			},
		},
	}
	rSame, _ := newDefragControllerTestReconciler(t, pool, samePoolSource)
	blocked, err = rSame.hasActiveDefragSourceNodes(context.Background(), pool)
	if err != nil {
		t.Fatalf("check source nodes for same pool: %v", err)
	}
	if !blocked {
		t.Fatalf("source node from same pool should block the next defrag step")
	}
}

func TestEvictedPodMarkerTTL_IndependentFromMaxDuration(t *testing.T) {
	// MaxDuration is generous (2h) but EvictedPodMarkerTTL is short (10m).
	// A pod marked 30m ago must therefore be treated as stale and cleaned,
	// proving the TTL is read from its own field rather than reusing
	// MaxDuration.
	pool := newDefragTestPool()
	pool.Spec.NodeManagerConfig.NodeCompaction.Defrag.MaxDuration = "2h"
	pool.Spec.NodeManagerConfig.NodeCompaction.Defrag.EvictedPodMarkerTTL = "10m"

	stalePod := newPod("stale-evicted-worker")
	stalePod.Labels = map[string]string{
		constants.DefragEvictedPodLabel: constants.TrueStringValue,
	}
	stalePod.Annotations = map[string]string{
		constants.DefragEvictedPodPoolAnnotation:  pool.Name,
		constants.DefragEvictedPodSinceAnnotation: time.Now().Add(-30 * time.Minute).Format(time.RFC3339),
	}

	r, kubeClient := newDefragControllerTestReconciler(t, pool, stalePod)
	if err := r.sweepStaleDefragEvictedPodMarkers(context.Background(), pool); err != nil {
		t.Fatalf("sweep evicted pod markers: %v", err)
	}
	blocked, err := r.hasActiveDefragEvictedPods(context.Background(), pool)
	if err != nil {
		t.Fatalf("check evicted pod marker TTL: %v", err)
	}
	if blocked {
		t.Fatalf("pod older than EvictedPodMarkerTTL should not block; MaxDuration must not leak in")
	}

	updated := &corev1.Pod{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Namespace: stalePod.Namespace, Name: stalePod.Name}, updated); err != nil {
		t.Fatalf("get stale pod after cleanup: %v", err)
	}
	if updated.Labels[constants.DefragEvictedPodLabel] == constants.TrueStringValue {
		t.Fatalf("stale defrag-evicted label should be cleared, labels=%v", updated.Labels)
	}
}

func TestEvictWorkerPods_RealFailurePlacesNodeOnEvictSkipList(t *testing.T) {
	// EvictV1 returns a real (non-NotFound) error on the first pod. The
	// node must end up on the same-pool evict-skip list and the source-node
	// marker must be cleared so this pool's other defrag candidates are not
	// blocked by hasActiveDefragSourceNodes.
	pool := newDefragTestPool()
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: testDefragNodeA}}

	fakeClient := clientgofake.NewSimpleClientset(newPod("p1"))
	fakeClient.PrependReactor("create", "pods", func(a k8stesting.Action) (bool, runtime.Object, error) {
		if a.GetSubresource() != testEvictionSubresource {
			return false, nil, nil
		}
		return true, nil, errors.New("simulated webhook reject")
	})
	r, kubeClient := newDefragControllerTestReconciler(t, pool, node)
	r.KubeClient = fakeClient

	cand := &defragCandidate{
		nodeName:   testDefragNodeA,
		workerPods: []*corev1.Pod{newPod("p1")},
	}
	stats := &defragRunStats{}
	if r.evictWorkerPods(context.Background(), pool, cand, stats, &errLogger{}) {
		t.Fatalf("expected abort=false")
	}
	if stats.EvictionFailures != 1 {
		t.Fatalf("evictionFailures=%d want 1", stats.EvictionFailures)
	}

	updated := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: testDefragNodeA}, updated); err != nil {
		t.Fatalf("get node after evict-skip mark: %v", err)
	}
	if updated.Labels[constants.DefragEvictSkipNodeLabel] != constants.TrueStringValue {
		t.Fatalf("expected evict-skip label on node, labels=%v", updated.Labels)
	}
	if owner := updated.Annotations[constants.DefragEvictSkipNodePoolAnnotation]; owner != pool.Name {
		t.Fatalf("expected evict-skip pool annotation %q, got %q (annotations=%v)", pool.Name, owner, updated.Annotations)
	}
	if updated.Annotations[constants.DefragEvictSkipNodeReasonAnnotation] == "" {
		t.Fatalf("expected evict-skip reason annotation, annotations=%v", updated.Annotations)
	}
	if rawSince := updated.Annotations[constants.DefragEvictSkipNodeSinceAnnotation]; rawSince == "" {
		t.Fatalf("expected evict-skip since annotation, annotations=%v", updated.Annotations)
	} else if _, err := time.Parse(time.RFC3339, rawSince); err != nil {
		t.Fatalf("evict-skip since annotation should be RFC3339, got %q: %v", rawSince, err)
	}
	if updated.Labels[constants.DefragSourceNodeLabel] == constants.TrueStringValue {
		t.Fatalf("source-node marker should be cleared after evict-skip mark, labels=%v", updated.Labels)
	}
}

func TestCollectDefragCandidates_SkipsEvictSkipNode(t *testing.T) {
	now := time.Now()
	pool := newDefragTestPool()
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testDefragNodeA,
			Labels: map[string]string{
				constants.DefragEvictSkipNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				constants.DefragEvictSkipNodePoolAnnotation:  pool.Name,
				constants.DefragEvictSkipNodeSinceAnnotation: now.Format(time.RFC3339),
			},
		},
	}
	gpuNode := newDefragGPUNode(testDefragNodeA)
	worker := newDefragWorkerPod("worker-old", testDefragNodeA, now.Add(-2*time.Hour))
	objects := []ctrlclient.Object{pool, node, gpuNode, worker}
	objects = append(objects, newDefragNodeGPUs(testDefragNodeA)...)

	r, _ := newDefragControllerTestReconciler(t, objects...)
	if err := r.Allocator.InitGPUAndQuotaStore(); err != nil {
		t.Fatalf("init GPU store: %v", err)
	}
	r.Allocator.ReconcileAllocationStateForTesting()

	candidates, err := r.collectDefragCandidates(context.Background(), pool, &defragRunStats{})
	if err != nil {
		t.Fatalf("collect defrag candidates: %v", err)
	}
	if len(candidates) != 0 {
		t.Fatalf("evict-skip node must not appear as candidate, got %d", len(candidates))
	}

	// Cross-pool evict-skip must NOT block this pool: another node carrying
	// the marker for a different pool should still be selectable here.
	otherEvictSkipNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-b",
			Labels: map[string]string{
				constants.DefragEvictSkipNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				constants.DefragEvictSkipNodePoolAnnotation:  "pool-b",
				constants.DefragEvictSkipNodeSinceAnnotation: now.Format(time.RFC3339),
			},
		},
	}
	otherGPUNode := newDefragGPUNode("node-b")
	otherWorker := newDefragWorkerPod("worker-old-b", "node-b", now.Add(-2*time.Hour))
	moreObjects := []ctrlclient.Object{otherEvictSkipNode, otherGPUNode, otherWorker}
	moreObjects = append(moreObjects, newDefragNodeGPUs("node-b")...)
	moreObjects = append(moreObjects, objects...)

	r2, _ := newDefragControllerTestReconciler(t, moreObjects...)
	r2.KubeClient = clientgofake.NewSimpleClientset(newDefragWorkerPDB("pdb-worker", "ns1"))
	if err := r2.Allocator.InitGPUAndQuotaStore(); err != nil {
		t.Fatalf("init GPU store: %v", err)
	}
	r2.Allocator.ReconcileAllocationStateForTesting()
	candidates, err = r2.collectDefragCandidates(context.Background(), pool, &defragRunStats{})
	if err != nil {
		t.Fatalf("collect defrag candidates (cross-pool): %v", err)
	}
	if len(candidates) != 1 || candidates[0].nodeName != "node-b" {
		t.Fatalf("expected node-b to be the only candidate, got %+v", candidates)
	}
}

func TestCleanupStaleDefragEvictSkipMarkers_ReleasesEmptyNode(t *testing.T) {
	// Node has the evict-skip marker but no active TF worker -- the marker
	// has done its job and must be released so the next campaign can
	// re-evaluate the node.
	pool := newDefragTestPool()
	skippedNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testDefragNodeA,
			Labels: map[string]string{
				constants.DefragEvictSkipNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				constants.DefragEvictSkipNodePoolAnnotation:   pool.Name,
				constants.DefragEvictSkipNodeSinceAnnotation:  time.Now().Add(-time.Hour).Format(time.RFC3339),
				constants.DefragEvictSkipNodeReasonAnnotation: "evict failed: timeout",
			},
		},
	}

	r, kubeClient := newDefragControllerTestReconciler(t, pool, skippedNode)
	if err := r.cleanupStaleDefragEvictSkipMarkers(context.Background(), pool); err != nil {
		t.Fatalf("cleanup evict-skip markers: %v", err)
	}

	updated := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: testDefragNodeA}, updated); err != nil {
		t.Fatalf("get node after cleanup: %v", err)
	}
	if updated.Labels[constants.DefragEvictSkipNodeLabel] == constants.TrueStringValue {
		t.Fatalf("evict-skip label should be cleared on empty node, labels=%v", updated.Labels)
	}
	if updated.Annotations[constants.DefragEvictSkipNodePoolAnnotation] != "" ||
		updated.Annotations[constants.DefragEvictSkipNodeSinceAnnotation] != "" ||
		updated.Annotations[constants.DefragEvictSkipNodeReasonAnnotation] != "" {
		t.Fatalf("evict-skip annotations should be cleared, annotations=%v", updated.Annotations)
	}
}

func TestClearAllDefragEvictSkipMarkersForPool_ScopedToPool(t *testing.T) {
	// Campaign-scoped release: every evict-skip marker owned by THIS pool
	// must drop, even if its node still has a stuck TF worker. Markers
	// owned by another pool must remain untouched.
	pool := newDefragTestPool()

	stuckWorker := newDefragWorkerPod("stuck", testDefragNodeA, time.Now().Add(-time.Hour))
	samePoolNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testDefragNodeA,
			Labels: map[string]string{
				constants.DefragEvictSkipNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				constants.DefragEvictSkipNodePoolAnnotation:  pool.Name,
				constants.DefragEvictSkipNodeSinceAnnotation: time.Now().Format(time.RFC3339),
			},
		},
	}
	otherPoolNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-b",
			Labels: map[string]string{
				constants.DefragEvictSkipNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				constants.DefragEvictSkipNodePoolAnnotation:  "pool-b",
				constants.DefragEvictSkipNodeSinceAnnotation: time.Now().Format(time.RFC3339),
			},
		},
	}

	r, kubeClient := newDefragControllerTestReconciler(t, pool, samePoolNode, otherPoolNode, stuckWorker)
	if err := r.clearAllDefragEvictSkipMarkersForPool(context.Background(), pool); err != nil {
		t.Fatalf("clear evict-skip markers: %v", err)
	}

	updatedSame := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: testDefragNodeA}, updatedSame); err != nil {
		t.Fatalf("get same-pool node after clear: %v", err)
	}
	if updatedSame.Labels[constants.DefragEvictSkipNodeLabel] == constants.TrueStringValue {
		t.Fatalf("same-pool evict-skip label must be cleared even with active worker, labels=%v", updatedSame.Labels)
	}
	if updatedSame.Annotations[constants.DefragEvictSkipNodePoolAnnotation] != "" ||
		updatedSame.Annotations[constants.DefragEvictSkipNodeSinceAnnotation] != "" {
		t.Fatalf("same-pool evict-skip annotations must be cleared, annotations=%v", updatedSame.Annotations)
	}

	updatedOther := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: "node-b"}, updatedOther); err != nil {
		t.Fatalf("get other-pool node after clear: %v", err)
	}
	if updatedOther.Labels[constants.DefragEvictSkipNodeLabel] != constants.TrueStringValue {
		t.Fatalf("cross-pool evict-skip label must be preserved, labels=%v", updatedOther.Labels)
	}
	if updatedOther.Annotations[constants.DefragEvictSkipNodePoolAnnotation] != "pool-b" {
		t.Fatalf("cross-pool evict-skip pool annotation must be preserved, annotations=%v", updatedOther.Annotations)
	}
}

func TestCleanupStaleDefragEvictSkipMarkers_KeepsNodeWithActiveWorker(t *testing.T) {
	// Node still has an active TF worker (the eviction failure left a
	// stuck pod behind). The evict-skip marker must stick around so this
	// pool's defrag keeps skipping the node.
	pool := newDefragTestPool()
	skippedNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testDefragNodeA,
			Labels: map[string]string{
				constants.DefragEvictSkipNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				constants.DefragEvictSkipNodePoolAnnotation:  pool.Name,
				constants.DefragEvictSkipNodeSinceAnnotation: time.Now().Add(-time.Hour).Format(time.RFC3339),
			},
		},
	}
	stuckWorker := newDefragWorkerPod("stuck-worker", testDefragNodeA, time.Now().Add(-time.Hour))

	r, kubeClient := newDefragControllerTestReconciler(t, pool, skippedNode, stuckWorker)
	primeAllocatorWorkerStore(t, r)
	if err := r.cleanupStaleDefragEvictSkipMarkers(context.Background(), pool); err != nil {
		t.Fatalf("cleanup evict-skip markers: %v", err)
	}

	updated := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: testDefragNodeA}, updated); err != nil {
		t.Fatalf("get node after cleanup: %v", err)
	}
	if updated.Labels[constants.DefragEvictSkipNodeLabel] != constants.TrueStringValue {
		t.Fatalf("evict-skip label must be kept while worker is still on node, labels=%v", updated.Labels)
	}
}

func TestSourceNodeMarkerTTL_IndependentFromMaxDuration(t *testing.T) {
	// Symmetric to the evicted-pod case: SourceNodeMarkerTTL must drive
	// stale source-node cleanup independently of MaxDuration.
	pool := newDefragTestPool()
	pool.Spec.NodeManagerConfig.NodeCompaction.Defrag.MaxDuration = "2h"
	pool.Spec.NodeManagerConfig.NodeCompaction.Defrag.SourceNodeMarkerTTL = "10m"

	staleNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testDefragNodeA,
			Labels: map[string]string{
				constants.DefragSourceNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				testDefragSourceNodePoolAnnotation:  pool.Name,
				testDefragSourceNodeSinceAnnotation: time.Now().Add(-30 * time.Minute).Format(time.RFC3339),
			},
		},
	}
	// Active worker forces hasActiveTensorFusionWorkerOnNode==true so that
	// stale cleanup must rely solely on the TTL (not the no-workers shortcut).
	activeWorker := newDefragWorkerPod("worker-active", testDefragNodeA, time.Now().Add(-time.Hour))

	r, kubeClient := newDefragControllerTestReconciler(t, pool, staleNode, activeWorker)
	primeAllocatorWorkerStore(t, r)
	if err := r.cleanupStaleDefragSourceMarkers(context.Background(), pool); err != nil {
		t.Fatalf("cleanup stale source markers: %v", err)
	}

	updated := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: testDefragNodeA}, updated); err != nil {
		t.Fatalf("get stale source node after cleanup: %v", err)
	}
	if updated.Labels[constants.DefragSourceNodeLabel] == constants.TrueStringValue {
		t.Fatalf("stale defrag-source label should be cleared by SourceNodeMarkerTTL, labels=%v", updated.Labels)
	}
}

func TestRunDefragCandidateLoop_StopsAfterFirstEvictedNode(t *testing.T) {
	candidates := []*defragCandidate{
		{nodeName: testDefragNodeA},
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
	if len(visited) != 1 || visited[0] != testDefragNodeA {
		t.Fatalf("visited=%v want only node-a", visited)
	}
}

func TestRunDefragCandidateLoop_SkipsAndContinuesToNextCandidate(t *testing.T) {
	candidates := []*defragCandidate{
		{nodeName: testDefragNodeA},
		{nodeName: "node-b"},
	}
	stats := &defragRunStats{}
	visited := []string{}

	result := runDefragCandidateLoop(context.Background(), candidates, stats, func(cand *defragCandidate) defragCandidateOutcome {
		visited = append(visited, cand.nodeName)
		if cand.nodeName == testDefragNodeA {
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
		// Empty clientset so PDB-required filter has something to query.
		// Tests that need real PDBs override r.KubeClient explicitly.
		KubeClient: clientgofake.NewSimpleClientset(),
	}, kubeClient
}

// primeAllocatorWorkerStore initializes the allocator's GPU + worker stores
// from the fake client so tests that exercise hasActiveTensorFusionWorkerOnNode
// (which now reads from nodeWorkerStore) see the seeded worker pods.
func primeAllocatorWorkerStore(t *testing.T, r *GPUPoolCompactionReconciler) {
	t.Helper()
	if err := r.Allocator.InitGPUAndQuotaStore(); err != nil {
		t.Fatalf("init GPU store: %v", err)
	}
	r.Allocator.ReconcileAllocationStateForTesting()
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

// newDefragNodeGPUs builds a fixed 4-GPU layout for the defrag test pool.
// All defrag tests share the same "pool-a" defined by newDefragTestPool,
// so the pool name is hardcoded rather than parameterized.
func newDefragNodeGPUs(nodeName string) []ctrlclient.Object {
	out := make([]ctrlclient.Object, 0, 4)
	for i := 0; i < 4; i++ {
		availableTflops := "100"
		if i == 0 {
			availableTflops = "50"
		}
		out = append(out, gpuWithUsageOnNode(
			fmt.Sprintf("%s-gpu-%d", nodeName, i),
			"pool-a",
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

// newDefragGPUNode is hardcoded to "pool-a" because every defrag test
// uses the same pool defined by newDefragTestPool.
func newDefragGPUNode(name string) *tfv1.GPUNode {
	return &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, "pool-a"): constants.TrueStringValue,
			},
		},
		Status: tfv1.GPUNodeStatus{
			Phase: tfv1.TensorFusionGPUNodePhaseRunning,
		},
	}
}

// newDefragWorkerPDB returns a PDB whose selector matches the labels set
// by newDefragWorkerPod, so candidate filtering treats the worker as
// PDB-covered.
func newDefragWorkerPDB(name, namespace string) *policyv1.PodDisruptionBudget {
	return &policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: policyv1.PodDisruptionBudgetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					constants.LabelComponent: constants.ComponentWorker,
					constants.WorkloadKey:    "workload-a",
				},
			},
		},
		Status: policyv1.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 1,
		},
	}
}

// assertWarningEvent drains the FakeRecorder and fails the test unless one
// of the events is a Warning matching `reason` and contains `substr`.
func assertWarningEvent(t *testing.T, recorder record.EventRecorder, reason, substr string) {
	t.Helper()
	fr, ok := recorder.(*record.FakeRecorder)
	if !ok {
		t.Fatalf("recorder is %T, want *record.FakeRecorder", recorder)
	}
	for {
		select {
		case ev := <-fr.Events:
			if strings.Contains(ev, "Warning "+reason) && strings.Contains(ev, substr) {
				return
			}
		default:
			t.Fatalf("no Warning %q event matched substr %q", reason, substr)
		}
	}
}

// assertNoEvent drains the FakeRecorder and fails the test if any event
// carries the given reason.
func assertNoEvent(t *testing.T, recorder record.EventRecorder, reason string) {
	t.Helper()
	fr, ok := recorder.(*record.FakeRecorder)
	if !ok {
		t.Fatalf("recorder is %T, want *record.FakeRecorder", recorder)
	}
	for {
		select {
		case ev := <-fr.Events:
			if strings.Contains(ev, reason) {
				t.Fatalf("unexpected event %q", ev)
			}
		default:
			return
		}
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

func TestRunDefragSafetySweep_RunsAllThreeEvenWhenEvictedPodActive(t *testing.T) {
	// Core fix this refactor delivers: an active (non-stale) defrag-evicted
	// pod must NOT short-circuit the safety sweep. Stale source-node /
	// evict-skip markers belonging to the same pool must still be cleaned
	// in this reconcile, so the next campaign has a clean slate.
	now := time.Now()
	pool := newDefragTestPool()

	activeEvictedPod := newPod("active-evicted")
	activeEvictedPod.Labels = map[string]string{
		constants.DefragEvictedPodLabel: constants.TrueStringValue,
	}
	activeEvictedPod.Annotations = map[string]string{
		constants.DefragEvictedPodPoolAnnotation:  pool.Name,
		constants.DefragEvictedPodSinceAnnotation: now.Format(time.RFC3339),
	}

	staleSourceNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testDefragNodeA,
			Labels: map[string]string{
				constants.DefragSourceNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				testDefragSourceNodePoolAnnotation:  pool.Name,
				testDefragSourceNodeSinceAnnotation: now.Add(-2 * time.Hour).Format(time.RFC3339),
			},
		},
	}

	staleEvictSkipNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-evict-skip",
			Labels: map[string]string{
				constants.DefragEvictSkipNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				constants.DefragEvictSkipNodePoolAnnotation: pool.Name,
				constants.DefragEvictSkipNodeSinceAnnotation: now.Add(-time.Minute).
					Format(time.RFC3339),
				constants.DefragEvictSkipNodeReasonAnnotation: "simulated-eviction-failure",
			},
		},
	}

	r, kubeClient := newDefragControllerTestReconciler(t,
		pool, activeEvictedPod, staleSourceNode, staleEvictSkipNode)

	r.runDefragSafetySweep(context.Background(), pool)

	updatedSource := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: testDefragNodeA}, updatedSource); err != nil {
		t.Fatalf("get source node after sweep: %v", err)
	}
	if updatedSource.Labels[constants.DefragSourceNodeLabel] == constants.TrueStringValue {
		t.Fatalf("stale source-node marker must clear even when evicted pod is active, labels=%v",
			updatedSource.Labels)
	}

	updatedEvictSkip := &corev1.Node{}
	if err := kubeClient.Get(context.Background(), types.NamespacedName{Name: "node-evict-skip"}, updatedEvictSkip); err != nil {
		t.Fatalf("get evict-skip node after sweep: %v", err)
	}
	if updatedEvictSkip.Labels[constants.DefragEvictSkipNodeLabel] == constants.TrueStringValue {
		t.Fatalf("idle evict-skip marker must clear even when evicted pod is active, labels=%v",
			updatedEvictSkip.Labels)
	}

	updatedPod := &corev1.Pod{}
	if err := kubeClient.Get(context.Background(),
		types.NamespacedName{Namespace: activeEvictedPod.Namespace, Name: activeEvictedPod.Name},
		updatedPod); err != nil {
		t.Fatalf("get active evicted pod after sweep: %v", err)
	}
	if updatedPod.Labels[constants.DefragEvictedPodLabel] != constants.TrueStringValue {
		t.Fatalf("non-stale defrag-evicted pod must keep its marker, labels=%v",
			updatedPod.Labels)
	}

	guard := r.evaluateDefragGuards(context.Background(), pool)
	if guard.err != nil {
		t.Fatalf("guard error: %v", guard.err)
	}
	if !guard.blockedByEvictedPod {
		t.Fatalf("expected guard to keep reporting blockedByEvictedPod, got %+v", guard)
	}
}

func TestEvaluateDefragGuards_DoesNotMutateMarkers(t *testing.T) {
	// Guards are pure: even if we feed them stale markers (which sweep
	// would have cleaned), guard must NOT clean them itself. Otherwise the
	// sweep/guard split has no value -- callers wouldn't be able to rely on
	// "guard is read-only".
	now := time.Now()
	pool := newDefragTestPool()

	stalePod := newPod("stale-evicted-worker")
	stalePod.Labels = map[string]string{
		constants.DefragEvictedPodLabel: constants.TrueStringValue,
	}
	stalePod.Annotations = map[string]string{
		constants.DefragEvictedPodPoolAnnotation:  pool.Name,
		constants.DefragEvictedPodSinceAnnotation: now.Add(-2 * time.Hour).Format(time.RFC3339),
	}

	staleSourceNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testDefragNodeA,
			Labels: map[string]string{
				constants.DefragSourceNodeLabel: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				testDefragSourceNodePoolAnnotation:  pool.Name,
				testDefragSourceNodeSinceAnnotation: now.Add(-2 * time.Hour).Format(time.RFC3339),
			},
		},
	}

	r, kubeClient := newDefragControllerTestReconciler(t, pool, stalePod, staleSourceNode)

	guard := r.evaluateDefragGuards(context.Background(), pool)
	if guard.err != nil {
		t.Fatalf("guard error: %v", guard.err)
	}
	// Stale evicted pod -> hasActiveDefragEvictedPods correctly returns false.
	// Stale source node -> hasActiveDefragSourceNodes returns true (it does
	// not check staleness). Either way, the guard MUST NOT have written
	// anything: re-read both objects and assert markers are still present.
	updatedPod := &corev1.Pod{}
	if err := kubeClient.Get(context.Background(),
		types.NamespacedName{Namespace: stalePod.Namespace, Name: stalePod.Name},
		updatedPod); err != nil {
		t.Fatalf("get stale pod after guard: %v", err)
	}
	if updatedPod.Labels[constants.DefragEvictedPodLabel] != constants.TrueStringValue {
		t.Fatalf("guard must not clear evicted-pod marker, labels=%v", updatedPod.Labels)
	}

	updatedNode := &corev1.Node{}
	if err := kubeClient.Get(context.Background(),
		types.NamespacedName{Name: testDefragNodeA}, updatedNode); err != nil {
		t.Fatalf("get stale source node after guard: %v", err)
	}
	if updatedNode.Labels[constants.DefragSourceNodeLabel] != constants.TrueStringValue {
		t.Fatalf("guard must not clear source-node marker, labels=%v", updatedNode.Labels)
	}
}

// ---- defrag monotonicity gates ----------------------------------------

// gpuFullyFreeOnNode returns a GPU whose Available == Capacity so
// countPoolGPUUsage classifies it as "free" and isGPUFullyAvailable
// returns true. Useful as fixture noise reduction in monotonicity tests.
func gpuFullyFreeOnNode(name, poolName, nodeName string) *tfv1.GPU {
	g := gpuWithUsage(name, poolName, "100", "10Gi", tfv1.UsedByTensorFusion)
	g.Status.NodeSelector = map[string]string{
		constants.KubernetesHostNameLabel: nodeName,
	}
	return g
}

func TestBuildDefragNodeBudgets_DropsEmptyTarget(t *testing.T) {
	// Layout: 1 source (excluded), 1 partially-used buddy, 1 fully-empty
	// pool node. The empty node MUST be filtered so defrag does not waste
	// a disruption shuffling workers onto a previously empty machine.
	const pool = "pool-a"
	source := "node-source"
	buddy := "node-buddy"
	empty := "node-empty"

	nodeGpuStore := map[string]map[string]*tfv1.GPU{
		source: {
			"src-gpu-0": gpuWithUsageOnNode("src-gpu-0", pool, source, "50", "5Gi"),
		},
		buddy: {
			"buddy-gpu-0": gpuWithUsageOnNode("buddy-gpu-0", pool, buddy, "60", "8Gi"),
			"buddy-gpu-1": gpuFullyFreeOnNode("buddy-gpu-1", pool, buddy),
		},
		empty: {
			"empty-gpu-0": gpuFullyFreeOnNode("empty-gpu-0", pool, empty),
			"empty-gpu-1": gpuFullyFreeOnNode("empty-gpu-1", pool, empty),
		},
	}
	lister := &fakeNodeInfoLister{infos: map[string]fwk.NodeInfo{
		source: newFrameworkNodeInfo(source, nil, nil),
		buddy:  newFrameworkNodeInfo(buddy, nil, nil),
		empty:  newFrameworkNodeInfo(empty, nil, nil),
	}}

	budgets := buildDefragNodeBudgets(pool, source, nodeGpuStore, nil, lister)

	if _, ok := budgets[source]; ok {
		t.Fatalf("source node must not appear in budgets, got %v", budgets)
	}
	if _, ok := budgets[empty]; ok {
		t.Fatalf("empty pool node must be filtered from budgets, got %v", budgets)
	}
	nb, ok := budgets[buddy]
	if !ok {
		t.Fatalf("buddy node missing from budgets, got %v", budgets)
	}
	if nb.totalGPUs != 2 || nb.usedGPUs != 1 {
		t.Fatalf("buddy budget total/used = %d/%d, want 2/1", nb.totalGPUs, nb.usedGPUs)
	}
}

func TestBuildDefragNodeBudgets_SkipsExcludedNodeLabels(t *testing.T) {
	// Regression: NodeDeletionMark / DefragSourceNodeLabel should still be
	// filtered out independently of the new emptiness gate. A node
	// labelled as source-marker also having real usage must NOT leak in.
	const pool = "pool-a"
	source := "node-source"
	marked := "node-marked"
	nodeGpuStore := map[string]map[string]*tfv1.GPU{
		source: {"src": gpuWithUsageOnNode("src", pool, source, "50", "5Gi")},
		marked: {"mk": gpuWithUsageOnNode("mk", pool, marked, "60", "5Gi")},
	}
	lister := &fakeNodeInfoLister{infos: map[string]fwk.NodeInfo{
		source: newFrameworkNodeInfo(source, nil, nil),
		marked: newFrameworkNodeInfo(marked, map[string]string{
			constants.DefragSourceNodeLabel: constants.TrueStringValue,
		}, nil),
	}}
	budgets := buildDefragNodeBudgets(pool, source, nodeGpuStore, nil, lister)
	if _, ok := budgets[marked]; ok {
		t.Fatalf("DefragSourceNodeLabel must keep node out of budgets, got %v", budgets)
	}
}

func TestIsGPUFullyAvailable(t *testing.T) {
	cases := []struct {
		name string
		g    *tfv1.GPU
		want bool
	}{
		{name: "nil", g: nil, want: false},
		{name: "missing-available", g: &tfv1.GPU{Status: tfv1.GPUStatus{Capacity: &tfv1.Resource{}}}, want: false},
		{name: "missing-capacity", g: &tfv1.GPU{Status: tfv1.GPUStatus{Available: &tfv1.Resource{}}}, want: false},
		{
			name: "fully-free",
			g:    gpuWithUsage("a", "p", "100", "10Gi", tfv1.UsedByTensorFusion),
			want: true,
		},
		{
			name: "tflops-half-used",
			g:    gpuWithUsage("a", "p", "50", "10Gi", tfv1.UsedByTensorFusion),
			want: false,
		},
		{
			name: "vram-half-used",
			g:    gpuWithUsage("a", "p", "100", "5Gi", tfv1.UsedByTensorFusion),
			want: false,
		},
		{
			// Capacity is 100/10Gi from gpuWithUsage; Available 0/0 is
			// strictly less, so this is "fully used", not fully available.
			name: "fully-used",
			g:    gpuWithUsage("a", "p", "0", "0", tfv1.UsedByTensorFusion),
			want: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := isGPUFullyAvailable(c.g); got != c.want {
				t.Fatalf("isGPUFullyAvailable=%v, want %v", got, c.want)
			}
		})
	}
}

func TestBudgetUtilizationPercent(t *testing.T) {
	cases := []struct {
		name string
		nb   *nodeBudget
		want float64
	}{
		{name: "nil", nb: nil, want: 0},
		{name: "zero-total", nb: &nodeBudget{}, want: 0},
		{name: "half", nb: &nodeBudget{totalGPUs: 4, usedGPUs: 2}, want: 50},
		{name: "all-used", nb: &nodeBudget{totalGPUs: 3, usedGPUs: 3}, want: 100},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := budgetUtilizationPercent(c.nb); got != c.want {
				t.Fatalf("budgetUtilizationPercent=%v, want %v", got, c.want)
			}
		})
	}
}

func TestApplyGPUPlacementToBudget_TracksUsedGPUTransition(t *testing.T) {
	// Validates the dynamic-util commit path: only fully-free -> partially
	// used transitions bump usedGPUs. A second placement onto the same GPU
	// must NOT double-count.
	const pool = "pool-a"
	node := "node-buddy"
	freeGPU := gpuFullyFreeOnNode("g-free", pool, node)
	usedGPU := gpuWithUsageOnNode("g-used", pool, node, "50", "5Gi") // half-used baseline

	nb := &nodeBudget{
		gpus:      map[string]*tfv1.GPU{"g-free": freeGPU, "g-used": usedGPU},
		totalGPUs: 2,
		usedGPUs:  1, // matches the half-used GPU above
	}
	req := &tfv1.AllocRequest{
		Request: tfv1.Resource{Tflops: resource.MustParse("10"), Vram: resource.MustParse("1Gi")},
	}

	// First placement: lands on the fully-free GPU → usedGPUs jumps to 2.
	applyGPUPlacementToBudget(nb, []*tfv1.GPU{freeGPU}, req)
	if nb.usedGPUs != 2 {
		t.Fatalf("after first placement usedGPUs=%d, want 2", nb.usedGPUs)
	}
	if isGPUFullyAvailable(nb.gpus["g-free"]) {
		t.Fatalf("free GPU should no longer be fully available after subtract")
	}

	// Second placement on the already-used GPU: no transition, usedGPUs stays.
	applyGPUPlacementToBudget(nb, []*tfv1.GPU{usedGPU}, req)
	if nb.usedGPUs != 2 {
		t.Fatalf("second placement bumped usedGPUs to %d; commit must not double-count", nb.usedGPUs)
	}

	// Budget utilization should now report 100% (2/2 used).
	if got := budgetUtilizationPercent(nb); got != 100 {
		t.Fatalf("budgetUtilizationPercent=%v, want 100", got)
	}
}

func TestApplyGPUPlacementToBudget_NilBudgetNoop(t *testing.T) {
	// Defensive: applyGPUPlacementToBudget(nil, ...) must not panic; this
	// shape never occurs in production but keeps the helper safe for reuse.
	req := &tfv1.AllocRequest{
		Request: tfv1.Resource{Tflops: resource.MustParse("10"), Vram: resource.MustParse("1Gi")},
	}
	applyGPUPlacementToBudget(nil, []*tfv1.GPU{
		gpuWithUsage("a", "p", "100", "10Gi", tfv1.UsedByTensorFusion),
	}, req)
}

// ---- compile-time sanity check ----------------------------------------
var _ = errors.New
var _ = fmt.Errorf
var _ ctrl.Result
