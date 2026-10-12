package controller

import (
	"context"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

const orphanTestNamespace = "orphan-ns"

// durationTolerance covers the time a test takes between building its objects
// and the reconciler reading the clock, and the one-second resolution of the
// recorded time.
const durationTolerance = 5 * time.Second

func newOrphanTestClient(t *testing.T, funcs *interceptor.Funcs, objs ...client.Object) client.WithWatch {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	require.NoError(t, tfv1.AddToScheme(scheme))
	builder := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).
		WithStatusSubresource(&tfv1.TensorFusionWorkload{})
	if funcs != nil {
		builder = builder.WithInterceptorFuncs(*funcs)
	}
	return builder.Build()
}

// webhookWorkload is a workload as the pod webhook leaves it for a pod without
// a controller: marked dynamic, no owner reference.
//
//nolint:unparam // the tests name the workload they act on
func webhookWorkload(name string, age time.Duration) *tfv1.TensorFusionWorkload {
	return &tfv1.TensorFusionWorkload{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         orphanTestNamespace,
			UID:               types.UID(name + "-uid"),
			CreationTimestamp: metav1.NewTime(time.Now().Add(-age)),
			Annotations: map[string]string{
				constants.WorkloadModeAnnotation: constants.WorkloadModeDynamic,
			},
		},
	}
}

func orphanTestPod(name, namespace string, labels, annotations map[string]string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace, Labels: labels, Annotations: annotations},
		Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "test-image"}}},
	}
}

//nolint:unparam // the tests name the workload they act on
func workloadExists(t *testing.T, c client.Client, name string) bool {
	t.Helper()
	err := c.Get(context.Background(), client.ObjectKey{Namespace: orphanTestNamespace, Name: name}, &tfv1.TensorFusionWorkload{})
	if apierrors.IsNotFound(err) {
		return false
	}
	require.NoError(t, err)
	return true
}

//nolint:unparam // the tests name the workload they act on
func collectOrphan(t *testing.T, r *TensorFusionWorkloadReconciler, name string) (bool, time.Duration) {
	t.Helper()
	deleted, recheck, err := r.collectOrphanWorkload(context.Background(),
		client.ObjectKey{Namespace: orphanTestNamespace, Name: name})
	require.NoError(t, err)
	return deleted, recheck
}

func TestOrphanClockStart(t *testing.T) {
	created := time.Now().Add(-time.Hour).Truncate(time.Second)
	workload := func(seen string) *tfv1.TensorFusionWorkload {
		w := &tfv1.TensorFusionWorkload{ObjectMeta: metav1.ObjectMeta{CreationTimestamp: metav1.NewTime(created)}}
		if seen != "" {
			w.Annotations = map[string]string{constants.WorkloadLastPodSeenAnnotation: seen}
		}
		return w
	}

	start, readable := orphanClockStart(workload(""))
	assert.True(t, readable)
	assert.True(t, start.Equal(created), "without a recorded time the period starts at creation")

	later := created.Add(20 * time.Minute)
	start, readable = orphanClockStart(workload(later.Format(time.RFC3339)))
	assert.True(t, readable)
	assert.True(t, start.Equal(later), "a later recorded time starts the period")

	start, readable = orphanClockStart(workload(created.Add(-time.Minute).Format(time.RFC3339)))
	assert.True(t, readable)
	assert.True(t, start.Equal(created), "a recorded time before creation does not move the start back")

	_, readable = orphanClockStart(workload("yesterday"))
	assert.False(t, readable)
}

func TestWorkloadOrphanCandidates(t *testing.T) {
	old := 2 * WorkloadOrphanGracePeriod
	owner := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "owner", Namespace: orphanTestNamespace, UID: "owner-uid"}}
	ownerRef := metav1.OwnerReference{APIVersion: "apps/v1", Kind: "Deployment", Name: "owner", UID: "owner-uid"}

	cases := []struct {
		name        string
		mutate      func(w *tfv1.TensorFusionWorkload)
		extra       []client.Object
		wantDeleted bool
	}{
		{
			name:        "created by the webhook, no owner, no pod, past the grace period: deleted",
			mutate:      func(*tfv1.TensorFusionWorkload) {},
			wantDeleted: true,
		},
		{
			name:   "created by a user (no workload mode): untouched",
			mutate: func(w *tfv1.TensorFusionWorkload) { w.Annotations = nil },
		},
		{
			name: "marked with another workload mode: untouched",
			mutate: func(w *tfv1.TensorFusionWorkload) {
				w.Annotations[constants.WorkloadModeAnnotation] = constants.WorkloadModeFixed
			},
		},
		{
			name:   "owner exists and has no pods (scaled to zero): untouched",
			mutate: func(w *tfv1.TensorFusionWorkload) { w.OwnerReferences = []metav1.OwnerReference{ownerRef} },
			extra:  []client.Object{owner},
		},
		{
			// The reference dangles, which is what makes the Kubernetes
			// garbage collector delete the workload; this controller stays out.
			name:   "owner is gone: left to the Kubernetes garbage collector",
			mutate: func(w *tfv1.TensorFusionWorkload) { w.OwnerReferences = []metav1.OwnerReference{ownerRef} },
		},
		{
			name: "already being deleted: untouched",
			mutate: func(w *tfv1.TensorFusionWorkload) {
				w.Finalizers = []string{constants.Finalizer}
				w.DeletionTimestamp = ptrTime(time.Now())
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			workload := webhookWorkload("w", old)
			tc.mutate(workload)
			c := newOrphanTestClient(t, nil, append(tc.extra, workload)...)
			r := &TensorFusionWorkloadReconciler{Client: c}

			deleted, recheck := collectOrphan(t, r, "w")

			assert.Equal(t, tc.wantDeleted, deleted)
			assert.Equal(t, !tc.wantDeleted, workloadExists(t, c, "w"))
			assert.Zero(t, recheck, "nothing to wait for either way")
		})
	}
}

func ptrTime(t time.Time) *metav1.Time {
	mt := metav1.NewTime(t)
	return &mt
}

func TestWorkloadOrphanKeptWhileAPodRefersToIt(t *testing.T) {
	old := 2 * WorkloadOrphanGracePeriod
	terminating := orphanTestPod("p", orphanTestNamespace, map[string]string{constants.WorkloadKey: "w"}, nil)
	terminating.Finalizers = []string{constants.Finalizer}
	terminating.DeletionTimestamp = ptrTime(time.Now())

	cases := []struct {
		name     string
		pod      *corev1.Pod
		wantKept bool
	}{
		{"worker or local-GPU client (label)",
			orphanTestPod("p", orphanTestNamespace, map[string]string{constants.WorkloadKey: "w"}, nil), true},
		{"client of a remote worker (selected-workload annotation)",
			orphanTestPod("p", orphanTestNamespace, nil, map[string]string{constants.SelectedWorkloadAnnotation: "w"}), true},
		{"pod about to become the owner (pending-owned-workload annotation)",
			orphanTestPod("p", orphanTestNamespace, nil, map[string]string{constants.SetPendingOwnedWorkloadAnnotation: "w"}), true},
		{"terminating pod", terminating, true},
		{"pod of another workload",
			orphanTestPod("p", orphanTestNamespace, map[string]string{constants.WorkloadKey: "other"}, nil), false},
		{"pod of a workload with the same name in another namespace",
			orphanTestPod("p", "elsewhere", map[string]string{constants.WorkloadKey: "w"}, nil), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := newOrphanTestClient(t, nil, webhookWorkload("w", old), tc.pod)
			r := &TensorFusionWorkloadReconciler{Client: c}

			deleted, recheck := collectOrphan(t, r, "w")

			assert.Equal(t, !tc.wantKept, deleted)
			assert.Equal(t, tc.wantKept, workloadExists(t, c, "w"))
			if tc.wantKept {
				assert.Equal(t, workloadPodSeenRefreshInterval, recheck, "looked at again without waiting for an event")
			}
		})
	}
}

func TestWorkloadOrphanGraceBoundary(t *testing.T) {
	t.Run("just inside the grace period: kept, looked at again when it ends", func(t *testing.T) {
		left := 30 * time.Second
		c := newOrphanTestClient(t, nil, webhookWorkload("w", WorkloadOrphanGracePeriod-left))
		r := &TensorFusionWorkloadReconciler{Client: c}

		deleted, recheck := collectOrphan(t, r, "w")

		assert.False(t, deleted)
		assert.True(t, workloadExists(t, c, "w"))
		assert.InDelta(t, float64(left+workloadOrphanRecheckSlack), float64(recheck), float64(durationTolerance))
		assert.Greater(t, recheck, time.Duration(0))
	})

	t.Run("just created: kept for the whole grace period", func(t *testing.T) {
		c := newOrphanTestClient(t, nil, webhookWorkload("w", 0))
		r := &TensorFusionWorkloadReconciler{Client: c}

		deleted, recheck := collectOrphan(t, r, "w")

		assert.False(t, deleted)
		assert.True(t, workloadExists(t, c, "w"))
		assert.InDelta(t, float64(WorkloadOrphanGracePeriod), float64(recheck), float64(durationTolerance))
	})

	t.Run("just past the grace period: deleted", func(t *testing.T) {
		c := newOrphanTestClient(t, nil, webhookWorkload("w", WorkloadOrphanGracePeriod+time.Second))
		r := &TensorFusionWorkloadReconciler{Client: c}

		deleted, _ := collectOrphan(t, r, "w")

		assert.True(t, deleted)
		assert.False(t, workloadExists(t, c, "w"))
	})

	t.Run("an injected grace period replaces the default", func(t *testing.T) {
		c := newOrphanTestClient(t, nil, webhookWorkload("w", time.Minute))
		r := &TensorFusionWorkloadReconciler{Client: c, orphanGracePeriod: 30 * time.Second}

		deleted, _ := collectOrphan(t, r, "w")

		assert.True(t, deleted)
	})
}

// The period runs from the last time a pod was admitted or seen, when that is
// later than the creation.
func TestWorkloadOrphanGraceCountsFromLastPodSeen(t *testing.T) {
	old := 10 * WorkloadOrphanGracePeriod
	seenAgo := func(w *tfv1.TensorFusionWorkload, ago time.Duration) *tfv1.TensorFusionWorkload {
		w.Annotations[constants.WorkloadLastPodSeenAnnotation] = time.Now().Add(-ago).UTC().Format(time.RFC3339)
		return w
	}

	t.Run("old workload whose pod was seen inside the grace period: kept", func(t *testing.T) {
		left := time.Minute
		c := newOrphanTestClient(t, nil, seenAgo(webhookWorkload("w", old), WorkloadOrphanGracePeriod-left))
		r := &TensorFusionWorkloadReconciler{Client: c}

		deleted, recheck := collectOrphan(t, r, "w")

		assert.False(t, deleted)
		assert.True(t, workloadExists(t, c, "w"))
		assert.InDelta(t, float64(left), float64(recheck), float64(durationTolerance))
	})

	t.Run("old workload whose pod was last seen before the grace period: deleted", func(t *testing.T) {
		c := newOrphanTestClient(t, nil, seenAgo(webhookWorkload("w", old), WorkloadOrphanGracePeriod+2*time.Second))
		r := &TensorFusionWorkloadReconciler{Client: c}

		deleted, _ := collectOrphan(t, r, "w")

		assert.True(t, deleted)
		assert.False(t, workloadExists(t, c, "w"))
	})

	t.Run("seeing a pod is recorded, and the period starts when the pod is gone", func(t *testing.T) {
		pod := orphanTestPod("p", orphanTestNamespace, map[string]string{constants.WorkloadKey: "w"}, nil)
		c := newOrphanTestClient(t, nil, webhookWorkload("w", old), pod)
		r := &TensorFusionWorkloadReconciler{Client: c}

		deleted, _ := collectOrphan(t, r, "w")
		require.False(t, deleted)
		stored := &tfv1.TensorFusionWorkload{}
		require.NoError(t, c.Get(context.Background(), client.ObjectKey{Namespace: orphanTestNamespace, Name: "w"}, stored))
		seen, err := time.Parse(time.RFC3339, stored.Annotations[constants.WorkloadLastPodSeenAnnotation])
		require.NoError(t, err)
		assert.WithinDuration(t, time.Now(), seen, durationTolerance)

		require.NoError(t, c.Delete(context.Background(), pod))
		deleted, recheck := collectOrphan(t, r, "w")

		assert.False(t, deleted, "the workload is old, but its pod was there a moment ago")
		assert.True(t, workloadExists(t, c, "w"))
		assert.InDelta(t, float64(WorkloadOrphanGracePeriod), float64(recheck), float64(durationTolerance))
	})

	t.Run("a recent record is not written again", func(t *testing.T) {
		pod := orphanTestPod("p", orphanTestNamespace, map[string]string{constants.WorkloadKey: "w"}, nil)
		workload := seenAgo(webhookWorkload("w", old), workloadPodSeenRefreshInterval/2)
		recorded := workload.Annotations[constants.WorkloadLastPodSeenAnnotation]
		patches := 0
		c := newOrphanTestClient(t, &interceptor.Funcs{
			Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
				patches++
				return c.Patch(ctx, obj, patch, opts...)
			},
		}, workload, pod)
		r := &TensorFusionWorkloadReconciler{Client: c}

		deleted, _ := collectOrphan(t, r, "w")

		assert.False(t, deleted)
		assert.Zero(t, patches)
		stored := &tfv1.TensorFusionWorkload{}
		require.NoError(t, c.Get(context.Background(), client.ObjectKey{Namespace: orphanTestNamespace, Name: "w"}, stored))
		assert.Equal(t, recorded, stored.Annotations[constants.WorkloadLastPodSeenAnnotation])
	})

	t.Run("a new workload with a pod is not written to", func(t *testing.T) {
		// The pod controller is about to set the owner reference; a write
		// here would only get in its way.
		pod := orphanTestPod("p", orphanTestNamespace, map[string]string{constants.WorkloadKey: "w"}, nil)
		c := newOrphanTestClient(t, nil, webhookWorkload("w", time.Second), pod)
		r := &TensorFusionWorkloadReconciler{Client: c}

		deleted, _ := collectOrphan(t, r, "w")

		assert.False(t, deleted)
		stored := &tfv1.TensorFusionWorkload{}
		require.NoError(t, c.Get(context.Background(), client.ObjectKey{Namespace: orphanTestNamespace, Name: "w"}, stored))
		assert.NotContains(t, stored.Annotations, constants.WorkloadLastPodSeenAnnotation)
	})

	t.Run("an unreadable record starts the period now instead of deleting", func(t *testing.T) {
		workload := webhookWorkload("w", old)
		workload.Annotations[constants.WorkloadLastPodSeenAnnotation] = "not a time"
		c := newOrphanTestClient(t, nil, workload)
		r := &TensorFusionWorkloadReconciler{Client: c}

		deleted, recheck := collectOrphan(t, r, "w")

		assert.False(t, deleted)
		assert.InDelta(t, float64(WorkloadOrphanGracePeriod), float64(recheck), float64(durationTolerance))
		stored := &tfv1.TensorFusionWorkload{}
		require.NoError(t, c.Get(context.Background(), client.ObjectKey{Namespace: orphanTestNamespace, Name: "w"}, stored))
		seen, err := time.Parse(time.RFC3339, stored.Annotations[constants.WorkloadLastPodSeenAnnotation])
		require.NoError(t, err)
		assert.WithinDuration(t, time.Now(), seen, durationTolerance)
	})
}

// Everything the decision uses is stored on the workload, so a reconciler that
// has just started (an operator restart) decides exactly as the one before it.
func TestWorkloadOrphanRestartSafety(t *testing.T) {
	t.Run("a restart does not cut the grace period short", func(t *testing.T) {
		left := 2 * time.Minute
		c := newOrphanTestClient(t, nil, webhookWorkload("w", WorkloadOrphanGracePeriod-left))

		for range 3 {
			restarted := &TensorFusionWorkloadReconciler{Client: c}
			deleted, recheck := collectOrphan(t, restarted, "w")
			assert.False(t, deleted)
			assert.InDelta(t, float64(left), float64(recheck), float64(durationTolerance))
		}
		assert.True(t, workloadExists(t, c, "w"))
		stored := &tfv1.TensorFusionWorkload{}
		require.NoError(t, c.Get(context.Background(), client.ObjectKey{Namespace: orphanTestNamespace, Name: "w"}, stored))
		assert.NotContains(t, stored.Annotations, constants.WorkloadLastPodSeenAnnotation,
			"waiting writes nothing that could move the end of the period")
	})

	t.Run("a restart does not start the grace period again", func(t *testing.T) {
		// The reconciler before the restart saw this workload and waited; the
		// one after it finds the period over and does not wait a second time.
		workload := webhookWorkload("w", 10*WorkloadOrphanGracePeriod)
		workload.Annotations[constants.WorkloadLastPodSeenAnnotation] =
			time.Now().Add(-WorkloadOrphanGracePeriod - 2*time.Second).UTC().Format(time.RFC3339)
		c := newOrphanTestClient(t, nil, workload)

		restarted := &TensorFusionWorkloadReconciler{Client: c}
		deleted, _ := collectOrphan(t, restarted, "w")

		assert.True(t, deleted)
		assert.False(t, workloadExists(t, c, "w"))
	})
}

// A pod stored a moment ago may not be in the informer cache yet: the list
// that precedes the delete goes to the API server.
func TestWorkloadOrphanConfirmsWithAPIServerBeforeDeleting(t *testing.T) {
	workload := webhookWorkload("w", 2*WorkloadOrphanGracePeriod)
	pod := orphanTestPod("p", orphanTestNamespace, nil, map[string]string{constants.SelectedWorkloadAnnotation: "w"})
	cache := newOrphanTestClient(t, nil, workload.DeepCopy())
	apiServer := newOrphanTestClient(t, nil, workload.DeepCopy(), pod)
	r := &TensorFusionWorkloadReconciler{Client: cache, apiReader: apiServer}

	deleted, recheck := collectOrphan(t, r, "w")

	assert.False(t, deleted)
	assert.True(t, workloadExists(t, cache, "w"))
	assert.Equal(t, workloadPodSeenRefreshInterval, recheck)
}

// The webhook records an admission on the workload it reuses. A delete decided
// on the workload as it was before that record must not go through.
func TestWorkloadOrphanDeleteRefusedWhenWorkloadChangedSinceRead(t *testing.T) {
	workload := webhookWorkload("w", 2*WorkloadOrphanGracePeriod)
	admitted := false
	c := newOrphanTestClient(t, &interceptor.Funcs{
		Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
			if !admitted {
				admitted = true
				current := &tfv1.TensorFusionWorkload{}
				if err := c.Get(ctx, client.ObjectKeyFromObject(obj), current); err != nil {
					return err
				}
				patch := client.MergeFrom(current.DeepCopy())
				current.Annotations[constants.WorkloadLastPodSeenAnnotation] = time.Now().UTC().Format(time.RFC3339)
				if err := c.Patch(ctx, current, patch); err != nil {
					return err
				}
			}
			return c.Delete(ctx, obj, opts...)
		},
	}, workload)
	r := &TensorFusionWorkloadReconciler{Client: c}

	deleted, recheck := collectOrphan(t, r, "w")

	require.True(t, admitted)
	assert.False(t, deleted)
	assert.True(t, workloadExists(t, c, "w"))
	assert.Equal(t, workloadOrphanRecheckSlack, recheck, "decided again right away")

	// Decided again, the admission just recorded keeps the workload.
	deleted, recheck = collectOrphan(t, r, "w")
	assert.False(t, deleted)
	assert.True(t, workloadExists(t, c, "w"))
	assert.InDelta(t, float64(WorkloadOrphanGracePeriod), float64(recheck), float64(durationTolerance))
}

// Deleting goes through the workload's finalizer like any other deletion: the
// first reconcile deletes, the one triggered by that deletion takes the
// finalizer off, and nothing else is written.
func TestWorkloadOrphanReconcileDeletesThroughFinalizer(t *testing.T) {
	workload := webhookWorkload("w", 2*WorkloadOrphanGracePeriod)
	workload.Finalizers = []string{constants.Finalizer}
	c := newOrphanTestClient(t, nil, workload)
	r := &TensorFusionWorkloadReconciler{Client: c}
	req := ctrl.Request{NamespacedName: client.ObjectKey{Namespace: orphanTestNamespace, Name: "w"}}

	res, err := r.Reconcile(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, res)
	stored := &tfv1.TensorFusionWorkload{}
	require.NoError(t, c.Get(context.Background(), req.NamespacedName, stored))
	assert.False(t, stored.DeletionTimestamp.IsZero(), "deleted, waiting for its finalizer")

	res, err = r.Reconcile(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, res)
	assert.False(t, workloadExists(t, c, "w"))
}

// A workload that is waiting out its grace period is reconciled as usual and
// queued again for the end of the period.
func TestWorkloadOrphanReconcileRequeuesForEndOfGrace(t *testing.T) {
	left := 2 * time.Minute
	workload := webhookWorkload("w", WorkloadOrphanGracePeriod-left)
	workload.Finalizers = []string{constants.Finalizer}
	workload.Spec.PoolName = "pool"
	pool := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "pool"}}
	c := newOrphanTestClient(t, nil, workload, pool)
	r := &TensorFusionWorkloadReconciler{Client: c}
	req := ctrl.Request{NamespacedName: client.ObjectKey{Namespace: orphanTestNamespace, Name: "w"}}

	for range 3 {
		res, err := r.Reconcile(context.Background(), req)
		require.NoError(t, err)
		assert.InDelta(t, float64(left), float64(res.RequeueAfter), float64(durationTolerance))
	}
	assert.True(t, workloadExists(t, c, "w"))
}

var _ = Describe("TensorFusionWorkload orphan collection", func() {
	// The suite's own workload controller runs with the real grace period, so
	// it leaves these workloads alone apart from its usual handling (finalizer,
	// status). A second reconciler with a short grace period is called here
	// against the same API server; when it deletes, the suite's controller is
	// what takes the finalizer off.
	const grace = 4 * time.Second
	var tfEnv *TensorFusionEnv
	orphanKey := client.ObjectKey{Name: "orphan-no-pod", Namespace: "default"}
	usedKey := client.ObjectKey{Name: "orphan-with-pod", Namespace: "default"}
	podKey := client.ObjectKey{Name: "orphan-with-pod-client", Namespace: "default"}

	newWebhookWorkload := func(key client.ObjectKey, poolName string) *tfv1.TensorFusionWorkload {
		return &tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      key.Name,
				Namespace: key.Namespace,
				Labels:    map[string]string{constants.GpuPoolKey: poolName},
				Annotations: map[string]string{
					constants.WorkloadModeAnnotation: constants.WorkloadModeDynamic,
				},
			},
			Spec: tfv1.WorkloadProfileSpec{
				PoolName: poolName,
				Resources: tfv1.Resources{
					Requests: tfv1.Resource{Tflops: resource.MustParse("10"), Vram: resource.MustParse("1Gi")},
					Limits:   tfv1.Resource{Tflops: resource.MustParse("10"), Vram: resource.MustParse("1Gi")},
				},
				Qos:        constants.QoSLevelMedium,
				IsLocalGPU: true,
			},
		}
	}

	BeforeEach(func() {
		tfEnv = NewTensorFusionEnvBuilder().
			AddPoolWithNodeCount(1).SetGpuCountPerNode(1).
			Build()
	})
	AfterEach(func() {
		for _, obj := range []client.Object{
			&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: podKey.Name, Namespace: podKey.Namespace}},
			&tfv1.TensorFusionWorkload{ObjectMeta: metav1.ObjectMeta{Name: orphanKey.Name, Namespace: orphanKey.Namespace}},
			&tfv1.TensorFusionWorkload{ObjectMeta: metav1.ObjectMeta{Name: usedKey.Name, Namespace: usedKey.Namespace}},
		} {
			Expect(client.IgnoreNotFound(k8sClient.Delete(ctx, obj, client.GracePeriodSeconds(0)))).To(Succeed())
		}
		Eventually(func(g Gomega) {
			g.Expect(apierrors.IsNotFound(k8sClient.Get(ctx, orphanKey, &tfv1.TensorFusionWorkload{}))).To(BeTrue())
			g.Expect(apierrors.IsNotFound(k8sClient.Get(ctx, usedKey, &tfv1.TensorFusionWorkload{}))).To(BeTrue())
		}).Should(Succeed())
		tfEnv.Cleanup()
	})

	It("removes a webhook-created workload whose pod never appears and keeps one whose pod exists", func() {
		pool := tfEnv.GetGPUPool(0)
		r := &TensorFusionWorkloadReconciler{
			Client:            k8sClient,
			apiReader:         k8sClient,
			orphanGracePeriod: grace,
		}
		usedWorkload := newWebhookWorkload(usedKey, pool.Name)
		pod := orphanTestPod(podKey.Name, podKey.Namespace, map[string]string{constants.WorkloadKey: usedKey.Name}, nil)

		By("keeping the workload while its pod could still be on its way")
		Expect(k8sClient.Create(ctx, newWebhookWorkload(orphanKey, pool.Name))).To(Succeed())
		deleted, recheck, err := r.collectOrphanWorkload(ctx, orphanKey)
		Expect(err).NotTo(HaveOccurred())
		Expect(deleted).To(BeFalse())
		Expect(recheck).To(And(BeNumerically(">", 0), BeNumerically("<=", grace+workloadOrphanRecheckSlack)))
		Expect(k8sClient.Get(ctx, orphanKey, &tfv1.TensorFusionWorkload{})).To(Succeed())
		Expect(k8sClient.Create(ctx, usedWorkload)).To(Succeed())
		Expect(k8sClient.Create(ctx, pod)).To(Succeed())

		By("deleting it once the grace period has passed without a pod")
		Eventually(func(g Gomega) {
			deleted, _, err := r.collectOrphanWorkload(ctx, orphanKey)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(deleted).To(BeTrue())
		}).WithTimeout(30 * time.Second).WithPolling(500 * time.Millisecond).Should(Succeed())
		Eventually(func(g Gomega) {
			err := k8sClient.Get(ctx, orphanKey, &tfv1.TensorFusionWorkload{})
			g.Expect(apierrors.IsNotFound(err)).To(BeTrue(), "the finalizer is released and the workload is gone")
		}).WithTimeout(30 * time.Second).Should(Succeed())

		By("keeping the workload whose pod exists, well past the grace period")
		Consistently(func(g Gomega) {
			deleted, recheck, err := r.collectOrphanWorkload(ctx, usedKey)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(deleted).To(BeFalse())
			g.Expect(recheck).To(BeNumerically(">", 0))
		}).WithTimeout(2 * grace).WithPolling(500 * time.Millisecond).Should(Succeed())
		used := &tfv1.TensorFusionWorkload{}
		Expect(k8sClient.Get(ctx, usedKey, used)).To(Succeed())
		Expect(used.DeletionTimestamp.IsZero()).To(BeTrue())
		Expect(used.Annotations).To(HaveKey(constants.WorkloadLastPodSeenAnnotation))

		By("deleting that one too, a grace period after its pod is gone")
		Expect(k8sClient.Delete(ctx, pod, client.GracePeriodSeconds(0))).To(Succeed())
		Eventually(func(g Gomega) {
			g.Expect(apierrors.IsNotFound(k8sClient.Get(ctx, podKey, &corev1.Pod{}))).To(BeTrue())
		}).Should(Succeed())
		Eventually(func(g Gomega) {
			deleted, _, err := r.collectOrphanWorkload(ctx, usedKey)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(deleted).To(BeTrue())
		}).WithTimeout(30 * time.Second).WithPolling(500 * time.Millisecond).Should(Succeed())
		Eventually(func(g Gomega) {
			g.Expect(apierrors.IsNotFound(k8sClient.Get(ctx, usedKey, &tfv1.TensorFusionWorkload{}))).To(BeTrue())
		}).WithTimeout(30 * time.Second).Should(Succeed())
	})
})
