package v1

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

// The workload controller removes a workload this webhook created once it has
// had neither an owner reference nor a pod for a grace period. A pod admitted
// against such a workload is not stored yet, so the admission has to leave a
// mark on the workload.

type reuseTestWrites struct {
	creates, patches int
}

func newReuseTestMutator(t *testing.T, objs ...client.Object) (*TensorFusionPodMutator, client.Client, *reuseTestWrites) {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	require.NoError(t, tfv1.AddToScheme(scheme))
	writes := &reuseTestWrites{}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).
		WithInterceptorFuncs(interceptor.Funcs{
			Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
				writes.creates++
				return c.Create(ctx, obj, opts...)
			},
			Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
				writes.patches++
				return c.Patch(ctx, obj, patch, opts...)
			},
		}).Build()
	return &TensorFusionPodMutator{Client: c}, c, writes
}

func reuseTestInfo() *utils.TensorFusionInfo {
	return &utils.TensorFusionInfo{
		Profile:      &tfv1.WorkloadProfileSpec{PoolName: "pool"},
		WorkloadName: "standalone",
	}
}

func reuseTestPod() *corev1.Pod {
	return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "standalone", Namespace: "ns"}}
}

// reuseTestWorkload is the workload an earlier admission of the same pod name
// left behind: created by the webhook, no owner reference.
func reuseTestWorkload() *tfv1.TensorFusionWorkload {
	return &tfv1.TensorFusionWorkload{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "standalone",
			Namespace:         "ns",
			CreationTimestamp: metav1.NewTime(time.Now().Add(-time.Hour)),
			Annotations: map[string]string{
				constants.WorkloadModeAnnotation: constants.WorkloadModeDynamic,
			},
		},
		Spec: tfv1.WorkloadProfileSpec{PoolName: "pool"},
	}
}

func getReuseTestWorkload(t *testing.T, c client.Client) *tfv1.TensorFusionWorkload {
	t.Helper()
	workload := &tfv1.TensorFusionWorkload{}
	require.NoError(t, c.Get(context.Background(), client.ObjectKey{Namespace: "ns", Name: "standalone"}, workload))
	return workload
}

func TestCreateOrUpdateWorkloadRecordsAdmissionOnReusedWorkloadWithoutOwner(t *testing.T) {
	m, c, writes := newReuseTestMutator(t, reuseTestWorkload())
	before := getReuseTestWorkload(t, c)

	workload, err := m.createOrUpdateWorkload(context.Background(), reuseTestPod(), reuseTestInfo(), false)
	require.NoError(t, err)

	assert.Equal(t, 1, writes.patches)
	stored := getReuseTestWorkload(t, c)
	seen, err := time.Parse(time.RFC3339, stored.Annotations[constants.WorkloadLastPodSeenAnnotation])
	require.NoError(t, err)
	assert.WithinDuration(t, time.Now(), seen, 5*time.Second)
	assert.Equal(t, stored.Annotations, workload.Annotations)

	// A delete decided on the workload as it was before this admission fails.
	err = c.Delete(context.Background(), before, client.Preconditions{ResourceVersion: &before.ResourceVersion})
	assert.True(t, apierrors.IsConflict(err), "got %v", err)
}

func TestCreateOrUpdateWorkloadLeavesOtherWorkloadsUnmarked(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(w *tfv1.TensorFusionWorkload)
	}{
		{"has an owner reference", func(w *tfv1.TensorFusionWorkload) {
			w.OwnerReferences = []metav1.OwnerReference{{APIVersion: "apps/v1", Kind: "Deployment", Name: "d", UID: "d-uid"}}
		}},
		{"created by a user", func(w *tfv1.TensorFusionWorkload) { w.Annotations = nil }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			existing := reuseTestWorkload()
			tc.mutate(existing)
			m, c, writes := newReuseTestMutator(t, existing)

			_, err := m.createOrUpdateWorkload(context.Background(), reuseTestPod(), reuseTestInfo(), false)
			require.NoError(t, err)

			assert.Zero(t, writes.patches)
			assert.NotContains(t, getReuseTestWorkload(t, c).Annotations, constants.WorkloadLastPodSeenAnnotation)
		})
	}

	t.Run("just created by this admission", func(t *testing.T) {
		// Its creation time is what the grace period counts from.
		m, c, writes := newReuseTestMutator(t)

		_, err := m.createOrUpdateWorkload(context.Background(), reuseTestPod(), reuseTestInfo(), false)
		require.NoError(t, err)

		assert.Equal(t, 1, writes.creates)
		assert.Zero(t, writes.patches)
		assert.NotContains(t, getReuseTestWorkload(t, c).Annotations, constants.WorkloadLastPodSeenAnnotation)
	})
}

func TestCreateOrUpdateWorkloadDryRunRecordsNothing(t *testing.T) {
	m, c, writes := newReuseTestMutator(t, reuseTestWorkload())

	_, err := m.createOrUpdateWorkload(context.Background(), reuseTestPod(), reuseTestInfo(), true)
	require.NoError(t, err)

	assert.Zero(t, writes.patches)
	assert.NotContains(t, getReuseTestWorkload(t, c).Annotations, constants.WorkloadLastPodSeenAnnotation)
}

func TestCreateOrUpdateWorkloadRefusesWorkloadBeingDeleted(t *testing.T) {
	existing := reuseTestWorkload()
	existing.Finalizers = []string{constants.Finalizer}
	deletedAt := metav1.NewTime(time.Now())
	existing.DeletionTimestamp = &deletedAt
	m, _, _ := newReuseTestMutator(t, existing)

	_, err := m.createOrUpdateWorkload(context.Background(), reuseTestPod(), reuseTestInfo(), false)

	var terminating *workloadTerminatingError
	require.ErrorAs(t, err, &terminating)
	assert.Equal(t, "the previous TensorFusionWorkload ns/standalone is still being deleted, retry the request in a moment",
		err.Error())
}
