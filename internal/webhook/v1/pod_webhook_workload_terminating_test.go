package v1

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	admissionv1 "k8s.io/api/admission/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

// A pod whose workload is still being deleted is refused for a moment only.
// The status of the refusal decides whether its creator repeats the request.

const terminatingTestMessage = "the previous TensorFusionWorkload ns/standalone is still being deleted, " +
	"retry the request in a moment"

// handleStandalonePod runs Handle for a pod without an owner, named like the
// workload of reuseTestWorkload, against a pool and the given workloads.
func handleStandalonePod(t *testing.T, funcs interceptor.Funcs, workloads ...client.Object) admission.Response {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	require.NoError(t, tfv1.AddToScheme(scheme))
	pool := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "mock"}}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(append(workloads, pool)...).
		WithInterceptorFuncs(funcs).Build()
	m := &TensorFusionPodMutator{Client: c, decoder: admission.NewDecoder(scheme)}

	raw, err := json.Marshal(newAdmissionTestPod("ns", nil, ""))
	require.NoError(t, err)
	return m.Handle(context.Background(), admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
		Object:    runtime.RawExtension{Raw: raw},
		Operation: admissionv1.Create,
		Namespace: "ns",
	}})
}

func TestHandleAnswersTerminatingWorkloadWithRetriableStatus(t *testing.T) {
	existing := reuseTestWorkload()
	existing.Finalizers = []string{constants.Finalizer}
	deletedAt := metav1.NewTime(time.Now())
	existing.DeletionTimestamp = &deletedAt

	resp := handleStandalonePod(t, interceptor.Funcs{}, existing)

	assert.False(t, resp.Allowed)
	require.NotNil(t, resp.Result)
	assert.Equal(t, int32(http.StatusServiceUnavailable), resp.Result.Code)
	assert.Equal(t, metav1.StatusReasonServiceUnavailable, resp.Result.Reason)
	assert.Equal(t, metav1.StatusFailure, resp.Result.Status)
	assert.Equal(t, terminatingTestMessage, resp.Result.Message)
	require.NotNil(t, resp.Result.Details)
	assert.Equal(t, int32(workloadTerminatingRetryAfterSeconds), resp.Result.Details.RetryAfterSeconds)
	assert.Positive(t, resp.Result.Details.RetryAfterSeconds)
}

func TestHandleKeepsStatusOfOtherWorkloadErrors(t *testing.T) {
	resp := handleStandalonePod(t, interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			if _, ok := obj.(*tfv1.TensorFusionWorkload); ok {
				return errors.New("storage is unavailable")
			}
			return c.Get(ctx, key, obj, opts...)
		},
	})

	assert.False(t, resp.Allowed)
	require.NotNil(t, resp.Result)
	assert.Equal(t, int32(http.StatusInternalServerError), resp.Result.Code)
	assert.Empty(t, resp.Result.Reason)
	assert.Equal(t, "create tf workload: failed to get workload: storage is unavailable", resp.Result.Message)
	assert.Nil(t, resp.Result.Details)
}

// The retriable status is for the typed error only, also when it is wrapped.
func TestWorkloadErrorResponseRetriableOnlyForTerminatingWorkload(t *testing.T) {
	terminating := &workloadTerminatingError{namespace: "ns", name: "standalone"}
	for _, err := range []error{terminating, errors.Join(errors.New("context"), terminating)} {
		resp := workloadErrorResponse(err)
		assert.Equal(t, int32(http.StatusServiceUnavailable), resp.Result.Code)
		assert.Equal(t, terminatingTestMessage, resp.Result.Message)
		assert.Equal(t, int32(workloadTerminatingRetryAfterSeconds), resp.Result.Details.RetryAfterSeconds)
	}

	// Same words, not the typed error.
	resp := workloadErrorResponse(errors.New(terminatingTestMessage))
	assert.Equal(t, int32(http.StatusInternalServerError), resp.Result.Code)
	assert.Equal(t, "create tf workload: "+terminatingTestMessage, resp.Result.Message)
	assert.Nil(t, resp.Result.Details)
}

func terminatingTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	require.NoError(t, tfv1.AddToScheme(scheme))
	return scheme
}

func deploymentRef(uid string) *metav1.OwnerReference {
	isController := true
	return &metav1.OwnerReference{
		APIVersion: "apps/v1", Kind: "Deployment", Name: "standalone", UID: types.UID(uid), Controller: &isController,
	}
}

// Only a controller of the same kind, API group and name with another UID is a
// replaced owner.
func TestOwnerWasReplaced(t *testing.T) {
	with := func(mutate func(ref *metav1.OwnerReference)) *metav1.OwnerReference {
		ref := deploymentRef("old-uid")
		mutate(ref)
		return ref
	}
	notController := false
	for _, tc := range []struct {
		name          string
		workloadOwner *metav1.OwnerReference
		podController *metav1.OwnerReference
		want          bool
	}{
		{"same name and kind, another UID", deploymentRef("old-uid"), deploymentRef("new-uid"), true},
		{"another version of the same API group", with(func(ref *metav1.OwnerReference) { ref.APIVersion = "apps/v1beta2" }),
			deploymentRef("new-uid"), true},
		{"the pod's own controller", deploymentRef("uid"), deploymentRef("uid"), false},
		{"workload without an owner (a user's, or a pod's without a controller)", nil, deploymentRef("new-uid"), false},
		{"pod without a controller", deploymentRef("old-uid"), nil, false},
		{"another kind under the same name", with(func(ref *metav1.OwnerReference) { ref.Kind = "StatefulSet" }),
			deploymentRef("new-uid"), false},
		{"the same kind of another API group", with(func(ref *metav1.OwnerReference) { ref.APIVersion = "example.com/v1" }),
			deploymentRef("new-uid"), false},
		{"another name", with(func(ref *metav1.OwnerReference) { ref.Name = "other" }), deploymentRef("new-uid"), false},
		{"an owner reference that is not the controller", with(func(ref *metav1.OwnerReference) { ref.Controller = &notController }),
			deploymentRef("new-uid"), false},
		{"pod controller without a UID", deploymentRef("old-uid"), deploymentRef(""), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			workload := reuseTestWorkload()
			if tc.workloadOwner != nil {
				workload.OwnerReferences = []metav1.OwnerReference{*tc.workloadOwner}
			}
			assert.Equal(t, tc.want, ownerWasReplaced(workload, tc.podController))
		})
	}
}

func TestCreateOrUpdateWorkloadRefusesWorkloadOfReplacedOwner(t *testing.T) {
	existing := reuseTestWorkload()
	existing.OwnerReferences = []metav1.OwnerReference{*deploymentRef("old-uid")}
	m, c, writes := newReuseTestMutator(t, existing)
	info := reuseTestInfo()
	info.PodControllerRef = deploymentRef("new-uid")

	for _, dryRun := range []bool{false, true} {
		_, err := m.createOrUpdateWorkload(context.Background(), reuseTestPod(), info, dryRun)

		var terminating *workloadTerminatingError
		require.ErrorAs(t, err, &terminating)
		assert.Equal(t, "the previous TensorFusionWorkload ns/standalone belongs to a replaced owner of the same name "+
			"and is about to be deleted, retry the request in a moment", err.Error())
	}
	assert.Zero(t, writes.creates+writes.patches)
	assert.Equal(t, types.UID("old-uid"), getReuseTestWorkload(t, c).OwnerReferences[0].UID)

	// The same pod against the workload of its own controller is admitted.
	info.PodControllerRef = deploymentRef("old-uid")
	_, err := m.createOrUpdateWorkload(context.Background(), reuseTestPod(), info, false)
	require.NoError(t, err)
}

// The workload another admission created a moment ago is not in the informer
// cache yet; the read after the refused create has to go to the API server.
func TestCreateOrUpdateWorkloadReadsConcurrentlyCreatedWorkloadFromAPIServer(t *testing.T) {
	scheme := terminatingTestScheme(t)
	concurrent := reuseTestWorkload()
	concurrent.OwnerReferences = []metav1.OwnerReference{*deploymentRef("uid")}
	info := reuseTestInfo()
	info.PodControllerRef = deploymentRef("uid")
	// The cache: without the workload, and its create is refused because the
	// API server has one.
	newCache := func() client.Client {
		return fake.NewClientBuilder().WithScheme(scheme).WithInterceptorFuncs(interceptor.Funcs{
			Create: func(context.Context, client.WithWatch, client.Object, ...client.CreateOption) error {
				return apierrors.NewAlreadyExists(tfv1.GroupVersion.WithResource("tensorfusionworkloads").GroupResource(), "standalone")
			},
		}).Build()
	}

	t.Run("read from the API server", func(t *testing.T) {
		apiServer := fake.NewClientBuilder().WithScheme(scheme).WithObjects(concurrent.DeepCopy()).Build()
		m := &TensorFusionPodMutator{Client: newCache(), APIReader: apiServer}

		workload, err := m.createOrUpdateWorkload(context.Background(), reuseTestPod(), info, false)

		require.NoError(t, err)
		assert.Equal(t, types.UID("uid"), workload.OwnerReferences[0].UID)
	})

	t.Run("gone again by the time it is read: retriable", func(t *testing.T) {
		apiServer := fake.NewClientBuilder().WithScheme(scheme).Build()
		m := &TensorFusionPodMutator{Client: newCache(), APIReader: apiServer}

		_, err := m.createOrUpdateWorkload(context.Background(), reuseTestPod(), info, false)

		var terminating *workloadTerminatingError
		require.ErrorAs(t, err, &terminating)
		assert.Equal(t, "the previous TensorFusionWorkload ns/standalone was deleted while this pod was being admitted, "+
			"retry the request in a moment", err.Error())
		assert.Equal(t, int32(http.StatusServiceUnavailable), workloadErrorResponse(err).Result.Code)
	})

	t.Run("without an API reader the client is used", func(t *testing.T) {
		m := &TensorFusionPodMutator{Client: newCache()}

		_, err := m.createOrUpdateWorkload(context.Background(), reuseTestPod(), info, false)

		var terminating *workloadTerminatingError
		require.ErrorAs(t, err, &terminating)
	})
}

// The cache still holds a workload that the API server has already removed.
func TestHandleAnswersWorkloadGoneAtPatchWithRetriableStatus(t *testing.T) {
	resp := handleStandalonePod(t, interceptor.Funcs{
		Patch: func(_ context.Context, _ client.WithWatch, obj client.Object, _ client.Patch, _ ...client.PatchOption) error {
			return apierrors.NewNotFound(tfv1.GroupVersion.WithResource("tensorfusionworkloads").GroupResource(), obj.GetName())
		},
	}, reuseTestWorkload())

	assert.False(t, resp.Allowed)
	require.NotNil(t, resp.Result)
	assert.Equal(t, int32(http.StatusServiceUnavailable), resp.Result.Code)
	assert.Equal(t, metav1.StatusReasonServiceUnavailable, resp.Result.Reason)
	assert.Equal(t, "the previous TensorFusionWorkload ns/standalone was deleted while this pod was being admitted, "+
		"retry the request in a moment", resp.Result.Message)
	require.NotNil(t, resp.Result.Details)
	assert.Equal(t, int32(workloadTerminatingRetryAfterSeconds), resp.Result.Details.RetryAfterSeconds)
}
