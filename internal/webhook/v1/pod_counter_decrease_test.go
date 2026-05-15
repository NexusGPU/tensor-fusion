package v1

import (
	"context"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/assert"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func makeCounterPod(name string, ownerKey string) *corev1.Pod {
	tt := true
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns",
			Name:      name,
			Annotations: map[string]string{
				constants.TensorFusionPodCounterKeyAnnotation: ownerKey,
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: "apps/v1",
				Kind:       "ReplicaSet",
				Name:       "rs",
				UID:        "rs-uid",
				Controller: &tt,
			}},
		},
	}
}

// TestDecrease_RetriesOnConflict reproduces the failure mode where two
// Decrease callers concurrently bump the same owner annotation and the
// second Update fires before the first commits. Previously a single
// conflict propagated to pod_controller, which then SWALLOWED it and
// proceeded to remove the finalizer, leaking the counter past the pod's
// lifetime. The fix wraps Decrease in retry.RetryOnConflict so transient
// races recover and only persistent errors surface.
func TestDecrease_RetriesOnConflict(t *testing.T) {
	scheme := runtime.NewScheme()
	assert.NoError(t, corev1.AddToScheme(scheme))
	assert.NoError(t, appsv1.AddToScheme(scheme))

	rs := &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:   "ns",
			Name:        "rs",
			UID:         "rs-uid",
			Annotations: map[string]string{"counter-key": "5"},
		},
	}

	// First N Updates return Conflict, then the real fake-client write succeeds.
	const firstConflicts = 2
	var attempts int32
	conflictyClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(rs).
		WithInterceptorFuncs(interceptor.Funcs{
			Update: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.UpdateOption) error {
				if n := atomic.AddInt32(&attempts, 1); n <= firstConflicts {
					return apierrors.NewConflict(
						schema.GroupResource{Group: "apps", Resource: "replicasets"},
						"rs", assert.AnError)
				}
				return c.Update(ctx, obj, opts...)
			},
		}).
		Build()

	counter := &TensorFusionPodCounter{Client: conflictyClient}
	pod := makeCounterPod("pod-1", "counter-key")
	assert.NoError(t, counter.Decrease(context.Background(), pod),
		"Decrease should swallow transient conflicts via internal retry")

	// Verify the annotation was decremented exactly once (5 -> 4).
	got := &appsv1.ReplicaSet{}
	assert.NoError(t, conflictyClient.Get(context.Background(), client.ObjectKey{Namespace: "ns", Name: "rs"}, got))
	v, _ := strconv.Atoi(got.Annotations["counter-key"])
	assert.Equal(t, 4, v, "counter must end at 4 (5 - 1), not 3 or 5")
	assert.Equal(t, int32(firstConflicts+1), atomic.LoadInt32(&attempts),
		"retry.RetryOnConflict should have retried %d times before succeeding", firstConflicts)
}

// TestDecrease_PropagatesPersistentError ensures a non-conflict Update error
// (RBAC-style) is NOT silently swallowed: it must propagate so pod_controller
// can keep the finalizer in place and requeue, surfacing the failure to a
// human instead of leaking the counter on the owner.
func TestDecrease_PropagatesPersistentError(t *testing.T) {
	scheme := runtime.NewScheme()
	assert.NoError(t, corev1.AddToScheme(scheme))
	assert.NoError(t, appsv1.AddToScheme(scheme))

	rs := &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:   "ns",
			Name:        "rs",
			UID:         "rs-uid",
			Annotations: map[string]string{"counter-key": "5"},
		},
	}
	rbacClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(rs).
		WithInterceptorFuncs(interceptor.Funcs{
			Update: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.UpdateOption) error {
				return apierrors.NewForbidden(
					schema.GroupResource{Group: "apps", Resource: "replicasets"},
					"rs", assert.AnError)
			},
		}).
		Build()

	counter := &TensorFusionPodCounter{Client: rbacClient}
	pod := makeCounterPod("pod-2", "counter-key")
	err := counter.Decrease(context.Background(), pod)
	assert.Error(t, err, "persistent (non-conflict) errors must propagate so pod_controller keeps the finalizer")
}
