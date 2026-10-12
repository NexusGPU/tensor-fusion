package controller

import (
	"context"
	"testing"
	"time"

	webhookv1 "github.com/NexusGPU/tensor-fusion/internal/webhook/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// The owner counts three injected pods and two exist: the third was admitted
// and never stored. Reconciling one of the two gives the slot back once the
// owner's last reservation is older than the grace period, and until then asks
// to be queued again for that moment.
func TestPodReconcileReleasesLeakedCounterSlots(t *testing.T) {
	const counterKey = "tensor-fusion.ai/tf-counter-abc"
	for _, tc := range []struct {
		name         string
		reservedAgo  time.Duration
		counted      string
		wantCount    string
		wantRequeued bool
	}{
		{name: "after the grace period", reservedAgo: time.Hour, counted: "3", wantCount: "2"},
		{
			name: "inside the grace period", reservedAgo: 30 * time.Second,
			counted: "3", wantCount: "3", wantRequeued: true,
		},
		{name: "every slot has its pod", reservedAgo: time.Hour, counted: "2", wantCount: "2"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			require.NoError(t, appsv1.AddToScheme(scheme))

			reservedAt := time.Now().Add(-tc.reservedAgo).UTC().Format(time.RFC3339Nano)
			rs := &appsv1.ReplicaSet{ObjectMeta: metav1.ObjectMeta{
				Namespace: "default", Name: "rs", UID: "rs-uid",
				Annotations: map[string]string{
					counterKey: tc.counted,
					webhookv1.PodCounterLastReserveAnnotation: reservedAt,
				},
			}}
			isController := true
			injectedPod := func(name string) *corev1.Pod {
				return &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:   "default",
						Name:        name,
						Labels:      map[string]string{constants.LabelComponent: constants.ComponentClient},
						Annotations: map[string]string{constants.TensorFusionPodCounterKeyAnnotation: counterKey},
						OwnerReferences: []metav1.OwnerReference{{
							APIVersion: "apps/v1", Kind: "ReplicaSet", Name: rs.Name, UID: rs.UID, Controller: &isController,
						}},
					},
					Status: corev1.PodStatus{Phase: corev1.PodRunning},
				}
			}
			c := fake.NewClientBuilder().WithScheme(scheme).
				WithObjects(rs, injectedPod("pod-0"), injectedPod("pod-1")).Build()
			r := &PodReconciler{Client: c, Scheme: scheme}

			req := ctrl.Request{NamespacedName: client.ObjectKey{Namespace: "default", Name: "pod-0"}}
			res, err := r.Reconcile(ctx, req)
			require.NoError(t, err)

			got := &appsv1.ReplicaSet{}
			require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(rs), got))
			require.Equal(t, tc.wantCount, got.Annotations[counterKey])
			if tc.wantRequeued {
				require.Greater(t, res.RequeueAfter, time.Minute, "queued again for the end of the grace period")
				require.LessOrEqual(t, res.RequeueAfter, webhookv1.PodCounterLeakGracePeriod)
			} else {
				require.Zero(t, res.RequeueAfter)
			}
		})
	}
}
