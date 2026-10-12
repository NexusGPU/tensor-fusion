package kubernetes

import (
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// An Event the hypervisor records has to be attached to the Pod it is about,
// by UID, so that it shows up with that Pod and not with a later one of the
// same name.
func TestEmitPodEventRecordsAnEventOnThePod(t *testing.T) {
	c := fake.NewClientBuilder().WithScheme(scheme).Build()
	backend := &KubeletBackend{apiClient: NewAPIClient(t.Context(), c), nodeName: "node-1"}

	require.NoError(t, backend.EmitPodEvent(t.Context(),
		"team-a", "worker-pod", "pod-uid", corev1.EventTypeWarning, "LimiterStalledBucket", "message"))
	// A second Event of the same kind is an object of its own.
	require.NoError(t, backend.EmitPodEvent(t.Context(),
		"team-a", "worker-pod", "pod-uid", corev1.EventTypeWarning, "LimiterStalledBucket", "message"))

	var events corev1.EventList
	require.NoError(t, c.List(t.Context(), &events, client.InNamespace("team-a")))
	require.Len(t, events.Items, 2)
	for _, event := range events.Items {
		require.Equal(t, corev1.ObjectReference{
			APIVersion: "v1", Kind: "Pod", Namespace: "team-a", Name: "worker-pod", UID: types.UID("pod-uid"),
		}, event.InvolvedObject)
		require.Equal(t, corev1.EventTypeWarning, event.Type)
		require.Equal(t, "LimiterStalledBucket", event.Reason)
		require.Equal(t, "message", event.Message)
		require.Equal(t, podEventComponent, event.Source.Component)
		require.Equal(t, "node-1", event.Source.Host)
		require.EqualValues(t, 1, event.Count)
		require.False(t, event.FirstTimestamp.IsZero())
	}
}
