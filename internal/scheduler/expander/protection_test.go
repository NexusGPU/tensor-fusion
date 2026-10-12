package expander

import (
	"context"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/log"
	karpv1 "sigs.k8s.io/karpenter/pkg/apis/v1"
)

const protectionTestNamespace = "protection-ns"

func newProtectionTestClient(t *testing.T, objs ...client.Object) client.Client {
	t.Helper()
	require.NoError(t, tfv1.AddToScheme(scheme.Scheme))
	return fake.NewClientBuilder().WithScheme(scheme.Scheme).WithObjects(objs...).Build()
}

// restartedExpander is an expander as it is right after the operator started:
// it has a client and nothing in memory.
func restartedExpander(ctx context.Context, k8sClient client.Client) *NodeExpander {
	return &NodeExpander{client: k8sClient, ctx: ctx, logger: log.FromContext(ctx)}
}

func protectedAnnotations(pod *corev1.Pod, until time.Time) map[string]string {
	return map[string]string{
		karpv1.DoNotDisruptAnnotationKey:  constants.TrueStringValue,
		expansionProtectedUntilAnnotation: until.UTC().Format(time.RFC3339),
		expansionProtectedForAnnotation:   pod.Namespace + "/" + pod.Name + "/" + string(pod.UID),
		"example.com/note":                "keep-me",
	}
}

func protectedNodeClaimAndNode(name string, annotations map[string]string) (*karpv1.NodeClaim, *corev1.Node) {
	labels := map[string]string{constants.KarpenterExpansionLabel: name}
	nodeClaim := &karpv1.NodeClaim{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: labels, Annotations: annotations}}
	// Karpenter copies the NodeClaim's labels and annotations when the node registers.
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "ip-" + name, Labels: labels, Annotations: annotations}}
	return nodeClaim.DeepCopy(), node.DeepCopy()
}

func requireProtection(t *testing.T, k8sClient client.Client, obj client.Object, protected bool) {
	t.Helper()
	require.NoError(t, k8sClient.Get(context.Background(), client.ObjectKeyFromObject(obj), obj))
	annotations := obj.GetAnnotations()
	if protected {
		require.Equal(t, constants.TrueStringValue, annotations[karpv1.DoNotDisruptAnnotationKey], "%T %s", obj, obj.GetName())
		require.Contains(t, annotations, expansionProtectedUntilAnnotation)
		require.Contains(t, annotations, expansionProtectedForAnnotation)
	} else {
		require.NotContains(t, annotations, karpv1.DoNotDisruptAnnotationKey, "%T %s", obj, obj.GetName())
		require.NotContains(t, annotations, expansionProtectedUntilAnnotation)
		require.NotContains(t, annotations, expansionProtectedForAnnotation)
	}
	require.Equal(t, "keep-me", annotations["example.com/note"], "unrelated annotations stay")
}

// The NodeClaim the expander creates for a pending pod is shielded from
// Karpenter's disruption, with the pod and a deadline recorded on it, and the
// shield is lifted from the NodeClaim and its Node as soon as that pod is
// bound.
func TestExpansionNodeClaimIsProtectedUntilPodIsBound(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	source := &karpv1.NodeClaim{ObjectMeta: metav1.ObjectMeta{
		Name:        "source-claim",
		Annotations: map[string]string{"example.com/note": "keep-me"},
	}}
	pod := createTestTensorFusionPod("waiting", protectionTestNamespace, "100", "1Gi")
	k8sClient := newProtectionTestClient(t, source, pod.DeepCopy())
	allocator := gpuallocator.NewGpuAllocator(ctx, nil, k8sClient, time.Second)
	_, expander := NewUnscheduledPodHandler(ctx, nil, allocator, k8sClient, &events.FakeRecorder{})

	before := time.Now()
	preparedNode := createTestNode("expansion-1", source)
	require.NoError(t, expander.createKarpenterNodeClaimDirect(ctx, pod, preparedNode, source))

	nodeClaim := &karpv1.NodeClaim{ObjectMeta: metav1.ObjectMeta{Name: preparedNode.Name}}
	requireProtection(t, k8sClient, nodeClaim, true)
	require.Equal(t, protectionTestNamespace+"/waiting/"+string(pod.UID), nodeClaim.Annotations[expansionProtectedForAnnotation])
	until, err := time.Parse(time.RFC3339, nodeClaim.Annotations[expansionProtectedUntilAnnotation])
	require.NoError(t, err)
	require.WithinDuration(t, before.Add(expansionProtectionTTL), until, 5*time.Second)

	// The node registers and inherits the NodeClaim's labels and annotations.
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{
		Name: "ip-expansion-1", Labels: nodeClaim.Labels, Annotations: nodeClaim.Annotations,
	}}
	require.NoError(t, k8sClient.Create(ctx, node.DeepCopy()))

	// The pod is still pending: nothing is released.
	expander.releaseExpansionProtection(ctx, "")
	requireProtection(t, k8sClient, nodeClaim, true)
	requireProtection(t, k8sClient, node, true)

	// The pod is bound; the bind handler passes its UID because the client's
	// cache may not show the binding yet.
	expander.releaseExpansionProtection(ctx, pod.UID)
	requireProtection(t, k8sClient, nodeClaim, false)
	requireProtection(t, k8sClient, node, false)
}

// The shield ends without the operator having seen the pod being bound: every
// decision is taken from what is stored on the NodeClaim and the Node, so an
// operator started later lifts it as well.
func TestExpansionProtectionIsReleasedFromPersistedState(t *testing.T) {
	pending := createTestTensorFusionPod("pending", protectionTestNamespace, "100", "1Gi")
	bound := createTestTensorFusionPod("bound", protectionTestNamespace, "100", "1Gi")
	bound.Spec.NodeName = "some-node"
	deleted := createTestTensorFusionPod("deleted", protectionTestNamespace, "100", "1Gi")
	replaced := createTestTensorFusionPod("replaced", protectionTestNamespace, "100", "1Gi")
	replacedBefore := replaced.DeepCopy()
	replacedBefore.UID = types.UID("earlier-uid")

	inTime := time.Now().Add(10 * time.Minute)
	unreadable := protectedAnnotations(pending, inTime)
	unreadable[expansionProtectedUntilAnnotation] = "soon"

	tests := []struct {
		name        string
		annotations map[string]string
		released    bool
	}{
		{"pod still pending", protectedAnnotations(pending, inTime), false},
		{"pod bound", protectedAnnotations(bound, inTime), true},
		{"pod deleted", protectedAnnotations(deleted, inTime), true},
		{"pod replaced by one of the same name", protectedAnnotations(replacedBefore, inTime), true},
		{"deadline passed with the pod still pending", protectedAnnotations(pending, time.Now().Add(-time.Second)), true},
		{"unreadable deadline", unreadable, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodeClaim, node := protectedNodeClaimAndNode("expansion-x", tt.annotations)
			k8sClient := newProtectionTestClient(t, nodeClaim, node, pending.DeepCopy(), bound.DeepCopy(), replaced.DeepCopy())

			restartedExpander(context.Background(), k8sClient).releaseExpansionProtection(context.Background(), "")

			requireProtection(t, k8sClient, nodeClaim, !tt.released)
			requireProtection(t, k8sClient, node, !tt.released)
		})
	}
}

// A do-not-disrupt annotation the expander did not set is not its to remove.
func TestExpansionProtectionReleaseLeavesForeignAnnotationAlone(t *testing.T) {
	annotations := map[string]string{
		karpv1.DoNotDisruptAnnotationKey: constants.TrueStringValue,
		"example.com/note":               "keep-me",
	}
	nodeClaim, node := protectedNodeClaimAndNode("expansion-y", annotations)
	k8sClient := newProtectionTestClient(t, nodeClaim, node)

	restartedExpander(context.Background(), k8sClient).releaseExpansionProtection(context.Background(), "")

	for _, obj := range []client.Object{nodeClaim, node} {
		require.NoError(t, k8sClient.Get(context.Background(), client.ObjectKeyFromObject(obj), obj))
		require.Equal(t, constants.TrueStringValue, obj.GetAnnotations()[karpv1.DoNotDisruptAnnotationKey])
	}
}
