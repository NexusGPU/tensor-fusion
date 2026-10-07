package karpenter

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	karpv1 "sigs.k8s.io/karpenter/pkg/apis/v1"
)

func TestNodeClaimCreateAndTerminateCanBeRetried(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))
	scheme.AddKnownTypes(schema.GroupVersion{Group: "karpenter.sh", Version: "v1"}, &karpv1.NodeClaim{}, &karpv1.NodeClaimList{})
	nodeClass := &unstructured.Unstructured{}
	nodeClass.SetAPIVersion("karpenter.k8s.aws/v1")
	nodeClass.SetKind("EC2NodeClass")
	nodeClass.SetName("test-class")
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(nodeClass).Build()
	provider := KarpenterGPUNodeProvider{client: k8sClient, nodeManagerConfig: &tfv1.NodeManagerConfig{NodeProvisioner: &tfv1.NodeProvisioner{}}}
	claim := &tfv1.GPUNodeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "gpu-claim", UID: "first-claim"},
		Spec: tfv1.GPUNodeClaimSpec{
			NodeName:     "test-node",
			NodeClassRef: tfv1.GroupKindName{Group: "karpenter.k8s.aws", Version: "v1", Kind: "EC2NodeClass", Name: "test-class"},
		},
	}
	first, err := provider.CreateNode(t.Context(), claim)
	require.NoError(t, err)
	// A status write failure causes the reconciler to call CreateNode again.
	retry, err := provider.CreateNode(t.Context(), claim)
	require.NoError(t, err)
	require.Equal(t, first.InstanceID, retry.InstanceID)
	nodes := &karpv1.NodeClaimList{}
	require.NoError(t, k8sClient.List(t.Context(), nodes))
	require.Len(t, nodes.Items, 1)
	require.True(t, metav1.IsControlledBy(&nodes.Items[0], claim))

	replacement := claim.DeepCopy()
	replacement.UID = "replacement-claim"
	_, err = provider.CreateNode(t.Context(), replacement)
	require.Error(t, err, "a same-name claim must not adopt a previous claim's node")

	node := &karpv1.NodeClaim{}
	require.NoError(t, k8sClient.Get(t.Context(), client.ObjectKey{Name: first.InstanceID}, node))
	identity := &types.NodeIdentityParam{InstanceID: first.InstanceID}
	require.NoError(t, provider.TerminateNode(t.Context(), identity))
	require.NoError(t, provider.TerminateNode(t.Context(), identity))
}
