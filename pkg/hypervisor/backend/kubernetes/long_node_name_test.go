package kubernetes

import (
	"strings"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// A node name may be up to 253 characters, a label value 63. The node agent
// must still register the GPUs of such a node (the API server rejects the
// whole object for one over-long label value) and find them again.
func TestGPURegistrationOnLongNodeName(t *testing.T) {
	for _, nodeName := range []string{
		"gpu-node-01",
		"kwok-" + strings.Repeat("n", 95),        // 100
		strings.Repeat("ip-10-0-0-1.", 22)[:253], // the longest legal name
	} {
		t.Run(nodeName[:11], func(t *testing.T) {
			gpuNode := &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{
				Name:            nodeName,
				UID:             "node-uid",
				OwnerReferences: []metav1.OwnerReference{{Name: "test-gpu-pool"}},
			}}
			k8sClient := fake.NewClientBuilder().WithScheme(scheme).
				WithStatusSubresource(&tfv1.GPU{}).WithObjects(gpuNode).Build()
			backend := &KubeletBackend{
				ctx: t.Context(), apiClient: NewAPIClient(t.Context(), k8sClient), nodeName: nodeName,
			}
			device := capacityTestDevice(true)
			require.NoError(t, backend.apiClient.CreateOrUpdateGPU(nodeName, device.UUID,
				func(node *tfv1.GPUNode, gpu *tfv1.GPU) error {
					return backend.mutateGPUResourceState(device, node, gpu)
				}))

			gpu := &tfv1.GPU{}
			require.NoError(t, k8sClient.Get(t.Context(), client.ObjectKey{Name: device.UUID}, gpu))
			for key, value := range gpu.Labels {
				require.Empty(t, validation.IsValidLabelValue(value), "label %s=%q", key, value)
			}
			if len(nodeName) <= utils.MaxLabelValueLength {
				require.Equal(t, nodeName, gpu.Labels[constants.LabelKeyOwner], "short names keep their label")
			}
			require.Equal(t, nodeName, utils.GPUNodeNameOfGPU(gpu), "the full node name must be recoverable")
			require.Equal(t, nodeName, gpu.Status.NodeSelector[constants.KubernetesHostNameLabel])

			// What the operator's controllers use to find a node's GPUs.
			owned := &tfv1.GPUList{}
			require.NoError(t, k8sClient.List(t.Context(), owned, utils.GPUOwnerSelector(nodeName)))
			require.Len(t, owned.Items, 1)

			// The node agent's own selector: a device that is gone and idle is removed.
			require.NoError(t, backend.apiClient.CleanupStaleGPUs(nodeName, nil))
			err := k8sClient.Get(t.Context(), client.ObjectKey{Name: device.UUID}, gpu)
			require.True(t, apierrors.IsNotFound(err), "stale GPU of a long-named node was not found by the selector: %v", err)
		})
	}
}
