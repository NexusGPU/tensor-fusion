package kubernetes

import (
	"context"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/stretchr/testify/assert"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const missingTestGPUNodeName = "missing-test-gpu-node"

func newMissingTestGPU(name, owner string, phase tfv1.TensorFusionGPUPhase) *tfv1.GPU {
	return &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				constants.LabelKeyOwner: owner,
			},
			Annotations: map[string]string{
				constants.LastSyncTimeAnnotationKey: time.Now().Format(time.RFC3339),
			},
		},
		Status: tfv1.GPUStatus{
			Phase: phase,
			Capacity: &tfv1.Resource{
				Tflops: resource.MustParse("100"),
				Vram:   resource.MustParse("16Gi"),
			},
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("100"),
				Vram:   resource.MustParse("16Gi"),
			},
		},
	}
}

func TestCleanupStaleGPUs(t *testing.T) {
	ctx := context.Background()

	present := newMissingTestGPU("gpu-present", missingTestGPUNodeName, tfv1.TensorFusionGPUPhaseRunning)
	missing := newMissingTestGPU("gpu-missing", missingTestGPUNodeName, tfv1.TensorFusionGPUPhaseRunning)
	otherNodeGPU := newMissingTestGPU("gpu-other-node", "other-gpu-node", tfv1.TensorFusionGPUPhaseRunning)
	// a missing GPU still referenced by running apps: must be marked, not deleted
	busyMissing := newMissingTestGPU("gpu-missing-busy", missingTestGPUNodeName, tfv1.TensorFusionGPUPhaseRunning)
	busyMissing.Status.RunningApps = []*tfv1.RunningAppDetail{{Name: "app-1", Namespace: "default"}}

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPU{}).
		WithObjects(present, missing, otherNodeGPU, busyMissing).Build()
	apiClient := NewAPIClient(ctx, k8sClient)

	err := apiClient.CleanupStaleGPUs(missingTestGPUNodeName, []string{"gpu-present"})
	assert.NoError(t, err)

	// an idle missing GPU must be deleted immediately (ops always drain before pulling a card)
	got := &tfv1.GPU{}
	err = k8sClient.Get(ctx, client.ObjectKey{Name: "gpu-missing"}, got)
	assert.True(t, apierrors.IsNotFound(err), "idle missing GPU should be deleted directly, got err=%v", err)

	// a missing GPU still referenced by running apps must be marked, not deleted,
	// so a transient NVML hiccup cannot wipe allocation accounting
	assert.NoError(t, k8sClient.Get(ctx, client.ObjectKey{Name: "gpu-missing-busy"}, got))
	assert.Equal(t, tfv1.TensorFusionGPUPhaseUnknown, got.Status.Phase,
		"busy missing GPU should be marked Unknown")
	missingSince := got.Annotations[constants.GPUMissingSinceAnnotationKey]
	assert.NotEmpty(t, missingSince, "busy missing GPU should carry missing-since annotation")
	_, err = time.Parse(time.RFC3339, missingSince)
	assert.NoError(t, err, "missing-since should be a valid RFC3339 timestamp")

	// the GPU still enumerated must be untouched
	assert.NoError(t, k8sClient.Get(ctx, client.ObjectKey{Name: "gpu-present"}, got))
	assert.Equal(t, tfv1.TensorFusionGPUPhaseRunning, got.Status.Phase)
	assert.NotContains(t, got.Annotations, constants.GPUMissingSinceAnnotationKey)

	// GPUs owned by other nodes must not be touched
	assert.NoError(t, k8sClient.Get(ctx, client.ObjectKey{Name: "gpu-other-node"}, got))
	assert.Equal(t, tfv1.TensorFusionGPUPhaseRunning, got.Status.Phase)
	assert.NotContains(t, got.Annotations, constants.GPUMissingSinceAnnotationKey)

	// idempotent: re-marking must keep the original missing-since timestamp
	err = apiClient.CleanupStaleGPUs(missingTestGPUNodeName, []string{"gpu-present"})
	assert.NoError(t, err)
	assert.NoError(t, k8sClient.Get(ctx, client.ObjectKey{Name: "gpu-missing-busy"}, got))
	assert.Equal(t, missingSince, got.Annotations[constants.GPUMissingSinceAnnotationKey],
		"missing-since must not be refreshed on subsequent runs")
}

func TestDeleteOrMarkMissingGPU(t *testing.T) {
	ctx := context.Background()

	idle := newMissingTestGPU("gpu-idle", missingTestGPUNodeName, tfv1.TensorFusionGPUPhaseRunning)
	busy := newMissingTestGPU("gpu-busy", missingTestGPUNodeName, tfv1.TensorFusionGPUPhaseRunning)
	busy.Status.RunningApps = []*tfv1.RunningAppDetail{{Name: "app-1", Namespace: "default"}}

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPU{}).
		WithObjects(idle, busy).Build()
	apiClient := NewAPIClient(ctx, k8sClient)

	// idle GPU: deleted directly
	assert.NoError(t, apiClient.DeleteOrMarkMissingGPU("gpu-idle"))
	got := &tfv1.GPU{}
	err := k8sClient.Get(ctx, client.ObjectKey{Name: "gpu-idle"}, got)
	assert.True(t, apierrors.IsNotFound(err), "idle GPU should be deleted, got err=%v", err)

	// busy GPU: marked missing instead of deleted
	assert.NoError(t, apiClient.DeleteOrMarkMissingGPU("gpu-busy"))
	assert.NoError(t, k8sClient.Get(ctx, client.ObjectKey{Name: "gpu-busy"}, got))
	assert.Equal(t, tfv1.TensorFusionGPUPhaseUnknown, got.Status.Phase)
	assert.NotEmpty(t, got.Annotations[constants.GPUMissingSinceAnnotationKey])

	// already-deleted GPU: no error
	assert.NoError(t, apiClient.DeleteOrMarkMissingGPU("gpu-gone"))
}

func TestCreateOrUpdateGPU_RecoversMissingGPU(t *testing.T) {
	ctx := context.Background()

	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: missingTestGPUNodeName,
			OwnerReferences: []metav1.OwnerReference{
				{Name: "test-gpu-pool"},
			},
		},
	}
	uuid := "gpu-recovered"
	missingGPU := newMissingTestGPU(uuid, missingTestGPUNodeName, tfv1.TensorFusionGPUPhaseUnknown)
	missingGPU.Annotations[constants.GPUMissingSinceAnnotationKey] =
		time.Now().Add(-5 * time.Minute).Format(time.RFC3339)
	// simulate in-use accounting that must survive recovery
	missingGPU.Status.Available.Tflops = resource.MustParse("40")
	missingGPU.Status.RunningApps = []*tfv1.RunningAppDetail{{Name: "app-1", Namespace: "default"}}

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPU{}).
		WithObjects(gpuNode, missingGPU).Build()

	backend := &KubeletBackend{
		ctx:          ctx,
		apiClient:    NewAPIClient(ctx, k8sClient),
		nodeName:     missingTestGPUNodeName,
		deviceTflops: map[string]resource.Quantity{},
	}
	device := &api.DeviceInfo{
		UUID:             uuid,
		Model:            "NVIDIA-Test-GPU",
		TotalMemoryBytes: 16 * 1024 * 1024 * 1024,
		MaxTflops:        100,
	}

	err := backend.apiClient.CreateOrUpdateGPU(missingTestGPUNodeName, uuid,
		func(node *tfv1.GPUNode, gpu *tfv1.GPU) error {
			return backend.mutateGPUResourceState(device, node, gpu)
		})
	assert.NoError(t, err)

	gpu := &tfv1.GPU{}
	assert.NoError(t, k8sClient.Get(ctx, client.ObjectKey{Name: uuid}, gpu))
	assert.NotContains(t, gpu.Annotations, constants.GPUMissingSinceAnnotationKey,
		"missing marker must be cleared when the device reappears")
	assert.Equal(t, tfv1.TensorFusionGPUPhasePending, gpu.Status.Phase,
		"recovered GPU should go back to Pending for the controller to promote")
	assert.Equal(t, resource.MustParse("40"), gpu.Status.Available.Tflops,
		"existing allocation accounting must be preserved on recovery")
}
