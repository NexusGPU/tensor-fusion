package controller

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func newMissingSyncTestGPU(name, owner string, phase tfv1.TensorFusionGPUPhase) *tfv1.GPU {
	return &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				constants.LabelKeyOwner: owner,
			},
		},
		Status: tfv1.GPUStatus{
			Phase: phase,
		},
	}
}

func TestSyncStatusToGPUDevicesSkipsMissingGPUs(t *testing.T) {
	t.Helper()

	ctx := context.Background()
	s := runtime.NewScheme()
	if err := tfv1.AddToScheme(s); err != nil {
		t.Fatalf("add scheme: %v", err)
	}

	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{Name: "missing-sync-test-node"},
	}

	normalGPU := newMissingSyncTestGPU("gpu-normal", gpuNode.Name, tfv1.TensorFusionGPUPhasePending)

	missingGPU := newMissingSyncTestGPU("gpu-missing", gpuNode.Name, tfv1.TensorFusionGPUPhaseUnknown)
	missingGPU.Annotations = map[string]string{
		constants.GPUMissingSinceAnnotationKey: "2026-07-13T00:00:00Z",
	}

	kubeClient := fake.NewClientBuilder().WithScheme(s).
		WithStatusSubresource(&tfv1.GPU{}).
		WithObjects(gpuNode, normalGPU, missingGPU).Build()
	reconciler := &GPUNodeReconciler{Client: kubeClient, Scheme: s}

	if _, err := reconciler.syncStatusToGPUDevices(ctx, gpuNode, tfv1.TensorFusionGPUPhaseRunning); err != nil {
		t.Fatalf("syncStatusToGPUDevices: %v", err)
	}

	updated := &tfv1.GPU{}
	if err := kubeClient.Get(ctx, types.NamespacedName{Name: "gpu-normal"}, updated); err != nil {
		t.Fatalf("get normal GPU: %v", err)
	}
	if updated.Status.Phase != tfv1.TensorFusionGPUPhaseRunning {
		t.Fatalf("expected normal GPU phase %q, got %q", tfv1.TensorFusionGPUPhaseRunning, updated.Status.Phase)
	}

	if err := kubeClient.Get(ctx, types.NamespacedName{Name: "gpu-missing"}, updated); err != nil {
		t.Fatalf("get missing GPU: %v", err)
	}
	if updated.Status.Phase != tfv1.TensorFusionGPUPhaseUnknown {
		t.Fatalf("missing GPU phase must stay %q, got %q", tfv1.TensorFusionGPUPhaseUnknown, updated.Status.Phase)
	}
}
