package controller

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestReconcileSharedLegacyResources(t *testing.T) {
	for _, phase := range []corev1.PodPhase{corev1.PodRunning, corev1.PodFailed, corev1.PodSucceeded} {
		t.Run(string(phase), func(t *testing.T) {
			ctx := context.Background()
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			require.NoError(t, tfv1.AddToScheme(scheme))
			gpu := &tfv1.GPU{ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"}, Status: tfv1.GPUStatus{
				Capacity: &tfv1.Resource{Tflops: resource.MustParse("100"), Vram: resource.MustParse("24Gi")},
			}}
			pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
				Name: "shared", Namespace: "default", Finalizers: []string{constants.Finalizer},
				Annotations: map[string]string{
					constants.IsolationModeAnnotation: tfv1.IsolationModeShared,
					constants.GPUDeviceIDsAnnotation:  gpu.Name,
					constants.TFLOPSRequestAnnotation: "0",
					constants.VRAMRequestAnnotation:   "0",
				},
			}, Spec: corev1.PodSpec{NodeName: "node-1"}, Status: corev1.PodStatus{Phase: phase}}
			if phase != corev1.PodRunning {
				require.NoError(t, utils.ApplySharedLegacyResources(pod, []*tfv1.GPU{gpu}))
			}
			c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pod, gpu).Build()
			r := &PodReconciler{Client: c}
			require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(pod), pod))
			require.NoError(t, r.reconcileSharedLegacyResources(ctx, pod))
			persisted := &corev1.Pod{}
			require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(pod), persisted))
			if phase == corev1.PodRunning {
				require.Equal(t, "100", persisted.Annotations[constants.TFLOPSRequestAnnotation])
				require.Equal(t, "24Gi", persisted.Annotations[constants.VRAMRequestAnnotation])
				// Terminating is still active; retain occupancy until exit/deletion.
				require.NoError(t, c.Delete(ctx, persisted))
				require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(pod), persisted))
				require.False(t, persisted.DeletionTimestamp.IsZero())
				require.NoError(t, r.reconcileSharedLegacyResources(ctx, persisted))
				require.Equal(t, gpu.Name, persisted.Annotations[constants.GPUDeviceIDsAnnotation])
				require.Equal(t, "24Gi", persisted.Annotations[constants.VRAMRequestAnnotation])
			} else {
				require.NotContains(t, persisted.Annotations, constants.GPUDeviceIDsAnnotation)
				require.NotContains(t, persisted.Annotations, constants.SharedLegacyResourcesAnnotation)
				require.Equal(t, "0", persisted.Annotations[constants.TFLOPSRequestAnnotation])
			}
			require.NoError(t, r.reconcileSharedLegacyResources(ctx, persisted))
		})
	}
}
