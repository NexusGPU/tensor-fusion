package gpuallocator

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestRecoveryPreservesTerminatingWorkersUntilStopped(t *testing.T) {
	for _, mode := range []tfv1.IsolationModeType{tfv1.IsolationModeShared, tfv1.IsolationModeSoft, tfv1.IsolationModeHard} {
		for _, phase := range []corev1.PodPhase{corev1.PodRunning, corev1.PodPending, corev1.PodFailed, corev1.PodSucceeded} {
			for _, keepTF := range []bool{true, false} {
				name := "without-tf-finalizer"
				if keepTF {
					name = "with-tf-finalizer"
				}
				t.Run(string(mode)+"/"+string(phase)+"/"+name, func(t *testing.T) {
					s := newTestAllocator()
					s.isolationPolicy = tfv1.IsolationModePolicyDynamic
					gpu := sharedTestGPU("gpu-1", "node-1", tfv1.IsolationModeSoft)
					now := metav1.Now()
					pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
						Name: "victim", Namespace: "default", UID: "victim-uid", DeletionTimestamp: &now,
						Finalizers:  []string{"test.example/hold"},
						Labels:      map[string]string{constants.LabelComponent: constants.ComponentWorker, constants.WorkloadKey: "workload"},
						Annotations: map[string]string{constants.IsolationModeAnnotation: string(mode), constants.GPUDeviceIDsAnnotation: gpu.Name, constants.TFLOPSRequestAnnotation: "10", constants.VRAMRequestAnnotation: "2Gi"},
					}, Spec: corev1.PodSpec{NodeName: "node-1"}, Status: corev1.PodStatus{Phase: phase}}
					if keepTF {
						pod.Finalizers = append(pod.Finalizers, constants.Finalizer)
					}
					if mode == tfv1.IsolationModeShared {
						require.NoError(t, utils.ApplySharedLegacyResources(pod, []*tfv1.GPU{gpu}))
					}
					scheme := runtime.NewScheme()
					require.NoError(t, corev1.AddToScheme(scheme))
					require.NoError(t, tfv1.AddToScheme(scheme))
					s.Client = fake.NewClientBuilder().WithScheme(scheme).WithObjects(pod, gpu).Build()
					s.gpuStore[types.NamespacedName{Name: gpu.Name}] = gpu
					s.reconcileAllocationState()
					t.Logf("TF finalizer=%v recovered=%v available=%s/%s", keepTF, s.uniqueAllocation[string(pod.UID)] != nil, gpu.Status.Available.Tflops.String(), gpu.Status.Available.Vram.String())
					if phase == corev1.PodFailed || phase == corev1.PodSucceeded {
						require.Nil(t, s.uniqueAllocation[string(pod.UID)])
						require.True(t, gpu.Status.Available.Tflops.Equal(gpu.Status.Capacity.Tflops))
						require.True(t, gpu.Status.Available.Vram.Equal(gpu.Status.Capacity.Vram))
						require.Empty(t, gpu.Status.RunningApps)
						return
					}
					require.NotNil(t, s.uniqueAllocation[string(pod.UID)], "active terminating worker must retain occupancy")
					require.Len(t, gpu.Status.RunningApps, 1)
					require.Equal(t, mode, gpu.Status.ActiveIsolationMode)
					if mode == tfv1.IsolationModeShared {
						require.True(t, gpu.Status.Available.Tflops.IsZero())
						require.True(t, gpu.Status.Available.Vram.IsZero())
					} else {
						require.Equal(t, "90", gpu.Status.Available.Tflops.String())
						require.Equal(t, "22Gi", gpu.Status.Available.Vram.String())
					}
					// Terminal-state and deletion notifications may both request cleanup.
					for range 2 {
						s.DeallocByPodIdentifier(s.ctx, types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name})
						require.True(t, gpu.Status.Available.Tflops.Equal(gpu.Status.Capacity.Tflops))
						require.True(t, gpu.Status.Available.Vram.Equal(gpu.Status.Capacity.Vram))
						require.Empty(t, gpu.Status.RunningApps)
						require.Empty(t, gpu.Status.ActiveIsolationMode)
						require.Empty(t, s.uniqueAllocation)
						require.Empty(t, s.podNamespaceNsToPodUID)
						require.Empty(t, s.nodeWorkerStore["node-1"])
					}
				})
			}
		}
	}

}
