package workload

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// TestApplyRecommendationToWorker_RollbackRestoresLimitIndependently exercises
// the patch-failure rollback path with a recommendation that scales request and
// limit by different amounts. Before the fix, AdjustAllocation only returned a
// request-delta and the rollback used it to back out the limit as well, leaving
// QuotaStore.CurrentUsage drifted on every retry. The assertions below pin the
// quota state back to its pre-adjust baseline so any regression of that bug fails.
func TestApplyRecommendationToWorker_RollbackRestoresLimitIndependently(t *testing.T) {
	const (
		ns         = "rollback-test-ns"
		workload   = "rollback-test-workload"
		poolName   = "rollback-test-pool"
		podName    = "rollback-test-pod"
		podUID     = "rollback-test-pod-uid"
		gpuName    = "rollback-test-gpu"
		nodeName   = "rollback-test-node"
		hostName   = "rollback-test-host"
		workerName = podName // worker.Name == pod.Name
	)

	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: gpuName,
			Labels: map[string]string{
				constants.LabelKeyOwner: nodeName,
				constants.GpuPoolKey:    poolName,
			},
		},
		Status: tfv1.GPUStatus{
			Phase: tfv1.TensorFusionGPUPhaseRunning,
			Capacity: &tfv1.Resource{
				Tflops: *resource.NewQuantity(200, resource.DecimalSI),
				Vram:   *resource.NewQuantity(2000, resource.BinarySI),
			},
			Available: &tfv1.Resource{
				Tflops: *resource.NewQuantity(200, resource.DecimalSI),
				Vram:   *resource.NewQuantity(2000, resource.BinarySI),
			},
			NodeSelector: map[string]string{
				constants.KubernetesHostNameLabel: hostName,
			},
		},
	}
	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: poolName},
		Status:     tfv1.GPUPoolStatus{Phase: tfv1.TensorFusionPoolPhaseRunning},
	}
	maxWorkers := int32(10)
	quotaObj := &tfv1.GPUResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "q", Namespace: ns},
		Spec: tfv1.GPUResourceQuotaSpec{
			Total: tfv1.GPUResourceQuotaTotal{
				Requests: &tfv1.Resource{
					Tflops: *resource.NewQuantity(500, resource.DecimalSI),
					Vram:   *resource.NewQuantity(5000, resource.BinarySI),
				},
				Limits: &tfv1.Resource{
					Tflops: *resource.NewQuantity(500, resource.DecimalSI),
					Vram:   *resource.NewQuantity(5000, resource.BinarySI),
				},
				MaxWorkers: &maxWorkers,
			},
		},
	}

	// The worker pod is deliberately NOT registered in the fake client so the
	// inner h.Patch(...) returns NotFound — that triggers the rollback path.
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(gpu, pool).
		WithLists(&tfv1.GPUResourceQuotaList{Items: []tfv1.GPUResourceQuota{*quotaObj}}).
		Build()

	ctx := context.Background()
	allocator := gpuallocator.NewGpuAllocator(ctx, nil, fakeClient, 0)
	require.NoError(t, allocator.InitGPUAndQuotaStore())
	allocator.ReconcileAllocationState()
	allocator.SetAllocatorReady()

	allocReq := &tfv1.AllocRequest{
		PoolName: poolName,
		WorkloadNameNamespace: tfv1.NameNamespace{
			Namespace: ns,
			Name:      workload,
		},
		Request: tfv1.Resource{
			Tflops: *resource.NewQuantity(30, resource.DecimalSI),
			Vram:   *resource.NewQuantity(300, resource.BinarySI),
		},
		Limit: tfv1.Resource{
			Tflops: *resource.NewQuantity(30, resource.DecimalSI),
			Vram:   *resource.NewQuantity(300, resource.BinarySI),
		},
		Count: 1,
		PodMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      podName,
			UID:       podUID,
		},
	}
	allocatedGPUs, err := allocator.Alloc(allocReq)
	require.NoError(t, err)
	require.Len(t, allocatedGPUs, 1)

	qs := allocator.GetQuotaStore()
	baseline, ok := qs.GetQuotaStatus(ns)
	require.True(t, ok)
	require.Equal(t, int64(30), baseline.Requests.Tflops.Value())
	require.Equal(t, int64(30), baseline.Limits.Tflops.Value())

	worker := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      podName,
			UID:       podUID,
			Annotations: map[string]string{
				constants.TFLOPSRequestAnnotation: "30",
				constants.TFLOPSLimitAnnotation:   "30",
				constants.VRAMRequestAnnotation:   "300",
				constants.VRAMLimitAnnotation:     "300",
				constants.GpuCountAnnotation:      "1",
				constants.GpuPoolKey:              poolName,
			},
		},
	}

	state := NewWorkloadState()
	state.Namespace = ns
	state.Name = workload
	state.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
		AutoSetResources: &tfv1.AutoSetResources{
			Enable:         true,
			TargetResource: tfv1.ScalingTargetResourceAll,
		},
	}

	// Request grows by 10, limit grows by 30 — deliberately asymmetric so a
	// rollback that uses the request delta for the limit (the old bug) leaves
	// quota.Limits stuck at 50 instead of returning to the 30 baseline.
	recommendation := &tfv1.Resources{
		Requests: tfv1.Resource{
			Tflops: *resource.NewQuantity(40, resource.DecimalSI),
			Vram:   *resource.NewQuantity(400, resource.BinarySI),
		},
		Limits: tfv1.Resource{
			Tflops: *resource.NewQuantity(60, resource.DecimalSI),
			Vram:   *resource.NewQuantity(600, resource.BinarySI),
		},
	}

	h := &handler{
		Client:    fakeClient,
		allocator: allocator,
	}

	err = h.applyRecommendationToWorker(ctx, state, worker, recommendation)
	require.Error(t, err, "expected patch failure to bubble up so the caller can retry")

	after, ok := qs.GetQuotaStatus(ns)
	require.True(t, ok)
	require.Equal(t, int64(30), after.Requests.Tflops.Value(),
		"request usage should be restored to the pre-adjust baseline")
	require.Equal(t, int64(300), after.Requests.Vram.Value(),
		"request vram usage should be restored to the pre-adjust baseline")
	require.Equal(t, int64(30), after.Limits.Tflops.Value(),
		"limit usage should be restored to the pre-adjust baseline (regression: would be 50 if rollback used request delta for limit)")
	require.Equal(t, int64(300), after.Limits.Vram.Value(),
		"limit vram usage should be restored to the pre-adjust baseline")
}
