package gpuallocator

import (
	"context"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/quota"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func newTestAllocator() *GpuAllocator {
	ch := make(chan struct{})
	close(ch)
	ctx := context.Background()
	return &GpuAllocator{
		ctx:                         ctx,
		initializedCh:               ch,
		uniqueAllocation:            map[string]*tfv1.AllocRequest{},
		assumedAllocation:           map[string]*tfv1.AllocRequest{},
		assumedAllocationTimestamps: map[string]time.Time{},
		assumedAllocationTTL:        DefaultAssumedAllocationTTL,
		uniqueDeallocation:          map[string]struct{}{},
		gpuStore:                    map[types.NamespacedName]*tfv1.GPU{},
		nodeWorkerStore:             map[string]map[types.NamespacedName]struct{}{},
		podNamespaceNsToPodUID:      map[string]string{},
		dirtyQueue:                  map[types.NamespacedName]struct{}{},
		quotaStore:                  quota.NewQuotaStore(nil, ctx),
	}
}

func TestRollback(t *testing.T) {
	t.Run("non-existent podUID is a no-op and idempotent", func(t *testing.T) {
		s := newTestAllocator()
		assert.NoError(t, s.Rollback("nonexistent"))
		assert.NoError(t, s.Rollback("nonexistent"))
	})

	t.Run("committed allocation: GPU available restored, pod state cleared, no Dealloc marker", func(t *testing.T) {
		s := newTestAllocator()
		gpu := makeGPU("gpu-1", "100", "24Gi", "70", "16Gi")
		gpu.Status.NodeSelector = map[string]string{constants.KubernetesHostNameLabel: "node-1"}
		s.gpuStore[types.NamespacedName{Name: "gpu-1"}] = gpu

		req := &tfv1.AllocRequest{
			PodMeta:               metav1.ObjectMeta{Name: "pod-x", Namespace: "ns", UID: "pod-x-uid"},
			GPUNames:              []string{"gpu-1"},
			Request:               tfv1.Resource{Tflops: qty("30"), Vram: qty("8Gi")},
			WorkloadNameNamespace: tfv1.NameNamespace{Name: "wl", Namespace: "ns"},
		}
		s.uniqueAllocation["pod-x-uid"] = req
		s.nodeWorkerStore["node-1"] = map[types.NamespacedName]struct{}{
			{Name: "pod-x", Namespace: "ns"}: {},
		}
		s.podNamespaceNsToPodUID["ns/pod-x"] = "pod-x-uid"

		assert.NoError(t, s.Rollback("pod-x-uid"))

		// GPU resources restored (70 + 30 = 100, 16Gi + 8Gi = 24Gi)
		assert.Equal(t, "100", gpu.Status.Available.Tflops.String())
		assert.Equal(t, "24Gi", gpu.Status.Available.Vram.String())

		// pod state cleared
		_, ok := s.uniqueAllocation["pod-x-uid"]
		assert.False(t, ok, "uniqueAllocation should be cleared")
		_, ok = s.podNamespaceNsToPodUID["ns/pod-x"]
		assert.False(t, ok, "podNamespaceNsToPodUID should be cleared")
		_, ok = s.nodeWorkerStore["node-1"][types.NamespacedName{Name: "pod-x", Namespace: "ns"}]
		assert.False(t, ok, "nodeWorkerStore entry should be cleared")

		// uniqueDeallocation must NOT be set: re-scheduling the same pod must succeed.
		_, ok = s.uniqueDeallocation["pod-x-uid"]
		assert.False(t, ok, "uniqueDeallocation must not be marked by Rollback")

		// markGPUDirty must have queued the GPU.
		_, ok = s.dirtyQueue[types.NamespacedName{Name: "gpu-1"}]
		assert.True(t, ok, "rolled-back GPU should be marked dirty")
	})

	t.Run("rolling back a committed allocation twice is idempotent", func(t *testing.T) {
		s := newTestAllocator()
		gpu := makeGPU("gpu-1", "100", "24Gi", "70", "16Gi")
		gpu.Status.NodeSelector = map[string]string{constants.KubernetesHostNameLabel: "node-1"}
		s.gpuStore[types.NamespacedName{Name: "gpu-1"}] = gpu

		req := &tfv1.AllocRequest{
			PodMeta:  metav1.ObjectMeta{Name: "pod-y", Namespace: "ns", UID: "pod-y-uid"},
			GPUNames: []string{"gpu-1"},
			Request:  tfv1.Resource{Tflops: qty("30"), Vram: qty("8Gi")},
		}
		s.uniqueAllocation["pod-y-uid"] = req

		assert.NoError(t, s.Rollback("pod-y-uid"))
		// Second call should be a no-op (uniqueAllocation already cleared).
		assert.NoError(t, s.Rollback("pod-y-uid"))
		// Available must not have been double-credited.
		assert.Equal(t, "100", gpu.Status.Available.Tflops.String())
		assert.Equal(t, "24Gi", gpu.Status.Available.Vram.String())
	})

	t.Run("UnreserveOrRollback handles all three states", func(t *testing.T) {
		t.Run("no state -> no-op", func(t *testing.T) {
			s := newTestAllocator()
			assert.NoError(t, s.UnreserveOrRollback("nothing-here"))
		})

		t.Run("assumed only -> Forget runs, assumed cleared", func(t *testing.T) {
			s := newTestAllocator()
			req := &tfv1.AllocRequest{
				PodMeta:               metav1.ObjectMeta{Name: "pod-a", Namespace: "ns", UID: "pod-a-uid"},
				GPUNames:              []string{"gpu-1"},
				Request:               tfv1.Resource{Tflops: qty("10"), Vram: qty("4Gi")},
				WorkloadNameNamespace: tfv1.NameNamespace{Name: "wl", Namespace: "ns"},
			}
			s.assumedAllocation["pod-a-uid"] = req

			assert.NoError(t, s.UnreserveOrRollback("pod-a-uid"))
			_, ok := s.assumedAllocation["pod-a-uid"]
			assert.False(t, ok, "assumed entry must be cleared")
		})

		t.Run("committed -> Rollback runs, no Dealloc marker, GPU restored", func(t *testing.T) {
			s := newTestAllocator()
			gpu := makeGPU("gpu-1", "100", "24Gi", "70", "16Gi")
			gpu.Status.NodeSelector = map[string]string{constants.KubernetesHostNameLabel: "node-1"}
			s.gpuStore[types.NamespacedName{Name: "gpu-1"}] = gpu
			req := &tfv1.AllocRequest{
				PodMeta:               metav1.ObjectMeta{Name: "pod-c", Namespace: "ns", UID: "pod-c-uid"},
				GPUNames:              []string{"gpu-1"},
				Request:               tfv1.Resource{Tflops: qty("30"), Vram: qty("8Gi")},
				WorkloadNameNamespace: tfv1.NameNamespace{Name: "wl", Namespace: "ns"},
			}
			s.uniqueAllocation["pod-c-uid"] = req

			assert.NoError(t, s.UnreserveOrRollback("pod-c-uid"))
			_, ok := s.uniqueAllocation["pod-c-uid"]
			assert.False(t, ok)
			_, ok = s.uniqueDeallocation["pod-c-uid"]
			assert.False(t, ok, "rollback path must not mark uniqueDeallocation")
			assert.Equal(t, "100", gpu.Status.Available.Tflops.String())
		})
	})

	t.Run("sweepStaleAssumedAllocations TTL semantics", func(t *testing.T) {
		t.Run("fresh entry is preserved", func(t *testing.T) {
			s := newTestAllocator()
			s.assumedAllocation["fresh-uid"] = &tfv1.AllocRequest{
				PodMeta:               metav1.ObjectMeta{Name: "p", Namespace: "ns", UID: "fresh-uid"},
				WorkloadNameNamespace: tfv1.NameNamespace{Name: "wl", Namespace: "ns"},
			}
			s.assumedAllocationTimestamps["fresh-uid"] = time.Now()

			swept := s.sweepStaleAssumedAllocationsLocked(time.Now())
			assert.Equal(t, 0, swept)
			_, ok := s.assumedAllocation["fresh-uid"]
			assert.True(t, ok)
		})

		t.Run("entry older than TTL is swept", func(t *testing.T) {
			s := newTestAllocator()
			s.assumedAllocationTTL = 10 * time.Minute
			s.assumedAllocation["stale-uid"] = &tfv1.AllocRequest{
				PodMeta:               metav1.ObjectMeta{Name: "p", Namespace: "ns", UID: "stale-uid"},
				WorkloadNameNamespace: tfv1.NameNamespace{Name: "wl", Namespace: "ns"},
			}
			s.assumedAllocationTimestamps["stale-uid"] = time.Now().Add(-20 * time.Minute)

			swept := s.sweepStaleAssumedAllocationsLocked(time.Now())
			assert.Equal(t, 1, swept)
			_, ok := s.assumedAllocation["stale-uid"]
			assert.False(t, ok)
			_, ok = s.assumedAllocationTimestamps["stale-uid"]
			assert.False(t, ok)
		})

		t.Run("swept entry releases its Dynamic mode lock", func(t *testing.T) {
			s := newTestAllocator()
			s.isolationPolicy = tfv1.IsolationModePolicyDynamic
			s.assumedAllocationTTL = 10 * time.Minute
			gpu := makeGPU("gpu-dynamic", "100", "24Gi", "100", "24Gi")
			gpu.Status.IsolationPolicy = tfv1.IsolationModePolicyDynamic
			gpu.Status.ActiveIsolationMode = tfv1.IsolationModeSoft
			s.gpuStore[types.NamespacedName{Name: gpu.Name}] = gpu
			s.assumedAllocation["stale-dynamic"] = &tfv1.AllocRequest{
				PodMeta:               metav1.ObjectMeta{Name: "p", Namespace: "ns", UID: "stale-dynamic"},
				WorkloadNameNamespace: tfv1.NameNamespace{Name: "wl", Namespace: "ns"},
				GPUNames:              []string{gpu.Name},
				Isolation:             tfv1.IsolationModeSoft,
			}
			s.assumedAllocationTimestamps["stale-dynamic"] = time.Now().Add(-20 * time.Minute)

			swept := s.sweepStaleAssumedAllocationsLocked(time.Now())
			assert.Equal(t, 1, swept)
			assert.Empty(t, gpu.Status.ActiveIsolationMode)
		})

		t.Run("already-committed entry is dropped without TTL check", func(t *testing.T) {
			s := newTestAllocator()
			req := &tfv1.AllocRequest{
				PodMeta:               metav1.ObjectMeta{Name: "p", Namespace: "ns", UID: "committed-uid"},
				WorkloadNameNamespace: tfv1.NameNamespace{Name: "wl", Namespace: "ns"},
			}
			s.assumedAllocation["committed-uid"] = req
			s.assumedAllocationTimestamps["committed-uid"] = time.Now() // fresh
			s.uniqueAllocation["committed-uid"] = req

			_ = s.sweepStaleAssumedAllocationsLocked(time.Now())
			// committed pod's leftover assumed must be cleared regardless of age.
			_, ok := s.assumedAllocation["committed-uid"]
			assert.False(t, ok)
			_, ok = s.assumedAllocationTimestamps["committed-uid"]
			assert.False(t, ok)
		})

		t.Run("gang-waiting pod is preserved past TTL via probe", func(t *testing.T) {
			s := newTestAllocator()
			s.assumedAllocationTTL = 1 * time.Minute
			s.assumedAllocation["gang-uid"] = &tfv1.AllocRequest{
				PodMeta:               metav1.ObjectMeta{Name: "p", Namespace: "ns", UID: "gang-uid"},
				WorkloadNameNamespace: tfv1.NameNamespace{Name: "wl", Namespace: "ns"},
			}
			s.assumedAllocationTimestamps["gang-uid"] = time.Now().Add(-1 * time.Hour) // way past TTL
			s.gangWaitingProbe = func(uid string) bool { return uid == "gang-uid" }

			now := time.Now()
			swept := s.sweepStaleAssumedAllocationsLocked(now)
			assert.Equal(t, 0, swept)
			_, ok := s.assumedAllocation["gang-uid"]
			assert.True(t, ok, "gang-waiting entry must be preserved")
			// Probe path should also refresh the timestamp so the sweep window restarts.
			assert.True(t, !s.assumedAllocationTimestamps["gang-uid"].Before(now))
		})

		t.Run("entry without timestamp gets stamped (no immediate sweep)", func(t *testing.T) {
			s := newTestAllocator()
			s.assumedAllocation["legacy-uid"] = &tfv1.AllocRequest{
				PodMeta:               metav1.ObjectMeta{Name: "p", Namespace: "ns", UID: "legacy-uid"},
				WorkloadNameNamespace: tfv1.NameNamespace{Name: "wl", Namespace: "ns"},
			}
			// No timestamp.

			swept := s.sweepStaleAssumedAllocationsLocked(time.Now())
			assert.Equal(t, 0, swept)
			_, ok := s.assumedAllocationTimestamps["legacy-uid"]
			assert.True(t, ok, "missing timestamp should be filled in on first sweep")
		})
	})

	t.Run("partitioned allocation falls back to template-based release via releaseAllocationFromGPU", func(t *testing.T) {
		// Partitioned-mode release goes through deallocPartition, which is exercised
		// by partitioned_scheduling_test.go. Here we only verify that Rollback runs
		// without panicking on an empty AllocatedPartitions map and still cleans
		// the per-pod state.
		s := newTestAllocator()
		gpu := makeGPU("gpu-2", "100", "24Gi", "70", "16Gi")
		gpu.Status.NodeSelector = map[string]string{constants.KubernetesHostNameLabel: "node-1"}
		gpu.Status.AllocatedPartitions = map[string]tfv1.AllocatedPartition{}
		s.gpuStore[types.NamespacedName{Name: "gpu-2"}] = gpu

		req := &tfv1.AllocRequest{
			PodMeta:             metav1.ObjectMeta{Name: "pod-p", Namespace: "ns", UID: "pod-p-uid"},
			GPUNames:            []string{"gpu-2"},
			Request:             tfv1.Resource{Tflops: qty("30"), Vram: qty("8Gi")},
			Isolation:           tfv1.IsolationModePartitioned,
			PartitionTemplateID: "not-a-real-template",
		}
		s.uniqueAllocation["pod-p-uid"] = req

		assert.NoError(t, s.Rollback("pod-p-uid"))
		_, ok := s.uniqueAllocation["pod-p-uid"]
		assert.False(t, ok)
	})
}

func TestDynamicGangAssumptions(t *testing.T) {
	newDynamicGPU := func() *tfv1.GPU {
		gpu := makeGPU("gpu-1", "100", "24Gi", "100", "24Gi")
		gpu.Status.NodeSelector = map[string]string{constants.KubernetesHostNameLabel: "node-1"}
		gpu.Status.IsolationPolicy = tfv1.IsolationModePolicyDynamic
		return gpu
	}
	newGangRequest := func(uid string, isolation tfv1.IsolationModeType) *tfv1.AllocRequest {
		return &tfv1.AllocRequest{
			PodMeta:               metav1.ObjectMeta{Name: uid, Namespace: "ns", UID: types.UID(uid)},
			Request:               tfv1.Resource{Tflops: qty("10"), Vram: qty("1Gi")},
			Limit:                 tfv1.Resource{Tflops: qty("10"), Vram: qty("1Gi")},
			WorkloadNameNamespace: tfv1.NameNamespace{Name: "gang", Namespace: "ns"},
			Isolation:             isolation,
		}
	}

	t.Run("same-mode members retain the lock until the last unreserve", func(t *testing.T) {
		s := newTestAllocator()
		s.isolationPolicy = tfv1.IsolationModePolicyDynamic
		gpu := newDynamicGPU()
		s.gpuStore[types.NamespacedName{Name: gpu.Name}] = gpu

		assert.NoError(t, s.Assume([]string{gpu.Name}, newGangRequest("gang-1", tfv1.IsolationModeSoft)))
		assert.Equal(t, tfv1.IsolationModeType(tfv1.IsolationModeSoft), gpu.Status.ActiveIsolationMode)
		assert.NoError(t, s.Assume([]string{gpu.Name}, newGangRequest("gang-2", tfv1.IsolationModeSoft)))

		assert.NoError(t, s.UnreserveOrRollback("gang-1"))
		assert.Equal(t, tfv1.IsolationModeType(tfv1.IsolationModeSoft), gpu.Status.ActiveIsolationMode,
			"the remaining gang assumption must retain the GPU mode lock")

		assert.NoError(t, s.UnreserveOrRollback("gang-2"))
		assert.Empty(t, gpu.Status.ActiveIsolationMode,
			"the final gang unreserve must release the GPU mode lock")
	})

	t.Run("different-mode member cannot reserve a locked GPU", func(t *testing.T) {
		s := newTestAllocator()
		s.isolationPolicy = tfv1.IsolationModePolicyDynamic
		gpu := newDynamicGPU()
		s.gpuStore[types.NamespacedName{Name: gpu.Name}] = gpu

		assert.NoError(t, s.Assume([]string{gpu.Name}, newGangRequest("gang-soft", tfv1.IsolationModeSoft)))
		err := s.Assume([]string{gpu.Name}, newGangRequest("gang-hard", tfv1.IsolationModeHard))
		assert.ErrorContains(t, err, "locked to isolation mode soft")
	})
}

func TestIsolationPolicyCompatibility(t *testing.T) {
	gpu := func(policy tfv1.IsolationModePolicyType, runtimeMode, activeMode tfv1.IsolationModeType, conflict bool) *tfv1.GPU {
		return &tfv1.GPU{Status: tfv1.GPUStatus{
			IsolationPolicy:          policy,
			IsolationMode:            runtimeMode,
			ActiveIsolationMode:      activeMode,
			DynamicIsolationConflict: conflict,
		}}
	}

	tests := []struct {
		name      string
		expected  tfv1.IsolationModePolicyType
		gpu       *tfv1.GPU
		requested tfv1.IsolationModeType
		want      bool
	}{
		{name: "Static accepts legacy empty policy", expected: tfv1.IsolationModePolicyStatic, gpu: gpu("", tfv1.IsolationModeSoft, "", false), requested: tfv1.IsolationModeSoft, want: true},
		{name: "Static rejects Dynamic GPU", expected: tfv1.IsolationModePolicyStatic, gpu: gpu(tfv1.IsolationModePolicyDynamic, "", "", false), requested: tfv1.IsolationModeSoft, want: false},
		{name: "Static rejects runtime mode mismatch", expected: tfv1.IsolationModePolicyStatic, gpu: gpu(tfv1.IsolationModePolicyStatic, tfv1.IsolationModeHard, "", false), requested: tfv1.IsolationModeSoft, want: false},
		{name: "Dynamic accepts idle Dynamic GPU", expected: tfv1.IsolationModePolicyDynamic, gpu: gpu(tfv1.IsolationModePolicyDynamic, "", "", false), requested: tfv1.IsolationModeHard, want: true},
		{name: "Dynamic accepts matching active mode", expected: tfv1.IsolationModePolicyDynamic, gpu: gpu(tfv1.IsolationModePolicyDynamic, "", tfv1.IsolationModeSoft, false), requested: tfv1.IsolationModeSoft, want: true},
		{name: "Dynamic rejects missing advertised policy", expected: tfv1.IsolationModePolicyDynamic, gpu: gpu("", "", "", false), requested: tfv1.IsolationModeSoft, want: false},
		{name: "Dynamic rejects conflicting active mode", expected: tfv1.IsolationModePolicyDynamic, gpu: gpu(tfv1.IsolationModePolicyDynamic, "", tfv1.IsolationModeHard, false), requested: tfv1.IsolationModeSoft, want: false},
		{name: "Dynamic rejects recovered conflict", expected: tfv1.IsolationModePolicyDynamic, gpu: gpu(tfv1.IsolationModePolicyDynamic, "", "", true), requested: tfv1.IsolationModeSoft, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, isGPUIsolationCompatible(tt.expected, tt.gpu, tt.requested))
		})
	}
}

func TestAssumeRevalidatesAdvertisedIsolationPolicy(t *testing.T) {
	newRequest := func(uid string) *tfv1.AllocRequest {
		return &tfv1.AllocRequest{
			PodMeta:               metav1.ObjectMeta{Name: uid, Namespace: "ns", UID: types.UID(uid)},
			Request:               tfv1.Resource{Tflops: qty("10"), Vram: qty("1Gi")},
			Limit:                 tfv1.Resource{Tflops: qty("10"), Vram: qty("1Gi")},
			WorkloadNameNamespace: tfv1.NameNamespace{Name: "wl", Namespace: "ns"},
			Isolation:             tfv1.IsolationModeSoft,
		}
	}

	t.Run("Static scheduler rejects Dynamic GPU", func(t *testing.T) {
		s := newTestAllocator()
		s.isolationPolicy = tfv1.IsolationModePolicyStatic
		gpu := makeGPU("gpu-dynamic", "100", "24Gi", "100", "24Gi")
		gpu.Status.IsolationPolicy = tfv1.IsolationModePolicyDynamic
		gpu.Status.NodeSelector = map[string]string{constants.KubernetesHostNameLabel: "node-1"}
		s.gpuStore[types.NamespacedName{Name: gpu.Name}] = gpu

		assert.ErrorContains(t, s.Assume([]string{gpu.Name}, newRequest("static-request")), "incompatible")
	})

	t.Run("Dynamic scheduler rejects GPU that has not advertised Dynamic", func(t *testing.T) {
		s := newTestAllocator()
		s.isolationPolicy = tfv1.IsolationModePolicyDynamic
		gpu := makeGPU("gpu-static", "100", "24Gi", "100", "24Gi")
		gpu.Status.IsolationPolicy = tfv1.IsolationModePolicyStatic
		gpu.Status.NodeSelector = map[string]string{constants.KubernetesHostNameLabel: "node-1"}
		s.gpuStore[types.NamespacedName{Name: gpu.Name}] = gpu

		assert.ErrorContains(t, s.Assume([]string{gpu.Name}, newRequest("dynamic-request")), "has not advertised Dynamic")
	})
}

func TestRecoverDynamicIsolationModes(t *testing.T) {
	workers := []corev1.Pod{
		{ObjectMeta: metav1.ObjectMeta{UID: "soft", Annotations: map[string]string{constants.GPUDeviceIDsAnnotation: "gpu-soft"}}},
		{ObjectMeta: metav1.ObjectMeta{UID: "hard", Annotations: map[string]string{constants.GPUDeviceIDsAnnotation: "gpu-hard"}}},
		{ObjectMeta: metav1.ObjectMeta{UID: "partitioned", Annotations: map[string]string{constants.GPUDeviceIDsAnnotation: "gpu-partitioned"}}},
		{ObjectMeta: metav1.ObjectMeta{UID: "conflicting", Annotations: map[string]string{constants.GPUDeviceIDsAnnotation: "gpu-soft"}}},
	}
	allocations := map[string]*tfv1.AllocRequest{
		"soft":        {Isolation: tfv1.IsolationModeSoft},
		"hard":        {Isolation: tfv1.IsolationModeHard},
		"partitioned": {Isolation: tfv1.IsolationModePartitioned},
		"conflicting": {Isolation: tfv1.IsolationModeHard},
	}

	modes, conflicts := recoverDynamicIsolationModes(workers, allocations)
	assert.Equal(t, tfv1.IsolationModeType(tfv1.IsolationModeSoft), modes[types.NamespacedName{Name: "gpu-soft"}])
	assert.Equal(t, tfv1.IsolationModeType(tfv1.IsolationModeHard), modes[types.NamespacedName{Name: "gpu-hard"}])
	assert.True(t, conflicts[types.NamespacedName{Name: "gpu-soft"}])
	assert.True(t, conflicts[types.NamespacedName{Name: "gpu-partitioned"}])
}

func TestSyncGPUMetadataPreservesSchedulerOwnedDynamicStatus(t *testing.T) {
	old := &tfv1.GPU{Status: tfv1.GPUStatus{
		IsolationPolicy:          tfv1.IsolationModePolicyDynamic,
		ActiveIsolationMode:      tfv1.IsolationModeSoft,
		DynamicIsolationConflict: true,
	}}
	incoming := &tfv1.GPU{Status: tfv1.GPUStatus{
		IsolationPolicy:          tfv1.IsolationModePolicyDynamic,
		ActiveIsolationMode:      tfv1.IsolationModeHard,
		DynamicIsolationConflict: false,
		Message:                  "managed",
	}}

	syncGPUMetadataAndStatusFromCluster(old, incoming)
	assert.Equal(t, tfv1.IsolationModeType(tfv1.IsolationModeSoft), old.Status.ActiveIsolationMode)
	assert.True(t, old.Status.DynamicIsolationConflict)
	assert.Equal(t, "managed", old.Status.Message)
}
