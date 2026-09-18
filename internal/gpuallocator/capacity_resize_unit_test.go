package gpuallocator

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// A capacity downgrade can leave active requests larger than the new capacity.
// Releasing or shrinking one request must credit only the true remaining space,
// for both compute and VRAM, including when another deficit remains.
func TestCapacityShrinkThenReleaseOrAdjust(t *testing.T) {
	for _, mode := range []tfv1.IsolationModeType{tfv1.IsolationModeSoft, tfv1.IsolationModeHard} {
		for _, action := range []string{"dealloc", "rollback", "adjust"} {
			t.Run(string(mode)+"/"+action, func(t *testing.T) {
				ctx := context.Background()
				s := newTestAllocator()
				gpu := makeGPU(testGPU1Name, "989", "200Gi", "89", "40Gi")
				key := types.NamespacedName{Name: gpu.Name}
				s.gpuStore[key] = gpu
				a := makeAllocReq("a", []string{gpu.Name}, "450", "80Gi")
				b := makeAllocReq("b", []string{gpu.Name}, "450", "80Gi")
				a.Isolation, b.Isolation = mode, mode
				s.uniqueAllocation[string(a.PodMeta.UID)] = a
				s.uniqueAllocation[string(b.PodMeta.UID)] = b
				incoming := gpu.DeepCopy()
				incoming.Status.Capacity = &tfv1.Resource{Tflops: qty("835"), Vram: qty("141Gi")}
				scheme := runtime.NewScheme()
				require.NoError(t, tfv1.AddToScheme(scheme))
				s.Client = fake.NewClientBuilder().WithScheme(scheme).
					WithStatusSubresource(&tfv1.GPU{}).WithObjects(incoming).Build()
				s.handleGPUUpdate(ctx, incoming)
				s.SyncGPUsToK8s()
				persisted := &tfv1.GPU{}
				require.NoError(t, s.Get(ctx, key, persisted))
				require.True(t, persisted.Status.Available.Tflops.IsZero())
				require.True(t, persisted.Status.Available.Vram.IsZero())
				extra := makeAllocReq("extra", []string{gpu.Name}, "1", "1Gi")
				require.Error(t, s.applyAllocationToGPU(gpu, extra, gpu.Name))

				expectedTflops, expectedVram := "385", "61Gi"
				switch action {
				case "dealloc":
					s.Dealloc(a.WorkloadNameNamespace, a.GPUNames, a.PodMeta)
				case "rollback":
					require.NoError(t, s.Rollback(string(a.PodMeta.UID)))
				case "adjust":
					adjust := tfv1.AdjustRequest{PodUID: string(a.PodMeta.UID),
						NewRequest: tfv1.Resource{Tflops: qty("440"), Vram: qty("79Gi")}}
					_, _, _, err := s.AdjustAllocation(ctx, adjust, false)
					require.NoError(t, err)
					require.True(t, gpu.Status.Available.Tflops.IsZero(), "remaining requests still exceed capacity")
					require.True(t, gpu.Status.Available.Vram.IsZero())
					adjust.NewRequest = tfv1.Resource{Tflops: qty("300"), Vram: qty("50Gi")}
					_, _, _, err = s.AdjustAllocation(ctx, adjust, true)
					require.NoError(t, err)
					require.Equal(t, "440", a.Request.Tflops.String(), "dry-run must not update the ledger")
					require.True(t, gpu.Status.Available.Vram.IsZero())
					_, _, _, err = s.AdjustAllocation(ctx, adjust, false)
					require.NoError(t, err)
					expectedTflops, expectedVram = "85", "11Gi"
				}
				s.SyncGPUsToK8s()
				require.NoError(t, s.Get(ctx, key, persisted))
				require.Equal(t, expectedTflops, persisted.Status.Available.Tflops.String())
				require.Equal(t, expectedVram, persisted.Status.Available.Vram.String())
				if action == "adjust" {
					s.Dealloc(a.WorkloadNameNamespace, a.GPUNames, a.PodMeta)
				}
				s.Dealloc(b.WorkloadNameNamespace, b.GPUNames, b.PodMeta)
				require.Equal(t, "835", gpu.Status.Available.Tflops.String())
				require.Equal(t, "141Gi", gpu.Status.Available.Vram.String())
			})
		}
	}
}

func TestCapacityChangeRoundTripPublishesAvailable(t *testing.T) {
	for _, tc := range []struct {
		name                     string
		mode                     tfv1.IsolationModeType
		percent                  bool
		expected989, expected835 string
	}{
		{"idle", "", false, "989", "835"},
		{"soft_absolute", tfv1.IsolationModeSoft, false, "889", "735"},
		{"hard_absolute", tfv1.IsolationModeHard, false, "889", "735"},
		{"soft_percent", tfv1.IsolationModeSoft, true, "792", "668"},
		{"hard_percent", tfv1.IsolationModeHard, true, "792", "668"},
		{"shared", tfv1.IsolationModeShared, false, "0", "0"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			s := newTestAllocator()
			g := makeGPU("h200-background", "835", "141Gi", "835", "141Gi")
			key := types.NamespacedName{Name: g.Name}
			s.gpuStore[key] = g
			req := makeAllocReq("holder", []string{g.Name}, "100", "1Gi")
			req.Isolation = tc.mode
			if tc.percent {
				req.Request.ComputePercent = qty("20")
			}
			if tc.mode != "" {
				s.uniqueAllocation[string(req.PodMeta.UID)] = req
				s.recomputeGPUAvailableFromAllocations(g)
			}
			scheme := runtime.NewScheme()
			require.NoError(t, tfv1.AddToScheme(scheme))
			s.Client = fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&tfv1.GPU{}).WithObjects(g.DeepCopy()).Build()
			for _, step := range []struct{ capacity, expected string }{{"989", tc.expected989}, {"835", tc.expected835}} {
				incoming := &tfv1.GPU{}
				require.NoError(t, s.Get(ctx, key, incoming))
				incoming.Status.Capacity.Tflops = qty(step.capacity)
				require.NoError(t, s.Status().Update(ctx, incoming))
				s.handleGPUUpdate(ctx, incoming)
				s.SyncGPUsToK8s()
				persisted := &tfv1.GPU{}
				require.NoError(t, s.Get(ctx, key, persisted))
				require.Equal(t, step.expected, persisted.Status.Available.Tflops.String())
				require.Equal(t, step.capacity, persisted.Status.Capacity.Tflops.String())
			}
			if tc.mode != "" {
				s.Dealloc(req.WorkloadNameNamespace, req.GPUNames, req.PodMeta)
				s.SyncGPUsToK8s()
				persisted := &tfv1.GPU{}
				require.NoError(t, s.Get(ctx, key, persisted))
				require.Equal(t, "835", persisted.Status.Available.Tflops.String())
				require.Equal(t, "141Gi", persisted.Status.Available.Vram.String())
			}
		})
	}
}
