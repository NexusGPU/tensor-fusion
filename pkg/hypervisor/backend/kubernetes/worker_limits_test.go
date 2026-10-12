package kubernetes

import (
	"encoding/json"
	"maps"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// The GPU resource carries the TFLOPS capacity the device controller resolved
// for the device, exactly, and says whether it is a vendor-published FP16
// figure or the provider's estimate. The backend does not resolve anything
// itself: it is the only writer of status.capacity.tflops and writes what the
// device holds.
func TestGPUResourceCarriesDeviceTflopsCapacity(t *testing.T) {
	const (
		fromTable    = constants.GPUTflopsCapacitySourceProviderConfig
		fromEstimate = constants.GPUTflopsCapacitySourceEstimate
	)
	tests := []struct {
		name string
		// model and estimate are what the provider library reports; fp16Tflops is
		// the hardware table's figure for the model, zero when it lists none.
		model      string
		estimate   float64
		fp16Tflops float64
		want       string
		wantSource string
	}{
		{"RTX 3090 listed", "NVIDIA GeForce RTX 3090", 44.08, 71.2, "71.2", fromTable},
		{"RTX 4090 listed", "NVIDIA GeForce RTX 4090", 82.58, 165.16, "165.16", fromTable},
		{"A100 80GB listed", "NVIDIA A100-SXM4-80GB", 19.49184, 312, "312", fromTable},
		{"H100 80GB listed", "NVIDIA H100 80GB HBM3", 66.90816, 989, "989", fromTable},
		{"T4 listed", "Tesla T4", 8.1408, 65, "65", fromTable},
		{"MIG parent listed, no estimate", "NVIDIA A100-SXM4-80GB", 0, 312, "312", fromTable},
		{"RTX A6000 not listed", "NVIDIA RTX A6000", 38.7072, 0, "38.7072", fromEstimate},
		{"A100 80GB not listed", "NVIDIA A100-SXM4-80GB", 19.49184, 0, "19.49184", fromEstimate},
		{"not listed and no estimate", "NVIDIA RTX A6000", 0, 0, "0", fromEstimate},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &KubeletBackend{ctx: t.Context(), apiClient: &APIClient{}, nodeName: "node"}
			device := &api.DeviceInfo{
				UUID: "gpu-0", Vendor: constants.AcceleratorVendorNvidia, Model: tt.model,
				TotalMemoryBytes: 16 << 30, ProviderEstimatedTflops: tt.estimate,
			}
			device.ResolveTflopsCapacity(tt.fp16Tflops, tt.fp16Tflops > 0)

			gpuNode := &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{
				Name: "node", OwnerReferences: []metav1.OwnerReference{{Name: "pool"}},
			}}
			gpu := &tfv1.GPU{}
			require.NoError(t, backend.mutateGPUResourceState(device, gpuNode, gpu))
			require.True(t, resource.MustParse(tt.want).Equal(gpu.Status.Capacity.Tflops),
				"published %s, want %s", gpu.Status.Capacity.Tflops.String(), tt.want)
			require.Equal(t, tt.wantSource, gpu.Annotations[constants.GPUTflopsCapacitySourceAnnotation])
			require.True(t, gpu.Status.Capacity.Tflops.Equal(gpu.Status.Available.Tflops))

			// What a reader of the resource computes with is the device's capacity
			// to the last bit, also after the resource went through the API.
			raw, err := json.Marshal(gpu)
			require.NoError(t, err)
			read := &tfv1.GPU{}
			require.NoError(t, json.Unmarshal(raw, read))
			require.Equal(t, device.TflopsCapacity, read.Status.Capacity.Tflops.AsApproximateFloat64())
			require.True(t, api.TflopsQuantity(device.TflopsCapacity).Equal(read.Status.Capacity.Tflops))

			// The marker follows the device: once the model is listed it changes
			// with the next sync, and a device that was never resolved has none.
			listed := device.DeepCopy()
			listed.ResolveTflopsCapacity(312, true)
			require.NoError(t, backend.mutateGPUResourceState(listed, gpuNode, gpu))
			require.Equal(t, constants.GPUTflopsCapacitySourceProviderConfig,
				gpu.Annotations[constants.GPUTflopsCapacitySourceAnnotation])
			require.True(t, resource.MustParse("312").Equal(gpu.Status.Capacity.Tflops))
			unresolved := &api.DeviceInfo{UUID: "gpu-0", Model: tt.model}
			require.NoError(t, backend.mutateGPUResourceState(unresolved, gpuNode, gpu))
			require.NotContains(t, gpu.Annotations, constants.GPUTflopsCapacitySourceAnnotation)
		})
	}
}

// Node totals come from the device controller, which sums the same capacities;
// the backend passes them on unchanged.
func TestDiscoveryCompleteDoesNotRewriteNodeTotals(t *testing.T) {
	node := &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{Name: "node"}}
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPUNode{}).WithObjects(node).Build()
	backend := &KubeletBackend{ctx: t.Context(), apiClient: NewAPIClient(t.Context(), k8sClient), nodeName: "node"}

	backend.GetDeviceChangeHandler().OnDiscoveryComplete(&api.NodeInfo{
		TotalTFlops: 71.2 + 312, TotalVRAMBytes: 104 << 30, DeviceIDs: []string{"gpu-0", "gpu-1"},
	})

	updated := &tfv1.GPUNode{}
	require.NoError(t, k8sClient.Get(t.Context(), client.ObjectKey{Name: "node"}, updated))
	require.InDelta(t, 383.2, updated.Status.TotalTFlops.AsApproximateFloat64(), 1e-6)
	require.Equal(t, int32(2), updated.Status.TotalGPUs)
}

// The autoscaler adjusts a running worker by patching its resource
// annotations. The backend must report the new limits through OnUpdate.
func TestWorkerSyncReportsInPlaceLimitChange(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	pod := createTestPodWithIndex(1)
	added := make(chan *api.WorkerInfo, 1)
	type update struct{ old, new *api.WorkerInfo }
	updates := make(chan update, 4)
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnAdd:    func(info *api.WorkerInfo) { added <- info },
		OnUpdate: func(old, new *api.WorkerInfo) { updates <- update{old, new} },
	}))
	b.podCacher.onPodAdd(pod)
	var original *api.WorkerInfo
	select {
	case original = <-added:
	case <-time.After(time.Second):
		t.Fatal("Add was not delivered")
	}
	require.Equal(t, "20", original.Limits.Tflops.String())
	require.Equal(t, "2Gi", original.Limits.Vram.String())

	patched := pod.DeepCopy()
	maps.Copy(patched.Annotations, utils.GPUResourcesToAnnotations(&tfv1.Resources{
		Requests: tfv1.Resource{Tflops: resource.MustParse("5"), Vram: resource.MustParse("512Mi")},
		Limits:   tfv1.Resource{Tflops: resource.MustParse("8"), Vram: resource.MustParse("1Gi")},
	}))
	b.podCacher.onPodUpdate(pod, patched)

	select {
	case got := <-updates:
		require.Equal(t, original, got.old)
		require.Equal(t, original.WorkerUID, got.new.WorkerUID)
		require.Equal(t, "8", got.new.Limits.Tflops.String())
		require.Equal(t, "1Gi", got.new.Limits.Vram.String())
		require.Equal(t, "5", got.new.Requests.Tflops.String())
		require.Equal(t, original.AllocatedDevices, got.new.AllocatedDevices)
		require.Equal(t, original.IsolationMode, got.new.IsolationMode)
		workers := b.ListWorkers()
		require.Len(t, workers, 1)
		require.Equal(t, got.new, workers[0])
	case <-time.After(time.Second):
		t.Fatal("limit change was not reported to the worker handler")
	}
}
