package kubernetes

import (
	"context"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

const (
	capacityTestNodeName = "capacity-test-gpu-node"
	capacityTestGPUName  = "gpu-capacity-test"
	// What the provider library estimates for the test device, and what the
	// hardware table publishes for its model.
	capacityTestEstimate = 44.08
	capacityTestFp16     = 71.2
)

// capacityTestGPU is a GPU resource as it exists while workloads run on it:
// capacity from an earlier sync of the node agent, the allocation fields from
// the operator's allocator. An empty availableTflops leaves the status unset.
func capacityTestGPU(capacityTflops, availableTflops string, runningApps int) *tfv1.GPU {
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name:   capacityTestGPUName,
			Labels: map[string]string{constants.LabelKeyOwner: capacityTestNodeName},
			Annotations: map[string]string{
				constants.LastSyncTimeAnnotationKey: time.Now().Add(-time.Hour).Format(time.RFC3339),
			},
		},
	}
	if availableTflops == "" {
		return gpu
	}
	gpu.Status = tfv1.GPUStatus{
		Phase: tfv1.TensorFusionGPUPhaseRunning,
		Capacity: &tfv1.Resource{
			Tflops: resource.MustParse(capacityTflops),
			Vram:   resource.MustParse("24Gi"),
		},
		Available: &tfv1.Resource{
			Tflops: resource.MustParse(availableTflops),
			Vram:   resource.MustParse("8Gi"),
		},
	}
	for i := range runningApps {
		gpu.Status.RunningApps = append(gpu.Status.RunningApps, &tfv1.RunningAppDetail{
			Name: "app-" + string(rune('a'+i)), Namespace: "default", Count: 1,
		})
	}
	return gpu
}

// capacityTestDevice is the device as the device controller hands it to the
// backend: with the hardware table's figure when the model is listed there,
// with the provider's estimate otherwise.
func capacityTestDevice(listed bool) *api.DeviceInfo {
	device := &api.DeviceInfo{
		UUID:                    capacityTestGPUName,
		Vendor:                  constants.AcceleratorVendorNvidia,
		Model:                   "NVIDIA GeForce RTX 3090",
		TotalMemoryBytes:        24 << 30,
		ProviderEstimatedTflops: capacityTestEstimate,
	}
	if listed {
		device.ResolveTflopsCapacity(capacityTestFp16, true)
	} else {
		device.ResolveTflopsCapacity(0, false)
	}
	return device
}

func newCapacityTestBackend(
	t *testing.T, funcs interceptor.Funcs, gpus ...*tfv1.GPU,
) (*KubeletBackend, client.Client) {
	t.Helper()
	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name:            capacityTestNodeName,
			OwnerReferences: []metav1.OwnerReference{{Name: "test-gpu-pool"}},
		},
	}
	builder := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPU{}).
		WithInterceptorFuncs(funcs).
		WithObjects(gpuNode)
	for _, gpu := range gpus {
		builder = builder.WithObjects(gpu)
	}
	k8sClient := builder.Build()
	return &KubeletBackend{
		ctx:       t.Context(),
		apiClient: NewAPIClient(t.Context(), k8sClient),
		nodeName:  capacityTestNodeName,
	}, k8sClient
}

// syncCapacityTestDevice is one device sync of the node agent.
func syncCapacityTestDevice(t *testing.T, backend *KubeletBackend, device *api.DeviceInfo) {
	t.Helper()
	require.NoError(t, backend.apiClient.CreateOrUpdateGPU(capacityTestNodeName, device.UUID,
		func(node *tfv1.GPUNode, gpu *tfv1.GPU) error {
			return backend.mutateGPUResourceState(device, node, gpu)
		}))
}

func getCapacityTestGPU(t *testing.T, k8sClient client.Client) *tfv1.GPU {
	t.Helper()
	gpu := &tfv1.GPU{}
	require.NoError(t, k8sClient.Get(t.Context(), client.ObjectKey{Name: capacityTestGPUName}, gpu))
	return gpu
}

func requireQuantity(t *testing.T, want string, got resource.Quantity, what string) {
	t.Helper()
	require.True(t, resource.MustParse(want).Equal(got), "%s: got %s, want %s", what, got.String(), want)
}

// The rule the node agent applies to status.available: capacity minus what the
// resource records as allocated, never below zero and never above the
// capacity. It is the allocator's rule, so the two write the same figure.
func TestSettleGPUAvailable(t *testing.T) {
	res := func(tflops, vram string) *tfv1.Resource {
		return &tfv1.Resource{Tflops: resource.MustParse(tflops), Vram: resource.MustParse(vram)}
	}
	tests := []struct {
		name string
		// live is the status on the API server; desiredCapacity what the agent
		// is about to publish. desiredAvailable is what the agent's mutate left
		// in available: live's figure unless it reset it.
		liveCapacity, liveAvailable       *tfv1.Resource
		desiredCapacity, desiredAvailable *tfv1.Resource
		wantTflops, wantVram              string
	}{
		{
			name:            "first publication: everything is available",
			desiredCapacity: res("71.2", "24Gi"),
			wantTflops:      "71.2", wantVram: "24Gi",
		},
		{
			name:         "capacity grows, allocations stay allocated",
			liveCapacity: res("44.08", "24Gi"), liveAvailable: res("14.08", "8Gi"),
			desiredCapacity: res("71.2", "24Gi"), desiredAvailable: res("14.08", "8Gi"),
			wantTflops: "41.2", wantVram: "8Gi",
		},
		{
			name:         "capacity shrinks, allocations still fit",
			liveCapacity: res("71.2", "24Gi"), liveAvailable: res("41.2", "8Gi"),
			desiredCapacity: res("44.08", "24Gi"), desiredAvailable: res("41.2", "8Gi"),
			wantTflops: "14.08", wantVram: "8Gi",
		},
		{
			name:         "capacity shrinks below what is allocated: zero, not negative",
			liveCapacity: res("71.2", "24Gi"), liveAvailable: res("11.2", "8Gi"),
			desiredCapacity: res("44.08", "24Gi"), desiredAvailable: res("11.2", "8Gi"),
			wantTflops: "0", wantVram: "8Gi",
		},
		{
			name:         "both resources change",
			liveCapacity: res("71.2", "24Gi"), liveAvailable: res("41.2", "4Gi"),
			desiredCapacity: res("44.08", "16Gi"), desiredAvailable: res("41.2", "4Gi"),
			wantTflops: "14.08", wantVram: "0",
		},
		{
			name:         "idle GPU follows the capacity",
			liveCapacity: res("44.08", "24Gi"), liveAvailable: res("44.08", "24Gi"),
			desiredCapacity: res("71.2", "24Gi"), desiredAvailable: res("44.08", "24Gi"),
			wantTflops: "71.2", wantVram: "24Gi",
		},
		{
			name:         "unchanged capacity: the allocator's figure is left alone",
			liveCapacity: res("71.2", "24Gi"), liveAvailable: res("41.2", "8Gi"),
			desiredCapacity: res("71.2", "24Gi"), desiredAvailable: res("41.2", "8Gi"),
			wantTflops: "41.2", wantVram: "8Gi",
		},
		{
			name:         "negative figure left by an older operator is floored",
			liveCapacity: res("44.08", "24Gi"), liveAvailable: res("-15.92", "8Gi"),
			desiredCapacity: res("44.08", "24Gi"), desiredAvailable: res("-15.92", "8Gi"),
			wantTflops: "0", wantVram: "8Gi",
		},
		{
			name:         "negative figure and a larger capacity: the deficit is honoured",
			liveCapacity: res("44.08", "24Gi"), liveAvailable: res("-15.92", "8Gi"),
			desiredCapacity: res("71.2", "24Gi"), desiredAvailable: res("-15.92", "8Gi"),
			wantTflops: "11.2", wantVram: "8Gi",
		},
		{
			name:         "figure above the capacity is capped",
			liveCapacity: res("44.08", "24Gi"), liveAvailable: res("71.2", "48Gi"),
			desiredCapacity: res("44.08", "24Gi"), desiredAvailable: res("71.2", "48Gi"),
			wantTflops: "44.08", wantVram: "24Gi",
		},
		{
			name:         "reset by the agent keeps the reset value",
			liveCapacity: res("44.08", "24Gi"), liveAvailable: res("0", "0"),
			desiredCapacity: res("71.2", "24Gi"), desiredAvailable: res("71.2", "24Gi"),
			wantTflops: "71.2", wantVram: "24Gi",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			live := &tfv1.GPUStatus{Capacity: tt.liveCapacity, Available: tt.liveAvailable}
			desired := &tfv1.GPUStatus{Capacity: tt.desiredCapacity, Available: tt.desiredAvailable}

			settleGPUAvailable(live, desired)

			requireQuantity(t, tt.wantTflops, desired.Available.Tflops, "available tflops")
			requireQuantity(t, tt.wantVram, desired.Available.Vram, "available vram")
			require.GreaterOrEqual(t, desired.Available.Tflops.Sign(), 0)
			require.GreaterOrEqual(t, desired.Available.Vram.Sign(), 0)
			require.LessOrEqual(t, desired.Available.Tflops.Cmp(desired.Capacity.Tflops), 0)
			require.LessOrEqual(t, desired.Available.Vram.Cmp(desired.Capacity.Vram), 0)
		})
	}
}

// A device sync that changes the capacity of a GPU with running workloads
// writes capacity and available together. This is what a GPU looks like right
// after the node agent was upgraded, whichever operator version is running:
// an older operator does not republish available on a capacity change.
func TestCreateOrUpdateGPU_CapacityChangeKeepsAllocations(t *testing.T) {
	tests := []struct {
		name string
		// The GPU before the sync, and whether the hardware table lists the
		// model at this sync (71.2) or not (the estimate, 44.08).
		capacity, available string
		listed              bool
		wantCapacity        string
		wantAvailable       string
		wantSource          string
	}{
		{
			name:     "model is listed now: 44.08 -> 71.2 with 30 allocated",
			capacity: "44.08", available: "14.08", listed: true,
			wantCapacity: "71.2", wantAvailable: "41.2",
			wantSource: constants.GPUTflopsCapacitySourceProviderConfig,
		},
		{
			name:     "model is no longer listed: 71.2 -> 44.08 with 30 allocated",
			capacity: "71.2", available: "41.2", listed: false,
			wantCapacity: "44.08", wantAvailable: "14.08",
			wantSource: constants.GPUTflopsCapacitySourceEstimate,
		},
		{
			name:     "71.2 -> 44.08 with 60 allocated: over-committed, available is zero",
			capacity: "71.2", available: "11.2", listed: false,
			wantCapacity: "44.08", wantAvailable: "0",
			wantSource: constants.GPUTflopsCapacitySourceEstimate,
		},
		{
			name:     "same capacity: available is not touched",
			capacity: "71.2", available: "41.2", listed: true,
			wantCapacity: "71.2", wantAvailable: "41.2",
			wantSource: constants.GPUTflopsCapacitySourceProviderConfig,
		},
		{
			name:     "negative available left by an older operator",
			capacity: "44.08", available: "-15.92", listed: false,
			wantCapacity: "44.08", wantAvailable: "0",
			wantSource: constants.GPUTflopsCapacitySourceEstimate,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend, k8sClient := newCapacityTestBackend(t, interceptor.Funcs{},
				capacityTestGPU(tt.capacity, tt.available, 2))

			// Twice: the second sync finds its own figures and changes nothing.
			for range 2 {
				syncCapacityTestDevice(t, backend, capacityTestDevice(tt.listed))

				gpu := getCapacityTestGPU(t, k8sClient)
				requireQuantity(t, tt.wantCapacity, gpu.Status.Capacity.Tflops, "capacity tflops")
				requireQuantity(t, tt.wantAvailable, gpu.Status.Available.Tflops, "available tflops")
				requireQuantity(t, "24Gi", gpu.Status.Capacity.Vram, "capacity vram")
				requireQuantity(t, "8Gi", gpu.Status.Available.Vram, "available vram")
				require.GreaterOrEqual(t, gpu.Status.Available.Tflops.Sign(), 0)
				require.Len(t, gpu.Status.RunningApps, 2, "the allocator's running apps are kept")
				require.Equal(t, tfv1.TensorFusionGPUPhaseRunning, gpu.Status.Phase)
				require.Equal(t, tt.wantSource, gpu.Annotations[constants.GPUTflopsCapacitySourceAnnotation])
			}
		})
	}
}

func TestCreateOrUpdateGPU_NewGPUStartsFullyAvailable(t *testing.T) {
	backend, k8sClient := newCapacityTestBackend(t, interceptor.Funcs{})

	syncCapacityTestDevice(t, backend, capacityTestDevice(true))

	gpu := getCapacityTestGPU(t, k8sClient)
	requireQuantity(t, "71.2", gpu.Status.Capacity.Tflops, "capacity tflops")
	requireQuantity(t, "71.2", gpu.Status.Available.Tflops, "available tflops")
	require.True(t, gpu.Status.Capacity.Vram.Equal(gpu.Status.Available.Vram))
	require.Equal(t, tfv1.TensorFusionGPUPhasePending, gpu.Status.Phase)
}

// allocatorPublishes stands for the operator's allocator writing its fields
// while the node agent is in the middle of a device sync.
func allocatorPublishes(t *testing.T, ctx context.Context, c client.Client, availableTflops string, runningApps int) {
	t.Helper()
	gpu := &tfv1.GPU{}
	require.NoError(t, c.Get(ctx, client.ObjectKey{Name: capacityTestGPUName}, gpu))
	gpu.Status.Available.Tflops = resource.MustParse(availableTflops)
	gpu.Status.RunningApps = nil
	for i := range runningApps {
		gpu.Status.RunningApps = append(gpu.Status.RunningApps, &tfv1.RunningAppDetail{
			Name: "app-" + string(rune('a'+i)), Namespace: "default", Count: 1,
		})
	}
	require.NoError(t, c.Status().Update(ctx, gpu))
}

// The allocator owns available and runningApps. A device sync that read the
// GPU before the allocator published a new allocation must not write its older
// copy of those fields back, wherever in the sync the allocator's write lands.
func TestCreateOrUpdateGPU_DoesNotOverwriteAllocatorStatus(t *testing.T) {
	t.Run("allocator publishes after the sync read the GPU", func(t *testing.T) {
		gpuReads := 0
		backend, k8sClient := newCapacityTestBackend(t, interceptor.Funcs{
			Get: func(
				ctx context.Context, c client.WithWatch, key client.ObjectKey,
				obj client.Object, opts ...client.GetOption,
			) error {
				if err := c.Get(ctx, key, obj, opts...); err != nil {
					return err
				}
				if _, isGPU := obj.(*tfv1.GPU); isGPU {
					gpuReads++
					if gpuReads == 1 {
						// A second workload was just placed: 10 more TFLOPS.
						allocatorPublishes(t, ctx, c, "4.08", 2)
					}
				}
				return nil
			},
		}, capacityTestGPU("44.08", "14.08", 1))

		syncCapacityTestDevice(t, backend, capacityTestDevice(false))

		gpu := getCapacityTestGPU(t, k8sClient)
		requireQuantity(t, "44.08", gpu.Status.Capacity.Tflops, "capacity tflops")
		requireQuantity(t, "4.08", gpu.Status.Available.Tflops, "available tflops")
		require.Len(t, gpu.Status.RunningApps, 2)
	})

	t.Run("allocator publishes while the status write is on its way", func(t *testing.T) {
		statusWrites := 0
		backend, k8sClient := newCapacityTestBackend(t, interceptor.Funcs{
			SubResourcePatch: func(
				ctx context.Context, c client.Client, subResourceName string,
				obj client.Object, patch client.Patch, opts ...client.SubResourcePatchOption,
			) error {
				if _, isGPU := obj.(*tfv1.GPU); isGPU {
					statusWrites++
					if statusWrites == 1 {
						allocatorPublishes(t, ctx, c, "4.08", 2)
					}
				}
				return c.SubResource(subResourceName).Patch(ctx, obj, patch, opts...)
			},
		}, capacityTestGPU("44.08", "14.08", 1))

		// The capacity changes in the same sync: 40 are allocated by the time
		// the write goes through, so 71.2 leaves 31.2.
		syncCapacityTestDevice(t, backend, capacityTestDevice(true))

		gpu := getCapacityTestGPU(t, k8sClient)
		require.Equal(t, 2, statusWrites, "the write that lost the race is recomputed, not forced")
		requireQuantity(t, "71.2", gpu.Status.Capacity.Tflops, "capacity tflops")
		requireQuantity(t, "31.2", gpu.Status.Available.Tflops, "available tflops")
		require.Len(t, gpu.Status.RunningApps, 2)
	})
}

// UpdateGPUStatus is used by callers that read a GPU, change one field of
// their own and write the status back. The allocator's fields in their copy
// may be older than what is on the API server by then.
func TestUpdateGPUStatus_KeepsAllocatorOwnedFields(t *testing.T) {
	backend, k8sClient := newCapacityTestBackend(t, interceptor.Funcs{}, capacityTestGPU("71.2", "41.2", 1))

	stale, err := backend.apiClient.GetGPU(capacityTestGPUName)
	require.NoError(t, err)
	allocatorPublishes(t, t.Context(), k8sClient, "11.2", 2)

	stale.Status.UsedBy = tfv1.UsedByNvidiaDevicePlugin
	require.NoError(t, backend.apiClient.UpdateGPUStatus(stale))

	gpu := getCapacityTestGPU(t, k8sClient)
	require.Equal(t, tfv1.UsedByNvidiaDevicePlugin, gpu.Status.UsedBy)
	requireQuantity(t, "71.2", gpu.Status.Capacity.Tflops, "capacity tflops")
	requireQuantity(t, "11.2", gpu.Status.Available.Tflops, "available tflops")
	require.Len(t, gpu.Status.RunningApps, 2)
}
