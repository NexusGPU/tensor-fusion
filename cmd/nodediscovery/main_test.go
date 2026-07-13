package main

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/stretchr/testify/assert"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

type stubNVMLDevice struct {
	nvml.Device
	getNvLinkStateFunc            func(int) (nvml.EnableState, nvml.Return)
	getNvLinkVersionFunc          func(int) (uint32, nvml.Return)
	getNvLinkRemoteDeviceTypeFunc func(int) (nvml.IntNvLinkDeviceType, nvml.Return)
	getNvLinkRemotePciInfoFunc    func(int) (nvml.PciInfo, nvml.Return)
	getP2PStatusFunc              func(nvml.Device, nvml.GpuP2PCapsIndex) (nvml.GpuP2PStatus, nvml.Return)
}

const (
	testDeviceName  = "NVIDIA-Test-GPU"
	testNodeName    = "test-node"
	testGPUNodeName = "test-gpu-node"
)

func (d *stubNVMLDevice) GetNvLinkState(n int) (nvml.EnableState, nvml.Return) {
	return d.getNvLinkStateFunc(n)
}

func (d *stubNVMLDevice) GetNvLinkVersion(n int) (uint32, nvml.Return) {
	return d.getNvLinkVersionFunc(n)
}

func (d *stubNVMLDevice) GetNvLinkRemoteDeviceType(n int) (nvml.IntNvLinkDeviceType, nvml.Return) {
	return d.getNvLinkRemoteDeviceTypeFunc(n)
}

func (d *stubNVMLDevice) GetNvLinkRemotePciInfo(n int) (nvml.PciInfo, nvml.Return) {
	return d.getNvLinkRemotePciInfoFunc(n)
}

func (d *stubNVMLDevice) GetP2PStatus(device nvml.Device, idx nvml.GpuP2PCapsIndex) (nvml.GpuP2PStatus, nvml.Return) {
	return d.getP2PStatusFunc(device, idx)
}

func TestCreateOrUpdateTensorFusionGPU(t *testing.T) {
	// Setup test data
	ctx := context.Background()
	uuid := "test-uuid"
	memInfo := nvml.Memory_v2{Total: 16 * 1024 * 1024 * 1024} // 16 GiB
	tflops := resource.MustParse("100")
	deviceName := testDeviceName
	k8sNodeName := testNodeName
	gpuNodeName := testGPUNodeName

	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: gpuNodeName,
			OwnerReferences: []metav1.OwnerReference{
				{
					Name: "test-gpu-pool",
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	_ = tfv1.AddToScheme(scheme)

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&tfv1.GPU{}).Build()

	gpu, err := createOrUpdateTensorFusionGPU(
		k8sClient, ctx, k8sNodeName, gpuNode, uuid, deviceName, memInfo, tflops, 1, -1, nil)
	assert.NoError(t, err)

	// Assertions
	assert.NotNil(t, gpu, "GPU object should not be nil")
	assert.Equal(t, uuid, gpu.Name, "GPU name should match UUID")
	assert.Equal(t, deviceName, gpu.Status.GPUModel, "GPU model should match device name")
	assert.Equal(t, tflops, gpu.Status.Capacity.Tflops, "GPU TFlops should match")
	assert.Equal(t, resource.MustParse("16384Mi"), gpu.Status.Capacity.Vram, "GPU VRAM should match")
	assert.Equal(t, gpu.Status.Capacity, gpu.Status.Available, "Available resources should match capacity")
	assert.Equal(t, map[string]string{"kubernetes.io/hostname": k8sNodeName},
		gpu.Status.NodeSelector, "Node selector should match")

	// Verify labels and annotations
	assert.Equal(t, map[string]string{
		constants.LabelKeyOwner: gpuNodeName,
		constants.GpuPoolKey:    "test-gpu-pool",
	}, gpu.Labels, "GPU labels should match")
	assert.Contains(t, gpu.Annotations, constants.LastSyncTimeAnnotationKey,
		"GPU annotations should contain last report time")
	_, err = time.Parse(time.RFC3339, gpu.Annotations[constants.LastSyncTimeAnnotationKey])
	assert.NoError(t, err, "Last report time annotation should be a valid RFC3339 timestamp")

	// Verify the Available field does not change after the update
	gpu.Status.Available.Tflops.Sub(resource.MustParse("1000"))
	gpu.Status.Available.Vram.Sub(resource.MustParse("2000Mi"))
	err = k8sClient.Status().Update(ctx, gpu)
	assert.NoError(t, err)

	tflops.Add(resource.MustParse("100"))
	updatedGpu, err := createOrUpdateTensorFusionGPU(
		k8sClient, ctx, k8sNodeName, gpuNode, uuid, deviceName, memInfo, tflops, 1, -1, nil,
	)
	assert.NoError(t, err)
	assert.NotEqual(t, updatedGpu.Status.Capacity, gpu.Status.Capacity, "GPU capacity should not match")
	assert.Equal(t, updatedGpu.Status.Available.Tflops, gpu.Status.Available.Tflops, "GPU TFlops should match")
	assert.Equal(t, updatedGpu.Status.Available.Vram, gpu.Status.Available.Vram, "GPU VRAM should match")
}

func TestCreateOrUpdateTensorFusionGPU_PreservesUUIDAndNormalizesNvLinkPeerUUID(t *testing.T) {
	ctx := context.Background()
	uuid := "gpu-test-uuid"
	memInfo := nvml.Memory_v2{Total: 16 * 1024 * 1024 * 1024}
	tflops := resource.MustParse("100")
	deviceName := testDeviceName
	k8sNodeName := testNodeName
	gpuNodeName := testGPUNodeName

	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: gpuNodeName,
			OwnerReferences: []metav1.OwnerReference{
				{
					Name: "test-gpu-pool",
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	_ = tfv1.AddToScheme(scheme)

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&tfv1.GPU{}).Build()

	nvlink := &tfv1.GPUNvLinkStatus{
		Peers: []tfv1.GPUNvLinkPeer{
			{PeerUUID: "GPU-PEER-UUID "},
		},
	}
	gpu, err := createOrUpdateTensorFusionGPU(
		k8sClient, ctx, k8sNodeName, gpuNode, uuid, deviceName, memInfo, tflops, 1, -1, nvlink)
	assert.NoError(t, err)
	assert.NotNil(t, gpu)

	assert.Equal(t, uuid, gpu.Name)
	assert.Equal(t, uuid, gpu.Status.UUID)
	assert.NotNil(t, gpu.Status.NvLink)
	assert.Len(t, gpu.Status.NvLink.Peers, 1)
	assert.Equal(t, "gpu-peer-uuid", gpu.Status.NvLink.Peers[0].PeerUUID)
}

func TestParseLaptopGPU(t *testing.T) {
	deviceName := "NVIDIA-Test-GPU Laptop GPU"
	isLaptopGPU := strings.HasSuffix(deviceName, " Laptop GPU")
	assert.True(t, isLaptopGPU)
	deviceName = strings.ReplaceAll(deviceName, " Laptop GPU", "")
	assert.Equal(t, "NVIDIA-Test-GPU", deviceName)
	tflops := resource.MustParse("100.147")
	tflops = resource.MustParse(fmt.Sprintf("%.2f", tflops.AsApproximateFloat64()*constants.MobileGpuClockSpeedMultiplier))
	expected := resource.MustParse("75110m")
	assert.Equal(t, expected.String(), tflops.String())
}

func TestGPUControllerReference(t *testing.T) {
	// Setup test data
	ctx := context.Background()
	uuid := "test-uuid"
	memInfo := nvml.Memory_v2{Total: 16 * 1024 * 1024 * 1024} // 16 GiB
	tflops := resource.MustParse("100")
	deviceName := testDeviceName
	k8sNodeName := testNodeName
	gpuNodeName := testGPUNodeName

	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: gpuNodeName,
			UID:  "mock-uid",
			OwnerReferences: []metav1.OwnerReference{
				{
					Name: "test-gpu-pool",
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	_ = tfv1.AddToScheme(scheme)

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&tfv1.GPU{}).Build()

	gpu, err := createOrUpdateTensorFusionGPU(
		k8sClient, ctx, k8sNodeName, gpuNode, uuid, deviceName, memInfo, tflops, 1, -1, nil)
	assert.NoError(t, err)
	assert.True(t, metav1.IsControlledBy(gpu, gpuNode))

	newGpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "new-test-gpu-node",
			UID:  "new-mock-uid",
			OwnerReferences: []metav1.OwnerReference{
				{
					Name: "new-test-gpu-pool",
				},
			},
		},
	}

	gpu, err = createOrUpdateTensorFusionGPU(
		k8sClient, ctx, k8sNodeName, newGpuNode, uuid, deviceName, memInfo, tflops, 1, -1, nil)
	assert.NoError(t, err)
	assert.NotNil(t, gpu.OwnerReferences[0].Kind)
	assert.NotNil(t, gpu.OwnerReferences[0].APIVersion)
	assert.True(t, metav1.IsControlledBy(gpu, newGpuNode))
	assert.False(t, metav1.IsControlledBy(gpu, gpuNode))
}

func TestPatchGPUNodeStatus(t *testing.T) {
	tests := []struct {
		name           string
		setupGPUNode   func() *tfv1.GPUNode
		totalTFlops    resource.Quantity
		totalVRAM      resource.Quantity
		count          int32
		allDeviceIDs   []string
		expectError    bool
		validateResult func(t *testing.T, originalNode, patchedNode *tfv1.GPUNode)
	}{
		{
			name: "successful patch with empty phase",
			setupGPUNode: func() *tfv1.GPUNode {
				return &tfv1.GPUNode{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-gpu-node",
						Namespace: "default",
						OwnerReferences: []metav1.OwnerReference{
							{
								Name: "test-gpu-pool",
							},
						},
					},
					Status: tfv1.GPUNodeStatus{
						Phase:       "", // Empty phase should be set to pending
						TotalTFlops: resource.MustParse("50"),
						TotalVRAM:   resource.MustParse("8Gi"),
						TotalGPUs:   2,
					},
				}
			},
			totalTFlops:  resource.MustParse("100"),
			totalVRAM:    resource.MustParse("16Gi"),
			count:        4,
			allDeviceIDs: []string{"gpu-0", "gpu-1", "gpu-2", "gpu-3"},
			expectError:  false,
			validateResult: func(t *testing.T, originalNode, patchedNode *tfv1.GPUNode) {
				// Verify status fields were updated
				assert.Equal(t, resource.MustParse("100"), patchedNode.Status.TotalTFlops)
				assert.Equal(t, resource.MustParse("16Gi"), patchedNode.Status.TotalVRAM)
				assert.Equal(t, int32(4), patchedNode.Status.TotalGPUs)
				assert.Equal(t, int32(4), patchedNode.Status.ManagedGPUs)
				assert.Equal(t, []string{"gpu-0", "gpu-1", "gpu-2", "gpu-3"}, patchedNode.Status.ManagedGPUDeviceIDs)
				assert.Equal(t, tfv1.TensorFusionGPUNodePhasePending, patchedNode.Status.Phase)
				// Verify NodeInfo was updated
				assert.True(t, patchedNode.Status.NodeInfo.RAMSize.Value() > 0)
				assert.True(t, patchedNode.Status.NodeInfo.DataDiskSize.Value() > 0)
			},
		},
		{
			name: "successful patch with existing phase preserved",
			setupGPUNode: func() *tfv1.GPUNode {
				return &tfv1.GPUNode{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-gpu-node-running",
						Namespace: "default",
						OwnerReferences: []metav1.OwnerReference{
							{
								Name: "test-gpu-pool",
							},
						},
					},
					Status: tfv1.GPUNodeStatus{
						Phase:       tfv1.TensorFusionGPUNodePhaseRunning,
						TotalTFlops: resource.MustParse("200"),
						TotalVRAM:   resource.MustParse("32Gi"),
						TotalGPUs:   8,
					},
				}
			},
			totalTFlops:  resource.MustParse("150"),
			totalVRAM:    resource.MustParse("24Gi"),
			count:        6,
			allDeviceIDs: []string{"gpu-0", "gpu-1", "gpu-2", "gpu-3", "gpu-4", "gpu-5"},
			expectError:  false,
			validateResult: func(t *testing.T, originalNode, patchedNode *tfv1.GPUNode) {
				// Verify status fields were updated
				assert.Equal(t, resource.MustParse("150"), patchedNode.Status.TotalTFlops)
				assert.Equal(t, resource.MustParse("24Gi"), patchedNode.Status.TotalVRAM)
				assert.Equal(t, int32(6), patchedNode.Status.TotalGPUs)
				assert.Equal(t, int32(6), patchedNode.Status.ManagedGPUs)
				assert.Equal(t, []string{"gpu-0", "gpu-1", "gpu-2", "gpu-3", "gpu-4", "gpu-5"},
					patchedNode.Status.ManagedGPUDeviceIDs)
				// Verify existing phase was preserved
				assert.Equal(t, tfv1.TensorFusionGPUNodePhaseRunning, patchedNode.Status.Phase)
			},
		},
		{
			name: "zero resources handled correctly",
			setupGPUNode: func() *tfv1.GPUNode {
				return &tfv1.GPUNode{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-gpu-node-zero",
						Namespace: "default",
						OwnerReferences: []metav1.OwnerReference{
							{
								Name: "test-gpu-pool",
							},
						},
					},
					Status: tfv1.GPUNodeStatus{
						Phase: "",
					},
				}
			},
			totalTFlops:  resource.MustParse("0"),
			totalVRAM:    resource.MustParse("0"),
			count:        0,
			allDeviceIDs: []string{},
			expectError:  false,
			validateResult: func(t *testing.T, originalNode, patchedNode *tfv1.GPUNode) {
				assert.Equal(t, resource.MustParse("0"), patchedNode.Status.TotalTFlops)
				assert.Equal(t, resource.MustParse("0"), patchedNode.Status.TotalVRAM)
				assert.Equal(t, int32(0), patchedNode.Status.TotalGPUs)
				assert.Equal(t, int32(0), patchedNode.Status.ManagedGPUs)
				assert.Empty(t, patchedNode.Status.ManagedGPUDeviceIDs)
				assert.Equal(t, tfv1.TensorFusionGPUNodePhasePending, patchedNode.Status.Phase)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			gpuNode := tt.setupGPUNode()

			// Setup fake client with the GPUNode
			scheme := runtime.NewScheme()
			_ = tfv1.AddToScheme(scheme)
			k8sClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithStatusSubresource(&tfv1.GPUNode{}).
				WithObjects(gpuNode).
				Build()

			// Store original state for comparison
			originalNode := gpuNode.DeepCopy()

			// Call the function under test
			err := patchGPUNodeStatus(k8sClient, ctx, gpuNode, tt.totalTFlops, tt.totalVRAM, tt.count, tt.allDeviceIDs)

			// Verify error expectation
			if tt.expectError {
				assert.Error(t, err, "Expected an error but got none")
				return
			}
			assert.NoError(t, err, "Unexpected error")

			// Get the updated GPUNode from the client to verify the patch was applied
			updatedNode := &tfv1.GPUNode{}
			err = k8sClient.Get(ctx, client.ObjectKeyFromObject(gpuNode), updatedNode)
			assert.NoError(t, err, "Failed to get updated GPUNode")

			// Run custom validation
			if tt.validateResult != nil {
				tt.validateResult(t, originalNode, updatedNode)
			}
		})
	}
}

func TestPatchGPUNodeStatus_ErrorScenarios(t *testing.T) {
	tests := []struct {
		name         string
		setupClient  func() client.Client
		setupGPUNode func() *tfv1.GPUNode
		expectedErr  string
	}{
		{
			name: "GPUNode not found error",
			setupClient: func() client.Client {
				// Create client without the GPUNode object
				scheme := runtime.NewScheme()
				_ = tfv1.AddToScheme(scheme)
				return fake.NewClientBuilder().
					WithScheme(scheme).
					WithStatusSubresource(&tfv1.GPUNode{}).
					Build()
			},
			setupGPUNode: func() *tfv1.GPUNode {
				return &tfv1.GPUNode{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "nonexistent-gpu-node",
						Namespace: "default",
						OwnerReferences: []metav1.OwnerReference{
							{
								Name: "test-gpu-pool",
							},
						},
					},
				}
			},
			expectedErr: "not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			k8sClient := tt.setupClient()
			gpuNode := tt.setupGPUNode()

			// Call the function under test
			err := patchGPUNodeStatus(k8sClient, ctx, gpuNode,
				resource.MustParse("100"),
				resource.MustParse("16Gi"),
				4,
				[]string{"gpu-0", "gpu-1", "gpu-2", "gpu-3"})

			// Verify the expected error occurred
			assert.Error(t, err, "Expected an error but got none")
			assert.Contains(t, err.Error(), tt.expectedErr, "Error message should contain expected text")
		})
	}
}

func newMarkMissingTestGPU(name, owner string, phase tfv1.TensorFusionGPUPhase) *tfv1.GPU {
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

func TestMarkMissingGPUs(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	_ = tfv1.AddToScheme(scheme)

	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: testGPUNodeName,
			OwnerReferences: []metav1.OwnerReference{
				{Name: "test-gpu-pool"},
			},
		},
	}
	present := newMarkMissingTestGPU("gpu-present", testGPUNodeName, tfv1.TensorFusionGPUPhaseRunning)
	missing := newMarkMissingTestGPU("gpu-missing", testGPUNodeName, tfv1.TensorFusionGPUPhaseRunning)
	otherNodeGPU := newMarkMissingTestGPU("gpu-other-node", "other-gpu-node", tfv1.TensorFusionGPUPhaseRunning)

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPU{}).
		WithObjects(present, missing, otherNodeGPU).Build()

	// a drained card (no running apps) that disappeared: nothing to lose, delete directly
	busyMissing := newMarkMissingTestGPU("gpu-missing-busy", testGPUNodeName, tfv1.TensorFusionGPUPhaseRunning)
	busyMissing.Status.RunningApps = []*tfv1.RunningAppDetail{{Name: "app-1", Namespace: "default"}}
	assert.NoError(t, k8sClient.Create(ctx, busyMissing))

	err := markMissingGPUs(k8sClient, ctx, gpuNode, []string{"gpu-present"})
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
	err = markMissingGPUs(k8sClient, ctx, gpuNode, []string{"gpu-present"})
	assert.NoError(t, err)
	assert.NoError(t, k8sClient.Get(ctx, client.ObjectKey{Name: "gpu-missing-busy"}, got))
	assert.Equal(t, missingSince, got.Annotations[constants.GPUMissingSinceAnnotationKey],
		"missing-since must not be refreshed on subsequent runs")
}

func TestCreateOrUpdateTensorFusionGPU_RecoversMissingGPU(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	_ = tfv1.AddToScheme(scheme)

	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: testGPUNodeName,
			OwnerReferences: []metav1.OwnerReference{
				{Name: "test-gpu-pool"},
			},
		},
	}
	uuid := "gpu-recovered"
	missingGPU := newMarkMissingTestGPU(uuid, testGPUNodeName, tfv1.TensorFusionGPUPhaseUnknown)
	missingGPU.Annotations[constants.GPUMissingSinceAnnotationKey] =
		time.Now().Add(-5 * time.Minute).Format(time.RFC3339)
	// simulate in-use accounting that must survive recovery
	missingGPU.Status.Available.Tflops = resource.MustParse("40")

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPU{}).
		WithObjects(missingGPU).Build()

	memInfo := nvml.Memory_v2{Total: 16 * 1024 * 1024 * 1024}
	gpu, err := createOrUpdateTensorFusionGPU(
		k8sClient, ctx, testNodeName, gpuNode, uuid, testDeviceName,
		memInfo, resource.MustParse("100"), 0, -1, nil)
	assert.NoError(t, err)

	assert.NotContains(t, gpu.Annotations, constants.GPUMissingSinceAnnotationKey,
		"missing marker must be cleared when the device reappears")
	assert.Equal(t, tfv1.TensorFusionGPUPhasePending, gpu.Status.Phase,
		"recovered GPU should go back to Pending for the controller to promote")
	assert.Equal(t, resource.MustParse("40"), gpu.Status.Available.Tflops,
		"existing allocation accounting must be preserved on recovery")
}

func TestEstimateNvLinkBandwidthMBps(t *testing.T) {
	assert.Equal(t, int64(50000), estimateNvLinkBandwidthMBps(4))
	assert.Equal(t, int64(25000), estimateNvLinkBandwidthMBps(99))
	assert.Equal(t, int64(0), estimateNvLinkBandwidthMBps(0))
}

func TestPciBusIDToString(t *testing.T) {
	pci := nvml.PciInfo{
		BusId: [32]uint8{'0', '0', '0', '0', ':', '0', '1', ':', '0', '0', '.', '0'},
	}
	assert.Equal(t, "0000:01:00.0", pciBusIDToString(pci))

	legacyOnly := nvml.PciInfo{
		BusIdLegacy: [16]uint8{'0', '0', '0', '0', ':', '0', '2', ':', '0', '0', '.', '0'},
	}
	assert.Equal(t, "0000:02:00.0", pciBusIDToString(legacyOnly))
}

func TestCloneNvLinkStatus(t *testing.T) {
	src := &tfv1.GPUNvLinkStatus{
		PeerCount:          1,
		TotalLinkCount:     2,
		TotalBandwidthMBps: 200000,
		Peers: []tfv1.GPUNvLinkPeer{
			{
				PeerUUID:      "gpu-1",
				LinkCount:     2,
				LinkVersion:   4,
				BandwidthMBps: 200000,
			},
		},
	}
	cloned := cloneNvLinkStatus(src)
	assert.NotNil(t, cloned)
	assert.Equal(t, src, cloned)

	src.Peers[0].PeerUUID = "changed"
	assert.Equal(t, "gpu-1", cloned.Peers[0].PeerUUID, "clone should not share peers slice")
}

func TestCloneNvLinkStatus_NormalizesPeerUUID(t *testing.T) {
	src := &tfv1.GPUNvLinkStatus{
		Peers: []tfv1.GPUNvLinkPeer{
			{
				PeerUUID: "GPU-1 ",
			},
		},
	}
	cloned := cloneNvLinkStatus(src)
	assert.NotNil(t, cloned)
	assert.Len(t, cloned.Peers, 1)
	assert.Equal(t, "gpu-1", cloned.Peers[0].PeerUUID)
}

func TestDiscoverNvLinkStatusDirectPeer(t *testing.T) {
	self := &stubNVMLDevice{}
	self.getNvLinkStateFunc = func(n int) (nvml.EnableState, nvml.Return) {
		if n == 0 {
			return nvml.FEATURE_ENABLED, nvml.SUCCESS
		}
		return nvml.FEATURE_DISABLED, nvml.SUCCESS
	}
	self.getNvLinkVersionFunc = func(n int) (uint32, nvml.Return) {
		return 4, nvml.SUCCESS
	}
	self.getNvLinkRemoteDeviceTypeFunc = func(n int) (nvml.IntNvLinkDeviceType, nvml.Return) {
		return nvml.NVLINK_DEVICE_TYPE_GPU, nvml.SUCCESS
	}
	self.getNvLinkRemotePciInfoFunc = func(n int) (nvml.PciInfo, nvml.Return) {
		return nvml.PciInfo{
			BusId: [32]uint8{'0', '0', '0', '0', ':', '0', '2', ':', '0', '0', '.', '0'},
		}, nvml.SUCCESS
	}

	status := discoverNvLinkStatus(
		self,
		"gpu-0",
		map[string]string{"0000:02:00.0": "gpu-1"},
		[]discoveredGPU{{device: self, uuid: "gpu-0"}},
	)

	assert.NotNil(t, status)
	assert.Equal(t, int32(1), status.PeerCount)
	assert.Equal(t, int32(1), status.TotalLinkCount)
	assert.Equal(t, int64(50000), status.TotalBandwidthMBps)
	assert.Len(t, status.Peers, 1)
	assert.Equal(t, "gpu-1", status.Peers[0].PeerUUID)
	assert.Equal(t, int32(1), status.Peers[0].LinkCount)
	assert.Equal(t, int32(4), status.Peers[0].LinkVersion)
}

func TestDiscoverNvLinkStatus_NormalizesResolvedPeerUUID(t *testing.T) {
	self := &stubNVMLDevice{}
	self.getNvLinkStateFunc = func(n int) (nvml.EnableState, nvml.Return) {
		if n == 0 {
			return nvml.FEATURE_ENABLED, nvml.SUCCESS
		}
		return nvml.FEATURE_DISABLED, nvml.SUCCESS
	}
	self.getNvLinkVersionFunc = func(n int) (uint32, nvml.Return) {
		return 4, nvml.SUCCESS
	}
	self.getNvLinkRemoteDeviceTypeFunc = func(n int) (nvml.IntNvLinkDeviceType, nvml.Return) {
		return nvml.NVLINK_DEVICE_TYPE_GPU, nvml.SUCCESS
	}
	self.getNvLinkRemotePciInfoFunc = func(n int) (nvml.PciInfo, nvml.Return) {
		return nvml.PciInfo{
			BusId: [32]uint8{'0', '0', '0', '0', ':', '0', '2', ':', '0', '0', '.', '0'},
		}, nvml.SUCCESS
	}

	status := discoverNvLinkStatus(
		self,
		"GPU-0 ",
		map[string]string{"0000:02:00.0": "GPU-1 "},
		[]discoveredGPU{{device: self, uuid: "GPU-0 "}},
	)

	assert.NotNil(t, status)
	assert.Len(t, status.Peers, 1)
	assert.Equal(t, "gpu-1", status.Peers[0].PeerUUID)
}

func TestDiscoverNvLinkStatusSwitchFallback(t *testing.T) {
	self := &stubNVMLDevice{}
	peer := &stubNVMLDevice{}

	self.getNvLinkStateFunc = func(n int) (nvml.EnableState, nvml.Return) {
		if n == 0 || n == 1 {
			return nvml.FEATURE_ENABLED, nvml.SUCCESS
		}
		return nvml.FEATURE_DISABLED, nvml.SUCCESS
	}
	self.getNvLinkVersionFunc = func(n int) (uint32, nvml.Return) {
		return 4, nvml.SUCCESS
	}
	self.getNvLinkRemoteDeviceTypeFunc = func(n int) (nvml.IntNvLinkDeviceType, nvml.Return) {
		return nvml.NVLINK_DEVICE_TYPE_SWITCH, nvml.SUCCESS
	}
	self.getP2PStatusFunc = func(device nvml.Device, idx nvml.GpuP2PCapsIndex) (nvml.GpuP2PStatus, nvml.Return) {
		assert.Equal(t, nvml.P2P_CAPS_INDEX_NVLINK, idx)
		assert.Equal(t, peer, device)
		return nvml.P2P_STATUS_OK, nvml.SUCCESS
	}

	status := discoverNvLinkStatus(
		self,
		"gpu-0",
		nil,
		[]discoveredGPU{
			{device: self, uuid: "gpu-0"},
			{device: peer, uuid: "gpu-1"},
		},
	)

	assert.NotNil(t, status)
	assert.Equal(t, int32(1), status.PeerCount)
	assert.Equal(t, int32(2), status.TotalLinkCount)
	assert.Equal(t, int64(100000), status.TotalBandwidthMBps)
	assert.Len(t, status.Peers, 1)
	assert.Equal(t, "gpu-1", status.Peers[0].PeerUUID)
	assert.Equal(t, int32(2), status.Peers[0].LinkCount)
	assert.Equal(t, int32(4), status.Peers[0].LinkVersion)
	assert.Equal(t, int64(100000), status.Peers[0].BandwidthMBps)
}

func TestPatchGPUNodeStatus_Integration(t *testing.T) {
	// Integration test that verifies the complete flow
	ctx := context.Background()

	// Setup initial GPUNode
	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "integration-test-node",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{
				{
					Name: "test-gpu-pool",
				},
			},
		},
		Status: tfv1.GPUNodeStatus{
			Phase:               "",
			TotalTFlops:         resource.MustParse("10"),
			TotalVRAM:           resource.MustParse("2Gi"),
			TotalGPUs:           1,
			ManagedGPUs:         0, // Different from TotalGPUs to test sync
			ManagedGPUDeviceIDs: []string{"old-device"},
			NodeInfo: tfv1.GPUNodeInfo{
				RAMSize:      resource.MustParse("1Gi"),
				DataDiskSize: resource.MustParse("1Gi"),
			},
		},
	}

	// Setup fake client
	scheme := runtime.NewScheme()
	_ = tfv1.AddToScheme(scheme)
	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPUNode{}).
		WithObjects(gpuNode).
		Build()

	// Test multiple sequential patches to verify state consistency
	updates := []struct {
		totalTFlops  resource.Quantity
		totalVRAM    resource.Quantity
		count        int32
		allDeviceIDs []string
	}{
		{
			totalTFlops:  resource.MustParse("100"),
			totalVRAM:    resource.MustParse("16Gi"),
			count:        4,
			allDeviceIDs: []string{"gpu-0", "gpu-1", "gpu-2", "gpu-3"},
		},
		{
			totalTFlops:  resource.MustParse("200"),
			totalVRAM:    resource.MustParse("32Gi"),
			count:        8,
			allDeviceIDs: []string{"gpu-0", "gpu-1", "gpu-2", "gpu-3", "gpu-4", "gpu-5", "gpu-6", "gpu-7"},
		},
		{
			totalTFlops:  resource.MustParse("50"),
			totalVRAM:    resource.MustParse("8Gi"),
			count:        2,
			allDeviceIDs: []string{"gpu-0", "gpu-1"},
		},
	}

	for i, update := range updates {
		t.Run(fmt.Sprintf("update_%d", i+1), func(t *testing.T) {
			// Apply the patch
			err := patchGPUNodeStatus(k8sClient, ctx, gpuNode, update.totalTFlops,
				update.totalVRAM, update.count, update.allDeviceIDs)
			assert.NoError(t, err, "Patch should succeed")

			// Verify the update was applied
			updatedNode := &tfv1.GPUNode{}
			err = k8sClient.Get(ctx, client.ObjectKeyFromObject(gpuNode), updatedNode)
			assert.NoError(t, err, "Should be able to get updated node")

			// Verify all fields were updated correctly
			assert.Equal(t, update.totalTFlops, updatedNode.Status.TotalTFlops)
			assert.Equal(t, update.totalVRAM, updatedNode.Status.TotalVRAM)
			assert.Equal(t, update.count, updatedNode.Status.TotalGPUs)
			assert.Equal(t, update.count, updatedNode.Status.ManagedGPUs)
			assert.Equal(t, update.allDeviceIDs, updatedNode.Status.ManagedGPUDeviceIDs)

			// Phase should be set to pending only on first update
			if i == 0 {
				assert.Equal(t, tfv1.TensorFusionGPUNodePhasePending, updatedNode.Status.Phase)
			} else {
				// Should remain pending on subsequent updates
				assert.Equal(t, tfv1.TensorFusionGPUNodePhasePending, updatedNode.Status.Phase)
			}

			// NodeInfo should be updated with system values
			assert.True(t, updatedNode.Status.NodeInfo.RAMSize.Value() > 0)
			assert.True(t, updatedNode.Status.NodeInfo.DataDiskSize.Value() > 0)
		})
	}
}
