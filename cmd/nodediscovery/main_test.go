package main

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestCreateOrUpdateTensorFusionGPU(t *testing.T) {
	// Setup test data
	ctx := context.Background()
	uuid := "test-uuid"
	memInfo := nvml.Memory_v2{Total: 16 * 1024 * 1024 * 1024} // 16 GiB
	tflops := resource.MustParse("100")
	deviceName := "NVIDIA-Test-GPU"
	k8sNodeName := "test-node"
	gpuNodeName := "test-gpu-node"

	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: gpuNodeName,
		},
	}

	scheme := runtime.NewScheme()
	_ = tfv1.AddToScheme(scheme)

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&tfv1.GPU{}).Build()

	gpu := createOrUpdateTensorFusionGPU(k8sClient, ctx, k8sNodeName, gpuNode, uuid, deviceName, memInfo, tflops)

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
	assert.Equal(t, map[string]string{constants.LabelKeyOwner: gpuNodeName}, gpu.Labels, "GPU labels should match")
	assert.Contains(t, gpu.Annotations, constants.LastSyncTimeAnnotationKey,
		"GPU annotations should contain last report time")
	_, err := time.Parse(time.RFC3339, gpu.Annotations[constants.LastSyncTimeAnnotationKey])
	assert.NoError(t, err, "Last report time annotation should be a valid RFC3339 timestamp")

	// Verify the Available field does not change after the update
	gpu.Status.Available.Tflops.Sub(resource.MustParse("1000"))
	gpu.Status.Available.Vram.Sub(resource.MustParse("2000Mi"))
	err = k8sClient.Status().Update(ctx, gpu)
	assert.NoError(t, err)

	tflops.Add(resource.MustParse("100"))
	updatedGpu := createOrUpdateTensorFusionGPU(k8sClient, ctx, k8sNodeName, gpuNode, uuid, deviceName, memInfo, tflops)
	assert.NotEqual(t, updatedGpu.Status.Capacity, gpu.Status.Capacity, "GPU capacity should not match")
	assert.Equal(t, updatedGpu.Status.Available.Tflops, gpu.Status.Available.Tflops, "GPU TFlops should match")
	assert.Equal(t, updatedGpu.Status.Available.Vram, gpu.Status.Available.Vram, "GPU VRAM should match")
}

func TestGPUControllerReference(t *testing.T) {
	// Setup test data
	ctx := context.Background()
	uuid := "test-uuid"
	memInfo := nvml.Memory_v2{Total: 16 * 1024 * 1024 * 1024} // 16 GiB
	tflops := resource.MustParse("100")
	deviceName := "NVIDIA-Test-GPU"
	k8sNodeName := "test-node"
	gpuNodeName := "test-gpu-node"

	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: gpuNodeName,
			UID:  "mock-uid",
		},
	}

	scheme := runtime.NewScheme()
	_ = tfv1.AddToScheme(scheme)

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&tfv1.GPU{}).Build()

	gpu := createOrUpdateTensorFusionGPU(k8sClient, ctx, k8sNodeName, gpuNode, uuid, deviceName, memInfo, tflops)
	assert.True(t, metav1.IsControlledBy(gpu, gpuNode))

	newGpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "new-test-gpu-node",
			UID:  "new-mock-uid",
		},
	}

	gpu = createOrUpdateTensorFusionGPU(k8sClient, ctx, k8sNodeName, newGpuNode, uuid, deviceName, memInfo, tflops)
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

func TestPatchGPUNodeStatus_Integration(t *testing.T) {
	// Integration test that verifies the complete flow
	ctx := context.Background()

	// Setup initial GPUNode
	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "integration-test-node",
			Namespace: "default",
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
