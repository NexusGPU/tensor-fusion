package main

import (
	"context"
	"testing"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
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
	assert.Equal(t, tfv1.TensorFusionGPUPhaseRunning, gpu.Status.Phase, "GPU phase should be running")

	// Verify labels and annotations
	assert.Equal(t, map[string]string{constants.LabelKeyOwner: gpuNodeName}, gpu.Labels, "GPU labels should match")
	assert.Contains(t, gpu.Annotations, constants.GPULastReportTimeAnnotationKey,
		"GPU annotations should contain last report time")
	_, err := time.Parse(time.RFC3339, gpu.Annotations[constants.GPULastReportTimeAnnotationKey])
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
