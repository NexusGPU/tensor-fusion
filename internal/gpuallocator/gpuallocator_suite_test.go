/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package gpuallocator

import (
	"context"
	"fmt"
	"path/filepath"
	"runtime"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

// These tests use Ginkgo (BDD-style Go testing framework). Refer to
// http://onsi.github.io/ginkgo/ to learn more about Ginkgo.

var (
	cancel    context.CancelFunc
	cfg       *rest.Config
	ctx       context.Context
	k8sClient client.Client
	testEnv   *envtest.Environment
	mgr       ctrl.Manager
)

func TestGPUAllocator(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "GPU Allocator Suite")
}

var _ = BeforeSuite(func() {
	logf.SetLogger(zap.New(zap.WriteTo(GinkgoWriter), zap.UseDevMode(true)))

	ctx, cancel = context.WithCancel(context.TODO())

	By("bootstrapping test environment")
	testEnv = &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing: false,
		BinaryAssetsDirectory: filepath.Join("..", "..", "bin", "k8s",
			fmt.Sprintf("1.31.0-%s-%s", runtime.GOOS, runtime.GOARCH)),
	}

	var err error
	// cfg is defined in this file globally.
	cfg, err = testEnv.Start()
	Expect(err).NotTo(HaveOccurred())
	Expect(cfg).NotTo(BeNil())

	err = tfv1.AddToScheme(scheme.Scheme)
	Expect(err).NotTo(HaveOccurred())

	err = corev1.AddToScheme(scheme.Scheme)
	Expect(err).NotTo(HaveOccurred())

	// Create a Kubernetes client
	k8sClient, err = client.New(cfg, client.Options{Scheme: scheme.Scheme})
	Expect(err).NotTo(HaveOccurred())
	Expect(k8sClient).NotTo(BeNil())

	mgr, err = ctrl.NewManager(cfg, ctrl.Options{
		Scheme: scheme.Scheme,
		Metrics: metricsserver.Options{
			// disable metricsserver
			BindAddress: "0",
		},
	})
	Expect(err).NotTo(HaveOccurred())

	// Create test scheduling config template
	schedulingConfig := &tfv1.SchedulingConfigTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name: "default-config",
		},
		Spec: tfv1.SchedulingConfigTemplateSpec{
			Placement: tfv1.PlacementConfig{
				Mode: tfv1.PlacementModeCompactFirst,
			},
		},
	}
	err = k8sClient.Create(ctx, schedulingConfig)
	Expect(err).NotTo(HaveOccurred())

	// Create test GPU pool
	schedulingConfigTemplate := "default-config"
	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-pool",
		},
		Spec: tfv1.GPUPoolSpec{
			SchedulingConfigTemplate: &schedulingConfigTemplate,
		},
	}
	err = k8sClient.Create(ctx, pool)
	Expect(err).NotTo(HaveOccurred())

	// Create other pool for testing
	otherPool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{
			Name: "other-pool",
		},
		Spec: tfv1.GPUPoolSpec{},
	}
	err = k8sClient.Create(ctx, otherPool)
	Expect(err).NotTo(HaveOccurred())

	// Create test GPUs with metadata only first
	gpus := []tfv1.GPU{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "gpu-1",
				Namespace: "default",
				Labels: map[string]string{
					constants.GpuPoolKey:    "test-pool",
					constants.LabelKeyOwner: "node-1",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "gpu-2",
				Namespace: "default",
				Labels: map[string]string{
					constants.GpuPoolKey:    "test-pool",
					constants.LabelKeyOwner: "node-1",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "gpu-3",
				Namespace: "default",
				Labels: map[string]string{
					constants.GpuPoolKey:    "other-pool",
					constants.LabelKeyOwner: "node-2",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "gpu-4",
				Namespace: "default",
				Labels: map[string]string{
					constants.GpuPoolKey:    "test-pool",
					constants.LabelKeyOwner: "node-3",
				},
			},
		},
	}

	// First create the GPUs without status
	for i := range gpus {
		err = k8sClient.Create(ctx, &gpus[i])
		Expect(err).NotTo(HaveOccurred())
	}

	// Then update the status for each GPU
	gpuStatuses := []struct {
		name   string
		status tfv1.GPUStatus
	}{
		{
			name: "gpu-1",
			status: tfv1.GPUStatus{
				Phase: tfv1.TensorFusionGPUPhaseRunning,
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("100"),
					Vram:   resource.MustParse("16Gi"),
				},
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("100"),
					Vram:   resource.MustParse("16Gi"),
				},
				GPUModel:     "NVIDIA A100",
				NodeSelector: map[string]string{constants.KubernetesHostNameLabel: "node-1"},
			},
		},
		{
			name: "gpu-2",
			status: tfv1.GPUStatus{
				Phase: tfv1.TensorFusionGPUPhaseRunning,
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("80"),
					Vram:   resource.MustParse("32Gi"),
				},
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("100"),
					Vram:   resource.MustParse("32Gi"),
				},
				GPUModel:     "NVIDIA A100",
				NodeSelector: map[string]string{constants.KubernetesHostNameLabel: "node-1"},
			},
		},
		{
			name: "gpu-3",
			status: tfv1.GPUStatus{
				Phase: tfv1.TensorFusionGPUPhaseRunning,
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("120"),
					Vram:   resource.MustParse("24Gi"),
				},
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("120"),
					Vram:   resource.MustParse("24Gi"),
				},
				GPUModel:     "NVIDIA A100",
				NodeSelector: map[string]string{constants.KubernetesHostNameLabel: "node-2"},
			},
		},
		{
			name: "gpu-4",
			status: tfv1.GPUStatus{
				Phase: tfv1.TensorFusionGPUPhasePending,
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("150"),
					Vram:   resource.MustParse("48Gi"),
				},
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("150"),
					Vram:   resource.MustParse("48Gi"),
				},
				GPUModel:     "NVIDIA H100",
				NodeSelector: map[string]string{constants.KubernetesHostNameLabel: "node-3"},
			},
		},
	}

	for _, gs := range gpuStatuses {
		gpu := &tfv1.GPU{}
		err = k8sClient.Get(ctx, types.NamespacedName{Name: gs.name, Namespace: "default"}, gpu)
		Expect(err).NotTo(HaveOccurred())

		gpu.Status = gs.status
		err = k8sClient.Status().Update(ctx, gpu)
		Expect(err).NotTo(HaveOccurred())
	}

	nodes := []tfv1.GPUNode{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "node-1",
				Labels: map[string]string{
					constants.LabelKeyOwner: "test-pool",
				},
			},
			Spec: tfv1.GPUNodeSpec{
				ManageMode: tfv1.GPUNodeManageModeAutoSelect,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "node-2",
				Labels: map[string]string{
					constants.LabelKeyOwner: "test-pool",
				},
			},
			Spec: tfv1.GPUNodeSpec{
				ManageMode: tfv1.GPUNodeManageModeAutoSelect,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "node-3",
				Labels: map[string]string{
					constants.LabelKeyOwner: "test-pool",
				},
			},
			Spec: tfv1.GPUNodeSpec{
				ManageMode: tfv1.GPUNodeManageModeAutoSelect,
			},
		},
	}
	for _, node := range nodes {
		err = k8sClient.Create(ctx, &node)
		Expect(err).NotTo(HaveOccurred())
	}

	gpuNodeStatuses := []struct {
		name   string
		status tfv1.GPUNodeStatus
	}{
		{
			name: "node-1",
			status: tfv1.GPUNodeStatus{
				Phase:           tfv1.TensorFusionGPUNodePhaseRunning,
				TotalTFlops:     resource.MustParse("200"),
				TotalVRAM:       resource.MustParse("48Gi"),
				AvailableTFlops: resource.MustParse("180"),
				AvailableVRAM:   resource.MustParse("48Gi"),
			},
		},
		{
			name: "node-2",
			status: tfv1.GPUNodeStatus{
				Phase:           tfv1.TensorFusionGPUNodePhaseRunning,
				TotalTFlops:     resource.MustParse("120"),
				TotalVRAM:       resource.MustParse("24Gi"),
				AvailableTFlops: resource.MustParse("120"),
				AvailableVRAM:   resource.MustParse("24Gi"),
			},
		},
		{
			name: "node-3",
			status: tfv1.GPUNodeStatus{
				Phase:           tfv1.TensorFusionGPUNodePhaseRunning,
				TotalTFlops:     resource.MustParse("150"),
				TotalVRAM:       resource.MustParse("48Gi"),
				AvailableTFlops: resource.MustParse("150"),
				AvailableVRAM:   resource.MustParse("48Gi"),
			},
		},
	}

	for _, gpuNodeStatus := range gpuNodeStatuses {
		gpuNode := &tfv1.GPUNode{}
		err = k8sClient.Get(ctx, types.NamespacedName{Name: gpuNodeStatus.name, Namespace: "default"}, gpuNode)
		Expect(err).NotTo(HaveOccurred())
		gpuNode.Status = gpuNodeStatus.status
		err = k8sClient.Status().Update(ctx, gpuNode)
		Expect(err).NotTo(HaveOccurred())
	}

	go func() {
		defer GinkgoRecover()
		err = mgr.Start(ctx)
		Expect(err).ToNot(HaveOccurred(), "failed to run manager")
	}()
})

var _ = AfterSuite(func() {
	By("tearing down the test environment")
	cancel()
	err := testEnv.Stop()
	Expect(err).NotTo(HaveOccurred())
})

// Helper function to get a GPU from the API server
func getGPU(name string) *tfv1.GPU {
	gpu := &tfv1.GPU{}
	key := types.NamespacedName{Name: name}
	err := k8sClient.Get(ctx, key, gpu)
	ExpectWithOffset(1, err).NotTo(HaveOccurred())
	return gpu
}

func getGPUNode(gpu *tfv1.GPU) *tfv1.GPUNode {
	gpuNode := &tfv1.GPUNode{}
	key := types.NamespacedName{Name: gpu.Labels[constants.LabelKeyOwner]}
	err := k8sClient.Get(ctx, key, gpuNode)
	ExpectWithOffset(1, err).NotTo(HaveOccurred())
	return gpuNode
}
