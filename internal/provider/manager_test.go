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

package provider

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestNewManager(t *testing.T) {
	mgr := NewManager(nil)
	assert.NotNil(t, mgr)
	assert.NotNil(t, mgr.providers)
	assert.NotNil(t, mgr.gpuInfoCache)
	assert.NotNil(t, mgr.inUseResources)
	assert.Equal(t, 0, mgr.ProviderCount())
}

func TestUpdateAndGetProvider(t *testing.T) {
	mgr := NewManager(nil)

	provider := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "nvidia-provider",
		},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "NVIDIA",
			Images: tfv1.ProviderImages{
				Middleware:   "nvidia-hypervisor:latest",
				RemoteClient: "nvidia-client:latest",
				RemoteWorker: "nvidia-worker:latest",
			},
			InUseResourceNames: []string{
				"nvidia.com/gpu",
				"nvidia.com/mig-1g.5gb",
			},
			HardwareMetadata: []tfv1.HardwareModelInfo{
				{
					Model:         "A100_SXM_80G",
					FullModelName: "NVIDIA A100-SXM4-80GB",
					CostPerHour:   "1.89",
					Fp16TFlops:    resource.MustParse("312"),
				},
			},
		},
	}

	mgr.UpdateProvider(provider)

	// Test GetProvider
	got, ok := mgr.GetProvider("NVIDIA")
	assert.True(t, ok)
	assert.Equal(t, "NVIDIA", got.Spec.Vendor)
	assert.Equal(t, "nvidia-hypervisor:latest", got.Spec.Images.Middleware)

	// Test GetMiddlewareImage
	image := mgr.GetMiddlewareImage("NVIDIA", "default-image")
	assert.Equal(t, "nvidia-hypervisor:latest", image)

	// Test GetRemoteClientImage
	clientImage := mgr.GetRemoteClientImage("NVIDIA", "default-image")
	assert.Equal(t, "nvidia-client:latest", clientImage)

	// Test GetRemoteWorkerImage
	workerImage := mgr.GetRemoteWorkerImage("NVIDIA", "default-image")
	assert.Equal(t, "nvidia-worker:latest", workerImage)

	// Test GetInUseResourceNames
	resourceNames := mgr.GetInUseResourceNames("NVIDIA")
	assert.Len(t, resourceNames, 2)
	assert.Contains(t, resourceNames, corev1.ResourceName("nvidia.com/gpu"))

	// Test GetGpuInfos
	gpuInfos := mgr.GetGpuInfos("NVIDIA")
	assert.Len(t, gpuInfos, 1)
	assert.Equal(t, "A100_SXM_80G", gpuInfos[0].Model)
	assert.Equal(t, 1.89, gpuInfos[0].CostPerHour)
}

func TestDeleteProvider(t *testing.T) {
	mgr := NewManager(nil)

	provider := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "nvidia-provider",
		},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "NVIDIA",
			Images: tfv1.ProviderImages{
				Middleware:   "nvidia-hypervisor:latest",
				RemoteClient: "nvidia-client:latest",
				RemoteWorker: "nvidia-worker:latest",
			},
		},
	}

	mgr.UpdateProvider(provider)
	assert.Equal(t, 1, mgr.ProviderCount())

	mgr.DeleteProvider("NVIDIA")
	assert.Equal(t, 0, mgr.ProviderCount())

	_, ok := mgr.GetProvider("NVIDIA")
	assert.False(t, ok)
}

func TestGetProviderOrDefault(t *testing.T) {
	mgr := NewManager(nil)

	nvidiaProvider := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "nvidia-provider",
		},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "NVIDIA",
			Images: tfv1.ProviderImages{
				Middleware:   "nvidia-hypervisor:latest",
				RemoteClient: "nvidia-client:latest",
				RemoteWorker: "nvidia-worker:latest",
			},
		},
	}

	mgr.UpdateProvider(nvidiaProvider)

	// Test getting existing provider
	got := mgr.GetProviderOrDefault("NVIDIA")
	assert.NotNil(t, got)
	assert.Equal(t, "NVIDIA", got.Spec.Vendor)

	// Test fallback to NVIDIA when AMD doesn't exist
	got = mgr.GetProviderOrDefault("AMD")
	assert.NotNil(t, got)
	assert.Equal(t, "NVIDIA", got.Spec.Vendor)
}

func TestImageFallback(t *testing.T) {
	mgr := NewManager(nil)

	// Test fallback when provider doesn't exist
	image := mgr.GetMiddlewareImage("NonExistent", "default-image")
	assert.Equal(t, "default-image", image)

	// Test fallback when image is empty
	provider := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "empty-provider",
		},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "Empty",
			Images: tfv1.ProviderImages{
				Middleware:   "",
				RemoteClient: "",
				RemoteWorker: "",
			},
		},
	}
	mgr.UpdateProvider(provider)

	image = mgr.GetMiddlewareImage("Empty", "default-image")
	assert.Equal(t, "default-image", image)
}

func TestGetAllProviders(t *testing.T) {
	mgr := NewManager(nil)

	nvidiaProvider := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "nvidia-provider",
		},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "NVIDIA",
			Images: tfv1.ProviderImages{
				Middleware:   "nvidia-hypervisor:latest",
				RemoteClient: "nvidia-client:latest",
				RemoteWorker: "nvidia-worker:latest",
			},
		},
	}

	amdProvider := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "amd-provider",
		},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "AMD",
			Images: tfv1.ProviderImages{
				Middleware:   "amd-hypervisor:latest",
				RemoteClient: "amd-client:latest",
				RemoteWorker: "amd-worker:latest",
			},
		},
	}

	mgr.UpdateProvider(nvidiaProvider)
	mgr.UpdateProvider(amdProvider)

	all := mgr.GetAllProviders()
	assert.Len(t, all, 2)
	assert.Contains(t, all, "NVIDIA")
	assert.Contains(t, all, "AMD")
}

func TestGetGPUPricingMap(t *testing.T) {
	mgr := NewManager(nil)

	provider := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "nvidia-provider",
		},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "NVIDIA",
			Images: tfv1.ProviderImages{
				Middleware:   "nvidia-hypervisor:latest",
				RemoteClient: "nvidia-client:latest",
				RemoteWorker: "nvidia-worker:latest",
			},
			HardwareMetadata: []tfv1.HardwareModelInfo{
				{
					Model:         "A100",
					FullModelName: "NVIDIA A100",
					CostPerHour:   "1.89",
					Fp16TFlops:    resource.MustParse("312"),
				},
				{
					Model:         "H100",
					FullModelName: "NVIDIA H100",
					CostPerHour:   "2.99",
					Fp16TFlops:    resource.MustParse("989"),
				},
			},
		},
	}

	mgr.UpdateProvider(provider)

	pricingMap := mgr.GetGPUPricingMap()
	assert.Len(t, pricingMap, 2)
	assert.Equal(t, 1.89, pricingMap["NVIDIA A100"])
	assert.Equal(t, 2.99, pricingMap["NVIDIA H100"])
}

func TestGetAllInUseResourceNames(t *testing.T) {
	mgr := NewManager(nil)

	nvidiaProvider := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "nvidia-provider",
		},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "NVIDIA",
			Images: tfv1.ProviderImages{
				Middleware:   "nvidia-hypervisor:latest",
				RemoteClient: "nvidia-client:latest",
				RemoteWorker: "nvidia-worker:latest",
			},
			InUseResourceNames: []string{
				"nvidia.com/gpu",
			},
		},
	}

	amdProvider := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "amd-provider",
		},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "AMD",
			Images: tfv1.ProviderImages{
				Middleware:   "amd-hypervisor:latest",
				RemoteClient: "amd-client:latest",
				RemoteWorker: "amd-worker:latest",
			},
			InUseResourceNames: []string{
				"amd.com/gpu",
			},
		},
	}

	mgr.UpdateProvider(nvidiaProvider)
	mgr.UpdateProvider(amdProvider)

	allResources := mgr.GetAllInUseResourceNames()
	assert.Len(t, allResources, 2)
	assert.Contains(t, allResources, corev1.ResourceName("nvidia.com/gpu"))
	assert.Contains(t, allResources, corev1.ResourceName("amd.com/gpu"))
}

func TestGetGpuInfoByModel(t *testing.T) {
	mgr := NewManager(nil)

	provider := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "nvidia-provider",
		},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "NVIDIA",
			Images: tfv1.ProviderImages{
				Middleware:   "nvidia-hypervisor:latest",
				RemoteClient: "nvidia-client:latest",
				RemoteWorker: "nvidia-worker:latest",
			},
			HardwareMetadata: []tfv1.HardwareModelInfo{
				{
					Model:         "A100_SXM_80G",
					FullModelName: "NVIDIA A100-SXM4-80GB",
					CostPerHour:   "1.89",
					Fp16TFlops:    resource.MustParse("312"),
				},
			},
		},
	}

	mgr.UpdateProvider(provider)

	// Test by model name
	info := mgr.GetGpuInfoByModel("A100_SXM_80G")
	assert.NotNil(t, info)
	assert.Equal(t, "A100_SXM_80G", info.Model)

	// Test by full model name
	info = mgr.GetGpuInfoByModel("NVIDIA A100-SXM4-80GB")
	assert.NotNil(t, info)
	assert.Equal(t, "NVIDIA A100-SXM4-80GB", info.FullModelName)

	// Test non-existent
	info = mgr.GetGpuInfoByModel("NonExistent")
	assert.Nil(t, info)
}

func TestVirtualizationTemplates(t *testing.T) {
	mgr := NewManager(nil)

	provider := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "nvidia-provider",
		},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "NVIDIA",
			Images: tfv1.ProviderImages{
				Middleware:   "nvidia-hypervisor:latest",
				RemoteClient: "nvidia-client:latest",
				RemoteWorker: "nvidia-worker:latest",
			},
			VirtualizationTemplates: []tfv1.VirtualizationTemplate{
				{
					ID:              "mig-1g-10gb",
					Name:            "1g.10gb",
					MemoryGigabytes: 10,
					ComputePercent:  "14.2857",
					MaxInstances:    7,
				},
				{
					ID:              "mig-2g-20gb",
					Name:            "2g.20gb",
					MemoryGigabytes: 20,
					ComputePercent:  "28.5714",
					MaxInstances:    3,
				},
			},
			HardwareMetadata: []tfv1.HardwareModelInfo{
				{
					Model:                 "A100_SXM_80G",
					FullModelName:         "NVIDIA A100-SXM4-80GB",
					CostPerHour:           "1.89",
					Fp16TFlops:            resource.MustParse("312"),
					MaxPartitions:         7,
					MaxPlacementSlots:     8,
					PartitionTemplateRefs: []string{"mig-1g-10gb", "mig-2g-20gb"},
				},
			},
		},
	}

	mgr.UpdateProvider(provider)

	gpuInfos := mgr.GetGpuInfos("NVIDIA")
	assert.Len(t, gpuInfos, 1)
	assert.Len(t, gpuInfos[0].PartitionTemplates, 2)
	assert.Equal(t, "1g.10gb", gpuInfos[0].PartitionTemplates[0].Name)
	assert.InDelta(t, 14.2857, gpuInfos[0].PartitionTemplates[0].ComputePercent, 0.0001)
}

func TestHasProvider(t *testing.T) {
	mgr := NewManager(nil)

	assert.False(t, mgr.HasProvider("NVIDIA"))

	provider := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "nvidia-provider",
		},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "NVIDIA",
			Images: tfv1.ProviderImages{
				Middleware:   "nvidia-hypervisor:latest",
				RemoteClient: "nvidia-client:latest",
				RemoteWorker: "nvidia-worker:latest",
			},
		},
	}
	mgr.UpdateProvider(provider)

	assert.True(t, mgr.HasProvider("NVIDIA"))
	assert.False(t, mgr.HasProvider("AMD"))
}
