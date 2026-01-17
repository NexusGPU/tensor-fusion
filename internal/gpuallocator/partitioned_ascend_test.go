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
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const ascendGPUModel = "Ascend_310P3"

func ptrUint32(v uint32) *uint32 { return &v }

func setupAscendTestConfig() {
	mu.Lock()
	defer mu.Unlock()
	GPUCapacityMap[ascendGPUModel] = tfv1.Resource{Tflops: resource.MustParse("22"), Vram: resource.MustParse("21Gi")}
	MaxPartitionsMap[ascendGPUModel] = 7
	MaxIsolationGroupsMap[ascendGPUModel] = 4
	TotalExtendedResourcesMap[ascendGPUModel] = map[string]uint32{"AICORE": 8, "AICPU": 7}
	PartitionTemplateMap[ascendGPUModel] = map[string]config.PartitionTemplateInfo{
		"vir01":    {TemplateID: "vir01", Name: "vir01", MemoryGigabytes: 3, ComputePercent: 12.5, MaxPartition: 8, IsolationGroupSharing: config.IsolationGroupSharingShared, MaxPartitionsPerIsolationGroup: 2, ExtendedResources: map[string]uint32{"AICORE": 1, "AICPU": 1}},
		"vir02":    {TemplateID: "vir02", Name: "vir02", MemoryGigabytes: 6, ComputePercent: 25.0, MaxPartition: 4, IsolationGroupSharing: config.IsolationGroupSharingExclusive, ExtendedResources: map[string]uint32{"AICORE": 2, "AICPU": 2}},
		"vir04":    {TemplateID: "vir04", Name: "vir04", MemoryGigabytes: 12, ComputePercent: 50.0, MaxPartition: 2, IsolationGroupSharing: config.IsolationGroupSharingExclusive, ExtendedResources: map[string]uint32{"AICORE": 4, "AICPU": 4}},
		"vir04_3c": {TemplateID: "vir04_3c", Name: "vir04_3c", MemoryGigabytes: 12, ComputePercent: 50.0, MaxPartition: 2, IsolationGroupSharing: config.IsolationGroupSharingExclusive, ExtendedResources: map[string]uint32{"AICORE": 4, "AICPU": 3}},
	}
}

func createAscendGPU(partitions map[string]tfv1.AllocatedPartition) *tfv1.GPU {
	return &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{Name: "npu-0"},
		Status: tfv1.GPUStatus{
			GPUModel:            ascendGPUModel,
			Vendor:              "Ascend",
			Capacity:            &tfv1.Resource{Tflops: resource.MustParse("22"), Vram: resource.MustParse("21Gi")},
			Available:           &tfv1.Resource{Tflops: resource.MustParse("22"), Vram: resource.MustParse("21Gi")},
			AllocatedPartitions: partitions,
		},
	}
}

func getAscendGpuConfig() *config.GpuInfo {
	templates := []config.PartitionTemplateInfo{}
	for _, t := range PartitionTemplateMap[ascendGPUModel] {
		templates = append(templates, t)
	}
	return &config.GpuInfo{
		Model:                  ascendGPUModel,
		Vendor:                 "Ascend",
		MaxPartitions:          7,
		MaxIsolationGroups:     4,
		TotalExtendedResources: map[string]uint32{"AICORE": 8, "AICPU": 7},
		PartitionTemplates:     templates,
	}
}

var _ = Describe("Ascend Partition Strategy", func() {
	var strategy *AscendPartitionStrategy
	var gpuConfig *config.GpuInfo

	BeforeEach(func() {
		setupAscendTestConfig()
		strategy = &AscendPartitionStrategy{}
		gpuConfig = getAscendGpuConfig()
	})

	Describe("Name", func() {
		It("should return ascend-vnpu", func() {
			Expect(strategy.Name()).To(Equal("ascend-vnpu"))
		})
	})

	Describe("GetPartitionStrategy", func() {
		It("should return Ascend strategy for Ascend vendor", func() {
			Expect(GetPartitionStrategy("Ascend").Name()).To(Equal("ascend-vnpu"))
		})

		It("should return NVIDIA strategy for NVIDIA vendor", func() {
			Expect(GetPartitionStrategy("NVIDIA").Name()).To(Equal("nvidia-mig"))
		})
	})

	Describe("CheckAvailability", func() {
		Context("when allocating first partition", func() {
			It("should succeed for vir01", func() {
				err := strategy.CheckAvailability(createAscendGPU(nil), PartitionTemplateMap[ascendGPUModel]["vir01"], gpuConfig)
				Expect(err).NotTo(HaveOccurred())
			})
		})

		Context("when combining different templates", func() {
			It("should succeed for vir04+vir04_3c", func() {
				gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{
					"pod-1": {TemplateID: "vir04", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)},
				})
				err := strategy.CheckAvailability(gpu, PartitionTemplateMap[ascendGPUModel]["vir04_3c"], gpuConfig)
				Expect(err).NotTo(HaveOccurred())
			})
		})

		Context("when all vGroups are occupied", func() {
			It("should fail when no isolation group available", func() {
				gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{
					"pod-1": {TemplateID: "vir02", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)},
					"pod-2": {TemplateID: "vir02", PodUID: "pod-2", IsolationGroupID: ptrUint32(1)},
					"pod-3": {TemplateID: "vir02", PodUID: "pod-3", IsolationGroupID: ptrUint32(2)},
					"pod-4": {TemplateID: "vir02", PodUID: "pod-4", IsolationGroupID: ptrUint32(3)},
				})
				err := strategy.CheckAvailability(gpu, PartitionTemplateMap[ascendGPUModel]["vir02"], gpuConfig)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("maximum partition count"))
			})
		})

		Context("when exceeding template-specific limits", func() {
			It("should fail when max vir04 partitions exceeded", func() {
				gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{
					"pod-1": {TemplateID: "vir04", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)},
					"pod-2": {TemplateID: "vir04", PodUID: "pod-2", IsolationGroupID: ptrUint32(1)},
				})
				err := strategy.CheckAvailability(gpu, PartitionTemplateMap[ascendGPUModel]["vir04"], gpuConfig)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("maximum partition count"))
			})
		})

		Context("when using shared templates", func() {
			It("should allow vir01 sharing within same group", func() {
				gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{
					"pod-1": {TemplateID: "vir01", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)},
				})
				err := strategy.CheckAvailability(gpu, PartitionTemplateMap[ascendGPUModel]["vir01"], gpuConfig)
				Expect(err).NotTo(HaveOccurred())
			})
		})

		Context("when extended resources are exhausted", func() {
			It("should fail when AICORE is insufficient", func() {
				gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{
					"pod-1": {TemplateID: "vir04", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)},
					"pod-2": {TemplateID: "vir04", PodUID: "pod-2", IsolationGroupID: ptrUint32(1)},
				})
				err := strategy.CheckAvailability(gpu, PartitionTemplateMap[ascendGPUModel]["vir01"], gpuConfig)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("insufficient AICORE"))
			})
		})
	})

	Describe("AllocateSlot", func() {
		Context("when allocating first partition", func() {
			It("should allocate vGroup 0", func() {
				groupID, slotStart, slotEnd, err := strategy.AllocateSlot(createAscendGPU(nil), PartitionTemplateMap[ascendGPUModel]["vir04"], gpuConfig)
				Expect(err).NotTo(HaveOccurred())
				Expect(groupID).NotTo(BeNil())
				Expect(*groupID).To(Equal(uint32(0)))
				Expect(slotStart).To(BeNil())
				Expect(slotEnd).To(BeNil())
			})
		})

		Context("when using shared templates", func() {
			It("should join existing group for shared vir01", func() {
				gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{
					"pod-1": {TemplateID: "vir01", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)},
				})
				groupID, _, _, err := strategy.AllocateSlot(gpu, PartitionTemplateMap[ascendGPUModel]["vir01"], gpuConfig)
				Expect(err).NotTo(HaveOccurred())
				Expect(*groupID).To(Equal(uint32(0)))
			})

			It("should get new group when shared group is full", func() {
				gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{
					"pod-1": {TemplateID: "vir01", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)},
					"pod-2": {TemplateID: "vir01", PodUID: "pod-2", IsolationGroupID: ptrUint32(0)},
				})
				groupID, _, _, err := strategy.AllocateSlot(gpu, PartitionTemplateMap[ascendGPUModel]["vir01"], gpuConfig)
				Expect(err).NotTo(HaveOccurred())
				Expect(*groupID).To(Equal(uint32(1)))
			})
		})
	})
})
