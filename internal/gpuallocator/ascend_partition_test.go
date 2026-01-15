package gpuallocator

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/stretchr/testify/assert"
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
	return &tfv1.GPU{ObjectMeta: metav1.ObjectMeta{Name: "npu-0"}, Status: tfv1.GPUStatus{GPUModel: ascendGPUModel, Vendor: "Ascend", Capacity: &tfv1.Resource{Tflops: resource.MustParse("22"), Vram: resource.MustParse("21Gi")}, Available: &tfv1.Resource{Tflops: resource.MustParse("22"), Vram: resource.MustParse("21Gi")}, AllocatedPartitions: partitions}}
}

func getAscendGpuConfig() *config.GpuInfo {
	templates := []config.PartitionTemplateInfo{}
	for _, t := range PartitionTemplateMap[ascendGPUModel] {
		templates = append(templates, t)
	}
	return &config.GpuInfo{Model: ascendGPUModel, Vendor: "Ascend", MaxPartitions: 7, MaxIsolationGroups: 4, TotalExtendedResources: map[string]uint32{"AICORE": 8, "AICPU": 7}, PartitionTemplates: templates}
}

func TestAscendPartitionStrategy_Name(t *testing.T) {
	assert.Equal(t, "ascend-vnpu", (&AscendPartitionStrategy{}).Name())
}

func TestGetPartitionStrategy_Ascend(t *testing.T) {
	assert.Equal(t, "ascend-vnpu", GetPartitionStrategy("Ascend").Name())
	assert.Equal(t, "nvidia-mig", GetPartitionStrategy("NVIDIA").Name())
}

func TestAscendPartitionStrategy_CheckAvailability(t *testing.T) {
	setupAscendTestConfig()
	strategy := &AscendPartitionStrategy{}
	gpuConfig := getAscendGpuConfig()
	t.Run("first vir01 succeeds", func(t *testing.T) {
		assert.NoError(t, strategy.CheckAvailability(createAscendGPU(nil), PartitionTemplateMap[ascendGPUModel]["vir01"], gpuConfig))
	})
	t.Run("vir04+vir04_3c succeeds", func(t *testing.T) {
		gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{"pod-1": {TemplateID: "vir04", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)}})
		assert.NoError(t, strategy.CheckAvailability(gpu, PartitionTemplateMap[ascendGPUModel]["vir04_3c"], gpuConfig))
	})
	t.Run("all vGroups occupied fails", func(t *testing.T) {
		gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{
			"pod-1": {TemplateID: "vir02", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)},
			"pod-2": {TemplateID: "vir02", PodUID: "pod-2", IsolationGroupID: ptrUint32(1)},
			"pod-3": {TemplateID: "vir02", PodUID: "pod-3", IsolationGroupID: ptrUint32(2)},
			"pod-4": {TemplateID: "vir02", PodUID: "pod-4", IsolationGroupID: ptrUint32(3)},
		})
		err := strategy.CheckAvailability(gpu, PartitionTemplateMap[ascendGPUModel]["vir02"], gpuConfig)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "maximum partition count")
	})
	t.Run("exceed max vir04 partitions", func(t *testing.T) {
		gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{"pod-1": {TemplateID: "vir04", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)}, "pod-2": {TemplateID: "vir04", PodUID: "pod-2", IsolationGroupID: ptrUint32(1)}})
		err := strategy.CheckAvailability(gpu, PartitionTemplateMap[ascendGPUModel]["vir04"], gpuConfig)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "maximum partition count")
	})
	t.Run("vir01 sharing works", func(t *testing.T) {
		gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{"pod-1": {TemplateID: "vir01", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)}})
		assert.NoError(t, strategy.CheckAvailability(gpu, PartitionTemplateMap[ascendGPUModel]["vir01"], gpuConfig))
	})
	t.Run("AICORE exhausted", func(t *testing.T) {
		gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{"pod-1": {TemplateID: "vir04", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)}, "pod-2": {TemplateID: "vir04", PodUID: "pod-2", IsolationGroupID: ptrUint32(1)}})
		err := strategy.CheckAvailability(gpu, PartitionTemplateMap[ascendGPUModel]["vir01"], gpuConfig)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "insufficient AICORE")
	})
}

func TestAscendPartitionStrategy_AllocateSlot(t *testing.T) {
	setupAscendTestConfig()
	strategy := &AscendPartitionStrategy{}
	gpuConfig := getAscendGpuConfig()
	t.Run("first allocation gets vGroup 0", func(t *testing.T) {
		groupID, slotStart, slotEnd, err := strategy.AllocateSlot(createAscendGPU(nil), PartitionTemplateMap[ascendGPUModel]["vir04"], gpuConfig)
		assert.NoError(t, err)
		assert.NotNil(t, groupID)
		assert.Equal(t, uint32(0), *groupID)
		assert.Nil(t, slotStart)
		assert.Nil(t, slotEnd)
	})
	t.Run("shared vir01 joins existing group", func(t *testing.T) {
		gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{"pod-1": {TemplateID: "vir01", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)}})
		groupID, _, _, err := strategy.AllocateSlot(gpu, PartitionTemplateMap[ascendGPUModel]["vir01"], gpuConfig)
		assert.NoError(t, err)
		assert.Equal(t, uint32(0), *groupID)
	})
	t.Run("shared vir01 gets new group when full", func(t *testing.T) {
		gpu := createAscendGPU(map[string]tfv1.AllocatedPartition{"pod-1": {TemplateID: "vir01", PodUID: "pod-1", IsolationGroupID: ptrUint32(0)}, "pod-2": {TemplateID: "vir01", PodUID: "pod-2", IsolationGroupID: ptrUint32(0)}})
		groupID, _, _, err := strategy.AllocateSlot(gpu, PartitionTemplateMap[ascendGPUModel]["vir01"], gpuConfig)
		assert.NoError(t, err)
		assert.Equal(t, uint32(1), *groupID)
	})
}
