package gpuallocator

import (
	"sync"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
)

// TestPartitionTemplateMaps_ConcurrentReloadVsRead guards against the data
// race where providerconfig_controller / pricing reload calls
// LoadPartitionTemplatesFromConfig (which atomically publishes a fresh
// snapshot via atomic.Pointer[partitionConfig]) while the scheduler
// concurrently reads through GetPartitionTemplates / GetMaxPartitions /
// GetMaxPlacementSlots / GetMaxIsolationGroups / GetTotalExtendedResources /
// GetPartitionConfigSnapshot. The atomic-snapshot model means readers see
// either the previous or the new published *partitionConfig — never a
// torn intermediate state. Run with -race.
func TestPartitionTemplateMaps_ConcurrentReloadVsRead(t *testing.T) {
	configs := func(seed int) []config.GpuInfo {
		return []config.GpuInfo{
			{
				Model:              "M",
				FullModelName:      "M-full",
				MaxPartitions:      uint32(seed),
				MaxPlacementSlots:  uint32(seed),
				MaxIsolationGroups: uint32(seed),
				TotalExtendedResources: map[string]uint32{
					"AICORE": uint32(seed),
				},
				PartitionTemplates: []config.PartitionTemplateInfo{
					{TemplateID: "t1", Name: "t1", ComputePercent: 50, MemoryGigabytes: 16},
				},
			},
		}
	}
	// Seed once so readers don't trip the "no template configs" path.
	LoadPartitionTemplatesFromConfig(configs(1))

	gpuStatus := tfv1.GPUStatus{
		GPUModel: "M",
		Capacity: &tfv1.Resource{
			Tflops: resource.MustParse("100"),
			Vram:   resource.MustParse("80Gi"),
		},
	}
	gpu := &tfv1.GPU{}
	gpu.Name = "gpu-0"
	gpu.Status = gpuStatus
	gpu.Status.Available = &tfv1.Resource{
		Tflops: resource.MustParse("100"),
		Vram:   resource.MustParse("80Gi"),
	}

	stop := make(chan struct{})
	var readersWG, writerWG sync.WaitGroup

	// Writer: continuously reload templates until readers complete.
	writerWG.Add(1)
	go func() {
		defer writerWG.Done()
		for i := 1; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			LoadPartitionTemplatesFromConfig(configs(i))
		}
	}()

	// Readers: hammer every accessor used by partitioned_scheduling.go.
	const readers = 4
	readersWG.Add(readers)
	for i := 0; i < readers; i++ {
		go func() {
			defer readersWG.Done()
			req := &tfv1.AllocRequest{
				Request: tfv1.Resource{
					Tflops: resource.MustParse("10"),
					Vram:   resource.MustParse("4Gi"),
				},
			}
			for j := 0; j < 200; j++ {
				_, _ = MatchPartitionTemplate(gpuStatus, req)
				_, _, _ = CalculatePartitionResourceUsage(gpuStatus.Capacity.Tflops, "M", "t1")
				_ = CheckPartitionAvailability(gpu, "t1")
			}
		}()
	}

	readersWG.Wait()
	close(stop)
	writerWG.Wait()
}

// TestPartitionConfigSnapshot_PostPublishImmutability codifies the invariant
// that LoadPartitionTemplatesFromConfig must build a fresh *partitionConfig
// and Store it — never mutate an already-published one. If a future refactor
// reintroduces in-place mutation, a reader that captured the snapshot before
// the "update" would observe the mutated state, breaking this test.
func TestPartitionConfigSnapshot_PostPublishImmutability(t *testing.T) {
	LoadPartitionTemplatesFromConfig([]config.GpuInfo{{
		Model:         "M-v1",
		MaxPartitions: 7,
		PartitionTemplates: []config.PartitionTemplateInfo{
			{TemplateID: "t1", Name: "t1", ComputePercent: 50, MemoryGigabytes: 8},
		},
	}})

	// Capture a snapshot before the next publish.
	maxPartitionsV1, templatesV1 := GetPartitionConfigSnapshot()
	assert.Equal(t, uint32(7), maxPartitionsV1["M-v1"])
	assert.Contains(t, templatesV1, "M-v1")

	// Publish a totally different config.
	LoadPartitionTemplatesFromConfig([]config.GpuInfo{{
		Model:         "M-v2",
		MaxPartitions: 99,
		PartitionTemplates: []config.PartitionTemplateInfo{
			{TemplateID: "t2", Name: "t2", ComputePercent: 25, MemoryGigabytes: 4},
		},
	}})

	// The previously-captured snapshot must NOT see the new publish. Its maps
	// belong to a frozen *partitionConfig. If a refactor mutated the live maps
	// in place, "M-v1" would be gone and "M-v2" would appear here.
	assert.Equal(t, uint32(7), maxPartitionsV1["M-v1"],
		"old snapshot must retain MaxPartitions=7 for M-v1 even after LoadPartitionTemplatesFromConfig publishes a new config")
	assert.NotContains(t, maxPartitionsV1, "M-v2",
		"old snapshot must NOT pick up newly-published entries — that would mean the writer mutated the live map in place")
	assert.Contains(t, templatesV1, "M-v1",
		"old snapshot must retain M-v1 template entry")
	assert.NotContains(t, templatesV1, "M-v2",
		"old snapshot must NOT pick up newly-published M-v2 template")

	// New snapshot reflects the latest publish.
	maxPartitionsV2, templatesV2 := GetPartitionConfigSnapshot()
	assert.Equal(t, uint32(99), maxPartitionsV2["M-v2"])
	assert.Contains(t, templatesV2, "M-v2")
}
