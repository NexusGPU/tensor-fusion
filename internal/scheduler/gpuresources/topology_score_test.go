package gpuresources

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type fixedStrategy struct {
	scores map[string]int
}

func (f fixedStrategy) Score(gpu *tfv1.GPU, _ bool) int {
	return f.scores[gpu.Name]
}

func (f fixedStrategy) SelectGPUs(_ []*tfv1.GPU, _ uint) ([]*tfv1.GPU, error) {
	return nil, nil
}

var _ gpuallocator.Strategy = fixedStrategy{}

func TestSelectPreferredGPUsWithTopology_SingleGPUProtectsHighBandwidthCards(t *testing.T) {
	cfg := &config.GPUFitConfig{
		EnableNvLinkAware:      true,
		SingleGPUProtectWeight: 0.9,
	}
	strategy := fixedStrategy{
		scores: map[string]int{
			"gpu-a": 60,
			"gpu-b": 60,
		},
	}

	gpus := []*tfv1.GPU{
		makeTestGPU("gpu-a", "gpu-a-uuid", 400, nil),
		makeTestGPU("gpu-b", "gpu-b-uuid", 40, nil),
	}
	selected, topoScore, ok := selectPreferredGPUsWithTopology(gpus, 1, strategy, cfg)
	assert.True(t, ok)
	assert.Equal(t, []string{"gpu-b"}, selected)
	assert.Greater(t, topoScore, 0)
}

func TestSelectPreferredGPUsWithTopology_MultiGPUPrefersConnectedPair(t *testing.T) {
	cfg := &config.GPUFitConfig{
		EnableNvLinkAware:      true,
		ResourceScoreWeight:    0.1,
		TopologyScoreWeight:    0.9,
		SingleGPUProtectWeight: 0.6,
	}
	strategy := fixedStrategy{
		scores: map[string]int{
			"gpu-a": 60,
			"gpu-b": 60,
			"gpu-c": 60,
		},
	}

	gpuA := makeTestGPU("gpu-a", "gpu-a-uuid", 200000, []tfv1.GPUNvLinkPeer{
		{PeerUUID: "gpu-b-uuid", BandwidthMBps: 200000, LinkCount: 2, LinkVersion: 4},
	})
	gpuB := makeTestGPU("gpu-b", "gpu-b-uuid", 200000, []tfv1.GPUNvLinkPeer{
		{PeerUUID: "gpu-a-uuid", BandwidthMBps: 200000, LinkCount: 2, LinkVersion: 4},
	})
	gpuC := makeTestGPU("gpu-c", "gpu-c-uuid", 0, nil)

	selected, topoScore, ok := selectPreferredGPUsWithTopology([]*tfv1.GPU{gpuA, gpuB, gpuC}, 2, strategy, cfg)
	assert.True(t, ok)
	assert.Equal(t, []string{"gpu-a", "gpu-b"}, selected)
	assert.Greater(t, topoScore, 0)
}

func TestMergeResourceAndTopologyScore(t *testing.T) {
	cfg := &config.GPUFitConfig{
		EnableNvLinkAware:   true,
		ResourceScoreWeight: 0.8,
		TopologyScoreWeight: 0.2,
	}
	assert.Equal(t, 68, mergeResourceAndTopologyScore(80, 20, cfg))
	assert.Equal(t, 50, mergeResourceAndTopologyScore(50, 50, cfg))
}

func makeTestGPU(name, uuid string, totalBandwidth int64, peers []tfv1.GPUNvLinkPeer) *tfv1.GPU {
	return &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status: tfv1.GPUStatus{
			UUID: uuid,
			Capacity: &tfv1.Resource{
				Tflops: resource.MustParse("100"),
				Vram:   resource.MustParse("20Gi"),
			},
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("100"),
				Vram:   resource.MustParse("20Gi"),
			},
			NvLink: &tfv1.GPUNvLinkStatus{
				PeerCount:          int32(len(peers)),
				TotalLinkCount:     int32(len(peers)),
				TotalBandwidthMBps: totalBandwidth,
				Peers:              peers,
			},
		},
	}
}
