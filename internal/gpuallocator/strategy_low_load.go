package gpuallocator

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
)

// LowLoadFirst selects GPU with maximum available resources (least utilized)
// to distribute workloads more evenly across GPUs
type LowLoadFirst struct {
	cfg          *config.GPUFitConfig
	nodeGpuStore map[string]map[string]*tfv1.GPU
}

var _ Strategy = LowLoadFirst{}

// SelectGPUs selects multiple GPUs from the same node with the most available resources (least load)
func (l LowLoadFirst) SelectGPUs(gpus []*tfv1.GPU, count uint) ([]*tfv1.GPU, error) {
	return DefaultGPUSelector(l, l.nodeGpuStore, gpus, count)
}

// Score function is using by Kubernetes scheduler framework
func (l LowLoadFirst) Score(gpu *tfv1.GPU, _ bool) int {
	tflopsAvailablePercentage := gpu.Status.Available.Tflops.AsApproximateFloat64() /
		gpu.Status.Capacity.Tflops.AsApproximateFloat64() * 100
	vramAvailablePercentage := gpu.Status.Available.Vram.AsApproximateFloat64() /
		gpu.Status.Capacity.Vram.AsApproximateFloat64() * 100
	return normalizeScore(l.cfg, vramAvailablePercentage, tflopsAvailablePercentage)
}
