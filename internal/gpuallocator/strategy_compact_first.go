package gpuallocator

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
)

// CompactFirst selects GPU with minimum available resources (most utilized)
// to efficiently pack workloads and maximize GPU utilization
type CompactFirst struct {
	cfg          *config.GPUFitConfig
	nodeGpuStore map[string]map[string]*tfv1.GPU
}

var _ Strategy = CompactFirst{}

// SelectGPUs selects multiple GPUs from the same node with the least available resources (most packed)
func (c CompactFirst) SelectGPUs(gpus []*tfv1.GPU, count uint) ([]*tfv1.GPU, error) {
	return DefaultGPUSelector(c, c.nodeGpuStore, gpus, count)
}

// Score function is using by Kubernetes scheduler framework
func (c CompactFirst) Score(gpu *tfv1.GPU) int {
	tflopsUsedPercentage := 100 - gpu.Status.Available.Tflops.AsApproximateFloat64()/gpu.Status.Capacity.Tflops.AsApproximateFloat64()*100
	vramUsedPercentage := 100 - gpu.Status.Available.Vram.AsApproximateFloat64()/gpu.Status.Capacity.Vram.AsApproximateFloat64()*100
	return normalizeScore(c.cfg, vramUsedPercentage, tflopsUsedPercentage)
}
