package filter

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// GPUModelAndVendorFilter filters GPUs based on their model (e.g., A100, H100)
type GPUModelAndVendorFilter struct {
	requiredModel  string
	requiredVendor string
}

// NewGPUModelAndVendorFilter creates a new filter that matches GPUs with the specified model
func NewGPUModelAndVendorFilter(model string, vendor string) *GPUModelAndVendorFilter {
	return &GPUModelAndVendorFilter{
		requiredModel:  model,
		requiredVendor: vendor,
	}
}

// Filter implements GPUFilter interface
func (f *GPUModelAndVendorFilter) Filter(ctx context.Context, workerPodKey tfv1.NameNamespace, gpus []*tfv1.GPU) ([]*tfv1.GPU, error) {
	if f.requiredModel == "" && f.requiredVendor == "" {
		return gpus, nil
	}

	filtered := make([]*tfv1.GPU, 0, len(gpus))

	if f.requiredModel != "" {
		for _, gpu := range gpus {
			if gpu.Status.GPUModel == f.requiredModel {
				filtered = append(filtered, gpu)
			}
		}
	}
	if f.requiredVendor != "" {
		for _, gpu := range gpus {
			if gpu.Status.Vendor == f.requiredVendor {
				filtered = append(filtered, gpu)
			}
		}
	}
	return filtered, nil
}

func (f *GPUModelAndVendorFilter) Name() string {
	return "GPUModelAndVendorFilter"
}
