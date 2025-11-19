package filter

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// GPUVendorFilter filters GPUs based on their vendor
type GPUVendorFilter struct {
	requiredVendor string
}

// NewGPUVendorFilter creates a new filter that matches GPUs with the specified vendor
func NewGPUVendorFilter(vendor string) *GPUVendorFilter {
	return &GPUVendorFilter{
		requiredVendor: vendor,
	}
}

// Filter implements GPUFilter interface
func (f *GPUVendorFilter) Filter(ctx context.Context, workerPodKey tfv1.NameNamespace, gpus []*tfv1.GPU) ([]*tfv1.GPU, error) {
	if f.requiredVendor == "" {
		return gpus, nil
	}

	filtered := make([]*tfv1.GPU, 0, len(gpus))
	for _, gpu := range gpus {
		if gpu.Status.Vendor == f.requiredVendor {
			filtered = append(filtered, gpu)
		}
	}
	return filtered, nil
}

func (f *GPUVendorFilter) Name() string {
	return "GPUVendorFilter"
}
