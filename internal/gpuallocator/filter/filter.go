package filter

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// GPUFilter defines an interface for filtering GPU candidates
type GPUFilter interface {
	// Filter filters the list of GPUs and returns only those that pass the filter criteria
	// The implementation should not modify the input slice
	Filter(ctx context.Context, workerPodKey tfv1.NameNamespace, gpus []tfv1.GPU) ([]tfv1.GPU, error)
}

// FilterRegistry provides an immutable collection of GPU filters
// with methods to create new instances with additional filters
type FilterRegistry struct {
	parent  *FilterRegistry // Reference to parent registry
	filters []GPUFilter     // Only contains filters added at this level
}

// NewFilterRegistry creates a new empty filter registry
func NewFilterRegistry() *FilterRegistry {
	return &FilterRegistry{
		parent:  nil,
		filters: []GPUFilter{},
	}
}

// With creates a new FilterRegistry with the provided filters added
// The original FilterRegistry is not modified
func (fr *FilterRegistry) With(filters ...GPUFilter) *FilterRegistry {
	if len(filters) == 0 {
		return fr
	}

	// Create a new registry with the current one as parent
	return &FilterRegistry{
		parent:  fr,
		filters: filters,
	}
}

// Apply applies the filters in this registry to the given GPU list
// Filters are applied in the order they were added (parent filters first)
func (fr *FilterRegistry) Apply(ctx context.Context, workerPodKey tfv1.NameNamespace, gpus []tfv1.GPU) ([]tfv1.GPU, error) {
	// First apply parent filters (if any)
	filteredGPUs := gpus
	var err error

	if fr.parent != nil {
		filteredGPUs, err = fr.parent.Apply(ctx, workerPodKey, filteredGPUs)
		if err != nil {
			return nil, err
		}

		// If no GPUs left after parent filtering, return early
		if len(filteredGPUs) == 0 {
			return filteredGPUs, nil
		}
	}

	// Then apply filters at this level
	for _, filter := range fr.filters {
		filteredGPUs, err = filter.Filter(ctx, workerPodKey, filteredGPUs)
		if err != nil {
			return nil, err
		}

		// If no GPUs left after filtering, return early
		if len(filteredGPUs) == 0 {
			return filteredGPUs, nil
		}
	}

	return filteredGPUs, nil
}
