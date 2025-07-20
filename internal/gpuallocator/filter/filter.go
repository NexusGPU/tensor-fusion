package filter

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/samber/lo"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type FilterDetail struct {
	FilterName string
	Before     []string
	After      []string
	Diff       int
}

// GPUFilter defines an interface for filtering GPU candidates
type GPUFilter interface {
	// Filter filters the list of GPUs and returns only those that pass the filter criteria
	// The implementation should not modify the input slice
	Filter(ctx context.Context, workerPodKey tfv1.NameNamespace, gpus []tfv1.GPU) ([]tfv1.GPU, error)

	Name() string
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
func (fr *FilterRegistry) Apply(
	ctx context.Context,
	workerPodKey tfv1.NameNamespace,
	gpus []tfv1.GPU,
	isSimulateSchedule bool,
) ([]tfv1.GPU, []FilterDetail, error) {
	log := log.FromContext(ctx)
	if len(gpus) == 0 {
		log.Info("FilterRegistry - no GPUs to filter", "workerPodKey", workerPodKey)
		return gpus, nil, nil
	}

	// First apply parent filters (if any)
	filteredGPUs := gpus

	// only assigned when simulate schedule is true
	filterDetails := []FilterDetail{}
	var err error

	if fr.parent != nil {
		filteredGPUs, filterDetails, err = fr.parent.Apply(ctx, workerPodKey, filteredGPUs, isSimulateSchedule)
		if err != nil {
			return nil, nil, err
		}

		// If no GPUs left after parent filtering, return early
		if len(filteredGPUs) == 0 {
			log.Info("FilterRegistry - no GPUs left after parent filtering", "workerPodKey", workerPodKey)
			return filteredGPUs, filterDetails, nil
		}
	}

	// Then apply filters at this level
	for _, filter := range fr.filters {
		afterFilteredGPUs, err := filter.Filter(ctx, workerPodKey, filteredGPUs)
		if err != nil {
			return nil, nil, err
		}

		if isSimulateSchedule {
			detail := FilterDetail{
				FilterName: filter.Name(),
				Before:     lo.Map(filteredGPUs, func(gpu tfv1.GPU, _ int) string { return gpu.Name }),
				After:      lo.Map(afterFilteredGPUs, func(gpu tfv1.GPU, _ int) string { return gpu.Name }),
				Diff:       len(filteredGPUs) - len(afterFilteredGPUs),
			}
			filterDetails = append(filterDetails, detail)
		}
		filteredGPUs = afterFilteredGPUs

		// If no GPUs left after filtering, return early
		if len(filteredGPUs) == 0 {
			log.Info("FilterRegistry - no GPUs left after filtering", "workerPodKey", workerPodKey)
			return filteredGPUs, filterDetails, nil
		}
	}

	log.Info("FilterRegistry - filtered GPUs", "workerPodKey", workerPodKey,
		"before", len(gpus), "after", len(filteredGPUs))
	return filteredGPUs, filterDetails, nil
}
