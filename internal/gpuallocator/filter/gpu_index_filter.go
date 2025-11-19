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

package filter

import (
	"context"
	"slices"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/samber/lo"
)

// GPUIndexFilter filters GPUs based on required GPU indices
type GPUIndexFilter struct {
	requiredIndices []int32
}

// NewGPUIndexFilter creates a new GPUIndexFilter with the specified indices
func NewGPUIndexFilter(requiredIndices []int32) *GPUIndexFilter {
	return &GPUIndexFilter{
		requiredIndices: requiredIndices,
	}
}

// Filter implements GPUFilter.Filter
func (f *GPUIndexFilter) Filter(ctx context.Context, workerPodKey tfv1.NameNamespace, gpus []*tfv1.GPU) ([]*tfv1.GPU, error) {
	// If no indices specified, pass all GPUs
	if len(f.requiredIndices) == 0 {
		return gpus, nil
	}

	return lo.Filter(gpus, func(gpu *tfv1.GPU, _ int) bool {
		// Check GPU index
		if gpu.Status.Index != nil && slices.Contains(f.requiredIndices, *gpu.Status.Index) {
			return true
		}
		return false
	}), nil
}

func (f *GPUIndexFilter) Name() string {
	return "GPUIndexFilter"
}
