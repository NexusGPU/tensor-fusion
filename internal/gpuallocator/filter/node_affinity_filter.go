package filter

import (
	"context"
	"fmt"
	"sort"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	schedulingcorev1 "k8s.io/component-helpers/scheduling/corev1"
)

type NodeAffinityFilter struct {
	nodeSelector *corev1.NodeSelector
	preferred    []corev1.PreferredSchedulingTerm
}

type gpuScore struct {
	gpu   tfv1.GPU
	score int32
}

func NewNodeAffinityFilter(nodeSelector *corev1.NodeSelector, preferred []corev1.PreferredSchedulingTerm) *NodeAffinityFilter {
	return &NodeAffinityFilter{
		nodeSelector: nodeSelector,
		preferred:    preferred,
	}
}

// Filter
func (f *NodeAffinityFilter) Filter(ctx context.Context, gpus []tfv1.GPU) ([]tfv1.GPU, error) {
	if f.nodeSelector == nil && len(f.preferred) == 0 {
		return gpus, nil
	}

	// requiredDuringSchedulingIgnoredDuringExecution
	if f.nodeSelector != nil {
		var err error
		gpus, err = f.filterRequired(gpus)
		if err != nil {
			return nil, err
		}
	}

	// preferredDuringSchedulingIgnoredDuringExecution
	if len(f.preferred) > 0 {
		var err error
		gpus, err = f.filterPreferred(gpus)
		if err != nil {
			return nil, err
		}
	}

	return gpus, nil
}

// requiredDuringSchedulingIgnoredDuringExecution
func (f *NodeAffinityFilter) filterRequired(gpus []tfv1.GPU) ([]tfv1.GPU, error) {
	var filteredGPUs []tfv1.GPU
	for _, gpu := range gpus {
		nodeName, exists := gpu.Labels[constants.LabelKeyOwner]
		if !exists {
			continue
		}

		node := &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   nodeName,
				Labels: gpu.Status.NodeSelector,
			},
		}

		matches, err := schedulingcorev1.MatchNodeSelectorTerms(node, f.nodeSelector)
		if err != nil {
			return nil, fmt.Errorf("match node selector terms: %w", err)
		}

		if matches {
			filteredGPUs = append(filteredGPUs, gpu)
		}
	}
	return filteredGPUs, nil
}

// preferredDuringSchedulingIgnoredDuringExecution
func (f *NodeAffinityFilter) filterPreferred(gpus []tfv1.GPU) ([]tfv1.GPU, error) {
	gpuScores := make([]gpuScore, 0, len(gpus))
	for _, gpu := range gpus {
		nodeName, exists := gpu.Labels[constants.LabelKeyOwner]
		if !exists {
			continue
		}

		node := &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   nodeName,
				Labels: gpu.Status.NodeSelector,
			},
		}

		var totalScore int32
		// Only match the preferred node selector terms
		// If not match, the score is 0, that means if a lot of GPU not match the score will all be 0
		for _, term := range f.preferred {
			matches, err := schedulingcorev1.MatchNodeSelectorTerms(node, &corev1.NodeSelector{
				NodeSelectorTerms: []corev1.NodeSelectorTerm{term.Preference},
			})
			if err != nil {
				return nil, fmt.Errorf("match preferred node selector terms: %w", err)
			}
			if matches {
				totalScore += term.Weight
			}
		}

		gpuScores = append(gpuScores, gpuScore{
			gpu:   gpu,
			score: totalScore,
		})
	}

	// sort by score
	sort.Slice(gpuScores, func(i, j int) bool {
		return gpuScores[i].score > gpuScores[j].score
	})

	// change back to gpu list
	result := make([]tfv1.GPU, len(gpuScores))
	for i, score := range gpuScores {
		result[i] = score.gpu
	}
	return result, nil
}
