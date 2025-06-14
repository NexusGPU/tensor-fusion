package filter

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	schedulingcorev1 "k8s.io/component-helpers/scheduling/corev1"
)

func TestNodeAffinityFilter(t *testing.T) {
	tests := []struct {
		name         string
		nodeSelector *corev1.NodeSelector
		preferred    []corev1.PreferredSchedulingTerm
		gpus         []tfv1.GPU
		want         int
		wantErr      bool
	}{
		{
			name: "filter by required node affinity",
			nodeSelector: &corev1.NodeSelector{
				NodeSelectorTerms: []corev1.NodeSelectorTerm{
					{
						MatchExpressions: []corev1.NodeSelectorRequirement{
							{
								Key:      "gpu-type",
								Operator: corev1.NodeSelectorOpIn,
								Values:   []string{"a100"},
							},
						},
					},
				},
			},
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "gpu-1",
						Labels: map[string]string{
							constants.LabelKeyOwner: "node-1",
						},
					},
					Status: tfv1.GPUStatus{
						NodeSelector: map[string]string{
							"gpu-type": "a100",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "gpu-2",
						Labels: map[string]string{
							constants.LabelKeyOwner: "node-2",
						},
					},
					Status: tfv1.GPUStatus{
						NodeSelector: map[string]string{
							"gpu-type": "h100",
						},
					},
				},
			},
			want:    1,
			wantErr: false,
		},
		{
			name: "filter by preferred node affinity",
			preferred: []corev1.PreferredSchedulingTerm{
				{
					Weight: 100,
					Preference: corev1.NodeSelectorTerm{
						MatchExpressions: []corev1.NodeSelectorRequirement{
							{
								Key:      "zone",
								Operator: corev1.NodeSelectorOpIn,
								Values:   []string{"zone-1"},
							},
						},
					},
				},
			},
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "gpu-1",
						Labels: map[string]string{
							constants.LabelKeyOwner: "node-1",
						},
					},
					Status: tfv1.GPUStatus{
						NodeSelector: map[string]string{
							"zone": "zone-1",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "gpu-2",
						Labels: map[string]string{
							constants.LabelKeyOwner: "node-2",
						},
					},
					Status: tfv1.GPUStatus{
						NodeSelector: map[string]string{
							"zone": "zone-2",
						},
					},
				},
			},
			want:    2,
			wantErr: false,
		},
		{
			name: "no node affinity specified",
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "gpu-1",
						Labels: map[string]string{
							constants.LabelKeyOwner: "node-1",
						},
					},
					Status: tfv1.GPUStatus{
						NodeSelector: map[string]string{
							"gpu-type": "a100",
						},
					},
				},
			},
			want:    1,
			wantErr: false,
		},
		{
			name: "GPU without node label",
			nodeSelector: &corev1.NodeSelector{
				NodeSelectorTerms: []corev1.NodeSelectorTerm{
					{
						MatchExpressions: []corev1.NodeSelectorRequirement{
							{
								Key:      "gpu-type",
								Operator: corev1.NodeSelectorOpIn,
								Values:   []string{"a100"},
							},
						},
					},
				},
			},
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "gpu-1",
					},
					Status: tfv1.GPUStatus{
						NodeSelector: map[string]string{
							"gpu-type": "a100",
						},
					},
				},
			},
			want:    0,
			wantErr: false,
		},
		{
			name: "combined required and preferred affinity",
			nodeSelector: &corev1.NodeSelector{
				NodeSelectorTerms: []corev1.NodeSelectorTerm{
					{
						MatchExpressions: []corev1.NodeSelectorRequirement{
							{
								Key:      "gpu-type",
								Operator: corev1.NodeSelectorOpIn,
								Values:   []string{"a100"},
							},
						},
					},
				},
			},
			preferred: []corev1.PreferredSchedulingTerm{
				{
					Weight: 100,
					Preference: corev1.NodeSelectorTerm{
						MatchExpressions: []corev1.NodeSelectorRequirement{
							{
								Key:      "zone",
								Operator: corev1.NodeSelectorOpIn,
								Values:   []string{"zone-1"},
							},
						},
					},
				},
			},
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "gpu-1",
						Labels: map[string]string{
							constants.LabelKeyOwner: "node-1",
						},
					},
					Status: tfv1.GPUStatus{
						NodeSelector: map[string]string{
							"gpu-type": "a100",
							"zone":     "zone-1",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "gpu-2",
						Labels: map[string]string{
							constants.LabelKeyOwner: "node-2",
						},
					},
					Status: tfv1.GPUStatus{
						NodeSelector: map[string]string{
							"gpu-type": "a100",
							"zone":     "zone-2",
						},
					},
				},
			},
			want:    2,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewNodeAffinityFilter(tt.nodeSelector, tt.preferred)
			got, err := filter.Filter(context.Background(), tt.gpus)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Len(t, got, tt.want)

			// For soft affinity requirements, verify sorting
			if len(tt.preferred) > 0 {
				// Get the score of the first GPU
				firstScore := calculateScore(got[0], tt.preferred)
				// Verify that all GPUs have scores not higher than the first one
				for i := 1; i < len(got); i++ {
					score := calculateScore(got[i], tt.preferred)
					assert.LessOrEqual(t, score, firstScore)
				}
			}
		})
	}
}

// calculateScore calculates the score for a single GPU based on preferred scheduling terms
func calculateScore(gpu tfv1.GPU, preferred []corev1.PreferredSchedulingTerm) int32 {
	var totalScore int32
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   gpu.Labels[constants.LabelKeyOwner],
			Labels: gpu.Status.NodeSelector,
		},
	}

	for _, term := range preferred {
		matches, _ := schedulingcorev1.MatchNodeSelectorTerms(node, &corev1.NodeSelector{
			NodeSelectorTerms: []corev1.NodeSelectorTerm{term.Preference},
		})
		if matches {
			totalScore += term.Weight
		}
	}
	return totalScore
}
