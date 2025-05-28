package filter

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestGPUModelFilter(t *testing.T) {
	tests := []struct {
		name          string
		requiredModel string
		gpus          []tfv1.GPU
		want          int
		wantErr       bool
	}{
		{
			name:          "filter A100 GPUs",
			requiredModel: "A100",
			gpus: []tfv1.GPU{
				{
					Status: tfv1.GPUStatus{
						GPUModel: "A100",
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("100"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
				{
					Status: tfv1.GPUStatus{
						GPUModel: "H100",
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("200"),
							Vram:   resource.MustParse("80Gi"),
						},
					},
				},
			},
			want:    1,
			wantErr: false,
		},
		{
			name:          "no model specified",
			requiredModel: "",
			gpus: []tfv1.GPU{
				{
					Status: tfv1.GPUStatus{
						GPUModel: "A100",
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("100"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
			},
			want:    1,
			wantErr: false,
		},
		{
			name:          "non-existent model",
			requiredModel: "NonExistentModel",
			gpus: []tfv1.GPU{
				{
					Status: tfv1.GPUStatus{
						GPUModel: "A100",
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("100"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
			},
			want:    0,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewGPUModelFilter(tt.requiredModel)
			got, err := filter.Filter(context.Background(), tt.gpus)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Len(t, got, tt.want)
			if tt.want > 0 && tt.requiredModel != "" {
				for _, gpu := range got {
					assert.Equal(t, tt.requiredModel, gpu.Status.GPUModel)
				}
			}
		})
	}
}
