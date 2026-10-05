package v1

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

func TestApplyPodIndexResourceUsesZeroBasedEncoding(t *testing.T) {
	tests := []struct {
		name      string
		oneBased  int
		resource  string
		quantity  string
		wantIndex int
	}{
		{name: "first", oneBased: 1, resource: "tensor-fusion.ai/index_0", quantity: "1", wantIndex: 1},
		{name: "last in bucket", oneBased: 8, resource: "tensor-fusion.ai/index_0", quantity: "8", wantIndex: 8},
		{name: "first in second bucket", oneBased: 9, resource: "tensor-fusion.ai/index_1", quantity: "1", wantIndex: 9},
		{name: "last", oneBased: 128, resource: "tensor-fusion.ai/index_f", quantity: "8", wantIndex: 128},
		{name: "allocator fallback", oneBased: 0, resource: "tensor-fusion.ai/index_0", quantity: "1", wantIndex: 1},
		{name: "out of range fallback", oneBased: 129, resource: "tensor-fusion.ai/index_0", quantity: "1", wantIndex: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			container := &corev1.Container{}
			applyPodIndexResource(container, tt.oneBased)

			got, ok := container.Resources.Limits[corev1.ResourceName(tt.resource)]
			if !ok {
				t.Fatalf("resource %q was not set: %#v", tt.resource, container.Resources.Limits)
			}
			want := resource.MustParse(tt.quantity)
			if got.Cmp(want) != 0 {
				t.Fatalf("resource %q = %s, want %s", tt.resource, got.String(), want.String())
			}
			if len(container.Resources.Limits) != 1 {
				t.Fatalf("limits = %#v, want exactly one index resource", container.Resources.Limits)
			}

			pod := &corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{*container}}}
			gotIndex, err := utils.ParsePodIndexResourceClaim(pod)
			wantIndex := tt.wantIndex
			if err != nil || gotIndex != wantIndex {
				t.Fatalf("ParsePodIndexResourceClaim() = %d, %v; want %d", gotIndex, err, wantIndex)
			}
		})
	}

}
