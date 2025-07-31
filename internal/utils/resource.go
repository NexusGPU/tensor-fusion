package utils

import (
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"k8s.io/apimachinery/pkg/api/resource"
)

func CurrentResourcesFromAnnotations(annotations map[string]string) (*tfv1.Resources, error) {
	result := tfv1.Resources{}
	resInfo := []struct {
		key string
		dst *resource.Quantity
	}{
		{constants.TFLOPSRequestAnnotation, &result.Requests.Tflops},
		{constants.TFLOPSLimitAnnotation, &result.Limits.Tflops},
		{constants.VRAMRequestAnnotation, &result.Requests.Vram},
		{constants.VRAMLimitAnnotation, &result.Limits.Vram},
	}
	for _, info := range resInfo {
		annotation, ok := annotations[info.key]
		if !ok {
			continue
		}
		q, err := resource.ParseQuantity(annotation)
		if err != nil {
			return nil, fmt.Errorf("failed to parse %s: %v", info.key, err)
		}
		*info.dst = q
	}

	if result.IsZero() {
		return nil, nil
	}

	return &result, nil
}

func CurrentResourcesToAnnotations(resources *tfv1.Resources) map[string]string {
	return map[string]string{
		constants.TFLOPSRequestAnnotation: resources.Requests.Tflops.String(),
		constants.TFLOPSLimitAnnotation:   resources.Limits.Tflops.String(),
		constants.VRAMRequestAnnotation:   resources.Requests.Vram.String(),
		constants.VRAMLimitAnnotation:     resources.Limits.Vram.String(),
	}
}
