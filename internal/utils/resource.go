package utils

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/samber/lo"
	"k8s.io/apimachinery/pkg/api/resource"
	ctrl "sigs.k8s.io/controller-runtime"
)

func GPUResourcesFromAnnotations(annotations map[string]string) (*tfv1.Resources, error) {
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
			// Should not happen
			return nil, fmt.Errorf("missing gpu resource annotation %q", info.key)
		}
		q, err := resource.ParseQuantity(annotation)
		if err != nil {
			return nil, fmt.Errorf("failed to parse %q: %v", info.key, err)
		}
		*info.dst = q
	}

	return &result, nil
}

func GPUResourcesToAnnotations(resources *tfv1.Resources) map[string]string {
	return map[string]string{
		constants.TFLOPSRequestAnnotation: resources.Requests.Tflops.String(),
		constants.TFLOPSLimitAnnotation:   resources.Limits.Tflops.String(),
		constants.VRAMRequestAnnotation:   resources.Requests.Vram.String(),
		constants.VRAMLimitAnnotation:     resources.Limits.Vram.String(),
	}
}

func ComputePercentToTflops(gpuCapacity resource.Quantity, gpuResRequest tfv1.Resource) *resource.Quantity {
	requiredTflops := gpuResRequest.ComputePercent.AsApproximateFloat64() * gpuCapacity.AsApproximateFloat64() / 100
	return resource.NewQuantity(int64(requiredTflops), resource.DecimalSI)
}

// GetActualTflops returns the actual TFLOPs value from a Resource.
// If ComputePercent is set, it converts it to TFLOPs using the provided GPU capacity.
// Otherwise, it returns the Tflops value directly.
// Returns nil if both Tflops and ComputePercent are zero or if ComputePercent is set but gpuCapacity is zero.
func GetActualTflops(gpuCapacity resource.Quantity, res tfv1.Resource) *resource.Quantity {
	if !res.ComputePercent.IsZero() {
		if gpuCapacity.IsZero() {
			return nil
		}
		return ComputePercentToTflops(gpuCapacity, res)
	}
	if res.Tflops.IsZero() {
		return nil
	}
	result := res.Tflops.DeepCopy()
	return &result
}

// GetActualTflopsFloat64 returns the actual TFLOPs value as float64 from a Resource.
// Similar to GetActualTflops but returns float64 for precision-sensitive use cases like metrics.
// Returns 0 if both Tflops and ComputePercent are zero, or if ComputePercent is set but gpuCapacity is zero.
func GetActualTflopsFloat64(gpuCapacity resource.Quantity, res tfv1.Resource) float64 {
	if !res.ComputePercent.IsZero() {
		if gpuCapacity.IsZero() {
			return 0
		}
		return res.ComputePercent.AsApproximateFloat64() * gpuCapacity.AsApproximateFloat64() / 100
	}
	return res.Tflops.AsApproximateFloat64()
}

func ParseIndicesAnnotation(gpuIndicesStr string) ([]int32, bool) {
	if gpuIndicesStr == "" {
		return nil, false
	}
	gpuIndices := lo.Map(slices.Collect(strings.SplitSeq(gpuIndicesStr, ",")), func(index string, _ int) int32 {
		indexInt, err := strconv.Atoi(strings.TrimSpace(index))
		if err != nil {
			ctrl.Log.Error(err, "Invalid GPU index annotation", "index", index)
			return 0
		}
		if indexInt < math.MinInt32 || indexInt > math.MaxInt32 {
			ctrl.Log.Error(fmt.Errorf("out of range int32"), "Invalid GPU index range", "index", indexInt)
			return 0
		}
		return int32(indexInt)
	})
	return gpuIndices, false
}
