package utils

import (
	context "context"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const MaxGPUCounterPerAllocation = 128

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

func ComposeAllocationRequest(ctx context.Context, pod *corev1.Pod) (*tfv1.AllocRequest, string, error) {
	// allow Pods with no requests/limits to use TensorFusion, Pod webhook will ensure at least one request/limit is set
	gpuRequestResource, err := GetGPUResource(pod, true)
	if err != nil {
		log.FromContext(ctx).Error(err, "Invalid gpu request annotation", "pod", pod.Name, "namespace", pod.Namespace)
	}
	gpuLimitResource, err := GetGPUResource(pod, false)
	if err != nil {
		log.FromContext(ctx).Error(err, "Invalid gpu limit annotation", "pod", pod.Name, "namespace", pod.Namespace)
	}

	count := 1
	if gpuCountStr, exists := pod.Annotations[constants.GpuCountAnnotation]; exists {
		count, err = strconv.Atoi(gpuCountStr)
		if err != nil {
			return &tfv1.AllocRequest{}, "invalid gpu count annotation", err
		}
	}
	if count > MaxGPUCounterPerAllocation {
		return &tfv1.AllocRequest{}, "gpu count annotation is too large", nil
	}

	qosLevel := tfv1.QoSLevel(pod.Annotations[constants.QoSLevelAnnotation])
	if qosLevel == "" {
		qosLevel = tfv1.QoSMedium
	}

	gpuVendor := pod.Annotations[constants.GpuVendorAnnotation]

	gpuIndices, hasError := ParseIndicesAnnotation(pod.Annotations[constants.GpuIndicesAnnotation])
	if hasError {
		return &tfv1.AllocRequest{}, "invalid gpu-indices annotation",
			fmt.Errorf("can not parse gpu indices annotation")
	}

	// Read isolation mode
	isolationMode := tfv1.IsolationModeType(pod.Annotations[constants.IsolationModeAnnotation])
	if isolationMode == "" {
		isolationMode = tfv1.IsolationModeSoft
	}

	allocRequest := tfv1.AllocRequest{
		PoolName: pod.Annotations[constants.GpuPoolKey],
		Request:  gpuRequestResource,
		Limit:    gpuLimitResource,

		Count:      uint(count),
		GPUModel:   pod.Annotations[constants.GPUModelAnnotation],
		GPUIndices: gpuIndices,
		GPUVendor:  gpuVendor,
		Isolation:  isolationMode,
		WorkloadNameNamespace: tfv1.NameNamespace{
			Name:      pod.Labels[constants.WorkloadKey],
			Namespace: pod.Namespace,
		},
		PodMeta: pod.ObjectMeta,
		QoS:     qosLevel,
	}

	// Read partition template ID annotation if in partitioned mode
	if allocRequest.Isolation == tfv1.IsolationModePartitioned {
		if partitionTemplateID, ok := pod.Annotations[constants.PartitionTemplateIDAnnotation]; ok && partitionTemplateID != "" {
			allocRequest.PartitionTemplateID = partitionTemplateID
		}
	}

	// for already allocated workers, set the GPU device IDs for further scaling and retrieval
	if gpuIdStr, exists := pod.Annotations[constants.GPUDeviceIDsAnnotation]; exists {
		gpuIds := strings.SplitSeq(gpuIdStr, ",")
		allocRequest.GPUNames = slices.Collect(gpuIds)
	}

	return &allocRequest, "", nil
}

func ParsePodIndexResourceClaim(pod *corev1.Pod) (int, error) {
	for _, container := range pod.Spec.Containers {
		for indexKey, indexValue := range container.Resources.Limits {
			if strings.HasPrefix(string(indexKey), constants.PodIndexAnnotation+constants.PodIndexDelimiter) {
				indexStr := strings.Split(string(indexKey), constants.PodIndexDelimiter)[1]
				indexInt, err := strconv.ParseInt(indexStr, 16, 64)
				if err != nil {
					return 0, fmt.Errorf("failed to parse tensor fusion index of Pod resource limits: %v", err)
				}
				if indexInt < 0 || indexInt >= constants.IndexKeyLength {
					return 0, fmt.Errorf("tensor fusion index of Pod resource limits out of range: %d", indexInt)
				}
				return int(indexValue.Value()) + int(indexInt)*constants.IndexModLength, nil
			}
		}
	}
	return 0, fmt.Errorf("tensor fusion index of Pod resource limits is missing in any container")
}
