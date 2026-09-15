package utils

import (
	"encoding/json"
	"fmt"
	"maps"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

var sharedLegacyResourceKeys = []string{
	constants.TFLOPSRequestAnnotation, constants.TFLOPSLimitAnnotation,
	constants.VRAMRequestAnnotation, constants.VRAMLimitAnnotation,
	constants.ComputeRequestAnnotation, constants.ComputeLimitAnnotation,
}

// ApplySharedLegacyResources persists whole-card occupancy in annotations that
// v1 understands. v2 reads the saved requests for quota/metrics accounting.
func ApplySharedLegacyResources(pod *corev1.Pod, gpus []*tfv1.GPU) error {
	if len(gpus) == 0 {
		return fmt.Errorf("shared allocation has no GPUs")
	}
	maxTflops, maxVram := resource.Quantity{}, resource.Quantity{}
	for _, gpu := range gpus {
		if gpu == nil || gpu.Status.Capacity == nil ||
			gpu.Status.Capacity.Tflops.Sign() <= 0 || gpu.Status.Capacity.Vram.Sign() <= 0 {
			return fmt.Errorf("shared allocation requires positive capacity on every GPU")
		}
		if gpu.Status.Capacity.Tflops.Cmp(maxTflops) > 0 {
			maxTflops = gpu.Status.Capacity.Tflops.DeepCopy()
		}
		if gpu.Status.Capacity.Vram.Cmp(maxVram) > 0 {
			maxVram = gpu.Status.Capacity.Vram.DeepCopy()
		}
	}
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	if _, exists := pod.Annotations[constants.SharedLegacyResourcesAnnotation]; exists {
		if _, err := sharedOriginalResourceAnnotations(pod.Annotations); err != nil {
			return err
		}
	} else {
		original := map[string]string{}
		for _, key := range sharedLegacyResourceKeys {
			if value, exists := pod.Annotations[key]; exists {
				original[key] = value
			}
		}
		encoded, err := json.Marshal(original)
		if err != nil {
			return err
		}
		pod.Annotations[constants.SharedLegacyResourcesAnnotation] = string(encoded)
	}
	// v1 applies the same request to every selected GPU. The maximum capacity
	// conservatively reserves heterogeneous cards (smaller cards may have a
	// negative v1 Available until release). v1 Dealloc adds back the same request.
	pod.Annotations[constants.TFLOPSRequestAnnotation] = maxTflops.String()
	pod.Annotations[constants.TFLOPSLimitAnnotation] = maxTflops.String()
	pod.Annotations[constants.VRAMRequestAnnotation] = maxVram.String()
	pod.Annotations[constants.VRAMLimitAnnotation] = maxVram.String()
	delete(pod.Annotations, constants.ComputeRequestAnnotation)
	delete(pod.Annotations, constants.ComputeLimitAnnotation)
	return nil
}

func sharedOriginalResourceAnnotations(annotations map[string]string) (map[string]string, error) {
	var original map[string]string
	err := json.Unmarshal([]byte(annotations[constants.SharedLegacyResourcesAnnotation]), &original)
	if err != nil || original == nil {
		return nil, fmt.Errorf("invalid shared legacy resource backup")
	}
	return original, nil
}

// RestoreSharedLegacyResources is used for unbound retries and terminal Pods;
// a running or terminating worker must retain its legacy occupancy record.
func RestoreSharedLegacyResources(pod *corev1.Pod) (bool, error) {
	if _, exists := pod.Annotations[constants.SharedLegacyResourcesAnnotation]; !exists {
		return false, nil
	}
	original, err := sharedOriginalResourceAnnotations(pod.Annotations)
	if err != nil {
		return false, err
	}
	for _, key := range sharedLegacyResourceKeys {
		if value, exists := original[key]; exists {
			pod.Annotations[key] = value
		} else {
			delete(pod.Annotations, key)
		}
	}
	delete(pod.Annotations, constants.SharedLegacyResourcesAnnotation)
	return true, nil
}

// SharedLegacyResourceUpdates routes v2 resource adjustments into the backup
// while leaving the full-card legacy occupancy annotations intact.
func SharedLegacyResourceUpdates(pod *corev1.Pod, updates map[string]string) (map[string]string, error) {
	if _, exists := pod.Annotations[constants.SharedLegacyResourcesAnnotation]; !exists {
		return updates, nil
	}
	original, err := sharedOriginalResourceAnnotations(pod.Annotations)
	if err != nil {
		return nil, err
	}
	maps.Copy(original, updates)
	encoded, err := json.Marshal(original)
	if err != nil {
		return nil, err
	}
	return map[string]string{constants.SharedLegacyResourcesAnnotation: string(encoded)}, nil
}
