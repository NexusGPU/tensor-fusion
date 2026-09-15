package utils

import (
	"maps"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestSharedLegacyResourcesPreservesV2Accounting(t *testing.T) {
	for _, compute := range []string{"0", "10"} {
		t.Run(compute, func(t *testing.T) {
			pod := &corev1.Pod{}
			pod.Annotations = map[string]string{
				constants.IsolationModeAnnotation:  tfv1.IsolationModeShared,
				constants.ComputeRequestAnnotation: compute,
				constants.ComputeLimitAnnotation:   "50",
				constants.VRAMRequestAnnotation:    "2Gi",
				constants.VRAMLimitAnnotation:      "4Gi",
			}
			original := maps.Clone(pod.Annotations)
			request, err := GetGPUResource(pod, true)
			require.NoError(t, err)
			limit, err := GetGPUResource(pod, false)
			require.NoError(t, err)
			gpus := []*tfv1.GPU{{Status: tfv1.GPUStatus{Capacity: &tfv1.Resource{
				Tflops: resource.MustParse("100"), Vram: resource.MustParse("24Gi"),
			}}}}
			require.NoError(t, ApplySharedLegacyResources(pod, gpus))
			persisted := pod.DeepCopy()
			require.NoError(t, ApplySharedLegacyResources(pod, gpus))
			require.Equal(t, persisted, pod, "reconcile and scheduling retries must be idempotent")
			gotRequest, err := GetGPUResource(pod, true)
			require.NoError(t, err)
			gotLimit, err := GetGPUResource(pod, false)
			require.NoError(t, err)
			require.Equal(t, request, gotRequest)
			require.Equal(t, limit, gotLimit)
			require.Equal(t, persisted, pod, "resource reads must not mutate the Pod")
			_, err = RestoreSharedLegacyResources(pod)
			require.NoError(t, err)
			require.Equal(t, original, pod.Annotations)
		})
	}
}

func TestSharedLegacyResourcesAutoscalingKeepsOccupancy(t *testing.T) {
	pod := &corev1.Pod{}
	pod.Annotations = map[string]string{constants.IsolationModeAnnotation: tfv1.IsolationModeShared,
		constants.TFLOPSRequestAnnotation: "0", constants.VRAMRequestAnnotation: "0",
		constants.TFLOPSLimitAnnotation: "10", constants.VRAMLimitAnnotation: "4Gi"}
	gpus := []*tfv1.GPU{{Status: tfv1.GPUStatus{Capacity: &tfv1.Resource{
		Tflops: resource.MustParse("100"), Vram: resource.MustParse("24Gi"),
	}}}}
	require.NoError(t, ApplySharedLegacyResources(pod, gpus))
	updates, err := SharedLegacyResourceUpdates(pod, map[string]string{
		constants.TFLOPSRequestAnnotation: "5", constants.VRAMRequestAnnotation: "2Gi",
	})
	require.NoError(t, err)
	maps.Copy(pod.Annotations, updates)
	require.Equal(t, "100", pod.Annotations[constants.TFLOPSRequestAnnotation])
	require.Equal(t, "24Gi", pod.Annotations[constants.VRAMRequestAnnotation])
	request, err := GetGPUResource(pod, true)
	require.NoError(t, err)
	require.Equal(t, "5", request.Tflops.String())
	require.Equal(t, "2Gi", request.Vram.String())
	resources, err := GPUResourcesFromAnnotations(pod.Annotations)
	require.NoError(t, err)
	require.Equal(t, "5", resources.Requests.Tflops.String())
	require.Equal(t, "2Gi", resources.Requests.Vram.String())
	require.Equal(t, "10", resources.Limits.Tflops.String())
	require.Equal(t, "4Gi", resources.Limits.Vram.String())
}

func TestSharedLegacyResourcesRejectsMissingCapacityAndCorruptBackup(t *testing.T) {
	pod := &corev1.Pod{}
	require.Error(t, ApplySharedLegacyResources(pod, nil))
	require.Error(t, ApplySharedLegacyResources(pod, []*tfv1.GPU{{}}))
	pod.Annotations = map[string]string{constants.SharedLegacyResourcesAnnotation: "null"}
	_, err := RestoreSharedLegacyResources(pod)
	require.Error(t, err)
	require.Equal(t, "null", pod.Annotations[constants.SharedLegacyResourcesAnnotation])
}

func TestSharedLegacyResourcesRoundTrip(t *testing.T) {
	pod := &corev1.Pod{}
	pod.Annotations = map[string]string{
		constants.TFLOPSRequestAnnotation:  "2",
		constants.TFLOPSLimitAnnotation:    "3",
		constants.VRAMRequestAnnotation:    "4Gi",
		constants.VRAMLimitAnnotation:      "5Gi",
		constants.ComputeRequestAnnotation: "10",
		constants.ComputeLimitAnnotation:   "20",
	}
	gpus := []*tfv1.GPU{{Status: tfv1.GPUStatus{Capacity: &tfv1.Resource{Tflops: resource.MustParse("100"), Vram: resource.MustParse("24Gi")}}}}

	if err := ApplySharedLegacyResources(pod, gpus); err != nil {
		t.Fatal(err)
	}
	if got := pod.Annotations[constants.TFLOPSRequestAnnotation]; got != "100" {
		t.Fatalf("tflops request = %q, want 100", got)
	}
	if got := pod.Annotations[constants.VRAMRequestAnnotation]; got != "24Gi" {
		t.Fatalf("vram request = %q, want 24Gi", got)
	}
	if _, ok := pod.Annotations[constants.ComputeRequestAnnotation]; ok {
		t.Fatal("compute-percent request must be removed")
	}

	restored, err := RestoreSharedLegacyResources(pod)
	if err != nil {
		t.Fatal(err)
	}
	if !restored {
		t.Fatal("expected legacy resources to be restored")
	}
	if got := pod.Annotations[constants.TFLOPSRequestAnnotation]; got != "2" {
		t.Fatalf("restored tflops request = %q, want 2", got)
	}
	if got := pod.Annotations[constants.ComputeRequestAnnotation]; got != "10" {
		t.Fatalf("restored compute request = %q, want 10", got)
	}
	if _, ok := pod.Annotations[constants.SharedLegacyResourcesAnnotation]; ok {
		t.Fatal("backup annotation must be removed after restore")
	}
}

func TestSharedLegacyResourcesUsesLargestCapacity(t *testing.T) {
	pod := &corev1.Pod{}
	gpus := []*tfv1.GPU{
		{Status: tfv1.GPUStatus{Capacity: &tfv1.Resource{Tflops: resource.MustParse("80"), Vram: resource.MustParse("16Gi")}}},
		{Status: tfv1.GPUStatus{Capacity: &tfv1.Resource{Tflops: resource.MustParse("100"), Vram: resource.MustParse("24Gi")}}},
	}
	if err := ApplySharedLegacyResources(pod, gpus); err != nil {
		t.Fatal(err)
	}
	if got := pod.Annotations[constants.TFLOPSRequestAnnotation]; got != "100" {
		t.Fatalf("tflops request = %q, want largest capacity", got)
	}
	if got := pod.Annotations[constants.VRAMRequestAnnotation]; got != "24Gi" {
		t.Fatalf("vram request = %q, want largest capacity", got)
	}
}
