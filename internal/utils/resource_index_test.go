package utils

import (
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestParsePodIndexResourceClaimSupportsLegacyResource(t *testing.T) {
	pod := &corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{{
		Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
			corev1.ResourceName(constants.PodIndexAnnotation): resource.MustParse("512"),
		}},
	}}}}

	index, err := ParsePodIndexResourceClaim(pod)
	if err != nil {
		t.Fatalf("ParsePodIndexResourceClaim() error = %v", err)
	}
	if index != constants.LegacyIndexDeviceCount {
		t.Fatalf("index = %d, want %d", index, constants.LegacyIndexDeviceCount)
	}
}

func TestParsePodIndexResourceClaimPrefersV2ResourceWhenBothExist(t *testing.T) {
	pod := &corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{{
		Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
			corev1.ResourceName(constants.PodIndexAnnotation):                                     resource.MustParse("512"),
			corev1.ResourceName(constants.PodIndexAnnotation + constants.PodIndexDelimiter + "a"): resource.MustParse("3"),
		}},
	}}}}

	index, err := ParsePodIndexResourceClaim(pod)
	if err != nil {
		t.Fatalf("ParsePodIndexResourceClaim() error = %v", err)
	}
	want := 3 + 10*constants.IndexModLength
	if index != want {
		t.Fatalf("index = %d, want %d", index, want)
	}
}

func TestParsePodIndexResourceClaimRejectsInvalidLegacyResource(t *testing.T) {
	pod := &corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{{
		Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
			corev1.ResourceName(constants.PodIndexAnnotation): resource.MustParse("513"),
		}},
	}}}}

	if _, err := ParsePodIndexResourceClaim(pod); err == nil {
		t.Fatal("ParsePodIndexResourceClaim() error = nil, want out-of-range error")
	}
}
