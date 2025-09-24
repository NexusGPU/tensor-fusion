package gpuresources

import (
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

func TestIsDRAEnabled(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		expected    bool
	}{
		{
			name: "DRA enabled annotation",
			annotations: map[string]string{
				constants.DRAEnabledAnnotation: constants.TrueStringValue,
			},
			expected: true,
		},
		{
			name: "DRA disabled annotation",
			annotations: map[string]string{
				constants.DRAEnabledAnnotation: constants.FalseStringValue,
			},
			expected: false,
		},
		{
			name:     "no annotation",
			expected: false,
		},
		{
			name: "other annotations",
			annotations: map[string]string{
				"other.annotation": "value",
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: tt.annotations,
				},
			}

			result := isDRAEnabled(pod)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestHasDRAClaimScheduler(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected bool
	}{
		{
			name: "pod with resource claims",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					ResourceClaims: []corev1.PodResourceClaim{
						{Name: "gpu-claim"},
					},
				},
			},
			expected: true,
		},
		{
			name: "pod with multiple resource claims",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					ResourceClaims: []corev1.PodResourceClaim{
						{Name: "gpu-claim"},
						{Name: "other-claim"},
					},
				},
			},
			expected: true,
		},
		{
			name: "pod without resource claims",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{},
			},
			expected: false,
		},
		{
			name: "pod with empty resource claims",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					ResourceClaims: []corev1.PodResourceClaim{},
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := hasDRAClaim(tt.pod)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// Integration test for DRA detection logic
func TestDRADetectionIntegration(t *testing.T) {
	tests := []struct {
		name              string
		draAnnotation     string
		hasResourceClaims bool
		expectedDRA       bool
		expectedClaim     bool
	}{
		{
			name:              "DRA enabled with claims",
			draAnnotation:     constants.TrueStringValue,
			hasResourceClaims: true,
			expectedDRA:       true,
			expectedClaim:     true,
		},
		{
			name:              "DRA enabled without claims",
			draAnnotation:     constants.TrueStringValue,
			hasResourceClaims: false,
			expectedDRA:       true,
			expectedClaim:     false,
		},
		{
			name:              "DRA disabled with claims",
			draAnnotation:     constants.FalseStringValue,
			hasResourceClaims: true,
			expectedDRA:       false,
			expectedClaim:     true,
		},
		{
			name:              "no DRA annotation, no claims",
			hasResourceClaims: false,
			expectedDRA:       false,
			expectedClaim:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: make(map[string]string),
				},
				Spec: corev1.PodSpec{},
			}

			if tt.draAnnotation != "" {
				pod.Annotations[constants.DRAEnabledAnnotation] = tt.draAnnotation
			}

			if tt.hasResourceClaims {
				pod.Spec.ResourceClaims = []corev1.PodResourceClaim{
					{Name: "test-claim"},
				}
			}

			draEnabled := isDRAEnabled(pod)
			hasClaim := hasDRAClaim(pod)

			assert.Equal(t, tt.expectedDRA, draEnabled, "DRA enabled detection mismatch")
			assert.Equal(t, tt.expectedClaim, hasClaim, "Resource claim detection mismatch")
		})
	}
}

// Test the combination logic that scheduler uses
func TestSchedulerDRALogic(t *testing.T) {
	tests := []struct {
		name                string
		draAnnotation       string
		hasResourceClaims   bool
		shouldSkipScheduler bool
	}{
		{
			name:                "DRA enabled with claims - should skip",
			draAnnotation:       constants.TrueStringValue,
			hasResourceClaims:   true,
			shouldSkipScheduler: true,
		},
		{
			name:                "DRA enabled without claims - should not skip",
			draAnnotation:       constants.TrueStringValue,
			hasResourceClaims:   false,
			shouldSkipScheduler: false,
		},
		{
			name:                "DRA disabled with claims - should not skip",
			draAnnotation:       constants.FalseStringValue,
			hasResourceClaims:   true,
			shouldSkipScheduler: false,
		},
		{
			name:                "no DRA, no claims - should not skip",
			shouldSkipScheduler: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: make(map[string]string),
				},
				Spec: corev1.PodSpec{},
			}

			if tt.draAnnotation != "" {
				pod.Annotations[constants.DRAEnabledAnnotation] = tt.draAnnotation
			}

			if tt.hasResourceClaims {
				pod.Spec.ResourceClaims = []corev1.PodResourceClaim{
					{Name: "test-claim"},
				}
			}

			// This is the actual logic used in the scheduler
			shouldSkip := isDRAEnabled(pod) && hasDRAClaim(pod)
			assert.Equal(t, tt.shouldSkipScheduler, shouldSkip)
		})
	}
}
