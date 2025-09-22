/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

// DRAProcessor handles all DRA-related operations for pod admission
type DRAProcessor struct {
	client.Client
	enableDRA     bool
	resourceClass string // cached resource class to avoid repeated API calls
	configLoaded  bool   // tracks if configuration has been loaded
}

// generateUniqueID creates a random 8-character hex string for resource claim names
func generateUniqueID() string {
	bytes := make([]byte, 4)
	_, _ = rand.Read(bytes) // crypto/rand.Read always returns len(bytes), nil on success
	return hex.EncodeToString(bytes)
}

// NewDRAProcessor creates a new DRA processor
func NewDRAProcessor(client client.Client) *DRAProcessor {
	return &DRAProcessor{
		Client:    client,
		enableDRA: false,
	}
}

// InitializeDRAConfig is kept for backward compatibility but now does nothing
// Configuration is loaded lazily on first use
func (p *DRAProcessor) InitializeDRAConfig(ctx context.Context) error {
	// No-op - configuration is now loaded lazily
	if p.configLoaded {
		return nil
	}

	// Set defaults first
	p.enableDRA = false

	templateList := &tfv1.SchedulingConfigTemplateList{}
	// Use the provided context to respect cancellation
	err := p.List(ctx, templateList)
	if err != nil {
		// Log error but don't fail - fall back to defaults
		// This allows webhook to work even if templates are unavailable
		p.configLoaded = true
		return nil
	}

	// Check if any template has DRA enabled and cache the resource class
	for _, template := range templateList.Items {
		if template.Spec.DRA != nil {
			if template.Spec.DRA.Enable != nil && *template.Spec.DRA.Enable {
				p.enableDRA = true
			}
			// Cache the resource class from the template
			if template.Spec.DRA.ResourceClass != "" {
				p.resourceClass = template.Spec.DRA.ResourceClass
			}
		}
	}

	if p.enableDRA && p.resourceClass == "" {
		return fmt.Errorf("resource class is not set")
	}

	p.configLoaded = true
	return nil
}

// IsDRAEnabled checks if DRA is enabled for a specific pod
func (p *DRAProcessor) IsDRAEnabled(ctx context.Context, pod *corev1.Pod) bool {

	// Check pod-level annotation first (explicit override)
	if val, ok := pod.Annotations[constants.DRAEnabledAnnotation]; ok && val == constants.TrueStringValue {
		return true
	}

	// Check pod-level annotation for explicit disable
	if val, ok := pod.Annotations[constants.DRAEnabledAnnotation]; ok && val == constants.FalseStringValue {
		return false
	}

	// Fall back to global configuration
	return p.enableDRA
}

// HasDRAClaim checks if a pod has DRA ResourceClaim references
func HasDRAClaim(pod *corev1.Pod) bool {
	return len(pod.Spec.ResourceClaims) > 0
}

// convertToResourceClaim converts GPU resource requests to ResourceClaim
func (p *DRAProcessor) convertToResourceClaim(pod *corev1.Pod, tfInfo *utils.TensorFusionInfo) (*resourcev1beta2.ResourceClaim, error) {

	// Build CEL selector using DRA helper
	celSelector, err := BuildCELSelector(pod, tfInfo)
	if err != nil {
		return nil, fmt.Errorf("failed to build CEL selector: %w", err)
	}

	// Generate unique claim name with random suffix to avoid conflicts
	var baseName string

	if pod.GenerateName != "" {
		baseName = strings.TrimSuffix(pod.GenerateName, "-")
	} else if pod.Name != "" {
		baseName = pod.Name
	}

	uniqueID := generateUniqueID()
	claimName := fmt.Sprintf(constants.DRAResourceClaimName, baseName, uniqueID)

	// Use cached resource class instead of making API calls
	resourceClass := p.resourceClass

	claim := &resourcev1beta2.ResourceClaim{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "resource.k8s.io/v1beta2",
			Kind:       "ResourceClaim",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      claimName,
			Namespace: pod.Namespace,
			// Note: We cannot set OwnerReference here because pod.UID is empty during admission.
			// The controller will set the proper owner reference once the Pod is created.
		},
		Spec: resourcev1beta2.ResourceClaimSpec{
			Devices: resourcev1beta2.DeviceClaim{
				Requests: []resourcev1beta2.DeviceRequest{
					{
						Name: fmt.Sprintf(constants.DRAResourceClaimRequestName, generateUniqueID()),
						Exactly: &resourcev1beta2.ExactDeviceRequest{
							DeviceClassName: resourceClass,
							Selectors: []resourcev1beta2.DeviceSelector{
								{
									CEL: &resourcev1beta2.CELDeviceSelector{
										Expression: celSelector,
									},
								},
							},
							Count: int64(tfInfo.Profile.GPUCount),
						},
					},
				},
			},
		},
	}

	return claim, nil
}

// injectResourceClaimRef adds ResourceClaim reference to Pod spec
func (p *DRAProcessor) injectResourceClaimRef(pod *corev1.Pod, claim *resourcev1beta2.ResourceClaim, containerIndices []int) {
	// Add ResourceClaim reference to pod.Spec.ResourceClaims
	if pod.Spec.ResourceClaims == nil {
		pod.Spec.ResourceClaims = []corev1.PodResourceClaim{}
	}

	claimRef := corev1.PodResourceClaim{
		Name:              constants.DRAClaimDefineName,
		ResourceClaimName: &claim.Name,
	}

	// Check if the claim reference already exists to maintain idempotency
	claimExists := false
	for i, existingClaim := range pod.Spec.ResourceClaims {
		if existingClaim.Name == constants.DRAClaimDefineName {
			// Update existing claim to point to the new ResourceClaim name
			pod.Spec.ResourceClaims[i].ResourceClaimName = &claim.Name
			claimExists = true
			break
		}
	}

	if !claimExists {
		pod.Spec.ResourceClaims = append(pod.Spec.ResourceClaims, claimRef)
	}

	// Add resource claim consumption to containers
	for _, containerIndex := range containerIndices {
		container := &pod.Spec.Containers[containerIndex]
		if container.Resources.Claims == nil {
			container.Resources.Claims = []corev1.ResourceClaim{}
		}

		// Check if the container already has this claim to maintain idempotency
		hasGPUClaim := false
		for _, existingClaim := range container.Resources.Claims {
			if existingClaim.Name == constants.DRAClaimDefineName {
				hasGPUClaim = true
				break
			}
		}

		if !hasGPUClaim {
			container.Resources.Claims = append(container.Resources.Claims, corev1.ResourceClaim{
				Name: constants.DRAClaimDefineName,
			})
		}
	}
}

// createResourceClaim creates a ResourceClaim object with proper error handling and retries
func (p *DRAProcessor) createResourceClaim(ctx context.Context, claim *resourcev1beta2.ResourceClaim) error {
	// Try to create the ResourceClaim
	if err := p.Create(ctx, claim); err != nil {
		if errors.IsAlreadyExists(err) {
			// Check if the existing claim is for the same pod
			existingClaim := &resourcev1beta2.ResourceClaim{}
			getErr := p.Get(ctx, client.ObjectKey{Name: claim.Name, Namespace: claim.Namespace}, existingClaim)
			if getErr != nil {
				return fmt.Errorf("failed to check existing ResourceClaim: %w", getErr)
			}
			// Different pod or missing labels, this is an error
			return fmt.Errorf("ResourceClaim %s already exists for a different pod", claim.Name)
		}

		if errors.IsInvalid(err) {
			return fmt.Errorf("ResourceClaim is invalid: %w", err)
		}

		if errors.IsForbidden(err) {
			return fmt.Errorf("insufficient permissions to create ResourceClaim: %w", err)
		}
	}

	return nil
}

// Note: patchTFClientForDRA is temporarily handled in the main pod_webhook.go
// until we can properly abstract all the TF client patching logic

// HandleDRAAdmission handles the complete DRA admission process
func (p *DRAProcessor) HandleDRAAdmission(ctx context.Context, pod *corev1.Pod, tfInfo *utils.TensorFusionInfo, containerIndices []int) error {
	// Convert GPU resources to ResourceClaim
	resourceClaim, err := p.convertToResourceClaim(pod, tfInfo)
	if err != nil {
		return fmt.Errorf("failed to convert to ResourceClaim: %w", err)
	}

	// Create ResourceClaim
	if err := p.createResourceClaim(ctx, resourceClaim); err != nil {
		return fmt.Errorf("failed to create ResourceClaim: %w", err)
	}
	// Inject ResourceClaim reference to Pod
	p.injectResourceClaimRef(pod, resourceClaim, containerIndices)
	return nil
}

// TODO: support more attributes for filtering
func BuildCELSelector(pod *corev1.Pod, tfInfo *utils.TensorFusionInfo) (string, error) {
	var conditions []string

	// 1. Basic resource requirements using standard DRA quantity attributes
	requests := tfInfo.Profile.Resources.Requests
	if !requests.Tflops.IsZero() {
		conditions = append(conditions, fmt.Sprintf(`device.attributes["tflops"].quantity >= quantity("%s")`, requests.Tflops.String()))
	}
	if !requests.Vram.IsZero() {
		conditions = append(conditions, fmt.Sprintf(`device.attributes["vram"].quantity >= quantity("%s")`, requests.Vram.String()))
	}

	// 2. GPU model filter (if specified - basic attribute that should be widely supported)
	if tfInfo.Profile.GPUModel != "" {
		conditions = append(conditions, fmt.Sprintf(`device.attributes["model"] == "%s"`, tfInfo.Profile.GPUModel))
	}

	// Return a basic condition if no specific requirements
	if len(conditions) == 0 {
		// Simple condition that should work with most DRA drivers
		return `device.attributes.exists("type")`, nil
	}

	return strings.Join(conditions, " && "), nil
}
