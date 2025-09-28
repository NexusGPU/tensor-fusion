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
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

// DRAProcessor handles all DRA-related operations for pod admission
type DRAProcessor struct {
	client.Client
	enableDRA                 bool
	resourceClaimTemplateName string // cached ResourceClaimTemplate name
	configLoaded              bool   // tracks if configuration has been loaded
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
	p.resourceClaimTemplateName = constants.DRAResourceClaimTemplateName

	templateList := &tfv1.SchedulingConfigTemplateList{}
	// Use the provided context to respect cancellation
	err := p.List(ctx, templateList)
	if err != nil {
		// Log error but don't fail - fall back to defaults
		// This allows webhook to work even if templates are unavailable
		p.configLoaded = true
		return nil
	}

	// Check if any template has DRA enabled and cache the ResourceClaimTemplateName
	for _, template := range templateList.Items {
		if template.Spec.DRA != nil {
			if template.Spec.DRA.Enable != nil && *template.Spec.DRA.Enable {
				p.enableDRA = true
			}
			// Cache the ResourceClaimTemplateName from the template
			if template.Spec.DRA.ResourceClaimTemplateName != "" {
				p.resourceClaimTemplateName = template.Spec.DRA.ResourceClaimTemplateName
			}
		}
	}

	p.configLoaded = true
	return nil
}

// IsDRAEnabled checks if DRA is enabled for a specific pod
func (p *DRAProcessor) IsDRAEnabled(ctx context.Context, pod *corev1.Pod) bool {
	// Load configuration if not yet loaded (lazy loading)
	if !p.configLoaded {
		_ = p.InitializeDRAConfig(ctx) // Ignore error to maintain backward compatibility
	}

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

// HandleDRAAdmission handles the complete DRA admission process
func (p *DRAProcessor) HandleDRAAdmission(ctx context.Context, pod *corev1.Pod, tfInfo *utils.TensorFusionInfo, containerIndices []int) error {
	// Load DRA configuration if needed
	if err := p.InitializeDRAConfig(ctx); err != nil {
		return fmt.Errorf("failed to load DRA config: %w", err)
	}

	// Convert GPU resources to ResourceClaimTemplate reference and store CEL in annotation
	celSelector, err := BuildCELSelector(pod, tfInfo)
	if err != nil {
		return fmt.Errorf("failed to build CEL selector: %w", err)
	}

	// Inject ResourceClaimTemplate reference to Pod
	p.injectResourceClaimTemplateRef(pod)

	// Mark pod with DRA enabled annotation
	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}
	pod.Annotations[constants.DRAEnabledAnnotation] = constants.TrueStringValue
	pod.Annotations[constants.DRACelExpressionAnnotation] = celSelector

	return nil
}

// BuildCELSelector constructs a CEL expression for DRA device selection based on TensorFusion requirements
func BuildCELSelector(pod *corev1.Pod, tfInfo *utils.TensorFusionInfo) (string, error) {
	var conditions []string

	// 1. GPU model filter (if specified - basic attribute that should be widely supported)
	if tfInfo.Profile.GPUModel != "" {
		conditions = append(conditions, fmt.Sprintf(`device.attributes["model"] == "%s"`, tfInfo.Profile.GPUModel))
	}

	// 2. GPU count requirement (important for multi-GPU workloads)
	if tfInfo.Profile.GPUCount > 0 {
		conditions = append(conditions, fmt.Sprintf(`size(devices) >= %d`, tfInfo.Profile.GPUCount))
	}

	// 3. Pool name filter (for resource isolation and scheduling preferences)
	if tfInfo.Profile.PoolName != "" {
		conditions = append(conditions, fmt.Sprintf(`device.attributes["pool_name"] == "%s"`, tfInfo.Profile.PoolName))
	}

	// 4. Pod namespace filter (for namespace-based device isolation)
	if pod.Namespace != "" {
		conditions = append(conditions, fmt.Sprintf(`device.attributes["pod_namespace"] == "%s"`, pod.Namespace))
	}

	// Return a basic condition if no specific requirements
	if len(conditions) == 0 {
		// Simple condition that should work with most DRA drivers
		return `device.attributes.exists("type")`, nil
	}

	return strings.Join(conditions, " && "), nil
}

// injectResourceClaimTemplateRef adds ResourceClaimTemplate reference to Pod spec
func (p *DRAProcessor) injectResourceClaimTemplateRef(pod *corev1.Pod) {
	// Add ResourceClaimTemplate reference to pod.Spec.ResourceClaims
	if pod.Spec.ResourceClaims == nil {
		pod.Spec.ResourceClaims = []corev1.PodResourceClaim{}
	}

	// Use ResourceClaimTemplate instead of direct ResourceClaim
	claimRef := corev1.PodResourceClaim{
		Name:                      constants.DRAClaimDefineName,
		ResourceClaimTemplateName: &p.resourceClaimTemplateName,
	}

	// Check if the claim reference already exists to maintain idempotency
	claimExists := false
	for _, existingClaim := range pod.Spec.ResourceClaims {
		if existingClaim.Name == constants.DRAClaimDefineName {
			claimExists = true
			break
		}
	}

	if !claimExists {
		pod.Spec.ResourceClaims = append(pod.Spec.ResourceClaims, claimRef)
	}
}
