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

// PoolDRAConfig holds DRA configuration for a specific pool
type PoolDRAConfig struct {
	EnableDRA                 bool
	ResourceClaimTemplateName string
}

// DRAProcessor handles all DRA-related operations for pod admission
type DRAProcessor struct {
	client.Client
	poolConfigs map[string]*PoolDRAConfig // cached configurations per pool
}

// NewDRAProcessor creates a new DRA processor
func NewDRAProcessor(client client.Client) *DRAProcessor {
	return &DRAProcessor{
		Client:      client,
		poolConfigs: make(map[string]*PoolDRAConfig),
	}
}

// InitializeDRAConfig loads DRA configuration from the specified GPUPool
// This provides pool-level DRA control instead of global configuration
// Configuration is cached per pool for performance
func (p *DRAProcessor) InitializeDRAConfig(ctx context.Context, poolName string) error {
	// If no poolName specified, return error
	if poolName == "" {
		return fmt.Errorf("poolName is required for DRA configuration")
	}

	// Check if configuration is already cached for this pool
	if _, exists := p.poolConfigs[poolName]; exists {
		return nil
	}

	// Get the specific GPUPool
	pool := &tfv1.GPUPool{}
	poolKey := client.ObjectKey{Name: poolName}
	if err := p.Get(ctx, poolKey, pool); err != nil {
		// Log error but don't fail - cache default config
		// This allows webhook to work even if pool is unavailable
		p.poolConfigs[poolName] = &PoolDRAConfig{
			EnableDRA:                 false,
			ResourceClaimTemplateName: constants.DRAResourceClaimTemplateName,
		}
		return nil
	}

	// Create config with defaults
	config := &PoolDRAConfig{
		EnableDRA:                 false,
		ResourceClaimTemplateName: constants.DRAResourceClaimTemplateName,
	}

	// Read DRA configuration from GPUPool
	if pool.Spec.DRAConfig != nil {
		if pool.Spec.DRAConfig.Enable != nil && *pool.Spec.DRAConfig.Enable {
			config.EnableDRA = true
		}
		// Override ResourceClaimTemplateName if specified in pool
		if pool.Spec.DRAConfig.ResourceClaimTemplateName != "" {
			config.ResourceClaimTemplateName = pool.Spec.DRAConfig.ResourceClaimTemplateName
		}
	}

	// Cache the configuration for this pool
	p.poolConfigs[poolName] = config
	return nil
}

// IsDRAEnabled checks if DRA is enabled for a specific pod based on the GPUPool configuration
func (p *DRAProcessor) IsDRAEnabled(ctx context.Context, pod *corev1.Pod, poolName string) bool {
	// Load configuration if not yet loaded (lazy loading)
	_ = p.InitializeDRAConfig(ctx, poolName) // Ignore error to maintain backward compatibility

	// Check pod-level annotation first (explicit override)
	if val, ok := pod.Annotations[constants.DRAEnabledAnnotation]; ok && val == constants.TrueStringValue {
		return true
	}

	// Check pod-level annotation for explicit disable
	if val, ok := pod.Annotations[constants.DRAEnabledAnnotation]; ok && val == constants.FalseStringValue {
		return false
	}

	// Fall back to pool-level configuration
	if config, exists := p.poolConfigs[poolName]; exists {
		return config.EnableDRA
	}

	// Default to false if no configuration found
	return false
}

// HasDRAClaim checks if a pod has DRA ResourceClaim references
func HasDRAClaim(pod *corev1.Pod) bool {
	return len(pod.Spec.ResourceClaims) > 0
}

// HandleDRAAdmission handles the complete DRA admission process for a specific pool
func (p *DRAProcessor) HandleDRAAdmission(ctx context.Context, pod *corev1.Pod, tfInfo *utils.TensorFusionInfo, containerIndices []int) error {
	// Load DRA configuration if needed (using pool name from tfInfo)
	if err := p.InitializeDRAConfig(ctx, tfInfo.Profile.PoolName); err != nil {
		return fmt.Errorf("failed to load DRA config: %w", err)
	}

	// Check if user has provided a custom CEL expression
	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}

	userCEL := pod.Annotations[constants.DRACelExpressionAnnotation]

	// Only generate default CEL if user hasn't provided one
	if userCEL == "" {
		celSelector, err := BuildCELSelector(pod, tfInfo)
		if err != nil {
			return fmt.Errorf("failed to build CEL selector: %w", err)
		}
		pod.Annotations[constants.DRACelExpressionAnnotation] = celSelector
	}

	// Inject ResourceClaimTemplate reference to Pod
	p.injectResourceClaimTemplateRef(pod, tfInfo.Profile.PoolName)

	// Mark pod with DRA enabled annotation
	pod.Annotations[constants.DRAEnabledAnnotation] = constants.TrueStringValue

	return nil
}

// BuildCELSelector constructs a CEL expression for DRA device selection based on TensorFusion requirements
func BuildCELSelector(pod *corev1.Pod, tfInfo *utils.TensorFusionInfo) (string, error) {
	var conditions []string

	// 1. GPU model filter (if specified)
	if tfInfo.Profile.GPUModel != "" {
		conditions = append(conditions, fmt.Sprintf(`device.attributes["%s"] == "%s"`, constants.DRAAttributeModel, tfInfo.Profile.GPUModel))
	}

	// 2. Pool name filter (for resource isolation and scheduling preferences)
	if tfInfo.Profile.PoolName != "" {
		conditions = append(conditions, fmt.Sprintf(`device.attributes["%s"] == "%s"`, constants.DRAAttributePoolName, tfInfo.Profile.PoolName))
	}

	// 3. TFlops capacity requirement (if specified)
	if !tfInfo.Profile.Resources.Requests.Tflops.IsZero() {
		tflopsValue := tfInfo.Profile.Resources.Requests.Tflops.AsApproximateFloat64()
		conditions = append(conditions, fmt.Sprintf(`device.capacity["%s"].AsApproximateFloat64() >= %f`, constants.DRACapacityTFlops, tflopsValue))
	}

	// 4. VRAM capacity requirement (if specified)
	if !tfInfo.Profile.Resources.Requests.Vram.IsZero() {
		vramValue := tfInfo.Profile.Resources.Requests.Vram.AsApproximateFloat64()
		conditions = append(conditions, fmt.Sprintf(`device.capacity["%s"].AsApproximateFloat64() >= %f`, constants.DRACapacityVRAM, vramValue))
	}

	// 5. QoS level filter (if specified and not default)
	// This can be used for priority-based scheduling
	if tfInfo.Profile.Qos != "" {
		conditions = append(conditions, fmt.Sprintf(`device.attributes["%s"] == "%s"`, constants.DRAAttributeQoS, tfInfo.Profile.Qos))
	}

	// 6. GPU phase filter (select Running or Pending GPUs, consistent with PhaseFilter)
	conditions = append(conditions, fmt.Sprintf(
		`(device.attributes["%s"] == "%s" || device.attributes["%s"] == "%s")`,
		constants.DRAAttributePhase, constants.PhaseRunning,
		constants.DRAAttributePhase, constants.PhasePending,
	))

	return strings.Join(conditions, " && "), nil
}

// injectResourceClaimTemplateRef adds ResourceClaimTemplate reference to Pod spec
func (p *DRAProcessor) injectResourceClaimTemplateRef(pod *corev1.Pod, poolName string) {
	// Add ResourceClaimTemplate reference to pod.Spec.ResourceClaims
	if pod.Spec.ResourceClaims == nil {
		pod.Spec.ResourceClaims = []corev1.PodResourceClaim{}
	}

	// Get the ResourceClaimTemplate name from pool config
	templateName := constants.DRAResourceClaimTemplateName
	if config, exists := p.poolConfigs[poolName]; exists {
		templateName = config.ResourceClaimTemplateName
	}

	// Use ResourceClaimTemplate instead of direct ResourceClaim
	claimRef := corev1.PodResourceClaim{
		Name:                      constants.DRAClaimDefineName,
		ResourceClaimTemplateName: &templateName,
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
