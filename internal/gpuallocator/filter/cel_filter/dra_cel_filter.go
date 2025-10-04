package cel_filter

import (
	"context"
	"encoding/json"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/samber/lo"
	resourceapi "k8s.io/api/resource/v1"
	dracel "k8s.io/dynamic-resource-allocation/cel"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// DRACELFilter implements CEL filtering using k8s.io/dynamic-resource-allocation/cel
type DRACELFilter struct {
	name              string
	requiredPhases    []tfv1.TensorFusionGPUPhase
	userExpression    string
	cache             *dracel.Cache
	displayExpression string
}

// NewDRACELFilter creates a new DRA-based CEL filter from allocation request
func NewDRACELFilter(req *tfv1.AllocRequest, cache *dracel.Cache) (*DRACELFilter, error) {
	// Extract early filtering criteria
	var requiredPhases []tfv1.TensorFusionGPUPhase
	var userExpression, displayExpression string

	if req != nil {
		requiredPhases = []tfv1.TensorFusionGPUPhase{
			tfv1.TensorFusionGPUPhaseRunning,
			tfv1.TensorFusionGPUPhasePending,
		}
		userExpression = req.CELFilterExpression
		displayExpression = buildDisplayExpression(req)
	}

	// Handle nil request case
	name := "AllocRequest-unknown"
	if req != nil {
		name = fmt.Sprintf("AllocRequest-%s", req.WorkloadNameNamespace.String())
	}

	// Validate expression if provided
	if userExpression != "" && cache != nil {
		result := cache.Check(userExpression)
		if result.Error != nil {
			return nil, fmt.Errorf("failed to compile CEL expression %q: %w", userExpression, result.Error)
		}
	}

	return &DRACELFilter{
		name:              name,
		requiredPhases:    requiredPhases,
		userExpression:    userExpression,
		cache:             cache,
		displayExpression: displayExpression,
	}, nil
}

// Name returns the filter name
func (f *DRACELFilter) Name() string {
	return f.name
}

// Filter applies the CEL expression to filter GPUs
func (f *DRACELFilter) Filter(ctx context.Context, workerPodKey tfv1.NameNamespace, gpus []*tfv1.GPU) ([]*tfv1.GPU, error) {
	log := log.FromContext(ctx)
	if len(gpus) == 0 {
		return gpus, nil
	}

	// Early filtering phase: apply basic filters first
	earlyFilteredGPUs := make([]*tfv1.GPU, 0, len(gpus))
	for _, gpu := range gpus {
		// Progressive migration mode check
		if utils.IsProgressiveMigration() && gpu.Status.UsedBy != tfv1.UsedByTensorFusion {
			continue
		}

		// Fast path: check phase first (most common filter)
		if f.requiredPhases != nil && !lo.Contains(f.requiredPhases, gpu.Status.Phase) {
			continue
		}

		earlyFilteredGPUs = append(earlyFilteredGPUs, gpu)
	}

	// If no user expression, return early filtered results
	if f.userExpression == "" {
		log.V(1).Info("DRA CEL filter applied (early filtering only)",
			"filter", f.name,
			"inputGPUs", len(gpus),
			"outputGPUs", len(earlyFilteredGPUs))
		return earlyFilteredGPUs, nil
	}

	// If no GPUs passed early filtering, return empty result
	if len(earlyFilteredGPUs) == 0 {
		return earlyFilteredGPUs, nil
	}

	// Get compiled expression from cache
	compiledExpr := f.cache.GetOrCompile(f.userExpression)
	if compiledExpr.Error != nil {
		return nil, fmt.Errorf("failed to compile CEL expression %q: %w", f.userExpression, compiledExpr.Error)
	}

	// Apply CEL filtering using DRA
	filteredGPUs := make([]*tfv1.GPU, 0, len(earlyFilteredGPUs))
	for _, gpu := range earlyFilteredGPUs {
		// Convert GPU to DRA Device
		device, err := convertGPUToDevice(gpu)
		if err != nil {
			log.Error(err, "Failed to convert GPU to Device", "gpu", gpu.Name)
			continue
		}

		// Evaluate CEL expression
		matches, details, err := compiledExpr.DeviceMatches(ctx, device)
		if err != nil {
			log.Error(err, "CEL expression evaluation failed",
				"expression", f.userExpression,
				"gpu", gpu.Name,
				"details", details)
			// On error, exclude the GPU (fail-safe)
			continue
		}

		if matches {
			filteredGPUs = append(filteredGPUs, gpu)
		}
	}

	log.V(1).Info("DRA CEL filter applied",
		"filter", f.name,
		"displayExpression", f.displayExpression,
		"userExpression", f.userExpression,
		"inputGPUs", len(gpus),
		"earlyFilteredGPUs", len(earlyFilteredGPUs),
		"outputGPUs", len(filteredGPUs))

	return filteredGPUs, nil
}

// convertGPUToDevice converts tfv1.GPU to dracel.Device
func convertGPUToDevice(gpu *tfv1.GPU) (dracel.Device, error) {
	if gpu == nil {
		return dracel.Device{}, fmt.Errorf("GPU is nil")
	}

	allowMultiple := true
	device := dracel.Device{
		Driver:                   constants.DRADriverName,
		AllowMultipleAllocations: &allowMultiple,
		Attributes:               make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute),
		Capacity:                 make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity),
	}

	// Map basic attributes
	device.Attributes[GPUFieldName] = resourceapi.DeviceAttribute{StringValue: &gpu.Name}
	device.Attributes[GPUFieldNamespace] = resourceapi.DeviceAttribute{StringValue: &gpu.Namespace}
	model := gpu.Status.GPUModel
	device.Attributes[GPUFieldGPUModel] = resourceapi.DeviceAttribute{StringValue: &model}
	uuid := gpu.Status.UUID
	device.Attributes[GPUFieldUUID] = resourceapi.DeviceAttribute{StringValue: &uuid}
	usedBy := string(gpu.Status.UsedBy)
	device.Attributes[GPUFieldUsedBy] = resourceapi.DeviceAttribute{StringValue: &usedBy}
	message := gpu.Status.Message
	device.Attributes[GPUFieldMessage] = resourceapi.DeviceAttribute{StringValue: &message}

	// Map labels with prefix
	if len(gpu.Labels) > 0 {
		for k, v := range gpu.Labels {
			labelValue := v
			device.Attributes[resourceapi.QualifiedName(fmt.Sprintf("%s.%s", GPUFieldLabels, k))] = resourceapi.DeviceAttribute{StringValue: &labelValue}
		}
	}

	// Map annotations with prefix
	if len(gpu.Annotations) > 0 {
		for k, v := range gpu.Annotations {
			annotationValue := v
			device.Attributes[resourceapi.QualifiedName(fmt.Sprintf("%s.%s", GPUFieldAnnotations, k))] = resourceapi.DeviceAttribute{StringValue: &annotationValue}
		}
	}

	// Map nodeSelector with prefix
	if len(gpu.Status.NodeSelector) > 0 {
		for k, v := range gpu.Status.NodeSelector {
			selectorValue := v
			device.Attributes[resourceapi.QualifiedName(fmt.Sprintf("%s.%s", GPUFieldNodeSelector, k))] = resourceapi.DeviceAttribute{StringValue: &selectorValue}
		}
	}

	// Map runningApps as JSON string
	if len(gpu.Status.RunningApps) > 0 {
		appsJSON, err := json.Marshal(gpu.Status.RunningApps)
		if err != nil {
			return dracel.Device{}, fmt.Errorf("failed to marshal runningApps: %w", err)
		}
		appsStr := string(appsJSON)
		device.Attributes[GPUFieldRunningApps] = resourceapi.DeviceAttribute{StringValue: &appsStr}
	}

	// Map capacity (tflops and vram) - DRA experimental version maintains capacity state
	if gpu.Status.Capacity != nil {
		device.Capacity[ResourceFieldTFlops] = resourceapi.DeviceCapacity{Value: gpu.Status.Capacity.Tflops}
		device.Capacity[ResourceFieldVRAM] = resourceapi.DeviceCapacity{Value: gpu.Status.Capacity.Vram}
	}

	return device, nil
}
