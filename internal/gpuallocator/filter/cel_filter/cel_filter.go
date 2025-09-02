package cel_filter

import (
	"context"
	"fmt"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// AllocRequestCELFilter converts AllocRequest to CEL filter and executes it
type CELFilter struct {
	cache      *ExpressionCache
	expression string
	name       string
}

// NewAllocRequestCELFilter creates a new CEL filter from allocation request
func NewCELFilter(req *tfv1.AllocRequest, cache *ExpressionCache) (*CELFilter, error) {
	// Convert AllocRequest to CEL expression
	expression, err := convertAllocRequestToCEL(req)
	if err != nil {
		return nil, fmt.Errorf("failed to convert AllocRequest to CEL: %w", err)
	}

	// Handle nil request case
	name := "AllocRequest-unknown"
	if req != nil {
		name = fmt.Sprintf("AllocRequest-%s", req.WorkloadNameNamespace.String())
	}

	return &CELFilter{
		cache:      cache,
		expression: expression,
		name:       name,
	}, nil
}

// Name returns the filter name
func (f *CELFilter) Name() string {
	return f.name
}

// Filter applies the CEL expression derived from AllocRequest to filter GPUs
func (f *CELFilter) Filter(ctx context.Context, workerPodKey tfv1.NameNamespace, gpus []*tfv1.GPU) ([]*tfv1.GPU, error) {
	log := log.FromContext(ctx)
	if len(gpus) == 0 {
		return gpus, nil
	}

	if f.expression == "" {
		// If no expression, return all GPUs (no filtering needed)
		return gpus, nil
	}

	// Get compiled program from cache
	program, err := f.cache.GetOrCompileProgram(f.expression)
	if err != nil {
		return nil, fmt.Errorf("failed to get CEL program for expression %q: %w", f.expression, err)
	}

	var filteredGPUs []*tfv1.GPU

	for _, gpu := range gpus {
		// Create timeout context for CEL evaluation
		evalCtx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)

		// Create variables for CEL evaluation
		vars := createCELVariables(*gpu, workerPodKey)

		// Evaluate with timeout
		resultChan := make(chan evalResult, 1)
		go func() {
			result, _, evalErr := program.Eval(vars)
			resultChan <- evalResult{result: result, err: evalErr}
		}()

		select {
		case evalRes := <-resultChan:
			cancel()
			if evalRes.err != nil {
				log.Error(evalRes.err, "CEL expression evaluation failed",
					"expression", f.expression,
					"gpu", gpu.Name,
					"workerPodKey", workerPodKey)
				// On error, exclude the GPU (fail-safe)
				continue
			}

			// Convert result to boolean
			if boolResult, ok := evalRes.result.(types.Bool); ok {
				if bool(boolResult) {
					filteredGPUs = append(filteredGPUs, gpu)
				}
			} else {
				log.Error(nil, "CEL expression did not return boolean",
					"expression", f.expression,
					"result", evalRes.result,
					"gpu", gpu.Name)
				// On non-boolean result, exclude the GPU (fail-safe)
				continue
			}
		case <-evalCtx.Done():
			cancel()
			// Timeout - skip this GPU (fail-safe behavior)
			log.V(1).Info("CEL evaluation timeout", "gpu", gpu.Name, "expression", f.expression)
			continue
		}
	}

	log.V(1).Info("AllocRequest CEL filter applied",
		"filter", f.name,
		"expression", f.expression,
		"inputGPUs", len(gpus),
		"outputGPUs", len(filteredGPUs))

	return filteredGPUs, nil
}

type evalResult struct {
	result interface{}
	err    error
}

// convertAllocRequestToCEL converts an allocation request to a CEL expression
func convertAllocRequestToCEL(req *tfv1.AllocRequest) (string, error) {
	if req == nil {
		return "", nil
	}

	var conditions []string

	// Add custom CEL expression if provided by user
	if req.CELFilterExpression != "" {
		conditions = append(conditions, req.CELFilterExpression)
	}

	// Add GPU phase condition (must be Ready)
	conditions = append(conditions, "gpu.phase == 'Ready'")

	// Add GPU model filter if specified
	if req.GPUModel != "" {
		conditions = append(conditions, fmt.Sprintf("gpu.gpuModel == '%s'", req.GPUModel))
	}

	// If no conditions, return empty expression (no filtering)
	if len(conditions) == 0 {
		return "", nil
	}

	// Combine all conditions with AND
	if len(conditions) == 1 {
		return conditions[0], nil
	}

	expression := conditions[0]
	for i := 1; i < len(conditions); i++ {
		expression += " && " + conditions[i]
	}

	return expression, nil
}

// createCELEnvironment creates a CEL environment with GPU-related variables and functions
func createCELEnvironment() (*cel.Env, error) {
	return cel.NewEnv(
		// Define GPU object structure
		cel.Variable(CELVarGPU, cel.MapType(cel.StringType, cel.DynType)),
		// Define worker pod key
		cel.Variable(CELVarWorkerPodKey, cel.MapType(cel.StringType, cel.StringType)),
		// Define request object structure
		cel.Variable(CELVarRequest, cel.MapType(cel.StringType, cel.DynType)),
	)
}

// createCELVariables creates variables for CEL evaluation from GPU and request information
func createCELVariables(gpu tfv1.GPU, workerPodKey tfv1.NameNamespace) map[string]interface{} {
	// Convert GPU to a map for CEL evaluation
	gpuMap := map[string]interface{}{
		GPUFieldName:        gpu.Name,
		GPUFieldNamespace:   gpu.Namespace,
		GPUFieldGPUModel:    gpu.Status.GPUModel,
		GPUFieldUUID:        gpu.Status.UUID,
		GPUFieldPhase:       string(gpu.Status.Phase),
		GPUFieldUsedBy:      string(gpu.Status.UsedBy),
		GPUFieldMessage:     gpu.Status.Message,
		GPUFieldLabels:      gpu.Labels,
		GPUFieldAnnotations: gpu.Annotations,
	}

	// Add available information if available
	if gpu.Status.Available != nil {
		gpuMap[GPUFieldAvailable] = map[string]interface{}{
			ResourceFieldTFlops: gpu.Status.Available.Tflops.AsApproximateFloat64(),
			ResourceFieldVRAM:   gpu.Status.Available.Vram.AsApproximateFloat64(),
		}
	}

	// Add node selector information
	if gpu.Status.NodeSelector != nil {
		gpuMap[GPUFieldNodeSelector] = gpu.Status.NodeSelector
	}

	// Add running apps information (always set, even if empty)
	runningApps := make([]map[string]interface{}, len(gpu.Status.RunningApps))
	for i, app := range gpu.Status.RunningApps {
		runningApps[i] = map[string]interface{}{
			AppFieldName:      app.Name,
			AppFieldNamespace: app.Namespace,
			AppFieldCount:     app.Count,
		}
	}
	gpuMap[GPUFieldRunningApps] = runningApps

	// Worker pod key information
	workerPodKeyMap := map[string]string{
		PodKeyFieldName:      workerPodKey.Name,
		PodKeyFieldNamespace: workerPodKey.Namespace,
	}

	return map[string]interface{}{
		CELVarGPU:          gpuMap,
		CELVarWorkerPodKey: workerPodKeyMap,
	}
}
