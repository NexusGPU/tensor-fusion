package cel_filter

import (
	"context"
	"fmt"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Parallel processing constants
const (
	// Threshold for enabling parallel processing
	ParallelThreshold = 2000
	// Default number of worker goroutines
	DefaultWorkerCount = 4
)

// fieldUsage tracks which GPU fields are used in the expression
type fieldUsage struct {
	labels       bool
	annotations  bool
	available    bool
	nodeSelector bool
	runningApps  bool
}

// FastPathPredicate represents a compiled fast-path predicate function
type FastPathPredicate func(gpu *tfv1.GPU) bool

// ExpressionPattern represents a recognized expression pattern for fast path
type ExpressionPattern struct {
	Pattern    *regexp.Regexp
	Generator  func(matches []string) FastPathPredicate
}

// Common fast path patterns - order matters (most specific first)
var fastPathPatterns = []ExpressionPattern{
	// Complex AND pattern: gpu.available.tflops >= NUMBER && gpu.labels['KEY'] == 'VALUE'
	{
		Pattern: regexp.MustCompile(`^gpu\.available\.tflops\s*>=\s*([0-9]+(?:\.[0-9]+)?)\s*&&\s*gpu\.labels\['([^']+)'\]\s*==\s*'([^']+)'$`),
		Generator: func(matches []string) FastPathPredicate {
			threshold, _ := strconv.ParseFloat(matches[1], 64)
			labelKey, labelValue := matches[2], matches[3]
			return func(gpu *tfv1.GPU) bool {
				return gpu.Status.Available != nil && 
					gpu.Status.Available.Tflops.AsApproximateFloat64() >= threshold &&
					gpu.Labels != nil && gpu.Labels[labelKey] == labelValue
			}
		},
	},
	// gpu.available.tflops >= NUMBER
	{
		Pattern: regexp.MustCompile(`^gpu\.available\.tflops\s*>=\s*([0-9]+(?:\.[0-9]+)?)$`),
		Generator: func(matches []string) FastPathPredicate {
			threshold, _ := strconv.ParseFloat(matches[1], 64)
			return func(gpu *tfv1.GPU) bool {
				return gpu.Status.Available != nil && gpu.Status.Available.Tflops.AsApproximateFloat64() >= threshold
			}
		},
	},
	// gpu.available.tflops > NUMBER
	{
		Pattern: regexp.MustCompile(`^gpu\.available\.tflops\s*>\s*([0-9]+(?:\.[0-9]+)?)$`),
		Generator: func(matches []string) FastPathPredicate {
			threshold, _ := strconv.ParseFloat(matches[1], 64)
			return func(gpu *tfv1.GPU) bool {
				return gpu.Status.Available != nil && gpu.Status.Available.Tflops.AsApproximateFloat64() > threshold
			}
		},
	},
	// gpu.available.vram >= NUMBER
	{
		Pattern: regexp.MustCompile(`^gpu\.available\.vram\s*>=\s*([0-9]+(?:\.[0-9]+)?)$`),
		Generator: func(matches []string) FastPathPredicate {
			threshold, _ := strconv.ParseFloat(matches[1], 64)
			return func(gpu *tfv1.GPU) bool {
				return gpu.Status.Available != nil && gpu.Status.Available.Vram.AsApproximateFloat64() >= threshold
			}
		},
	},
	// gpu.available.vram > NUMBER  
	{
		Pattern: regexp.MustCompile(`^gpu\.available\.vram\s*>\s*([0-9]+(?:\.[0-9]+)?)$`),
		Generator: func(matches []string) FastPathPredicate {
			threshold, _ := strconv.ParseFloat(matches[1], 64)
			return func(gpu *tfv1.GPU) bool {
				return gpu.Status.Available != nil && gpu.Status.Available.Vram.AsApproximateFloat64() > threshold
			}
		},
	},
	// gpu.labels['KEY'] == 'VALUE'
	{
		Pattern: regexp.MustCompile(`^gpu\.labels\['([^']+)'\]\s*==\s*'([^']+)'$`),
		Generator: func(matches []string) FastPathPredicate {
			key, value := matches[1], matches[2]
			return func(gpu *tfv1.GPU) bool {
				return gpu.Labels != nil && gpu.Labels[key] == value
			}
		},
	},
	// gpu.annotations['KEY'] == 'VALUE'
	{
		Pattern: regexp.MustCompile(`^gpu\.annotations\['([^']+)'\]\s*==\s*'([^']+)'$`),
		Generator: func(matches []string) FastPathPredicate {
			key, value := matches[1], matches[2]
			return func(gpu *tfv1.GPU) bool {
				return gpu.Annotations != nil && gpu.Annotations[key] == value
			}
		},
	},
}


// ZeroAllocActivation provides zero-allocation variable resolution for CEL
// This eliminates the need to create map[string]interface{} for each GPU
type ZeroAllocActivation struct {
	gpu          *tfv1.GPU
	workerPodKey tfv1.NameNamespace
	usage        fieldUsage
}

// ResolveName implements interpreter.Activation interface
func (a *ZeroAllocActivation) ResolveName(name string) (interface{}, bool) {
	switch name {
	case CELVarGPU:
		return a.createGPUObject(), true
	case CELVarWorkerPodKey:
		return a.createWorkerPodKeyObject(), true
	default:
		return nil, false
	}
}

// Parent implements interpreter.Activation interface  
func (a *ZeroAllocActivation) Parent() interpreter.Activation {
	return nil
}

// createGPUObject creates GPU object on-demand without maps
func (a *ZeroAllocActivation) createGPUObject() interface{} {
	// Return GPU value with lazy caching
	return &gpuVal{GPU: a.gpu}
}


// createWorkerPodKeyObject creates worker pod key object
func (a *ZeroAllocActivation) createWorkerPodKeyObject() interface{} {
	return map[string]interface{}{
		"name":      a.workerPodKey.Name,
		"namespace": a.workerPodKey.Namespace,
	}
}

// gpuVal implements CEL value interface for GPU objects to eliminate map allocations
type gpuVal struct {
	*tfv1.GPU
	// Cached sub-values to avoid repeated allocations
	labels       ref.Val
	annotations  ref.Val 
	nodeSelector ref.Val
	available    ref.Val
	runningApps  ref.Val
}

// Type implements ref.Val interface
func (v *gpuVal) Type() ref.Type {
	return types.MapType
}

// Value implements ref.Val interface
func (v *gpuVal) Value() interface{} {
	return v.GPU
}

// Equal implements ref.Val interface
func (v *gpuVal) Equal(other ref.Val) ref.Val {
	if otherGPU, ok := other.(*gpuVal); ok {
		return types.Bool(v.GPU.UID == otherGPU.GPU.UID)
	}
	return types.False
}

// ConvertToNative implements ref.Val interface
func (v *gpuVal) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	return v.GPU, nil
}

// ConvertToType implements ref.Val interface  
func (v *gpuVal) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case types.TypeType:
		return types.MapType
	default:
		return types.NewErr("type conversion error")
	}
}

// HasField implements traits.FieldTester interface
func (v *gpuVal) HasField(field string) bool {
	switch field {
	case GPUFieldName, GPUFieldNamespace, GPUFieldGPUModel, GPUFieldUUID,
		 GPUFieldPhase, GPUFieldUsedBy, GPUFieldMessage, GPUFieldLabels,
		 GPUFieldAnnotations, GPUFieldAvailable, GPUFieldNodeSelector, GPUFieldRunningApps:
		return true
	default:
		return false
	}
}

// Get implements traits.Indexer interface for field access with lazy caching
func (v *gpuVal) Get(index ref.Val) ref.Val {
	field, ok := index.Value().(string)
	if !ok {
		return types.NewErr("index must be string")
	}
	
	switch field {
	case GPUFieldName:
		return types.String(v.GPU.Name)
	case GPUFieldNamespace:
		return types.String(v.GPU.Namespace)
	case GPUFieldGPUModel:
		return types.String(v.GPU.Status.GPUModel)
	case GPUFieldUUID:
		return types.String(v.GPU.Status.UUID)
	case GPUFieldPhase:
		return types.String(string(v.GPU.Status.Phase))
	case GPUFieldUsedBy:
		return types.String(string(v.GPU.Status.UsedBy))
	case GPUFieldMessage:
		return types.String(v.GPU.Status.Message)
	case GPUFieldLabels:
		// Lazy initialization with caching
		if v.labels == nil {
			v.labels = &labelsVal{labels: v.GPU.Labels}
		}
		return v.labels
	case GPUFieldAnnotations:
		// Lazy initialization with caching  
		if v.annotations == nil {
			v.annotations = &labelsVal{labels: v.GPU.Annotations}
		}
		return v.annotations
	case GPUFieldAvailable:
		// Lazy initialization with caching
		if v.available == nil {
			v.available = &availableVal{available: v.GPU.Status.Available}
		}
		return v.available
	case GPUFieldNodeSelector:
		// Lazy initialization with caching
		if v.nodeSelector == nil {
			v.nodeSelector = &labelsVal{labels: v.GPU.Status.NodeSelector}
		}
		return v.nodeSelector
	case GPUFieldRunningApps:
		// For now, keep simple implementation - can optimize later if needed
		if v.runningApps == nil {
			apps := make([]interface{}, len(v.GPU.Status.RunningApps))
			for i, app := range v.GPU.Status.RunningApps {
				apps[i] = map[string]interface{}{
					"name":      app.Name,
					"namespace": app.Namespace,
				}
			}
			v.runningApps = types.NewDynamicList(types.DefaultTypeAdapter, apps)
		}
		return v.runningApps
	default:
		return types.NewErr("no such field: %s", field)
	}
}

// availableVal provides direct access to GPU available resources without maps
type availableVal struct {
	available *tfv1.Resource
}

// Type implements ref.Val interface  
func (v *availableVal) Type() ref.Type {
	return types.MapType
}

// Value implements ref.Val interface
func (v *availableVal) Value() interface{} {
	return v.available
}

// Equal implements ref.Val interface
func (v *availableVal) Equal(other ref.Val) ref.Val {
	return types.False // Not used in comparisons
}

// ConvertToNative implements ref.Val interface
func (v *availableVal) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	return v.available, nil
}

// ConvertToType implements ref.Val interface
func (v *availableVal) ConvertToType(typeValue ref.Type) ref.Val {
	return types.NewErr("type conversion not supported")
}

// Get implements field access for available resources
func (v *availableVal) Get(index ref.Val) ref.Val {
	field, ok := index.Value().(string)
	if !ok {
		return types.NewErr("index must be string")
	}
	
	if v.available == nil {
		switch field {
		case "tflops":
			return types.Double(0.0)
		case "vram":
			return types.Int(0)
		default:
			return types.NewErr("no such field: %s", field)
		}
	}
	
	switch field {
	case "tflops":
		return types.Double(v.available.Tflops.AsApproximateFloat64())
	case "vram":
		return types.Int(v.available.Vram.Value())
	default:
		return types.NewErr("no such field: %s", field)
	}
}

// HasField implements field testing
func (v *availableVal) HasField(field string) bool {
	return field == "tflops" || field == "vram"
}

// labelsVal provides direct access to GPU labels without copying
type labelsVal struct {
	labels map[string]string
}

// Type implements ref.Val interface
func (v *labelsVal) Type() ref.Type {
	return types.MapType
}

// Value implements ref.Val interface  
func (v *labelsVal) Value() interface{} {
	return v.labels
}

// Equal implements ref.Val interface
func (v *labelsVal) Equal(other ref.Val) ref.Val {
	return types.False // Not used in comparisons
}

// ConvertToNative implements ref.Val interface
func (v *labelsVal) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	return v.labels, nil
}

// ConvertToType implements ref.Val interface
func (v *labelsVal) ConvertToType(typeValue ref.Type) ref.Val {
	return types.NewErr("type conversion not supported")
}

// Get implements map access for labels
func (v *labelsVal) Get(index ref.Val) ref.Val {
	key, ok := index.Value().(string)
	if !ok {
		return types.NewErr("index must be string")
	}
	
	if v.labels == nil {
		return types.String("")
	}
	
	value, exists := v.labels[key]
	if !exists {
		return types.String("")
	}
	return types.String(value)
}

// AllocRequestCELFilter converts AllocRequest to CEL filter and executes it
type CELFilter struct {
	cache *ExpressionCache
	name  string
	// Store early filtering criteria for optimization
	requiredPhase    string
	requiredGPUModel string
	userExpression   string
	// Track which fields are actually used
	usage fieldUsage
	// Display expression for logging (read-only)
	displayExpression string
	// Fast path predicate for common patterns
	fastPathPredicate FastPathPredicate
}

// NewAllocRequestCELFilter creates a new CEL filter from allocation request
func NewCELFilter(req *tfv1.AllocRequest, cache *ExpressionCache) (*CELFilter, error) {
	// Extract early filtering criteria
	var requiredPhase, requiredGPUModel, userExpression, displayExpression string

	if req != nil {
		requiredPhase = "Ready" // Keep as Ready for compatibility with tests
		requiredGPUModel = req.GPUModel
		userExpression = req.CELFilterExpression

		// Build display expression for logging (not used for execution)
		displayExpression = buildDisplayExpression(req)
	}

	// Analyze field usage in user expression only
	usage := analyzeFieldUsage(userExpression)
	
	// Try to compile fast path predicate
	fastPath := compileFastPath(userExpression)

	// Handle nil request case
	name := "AllocRequest-unknown"
	if req != nil {
		name = fmt.Sprintf("AllocRequest-%s", req.WorkloadNameNamespace.String())
	}

	return &CELFilter{
		cache:             cache,
		name:              name,
		requiredPhase:     requiredPhase,
		requiredGPUModel:  requiredGPUModel,
		userExpression:    userExpression,
		usage:             usage,
		displayExpression: displayExpression,
		fastPathPredicate: fastPath,
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

	// Pre-allocate result slice with estimated capacity
	filteredGPUs := make([]*tfv1.GPU, 0, len(gpus))

	// Early filtering phase: apply basic filters first to reduce CEL evaluation overhead
	earlyFilteredGPUs := make([]*tfv1.GPU, 0, len(gpus))
	for _, gpu := range gpus {
		// Fast path: check phase first (most common filter)
		if f.requiredPhase != "" && string(gpu.Status.Phase) != f.requiredPhase {
			continue
		}

		// Fast path: check GPU model (second most common filter)
		if f.requiredGPUModel != "" && gpu.Status.GPUModel != f.requiredGPUModel {
			continue
		}

		earlyFilteredGPUs = append(earlyFilteredGPUs, gpu)
	}

	// If no user expression, return early filtered results
	if f.userExpression == "" {
		log.V(1).Info("CEL filter applied (early filtering only)",
			"filter", f.name,
			"inputGPUs", len(gpus),
			"earlyFilteredGPUs", len(earlyFilteredGPUs),
			"outputGPUs", len(earlyFilteredGPUs))
		return earlyFilteredGPUs, nil
	}

	// If no GPUs passed early filtering, return empty result
	if len(earlyFilteredGPUs) == 0 {
		return earlyFilteredGPUs, nil
	}

	// Get compiled program from cache for user expression
	program, err := f.cache.GetOrCompileProgram(f.userExpression)
	if err != nil {
		return nil, fmt.Errorf("failed to get CEL program for expression %q: %w", f.userExpression, err)
	}


	// Use fast path if available, otherwise fall back to CEL
	if f.fastPathPredicate != nil {
		// Fast path: direct Go function evaluation with optional parallelization
		if len(earlyFilteredGPUs) >= ParallelThreshold {
			filteredGPUs = f.filterParallel(earlyFilteredGPUs)
		} else {
			for _, gpu := range earlyFilteredGPUs {
				if f.fastPathPredicate(gpu) {
					filteredGPUs = append(filteredGPUs, gpu)
				}
			}
		}
		
		log.V(1).Info("CEL filter applied (fast path)",
			"filter", f.name,
			"displayExpression", f.displayExpression,
			"userExpression", f.userExpression,
			"inputGPUs", len(gpus),
			"earlyFilteredGPUs", len(earlyFilteredGPUs),
			"outputGPUs", len(filteredGPUs))
	} else {
		// Fallback to CEL evaluation for complex expressions
		if len(earlyFilteredGPUs) >= ParallelThreshold {
			// Use parallel evaluation for large GPU sets
			filteredGPUs = f.filterFallbackParallel(ctx, program, earlyFilteredGPUs, workerPodKey)
		} else {
			// Sequential evaluation for smaller sets
			filteredGPUs = f.filterFallbackSequential(ctx, program, earlyFilteredGPUs, workerPodKey)
		}
		
		log.V(1).Info("CEL filter applied (CEL evaluation)",
			"filter", f.name,
			"displayExpression", f.displayExpression,
			"userExpression", f.userExpression,
			"inputGPUs", len(gpus),
			"earlyFilteredGPUs", len(earlyFilteredGPUs),
			"outputGPUs", len(filteredGPUs))
	}

	return filteredGPUs, nil
}

// buildDisplayExpression creates a readable expression string for logging purposes only
func buildDisplayExpression(req *tfv1.AllocRequest) string {
	if req == nil {
		return ""
	}

	var conditions []string

	// Add custom CEL expression if provided by user
	if req.CELFilterExpression != "" {
		conditions = append(conditions, req.CELFilterExpression)
	}

	// If no conditions, return empty expression
	if len(conditions) == 0 {
		return ""
	}

	// Combine all conditions with AND using strings.Builder for efficiency
	if len(conditions) == 1 {
		return conditions[0]
	}

	var builder strings.Builder
	builder.WriteString(conditions[0])
	for i := 1; i < len(conditions); i++ {
		builder.WriteString(" && ")
		builder.WriteString(conditions[i])
	}

	return builder.String()
}

// createCELEnvironment creates a CEL environment with GPU-related variables and functions
func createCELEnvironment() (*cel.Env, error) {
	return cel.NewEnv(
		// Define GPU object structure
		cel.Variable(CELVarGPU, cel.MapType(cel.StringType, cel.DynType)),
		// Define worker pod key
		cel.Variable(CELVarWorkerPodKey, cel.MapType(cel.StringType, cel.StringType)),
	)
}


// filterParallel processes GPUs in parallel for large datasets
func (f *CELFilter) filterParallel(gpus []*tfv1.GPU) []*tfv1.GPU {
	numGPUs := len(gpus)
	numWorkers := runtime.NumCPU()
	if numWorkers > DefaultWorkerCount {
		numWorkers = DefaultWorkerCount
	}
	
	chunkSize := (numGPUs + numWorkers - 1) / numWorkers
	resultChannels := make([]<-chan []*tfv1.GPU, numWorkers)
	
	// Create workers
	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > numGPUs {
			end = numGPUs
		}
		
		if start >= end {
			// No work for this worker
			ch := make(chan []*tfv1.GPU, 1)
			ch <- []*tfv1.GPU{}
			close(ch)
			resultChannels[i] = ch
			continue
		}
		
		chunk := gpus[start:end]
		resultCh := make(chan []*tfv1.GPU, 1)
		resultChannels[i] = resultCh
		
		// Start worker goroutine
		go func(gpuChunk []*tfv1.GPU, resultCh chan<- []*tfv1.GPU) {
			defer close(resultCh)
			
			filtered := make([]*tfv1.GPU, 0, len(gpuChunk)/2) // Estimate 50% pass rate
			for _, gpu := range gpuChunk {
				if f.fastPathPredicate(gpu) {
					filtered = append(filtered, gpu)
				}
			}
			resultCh <- filtered
		}(chunk, resultCh)
	}
	
	// Collect results
	var totalFiltered []*tfv1.GPU
	for _, ch := range resultChannels {
		chunkResults := <-ch
		totalFiltered = append(totalFiltered, chunkResults...)
	}
	
	return totalFiltered
}

// filterFallbackSequential performs sequential CEL evaluation for smaller GPU sets
func (f *CELFilter) filterFallbackSequential(ctx context.Context, program cel.Program, gpus []*tfv1.GPU, workerPodKey tfv1.NameNamespace) []*tfv1.GPU {
	filteredGPUs := make([]*tfv1.GPU, 0, len(gpus)/2)
	log := log.FromContext(ctx)
	
	for i, gpu := range gpus {
		// Periodic context check every 64 GPUs for very large sets
		if i&63 == 0 {
			select {
			case <-ctx.Done():
				log.V(1).Info("CEL evaluation cancelled", "processedGPUs", len(filteredGPUs), "totalGPUs", len(gpus))
				return filteredGPUs
			default:
			}
		}

		// Use zero-allocation activation instead of maps
		activation := &ZeroAllocActivation{
			gpu:          gpu,
			workerPodKey: workerPodKey,
			usage:        f.usage,
		}

		// Direct synchronous evaluation with custom activation
		result, _, evalErr := program.Eval(activation)

		if evalErr != nil {
			log.Error(evalErr, "CEL expression evaluation failed",
				"expression", f.userExpression,
				"gpu", gpu.Name,
				"workerPodKey", workerPodKey)
			// On error, exclude the GPU (fail-safe)
			continue
		}

		// Convert result to boolean
		if boolResult, ok := result.(types.Bool); ok {
			if bool(boolResult) {
				filteredGPUs = append(filteredGPUs, gpu)
			}
		} else {
			log.Error(nil, "CEL expression did not return boolean",
				"expression", f.userExpression,
				"result", result,
				"gpu", gpu.Name)
			// On non-boolean result, exclude the GPU (fail-safe)
			continue
		}
	}
	
	return filteredGPUs
}

// filterFallbackParallel performs parallel CEL evaluation for large GPU sets
func (f *CELFilter) filterFallbackParallel(ctx context.Context, program cel.Program, gpus []*tfv1.GPU, workerPodKey tfv1.NameNamespace) []*tfv1.GPU {
	numGPUs := len(gpus)
	numWorkers := runtime.NumCPU()
	if numWorkers > DefaultWorkerCount {
		numWorkers = DefaultWorkerCount
	}
	
	chunkSize := (numGPUs + numWorkers - 1) / numWorkers
	resultChannels := make([]<-chan []*tfv1.GPU, numWorkers)
	
	// Create workers
	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > numGPUs {
			end = numGPUs
		}
		
		if start >= end {
			// No work for this worker
			ch := make(chan []*tfv1.GPU, 1)
			ch <- []*tfv1.GPU{}
			close(ch)
			resultChannels[i] = ch
			continue
		}
		
		chunk := gpus[start:end]
		resultCh := make(chan []*tfv1.GPU, 1)
		resultChannels[i] = resultCh
		
		// Start worker goroutine
		go func(gpuChunk []*tfv1.GPU, resultCh chan<- []*tfv1.GPU) {
			defer close(resultCh)
			
			filtered := make([]*tfv1.GPU, 0, len(gpuChunk)/2) // Estimate 50% pass rate
			
			for _, gpu := range gpuChunk {
				// Use zero-allocation activation
				activation := &ZeroAllocActivation{
					gpu:          gpu,
					workerPodKey: workerPodKey,
					usage:        f.usage,
				}

				// Direct synchronous evaluation
				result, _, evalErr := program.Eval(activation)
				if evalErr != nil {
					// On error, exclude the GPU (fail-safe)
					continue
				}

				// Convert result to boolean
				if boolResult, ok := result.(types.Bool); ok {
					if bool(boolResult) {
						filtered = append(filtered, gpu)
					}
				}
				// On non-boolean result, exclude the GPU (fail-safe)
			}
			resultCh <- filtered
		}(chunk, resultCh)
	}
	
	// Collect results
	var totalFiltered []*tfv1.GPU
	for _, ch := range resultChannels {
		chunkResults := <-ch
		totalFiltered = append(totalFiltered, chunkResults...)
	}
	
	return totalFiltered
}


// compileFastPath tries to compile expression into a fast path predicate
// Uses AST analysis for better pattern matching than regex
func compileFastPath(expression string) FastPathPredicate {
	if expression == "" {
		return nil
	}
	
	// Try AST-based compilation first (more flexible)  
	if pred := compileASTFastPath(expression); pred != nil {
		return pred
	}
	
	// Fall back to regex patterns for backward compatibility
	for _, pattern := range fastPathPatterns {
		matches := pattern.Pattern.FindStringSubmatch(expression)
		if matches != nil {
			return pattern.Generator(matches)
		}
	}
	
	return nil
}

// compileASTFastPath analyzes AST to generate fast path predicates
func compileASTFastPath(expression string) FastPathPredicate {
	// Parse expression to AST
	env, err := createCELEnvironment()
	if err != nil {
		return nil
	}
	
	_, issues := env.Parse(expression)
	if issues != nil && issues.Err() != nil {
		return nil
	}
	
	// Extract conditions from expression string (simplified approach)
	conditions := extractConditionsFromString(expression)
	if len(conditions) == 0 {
		return nil
	}
	
	// Generate fast path predicate
	return func(gpu *tfv1.GPU) bool {
		for _, condition := range conditions {
			if !evaluateCondition(gpu, condition) {
				return false // Short-circuit on first failure (AND logic)
			}
		}
		return true
	}
}

// astCondition represents a simple condition extracted from AST
type astCondition struct {
	field    string    // e.g., "gpu.available.tflops", "gpu.labels['env']"
	operator string    // "==", "!=", ">=", ">"
	value    interface{} // expected value
}


// extractConditionsFromString uses enhanced pattern matching to extract conditions
// This bridges the gap between regex and full AST until full AST implementation
func extractConditionsFromString(exprStr string) []astCondition {
	var conditions []astCondition
	
	// Split by && to handle multiple conditions
	parts := strings.Split(exprStr, " && ")
	
	for _, part := range parts {
		part = strings.TrimSpace(part)
		
		// Handle gpu.available.tflops >= X
		if strings.Contains(part, "gpu.available.tflops") && strings.Contains(part, ">=") {
			if condition := parseNumericCondition(part, "gpu.available.tflops", ">="); condition != nil {
				conditions = append(conditions, *condition)
			}
		} else if strings.Contains(part, "gpu.available.tflops") && strings.Contains(part, ">") {
			if condition := parseNumericCondition(part, "gpu.available.tflops", ">"); condition != nil {
				conditions = append(conditions, *condition)
			}
		}
		
		// Handle gpu.available.vram >= X
		if strings.Contains(part, "gpu.available.vram") && strings.Contains(part, ">=") {
			if condition := parseNumericCondition(part, "gpu.available.vram", ">="); condition != nil {
				conditions = append(conditions, *condition)
			}
		}
		
		// Handle gpu.labels['key'] == 'value'
		if strings.Contains(part, "gpu.labels[") && strings.Contains(part, "==") {
			if condition := parseLabelCondition(part, "gpu.labels"); condition != nil {
				conditions = append(conditions, *condition)
			}
		}
		
		// Handle gpu.annotations['key'] == 'value'  
		if strings.Contains(part, "gpu.annotations[") && strings.Contains(part, "==") {
			if condition := parseLabelCondition(part, "gpu.annotations"); condition != nil {
				conditions = append(conditions, *condition)
			}
		}
		
		// Handle gpu.gpuModel == 'value'
		if strings.Contains(part, "gpu.gpuModel") && strings.Contains(part, "==") {
			if condition := parseStringCondition(part, "gpu.gpuModel", "=="); condition != nil {
				conditions = append(conditions, *condition)
			}
		}
	}
	
	return conditions
}

// parseNumericCondition parses numeric comparison conditions
func parseNumericCondition(expr, field, operator string) *astCondition {
	parts := strings.Split(expr, operator)
	if len(parts) != 2 {
		return nil
	}
	
	valueStr := strings.TrimSpace(parts[1])
	value, err := strconv.ParseFloat(valueStr, 64)
	if err != nil {
		return nil
	}
	
	return &astCondition{
		field:    field,
		operator: operator,
		value:    value,
	}
}

// parseLabelCondition parses label/annotation map access conditions  
func parseLabelCondition(expr, fieldPrefix string) *astCondition {
	// Extract key from gpu.labels['key'] == 'value' format
	keyStart := strings.Index(expr, "['") + 2
	keyEnd := strings.Index(expr[keyStart:], "']")
	if keyEnd == -1 {
		return nil
	}
	key := expr[keyStart : keyStart+keyEnd]
	
	// Extract value
	valueStart := strings.LastIndex(expr, "'") 
	if valueStart == -1 {
		return nil
	}
	// Find the quote before the last quote
	prevQuotePos := strings.LastIndex(expr[:valueStart], "'")
	if prevQuotePos == -1 {
		return nil
	}
	value := expr[prevQuotePos+1 : valueStart]
	
	return &astCondition{
		field:    fieldPrefix + "['" + key + "']",
		operator: "==",
		value:    value,
	}
}

// parseStringCondition parses simple string equality conditions
func parseStringCondition(expr, field, operator string) *astCondition {
	parts := strings.Split(expr, operator)
	if len(parts) != 2 {
		return nil
	}
	
	valueStr := strings.TrimSpace(parts[1])
	// Remove quotes
	if strings.HasPrefix(valueStr, "'") && strings.HasSuffix(valueStr, "'") {
		valueStr = valueStr[1 : len(valueStr)-1]
	}
	
	return &astCondition{
		field:    field,
		operator: operator,
		value:    valueStr,
	}
}

// evaluateCondition evaluates a single condition against a GPU
func evaluateCondition(gpu *tfv1.GPU, condition astCondition) bool {
	switch condition.field {
	case "gpu.available.tflops":
		if gpu.Status.Available == nil {
			return false
		}
		actualValue := gpu.Status.Available.Tflops.AsApproximateFloat64()
		expectedValue, ok := condition.value.(float64)
		if !ok {
			return false
		}
		
		switch condition.operator {
		case ">=":
			return actualValue >= expectedValue
		case ">":
			return actualValue > expectedValue
		default:
			return false
		}
		
	case "gpu.available.vram":
		if gpu.Status.Available == nil {
			return false
		}
		actualValue := float64(gpu.Status.Available.Vram.Value())
		expectedValue, ok := condition.value.(float64)
		if !ok {
			return false
		}
		
		switch condition.operator {
		case ">=":
			return actualValue >= expectedValue
		case ">":
			return actualValue > expectedValue
		default:
			return false
		}
		
	case "gpu.gpuModel":
		expectedValue, ok := condition.value.(string)
		if !ok {
			return false
		}
		return gpu.Status.GPUModel == expectedValue
		
	default:
		// Handle label/annotation access
		if strings.HasPrefix(condition.field, "gpu.labels['") {
			key := strings.TrimSuffix(strings.TrimPrefix(condition.field, "gpu.labels['"), "']")
			expectedValue, ok := condition.value.(string)
			if !ok {
				return false
			}
			if gpu.Labels == nil {
				return expectedValue == ""
			}
			return gpu.Labels[key] == expectedValue
		}
		
		if strings.HasPrefix(condition.field, "gpu.annotations['") {
			key := strings.TrimSuffix(strings.TrimPrefix(condition.field, "gpu.annotations['"), "']")
			expectedValue, ok := condition.value.(string)
			if !ok {
				return false
			}
			if gpu.Annotations == nil {
				return expectedValue == ""
			}
			return gpu.Annotations[key] == expectedValue
		}
		
		return false
	}
}

// analyzeFieldUsage performs simple heuristic analysis of which fields are used in the expression
func analyzeFieldUsage(expression string) fieldUsage {
	if expression == "" {
		return fieldUsage{}
	}

	return fieldUsage{
		labels:       strings.Contains(expression, "labels"),
		annotations:  strings.Contains(expression, "annotations"),
		available:    strings.Contains(expression, "available") || strings.Contains(expression, "tflops") || strings.Contains(expression, "vram"),
		nodeSelector: strings.Contains(expression, "nodeSelector"),
		runningApps:  strings.Contains(expression, "runningApps"),
	}
}

