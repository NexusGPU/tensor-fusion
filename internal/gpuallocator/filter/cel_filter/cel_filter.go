package cel_filter

import (
	"context"
	"fmt"
	"reflect"
	"regexp"
	"runtime"
	"strings"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/interpreter"
	"github.com/samber/lo"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Parallel processing constants
const (
	// Threshold for enabling parallel processing
	ParallelThreshold = 2000
	// Default number of worker goroutines
	DefaultWorkerCount = 4
)

// Global string pool for GPU Phase values to reduce allocations
var (
	gpuPhaseStringPool = sync.OnceValue(func() map[string]types.String {
		return map[string]types.String{
			constants.PhaseUnknown:    types.String(constants.PhaseUnknown),
			constants.PhasePending:    types.String(constants.PhasePending),
			constants.PhaseUpdating:   types.String(constants.PhaseUpdating),
			constants.PhaseRunning:    types.String(constants.PhaseRunning),
			constants.PhaseMigrating:  types.String(constants.PhaseMigrating),
			constants.PhaseDestroying: types.String(constants.PhaseDestroying),
		}
	})
)

// getPooledPhaseString returns a pooled CEL String for the given phase
func getPooledPhaseString(phase string) ref.Val {
	pool := gpuPhaseStringPool()
	if pooled, exists := pool[phase]; exists {
		return pooled
	}
	// Return error for unexpected phase values
	return types.NewErr("unknown GPU phase: %s", phase)
}

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
	Pattern   *regexp.Regexp
	Generator func(matches []string) FastPathPredicate
}

// ZeroAllocActivation provides zero-allocation variable resolution for CEL
// This eliminates the need to create map[string]interface{} for each GPU
type ZeroAllocActivation struct {
	gpuVal       gpuVal
	workerPodKey workerPodKeyVal
	usage        fieldUsage
}

func (a *ZeroAllocActivation) init(g *tfv1.GPU, k tfv1.NameNamespace, usage fieldUsage) {
	a.gpuVal.GPU = g
	a.gpuVal.labels = nil
	a.gpuVal.annotations = nil
	a.gpuVal.nodeSelector = nil
	a.gpuVal.available = nil
	a.gpuVal.runningApps = nil
	a.workerPodKey.name = k.Name
	a.workerPodKey.namespace = k.Namespace
	a.usage = usage
}

// ResolveName implements interpreter.Activation interface
func (a *ZeroAllocActivation) ResolveName(name string) (interface{}, bool) {
	switch name {
	case CELVarGPU:
		return &a.gpuVal, true
	case CELVarWorkerPodKey:
		return &a.workerPodKey, true
	default:
		return nil, false
	}
}

// Parent implements interpreter.Activation interface
func (a *ZeroAllocActivation) Parent() interpreter.Activation {
	return nil
}

type workerPodKeyVal struct {
	name      string
	namespace string
}

func (w *workerPodKeyVal) Type() ref.Type { return types.MapType }
func (w *workerPodKeyVal) Value() interface{} {
	return map[string]string{"name": w.name, "namespace": w.namespace}
}
func (w *workerPodKeyVal) Equal(other ref.Val) ref.Val { return types.False }
func (w *workerPodKeyVal) ConvertToNative(t reflect.Type) (interface{}, error) {
	return map[string]string{"name": w.name, "namespace": w.namespace}, nil
}
func (w *workerPodKeyVal) ConvertToType(typeValue ref.Type) ref.Val {
	return types.NewErr("type conversion not supported")
}
func (w *workerPodKeyVal) Get(index ref.Val) ref.Val {
	key, ok := index.Value().(string)
	if !ok {
		return types.NewErr("index must be string")
	}
	switch key {
	case GPUFieldName:
		return types.String(w.name)
	case GPUFieldNamespace:
		return types.String(w.namespace)
	default:
		return types.String("")
	}
}
func (w *workerPodKeyVal) HasField(field string) bool {
	return field == GPUFieldName || field == GPUFieldNamespace
}

type appVal struct {
	name      string
	namespace string
	count     int64
}

func (a *appVal) Type() ref.Type              { return types.MapType }
func (a *appVal) Value() interface{}          { return nil }
func (a *appVal) Equal(other ref.Val) ref.Val { return types.False }
func (a *appVal) ConvertToNative(t reflect.Type) (interface{}, error) {
	return map[string]interface{}{
		"name":      a.name,
		"namespace": a.namespace,
		"count":     a.count,
	}, nil
}
func (a *appVal) ConvertToType(typeValue ref.Type) ref.Val {
	return types.NewErr("type conversion not supported")
}
func (a *appVal) Get(index ref.Val) ref.Val {
	key, _ := index.Value().(string)
	switch key {
	case "name":
		return types.String(a.name)
	case "namespace":
		return types.String(a.namespace)
	case "count":
		return types.Int(a.count)
	default:
		return types.String("")
	}
}
func (a *appVal) HasField(field string) bool {
	return field == "name" || field == "namespace" || field == "count"
}

type runningAppsVal struct {
	apps []tfv1.RunningAppDetail
}

func (r *runningAppsVal) Type() ref.Type              { return types.ListType }
func (r *runningAppsVal) Value() interface{}          { return r.apps }
func (r *runningAppsVal) Equal(other ref.Val) ref.Val { return types.False }
func (r *runningAppsVal) ConvertToNative(t reflect.Type) (interface{}, error) {
	if t.Kind() == reflect.Slice {
		out := make([]map[string]interface{}, len(r.apps))
		for i, a := range r.apps {
			out[i] = map[string]interface{}{
				"name":      a.Name,
				"namespace": a.Namespace,
				"count":     a.Count,
			}
		}
		return out, nil
	}
	return r.apps, nil
}
func (r *runningAppsVal) ConvertToType(typeValue ref.Type) ref.Val {
	return types.NewErr("type conversion not supported")
}
func (r *runningAppsVal) Get(index ref.Val) ref.Val {
	i, ok := index.Value().(int)
	if !ok {
		if i64, ok2 := index.Value().(int64); ok2 {
			i = int(i64)
			ok = true
		}
	}
	if !ok || i < 0 || i >= len(r.apps) {
		return types.NewErr("index out of range")
	}
	app := r.apps[i]
	return &appVal{name: app.Name, namespace: app.Namespace, count: int64(app.Count)}
}

func (r *runningAppsVal) Size() ref.Val { return types.Int(len(r.apps)) }

func (r *runningAppsVal) Contains(elem ref.Val) ref.Val {
	av, ok := elem.(*appVal)
	if !ok {
		return types.False
	}
	for _, a := range r.apps {
		if a.Name == av.name && a.Namespace == av.namespace && int64(a.Count) == av.count {
			return types.True
		}
	}
	return types.False
}
func (r *runningAppsVal) Iterator() traits.Iterator {
	return &runningAppsIterator{apps: r.apps}
}
func (r *runningAppsVal) Add(elem ref.Val) ref.Val {
	return types.NewErr("runningApps list is read-only")
}

type runningAppsIterator struct {
	apps []tfv1.RunningAppDetail
	i    int
}

func (it *runningAppsIterator) Type() ref.Type              { return types.IteratorType }
func (it *runningAppsIterator) Value() interface{}          { return nil }
func (it *runningAppsIterator) Equal(other ref.Val) ref.Val { return types.False }
func (it *runningAppsIterator) ConvertToNative(t reflect.Type) (interface{}, error) {
	return nil, fmt.Errorf("iterator cannot convert to native")
}
func (it *runningAppsIterator) ConvertToType(typeValue ref.Type) ref.Val {
	return types.NewErr("type conversion not supported")
}
func (it *runningAppsIterator) HasNext() ref.Val {
	return types.Bool(it.i < len(it.apps))
}
func (it *runningAppsIterator) Next() ref.Val {
	if it.i >= len(it.apps) {
		return types.NewErr("iterator past end")
	}
	a := it.apps[it.i]
	it.i++
	return &appVal{name: a.Name, namespace: a.Namespace, count: int64(a.Count)}
}

var _ traits.Lister = (*runningAppsVal)(nil)
var _ traits.Iterator = (*runningAppsIterator)(nil)

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
		return types.Bool(v.UID == otherGPU.UID)
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
		return types.String(v.Name)
	case GPUFieldNamespace:
		return types.String(v.Namespace)
	case GPUFieldGPUModel:
		return types.String(v.Status.GPUModel)
	case GPUFieldUUID:
		return types.String(v.Status.UUID)
	case GPUFieldPhase:
		return getPooledPhaseString(string(v.Status.Phase))
	case GPUFieldUsedBy:
		return types.String(string(v.Status.UsedBy))
	case GPUFieldMessage:
		return types.String(v.Status.Message)
	case GPUFieldLabels:
		// Lazy initialization with caching
		if v.labels == nil {
			v.labels = &labelsVal{labels: v.Labels}
		}
		return v.labels
	case GPUFieldAnnotations:
		// Lazy initialization with caching
		if v.annotations == nil {
			v.annotations = &labelsVal{labels: v.Annotations}
		}
		return v.annotations
	case GPUFieldAvailable:
		// Lazy initialization with caching
		if v.available == nil {
			v.available = &availableVal{available: v.Status.Available}
		}
		return v.available
	case GPUFieldNodeSelector:
		// Lazy initialization with caching
		if v.nodeSelector == nil {
			v.nodeSelector = &labelsVal{labels: v.Status.NodeSelector}
		}
		return v.nodeSelector
	case GPUFieldRunningApps:
		// For now, keep simple implementation - can optimize later if needed
		if v.runningApps == nil {
			apps := make([]tfv1.RunningAppDetail, len(v.Status.RunningApps))
			for i, app := range v.Status.RunningApps {
				apps[i] = *app
			}
			v.runningApps = &runningAppsVal{apps: apps}
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
		case ResourceFieldTFlops:
			return types.Double(0.0)
		case ResourceFieldVRAM:
			return types.Double(0.0)
		default:
			return types.NewErr("no such field: %s", field)
		}
	}

	switch field {
	case ResourceFieldTFlops:
		return types.Double(v.available.Tflops.AsApproximateFloat64())
	case ResourceFieldVRAM:
		return types.Double(float64(v.available.Vram.Value()))
	default:
		return types.NewErr("no such field: %s", field)
	}
}

// HasField implements field testing
func (v *availableVal) HasField(field string) bool {
	return field == ResourceFieldTFlops || field == ResourceFieldVRAM
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
	requiredPhases   []tfv1.TensorFusionGPUPhase
	requiredGPUModel string
	userExpression   string
	// Track which fields are actually used
	usage fieldUsage
	// Display expression for logging (read-only)
	displayExpression string
}

// NewAllocRequestCELFilter creates a new CEL filter from allocation request
func NewCELFilter(req *tfv1.AllocRequest, cache *ExpressionCache) (*CELFilter, error) {
	// Extract early filtering criteria
	var requiredPhases []tfv1.TensorFusionGPUPhase
	var requiredGPUModel, userExpression, displayExpression string

	if req != nil {
		requiredPhases = []tfv1.TensorFusionGPUPhase{
			tfv1.TensorFusionGPUPhaseRunning,
			tfv1.TensorFusionGPUPhasePending,
		}
		requiredGPUModel = req.GPUModel
		userExpression = req.CELFilterExpression

		// Build display expression for logging (not used for execution)
		displayExpression = buildDisplayExpression(req)
	}

	// Analyze field usage in user expression only
	usage := analyzeFieldUsage(userExpression)

	// Handle nil request case
	name := "AllocRequest-unknown"
	if req != nil {
		name = fmt.Sprintf("AllocRequest-%s", req.WorkloadNameNamespace.String())
	}

	return &CELFilter{
		cache:             cache,
		name:              name,
		requiredPhases:    requiredPhases,
		requiredGPUModel:  requiredGPUModel,
		userExpression:    userExpression,
		usage:             usage,
		displayExpression: displayExpression,
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

	// Pre-allocate result slice with estimated capacity for early filtering
	var filteredGPUs []*tfv1.GPU

	// Early filtering phase: apply basic filters first to reduce CEL evaluation overhead
	earlyFilteredGPUs := make([]*tfv1.GPU, 0, len(gpus))
	for _, gpu := range gpus {
		// when running progressive migration mode, only return GPUs used by tensor-fusion
		if utils.IsProgressiveMigration() && gpu.Status.UsedBy != tfv1.UsedByTensorFusion {
			continue
		}
		// Fast path: check phase first (most common filter)
		if f.requiredPhases != nil && !lo.Contains(f.requiredPhases, gpu.Status.Phase) {
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

	// Fallback to CEL evaluation for complex expressions
	if len(earlyFilteredGPUs) >= ParallelThreshold {
		// Use parallel evaluation for large GPU sets
		filteredGPUs = f.filterFallbackParallel(program, earlyFilteredGPUs, workerPodKey)
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

// filterFallbackSequential performs sequential CEL evaluation for smaller GPU sets
func (f *CELFilter) filterFallbackSequential(ctx context.Context, program cel.Program, gpus []*tfv1.GPU, workerPodKey tfv1.NameNamespace) []*tfv1.GPU {
	filteredGPUs := make([]*tfv1.GPU, 0, len(gpus)/2)
	log := log.FromContext(ctx)
	var activation ZeroAllocActivation
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
		activation.init(gpu, workerPodKey, f.usage)

		// Direct synchronous evaluation with custom activation
		result, _, evalErr := program.Eval(&activation)

		if evalErr != nil {
			log.Error(evalErr, "CEL expression evaluation failed",
				"expression", f.userExpression,
				"gpu", gpu.Name,
				"workerPodKey", workerPodKey)
			// On error, exclude the GPU (fail-safe)
			continue
		}

		// Convert result to boolean
		if boolResult, ok := result.(types.Bool); ok && bool(boolResult) {
			filteredGPUs = append(filteredGPUs, gpu)
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
func (f *CELFilter) filterFallbackParallel(program cel.Program, gpus []*tfv1.GPU, workerPodKey tfv1.NameNamespace) []*tfv1.GPU {
	numGPUs := len(gpus)
	numWorkers := runtime.NumCPU()
	if numWorkers > DefaultWorkerCount {
		numWorkers = DefaultWorkerCount
	}

	chunkSize := (numGPUs + numWorkers - 1) / numWorkers
	resultChannels := make([]<-chan []*tfv1.GPU, numWorkers)
	var activation ZeroAllocActivation
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
				activation.init(gpu, workerPodKey, f.usage)

				// Direct synchronous evaluation
				result, _, evalErr := program.Eval(&activation)
				if evalErr != nil {
					// On error, exclude the GPU (fail-safe)
					continue
				}

				// Convert result to boolean
				if boolResult, ok := result.(types.Bool); ok && bool(boolResult) {
					filtered = append(filtered, gpu)
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
