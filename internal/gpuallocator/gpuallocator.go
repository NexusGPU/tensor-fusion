// Package gpuallocator handles GPU allocation
package gpuallocator

import (
	"context"
	"fmt"
	"maps"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	goerrors "github.com/pkg/errors"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator/filter"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/quota"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	fwk "k8s.io/kube-scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

const MaxGPUCounterPerAllocation = 128
const CleanUpCheckInterval = 3 * time.Minute

var mu sync.Mutex
var GPUCapacityMap = map[string]tfv1.Resource{}

type Strategy interface {
	// When isForNode = true, indicates each GPU's node level score
	// otherwise it's single GPU score inside one node
	Score(gpu *tfv1.GPU, isForNode bool) int

	SelectGPUs(gpus []*tfv1.GPU, count uint) ([]*tfv1.GPU, error)
}

// When called /api/simulate-schedule with Pod yaml as body, return detailed filter details
type SimulateSchedulingFilterDetail struct {
	FilterStageDetails []filter.FilterDetail
}

func (p *SimulateSchedulingFilterDetail) Clone() fwk.StateData {
	return p
}

// NewStrategy creates a strategy based on the placement mode
func NewStrategy(placementMode tfv1.PlacementMode, cfg *config.GPUFitConfig, nodeGpuStore map[string]map[string]*tfv1.GPU) Strategy {
	switch placementMode {
	case tfv1.PlacementModeLowLoadFirst:
		return LowLoadFirst{cfg: cfg, nodeGpuStore: nodeGpuStore}
	case tfv1.PlacementModeCompactFirst:
		return CompactFirst{cfg: cfg, nodeGpuStore: nodeGpuStore}
	default:
		return NodeCompactGPULowLoad{cfg: cfg, nodeGpuStore: nodeGpuStore}
	}
}

type GpuAllocator struct {
	client.Client
	filterRegistry *filter.FilterRegistry
	quotaStore     *quota.QuotaStore

	// In-memory store of GPUs
	gpuStore        map[types.NamespacedName]*tfv1.GPU
	nodeGpuStore    map[string]map[string]*tfv1.GPU
	poolGpuStore    map[string]map[string]*tfv1.GPU
	nodeWorkerStore map[string]map[types.NamespacedName]struct{}
	storeMutex      sync.RWMutex
	allocateMutex   sync.Mutex // serializes legacy Alloc() calls (CheckQuotaAndFilter + Bind)
	syncInterval    time.Duration
	cancel          context.CancelFunc
	ctx             context.Context

	// Queue for tracking modified GPUs that need to be synced
	dirtyQueue     map[types.NamespacedName]struct{}
	dirtyQueueLock sync.Mutex

	// each pod can only allocate and deallocate once, and deallocation must be after allocation
	uniqueAllocation       map[string]*tfv1.AllocRequest
	uniqueDeallocation     map[string]struct{}
	podNamespaceNsToPodUID map[string]string

	maxWorkerPerNode int

	initGPUStoreOnce    sync.Once
	reconcileWorkerOnce sync.Once
	initializedCh       chan struct{}

	bindHandlers []func(req *tfv1.AllocRequest)
}

func NewGpuAllocator(ctx context.Context, client client.Client, syncInterval time.Duration) *GpuAllocator {
	log := log.FromContext(ctx)

	if client == nil {
		log.Error(fmt.Errorf("client cannot be nil"), "Failed to create GPU allocator")
		return nil
	}

	// Create base filter store with common filters
	baseRegistry := filter.NewFilterRegistry().With(
		filter.NewPhaseFilter(tfv1.TensorFusionGPUPhaseRunning),
	)

	// Create quota store
	quotaStore := quota.NewQuotaStore(client, ctx)

	allocator := &GpuAllocator{
		Client:          client,
		filterRegistry:  baseRegistry,
		quotaStore:      quotaStore,
		gpuStore:        make(map[types.NamespacedName]*tfv1.GPU),
		nodeGpuStore:    make(map[string]map[string]*tfv1.GPU),
		poolGpuStore:    make(map[string]map[string]*tfv1.GPU),
		nodeWorkerStore: make(map[string]map[types.NamespacedName]struct{}),
		syncInterval:    syncInterval,
		dirtyQueue:      make(map[types.NamespacedName]struct{}),
		ctx:             ctx,

		uniqueAllocation:       make(map[string]*tfv1.AllocRequest, 512),
		uniqueDeallocation:     make(map[string]struct{}, 512),
		podNamespaceNsToPodUID: make(map[string]string, 512),
		initializedCh:          make(chan struct{}),
	}

	return allocator
}

func (s *GpuAllocator) RegisterBindHandler(handler func(req *tfv1.AllocRequest)) {
	s.bindHandlers = append(s.bindHandlers, handler)
}

func (s *GpuAllocator) GetAllocationInfo() (
	gpuStore map[types.NamespacedName]*tfv1.GPU,
	nodeWorkerStore map[string]map[types.NamespacedName]struct{},
	uniqueAllocation map[string]*tfv1.AllocRequest,
) {
	s.storeMutex.RLock()
	defer s.storeMutex.RUnlock()
	return s.gpuStore, s.nodeWorkerStore, s.uniqueAllocation
}

// GetNodeGpuStore returns a snapshot (shallow copy) of the node-to-GPU map.
// The caller can safely iterate the returned maps without holding any lock.
// The *tfv1.GPU pointers are shared with the allocator's internal state.
func (s *GpuAllocator) GetNodeGpuStore() map[string]map[string]*tfv1.GPU {
	s.storeMutex.RLock()
	defer s.storeMutex.RUnlock()
	result := make(map[string]map[string]*tfv1.GPU, len(s.nodeGpuStore))
	for nodeName, gpuMap := range s.nodeGpuStore {
		innerCopy := make(map[string]*tfv1.GPU, len(gpuMap))
		maps.Copy(innerCopy, gpuMap)
		result[nodeName] = innerCopy
	}
	return result
}

// AllocRequest encapsulates all parameters needed for GPU allocation
func (s *GpuAllocator) SetMaxWorkerPerNode(maxWorkerPerNode int) {
	s.maxWorkerPerNode = maxWorkerPerNode
}

var ScalingQuotaExceededError = goerrors.New("scaling quota exceeded")

func IsScalingQuotaExceededError(err error) bool {
	return goerrors.Is(err, ScalingQuotaExceededError)
}

// Filter applies filters to a pool of GPUs based on the provided request and returns selected GPUs.
// It does not modify the GPU resources, only filters and selects them.
func (s *GpuAllocator) Filter(
	req *tfv1.AllocRequest,
	toFilterGPUs []*tfv1.GPU,
	isSimulateSchedule bool,
) ([]*tfv1.GPU, []filter.FilterDetail, error) {
	// Add SameNodeFilter if count > 1 to ensure GPUs are from the same node
	filterRegistry := s.filterRegistry.With(filter.NewResourceFilter(req.Request, req.GPUIndices))

	// Add GPU model filter if specified
	if req.GPUModel != "" {
		filterRegistry = filterRegistry.With(filter.NewGPUModelAndVendorFilter(req.GPUModel, req.GPUVendor))
	}

	// NOTE: deprecated, use Kubernetes native spec template affinity way
	if req.NodeAffinity != nil {
		filterRegistry = filterRegistry.With(filter.NewNodeAffinityFilter(s.Client, req.NodeAffinity))
	}

	// Same node filter must be applied at final step
	if req.Count > 1 {
		filterRegistry = filterRegistry.With(filter.NewSameNodeFilter(req.Count))
	}

	// Apply the filters in sequence
	filteredGPUs, filterDetails, err := filterRegistry.Apply(s.ctx, req.WorkloadNameNamespace, toFilterGPUs, isSimulateSchedule)
	if err != nil {
		return nil, nil, fmt.Errorf("apply filters: %w", err)
	}

	return filteredGPUs, filterDetails, nil
}

func (s *GpuAllocator) FilterWithPreempt(
	req *tfv1.AllocRequest,
	preemptAllocRequests []*tfv1.AllocRequest,
	targetNodeNames ...string, // Optional: if provided, only consider GPUs from these nodes
) ([]*tfv1.GPU, []filter.FilterDetail, error) {
	toFilterGPUs := []*tfv1.GPU{}

	affectedNodes := make(map[string]bool)
	// use map for O(1) duplicate check instead of O(M) array iteration
	includedGPUs := make(map[string]bool)
	// if specified target nodes, only consider these nodes
	if len(targetNodeNames) > 0 {
		for _, nodeName := range targetNodeNames {
			affectedNodes[nodeName] = true
		}
	}

	// Step 1: Collect GPUs that will be released by preempted pods
	// Use a temporary map to accumulate releases for the same GPU (multiple victims on same GPU)
	gpuReleasedMap := make(map[string]*tfv1.GPU)

	log.FromContext(s.ctx).V(4).Info("[PREEMPT-DBG] FilterWithPreempt: starting to simulate release",
		"preemptAllocRequestsCount", len(preemptAllocRequests))

	for i, preemptAllocRequest := range preemptAllocRequests {
		log.FromContext(s.ctx).V(4).Info("[PREEMPT-DBG] Processing victim allocation",
			"victimIndex", i,
			"workload", preemptAllocRequest.WorkloadNameNamespace.Name,
			"gpuCount", preemptAllocRequest.Count,
			"gpuNames", preemptAllocRequest.GPUNames,
			"requestTflops", preemptAllocRequest.Request.Tflops.String(),
			"requestComputePercent", preemptAllocRequest.Request.ComputePercent.String(),
			"requestVram", preemptAllocRequest.Request.Vram.String())

		for _, gpuName := range preemptAllocRequest.GPUNames {
			// Get or create GPU copy for accumulating releases
			gpuCopy, exists := gpuReleasedMap[gpuName]
			var beforeTflops, beforeVram string
			if !exists {
				gpu := s.gpuStore[types.NamespacedName{Name: gpuName}]
				if gpu == nil {
					return nil, nil, fmt.Errorf("gpu %s not found", gpuName)
				}
				gpuCopy = gpu.DeepCopy()
				gpuReleasedMap[gpuName] = gpuCopy
				includedGPUs[gpuCopy.Name] = true
			}

			// Log GPU state before this release
			if gpuCopy.Status.Available != nil {
				beforeTflops = gpuCopy.Status.Available.Tflops.String()
				beforeVram = gpuCopy.Status.Available.Vram.String()
			}

			// Simulate releasing resources (accumulate if this GPU already had releases)
			var reqTflops resource.Quantity
			if !preemptAllocRequest.Request.ComputePercent.IsZero() {
				requiredTflops := utils.ComputePercentToTflops(gpuCopy.Status.Capacity.Tflops, preemptAllocRequest.Request)
				reqTflops = *requiredTflops
			} else {
				reqTflops = preemptAllocRequest.Request.Tflops
			}
			gpuCopy.Status.Available.Tflops.Add(reqTflops)
			gpuCopy.Status.Available.Vram.Add(preemptAllocRequest.Request.Vram)

			// Log GPU state after simulated release
			log.FromContext(s.ctx).V(4).Info("[PREEMPT-DBG] Simulated release on GPU",
				"gpu", gpuName,
				"node", gpuCopy.Status.NodeSelector[constants.KubernetesHostNameLabel],
				"victimWorkload", preemptAllocRequest.WorkloadNameNamespace.Name,
				"releasedTflops", reqTflops.String(),
				"releasedVram", preemptAllocRequest.Request.Vram.String(),
				"beforeAvailableTflops", beforeTflops,
				"afterAvailableTflops", gpuCopy.Status.Available.Tflops.String(),
				"beforeAvailableVram", beforeVram,
				"afterAvailableVram", gpuCopy.Status.Available.Vram.String(),
				"capacityTflops", gpuCopy.Status.Capacity.Tflops.String())

			// Add the node of the released GPU to affectedNodes if no target nodes are specified
			if len(targetNodeNames) == 0 {
				nodeName := gpuCopy.Status.NodeSelector[constants.KubernetesHostNameLabel]
				if nodeName != "" {
					affectedNodes[nodeName] = true
				}
			}
		}
	}

	// Convert map to slice for filtering
	for _, gpuCopy := range gpuReleasedMap {
		toFilterGPUs = append(toFilterGPUs, gpuCopy)
	}
	// Step2: Add other available GPUs from the same node(s) in the pool
	// This is critical for multi-GPU pods: preemption might release 1 GPU, but the node
	// might already have other free GPUs that together satisfy the requirement
	if len(affectedNodes) == 0 {
		// This should not happen: FilterWithPreempt requires either targetNodeNames or preemptAllocRequests
		return nil, nil, fmt.Errorf("FilterWithPreempt called with no affected nodes, invalid usage")
	}

	// only iterate the affected nodes, instead of the entire pool(O(N) instead of O(P))
	for nodeName := range affectedNodes {
		nodeGPUs := s.nodeGpuStore[nodeName]
		log.FromContext(s.ctx).V(4).Info("[PREEMPT-DBG] Adding other GPUs from affected node",
			"node", nodeName,
			"totalGPUsOnNode", len(nodeGPUs),
			"alreadyIncluded", len(includedGPUs))

		for _, gpu := range nodeGPUs {
			// Use map for O(1) duplicate check instead of O(M) array iteration
			if !includedGPUs[gpu.Name] {
				toFilterGPUs = append(toFilterGPUs, gpu.DeepCopy())
				includedGPUs[gpu.Name] = true
			}
		}

		log.FromContext(s.ctx).V(4).Info("[PREEMPT-DBG] Finished adding GPUs from node",
			"node", nodeName,
			"addedGPUs", len(toFilterGPUs)-len(gpuReleasedMap),
			"totalToFilterGPUs", len(toFilterGPUs))
	}

	// Step 3: Apply filters including SameNodeFilter for multi-GPU requirements
	log.FromContext(s.ctx).V(4).Info("[PREEMPT-DBG] FilterWithPreempt: applying filters",
		"toFilterGPUsCount", len(toFilterGPUs),
		"requiredGPUs", req.Count,
		"requiredTflops", req.Request.Tflops.String(),
		"requiredComputePercent", req.Request.ComputePercent.String(),
		"requiredVram", req.Request.Vram.String(),
		"gpuModel", req.GPUModel,
		"gpuVendor", req.GPUVendor,
		"gpuIndices", req.GPUIndices)

	filterRegistry := s.filterRegistry.With(filter.NewResourceFilter(req.Request, req.GPUIndices))

	// Add GPU model filter if specified
	if req.GPUModel != "" {
		filterRegistry = filterRegistry.With(filter.NewGPUModelAndVendorFilter(req.GPUModel, req.GPUVendor))
	}

	// Apply SameNodeFilter for multi-GPU pods to ensure all GPUs are from the same node
	if req.Count > 1 {
		filterRegistry = filterRegistry.With(filter.NewSameNodeFilter(req.Count))
	}

	filteredGPUs, filterDetails, err := filterRegistry.Apply(s.ctx, req.WorkloadNameNamespace, toFilterGPUs, false)
	if err != nil {
		return nil, nil, fmt.Errorf("apply filters: %w", err)
	}

	log.FromContext(s.ctx).V(4).Info("[PREEMPT-DBG] FilterWithPreempt: filter results",
		"filteredGPUsCount", len(filteredGPUs),
		"toFilterGPUsCount", len(toFilterGPUs),
		"filterDetailsCount", len(filterDetails))

	return filteredGPUs, filterDetails, nil
}

func (s *GpuAllocator) Select(req *tfv1.AllocRequest, filteredGPUs []*tfv1.GPU) ([]*tfv1.GPU, error) {
	pool := &tfv1.GPUPool{}
	if err := s.Get(s.ctx, client.ObjectKey{Name: req.PoolName}, pool); err != nil {
		return nil, fmt.Errorf("get pool %s: %w", req.PoolName, err)
	}

	schedulingConfigTemplate := &tfv1.SchedulingConfigTemplate{}
	if pool.Spec.SchedulingConfigTemplate != nil {
		if err := s.Get(s.ctx, client.ObjectKey{Name: *pool.Spec.SchedulingConfigTemplate}, schedulingConfigTemplate); err != nil {
			return nil, fmt.Errorf("get scheduling config template %s: %w", *pool.Spec.SchedulingConfigTemplate, err)
		}
	}

	strategy := NewStrategy(schedulingConfigTemplate.Spec.Placement.Mode, &config.GPUFitConfig{
		MaxWorkerPerNode: s.maxWorkerPerNode,
	}, s.nodeGpuStore)
	selectedGPUs, err := strategy.SelectGPUs(filteredGPUs, req.Count)
	if err != nil {
		return nil, fmt.Errorf("select GPU: %w", err)
	}

	// Return copies of the selected GPUs
	result := make([]*tfv1.GPU, len(selectedGPUs))
	for i, gpu := range selectedGPUs {
		result[i] = gpu.DeepCopy()
	}

	return result, nil
}

// Bind allocates resources on the provided GPUs for the given request.
// It updates the in-memory store and marks the GPUs as dirty for syncing.
func (s *GpuAllocator) Bind(
	gpuNames []string,
	req *tfv1.AllocRequest,
) ([]*tfv1.GPU, error) {
	<-s.initializedCh
	if len(gpuNames) == 0 {
		return nil, fmt.Errorf("no GPUs provided to bind")
	}

	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	if _, exists := s.uniqueAllocation[string(req.PodMeta.UID)]; exists {
		return nil, fmt.Errorf("pod %s has already allocated GPUs", req.PodMeta.UID)
	}

	// Proceed with GPU allocation
	gpuNodeName := ""
	for _, selectedGPU := range gpuNames {
		// Get the GPU from the store
		key := types.NamespacedName{Name: selectedGPU}
		gpu, exists := s.gpuStore[key]
		if !exists {
			return nil, fmt.Errorf("scheduled GPU %s not found in store", selectedGPU)
		}

		if gpuNodeName == "" {
			gpuNodeName = gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
		}

		// Double-check resource availability to prevent over-allocation
		if gpu.Status.Available == nil {
			return nil, fmt.Errorf("GPU %s has nil available resources", selectedGPU)
		}
		var reqTflops resource.Quantity
		if !req.Request.ComputePercent.IsZero() {
			requiredTflops := utils.ComputePercentToTflops(gpu.Status.Capacity.Tflops, req.Request)
			reqTflops = *requiredTflops
		} else {
			reqTflops = req.Request.Tflops
		}
		if gpu.Status.Available.Tflops.Cmp(reqTflops) < 0 {
			return nil, fmt.Errorf("GPU %s insufficient TFLOPs: available %s, requested %s",
				selectedGPU, gpu.Status.Available.Tflops.String(), reqTflops.String())
		}
		if gpu.Status.Available.Vram.Cmp(req.Request.Vram) < 0 {
			return nil, fmt.Errorf("GPU %s insufficient VRAM: available %s, requested %s",
				selectedGPU, gpu.Status.Available.Vram.String(), req.Request.Vram.String())
		}

		// reduce available resource on the GPU status

		if !req.Request.ComputePercent.IsZero() {
			requiredTflops := utils.ComputePercentToTflops(gpu.Status.Capacity.Tflops, req.Request)
			gpu.Status.Available.Tflops.Sub(*requiredTflops)
		} else {
			gpu.Status.Available.Tflops.Sub(req.Request.Tflops)
		}
		gpu.Status.Available.Vram.Sub(req.Request.Vram)

		addRunningApp(s.ctx, gpu, req)

		s.markGPUDirty(key)
	}

	// Allocate quota resources (atomic with GPU allocation)
	// Use actual allocated GPU count instead of requested count
	s.quotaStore.AllocateQuota(req.WorkloadNameNamespace.Namespace, req)
	s.addAllocationMap(gpuNodeName, req.PodMeta)
	metrics.SetSchedulerMetrics(req.PoolName, true)

	log.FromContext(s.ctx).Info("GPU allocation successful",
		"namespace", req.WorkloadNameNamespace.Namespace,
		"workload", req.WorkloadNameNamespace.Name,
		"podName", req.PodMeta.Name,
		"gpu_count", req.Count,
		"gpuNames", gpuNames,
		"node", gpuNodeName,
		"tflops", req.Request.Tflops.String(),
		"vram", req.Request.Vram.String())

	// Return copies of the bound GPUs from the store
	result := make([]*tfv1.GPU, req.Count)
	for i, gpuName := range gpuNames {
		key := types.NamespacedName{Name: gpuName}
		result[i] = s.gpuStore[key].DeepCopy()
	}
	req.GPUNames = gpuNames
	s.uniqueAllocation[string(req.PodMeta.UID)] = req
	delete(s.uniqueDeallocation, string(req.PodMeta.UID))
	s.podNamespaceNsToPodUID[req.PodMeta.Namespace+"/"+req.PodMeta.Name] = string(req.PodMeta.UID)

	for _, handler := range s.bindHandlers {
		handler(req)
	}

	return result, nil
}

// Alloc allocates a request to a gpu or multiple gpus from the same node.
// This is now implemented as a combination of Filter and Bind for backward compatibility.
func (s *GpuAllocator) Alloc(req *tfv1.AllocRequest) ([]*tfv1.GPU, error) {
	s.allocateMutex.Lock()
	defer s.allocateMutex.Unlock()

	filteredGPUs, _, err := s.CheckQuotaAndFilter(s.ctx, req, false)
	if err != nil {
		metrics.SetSchedulerMetrics(req.PoolName, false)
		return nil, err
	}
	selectedGPUs, err := s.Select(req, filteredGPUs)
	if err != nil {
		metrics.SetSchedulerMetrics(req.PoolName, false)
		return nil, err
	}

	// Then, bind resources to the selected GPUs
	gpuNames := lo.Map(selectedGPUs, func(gpu *tfv1.GPU, _ int) string {
		return gpu.Name
	})
	return s.Bind(gpuNames, req)
}

func (s *GpuAllocator) CheckQuotaAndFilter(ctx context.Context, req *tfv1.AllocRequest, isSimulateSchedule bool) ([]*tfv1.GPU, []filter.FilterDetail, error) {
	<-s.initializedCh

	// TODO not support compute percent quota check yet, percent is related to GPU capacity, will bypass all quota check
	// Using Percent to configure GPU computing request is NOT Recommended way, should offer a toggle to enable/disable it
	if err := s.quotaStore.CheckQuotaAvailable(req); err != nil {
		return nil, nil, err
	}

	// Get GPUs from the pool using the in-memory store
	if req.PoolName == "" {
		return nil, nil, fmt.Errorf("GPU Pool name is empty, can not find GPUs")
	}
	poolGPUs := s.listGPUsFromPool(req.PoolName)
	if len(poolGPUs) == 0 {
		return nil, nil, fmt.Errorf("no gpu devices in pool %s", req.PoolName)
	}
	filteredGPUs, filterDetails, err := s.Filter(req, poolGPUs, isSimulateSchedule)
	if err != nil {
		return nil, nil, err
	}
	if len(filteredGPUs) == 0 {
		return nil, filterDetails, fmt.Errorf("no gpus available or valid in pool %s after filtering", req.PoolName)
	}

	if s.maxWorkerPerNode > 0 {
		s.storeMutex.RLock()
		// First pass: check if any filtering is needed
		needsFiltering := false
		for _, gpu := range filteredGPUs {
			nodeName := gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
			if len(s.nodeWorkerStore[nodeName]) > s.maxWorkerPerNode {
				needsFiltering = true
				break
			}
		}

		// Only allocate and filter if necessary
		if needsFiltering {
			finalFilteredGPUs := make([]*tfv1.GPU, 0, len(filteredGPUs))
			for _, gpu := range filteredGPUs {
				nodeName := gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
				if len(s.nodeWorkerStore[nodeName]) <= s.maxWorkerPerNode {
					finalFilteredGPUs = append(finalFilteredGPUs, gpu)
				}
			}
			filteredGPUs = finalFilteredGPUs
		}
		s.storeMutex.RUnlock()
	}

	return filteredGPUs, filterDetails, nil
}

func (s *GpuAllocator) DeallocAsync(
	workloadNameNamespace tfv1.NameNamespace,
	gpus []string,
	podMeta metav1.ObjectMeta,
) {
	go func() {
		retry := 0
		for {
			pod := &v1.Pod{}
			if err := s.Get(s.ctx, client.ObjectKey{Namespace: podMeta.Namespace, Name: podMeta.Name}, pod); err != nil {
				if errors.IsNotFound(err) {
					s.Dealloc(workloadNameNamespace, gpus, podMeta)
					return
				}
			}
			time.Sleep(utils.CalculateExponentialBackoffWithJitter(int64(retry)))
			retry++
		}
	}()
}

// Dealloc a request from gpu to release available resources on it.
func (s *GpuAllocator) Dealloc(
	workloadNameNamespace tfv1.NameNamespace,
	gpus []string,
	podMeta metav1.ObjectMeta,
) {
	<-s.initializedCh
	podUID := string(podMeta.UID)
	log := log.FromContext(s.ctx)

	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	request, exists := s.uniqueAllocation[podUID]
	if !exists || request == nil {
		// should not block finalizer
		log.Error(fmt.Errorf("pod has not allocated GPUs"), "pod", podUID)
		return
	}

	if _, exists := s.uniqueDeallocation[podUID]; exists {
		delete(s.uniqueAllocation, podUID)
		// should not block finalizer
		log.Error(fmt.Errorf("pod has already deallocated GPUs"), "pod", podUID)
		return
	}

	nodeName := ""
	for _, gpu := range gpus {
		// Get the GPU from the store
		gpuNameNs := types.NamespacedName{Name: gpu}
		storeGPU, exists := s.gpuStore[gpuNameNs]
		if !exists {
			log.Error(fmt.Errorf("GPU not found in store"), "Failed to deallocate GPU", "name", gpu)
			continue
		}

		// Add resources back to the GPU
		if !request.Request.ComputePercent.IsZero() {
			requiredTflops := utils.ComputePercentToTflops(storeGPU.Status.Capacity.Tflops, request.Request)
			storeGPU.Status.Available.Tflops.Add(*requiredTflops)
		} else {
			storeGPU.Status.Available.Tflops.Add(request.Request.Tflops)
		}
		storeGPU.Status.Available.Vram.Add(request.Request.Vram)

		if nodeName == "" {
			nodeName = storeGPU.Status.NodeSelector[constants.KubernetesHostNameLabel]
		}

		removeRunningApp(s.ctx, storeGPU, request)

		s.markGPUDirty(gpuNameNs)
	}

	// remove pod from nodeWorkerStore
	delete(s.nodeWorkerStore[nodeName], types.NamespacedName{Name: podMeta.Name, Namespace: podMeta.Namespace})
	delete(s.uniqueAllocation, podUID)
	delete(s.podNamespaceNsToPodUID, podMeta.Namespace+"/"+podMeta.Name)
	s.uniqueDeallocation[podUID] = struct{}{}

	// Deallocate quota resources in memory (atomic operation)
	s.quotaStore.DeallocateQuota(workloadNameNamespace.Namespace, request)

	log.Info("GPU deallocation successful",
		"namespace", workloadNameNamespace.Namespace,
		"workload", workloadNameNamespace.Name,
		"podName", podMeta.Name,
		"gpu_count", len(gpus),
		"gpuNames", gpus,
		"tflops", request.Request.Tflops.String(),
		"vram", request.Request.Vram.String())
}

// Used for scale up decision, dryRun to pre-check capacity to determine if the allocation
// is valid when scaling up, return error and the max new requests/limits on existing GPU
// Auto scaler can directly call AdjustAllocation for scaling down decision
// it has to call AdjustAllocation with dryRun=true when scaling up,
// if return error is ScalingQuotaExceededError,
// it means the allocation is invalid, and it should scale up with another AdjustRequest
// to make sure not exceed quota, which returns in the first returned result
// retry until AdjustAllocation returns nil error, at most pre-configured maxRetry times
// returns remaining resource, delta resource, error
func (s *GpuAllocator) AdjustAllocation(ctx context.Context, adjustRequest tfv1.AdjustRequest, dryRun bool) (tfv1.Resource, tfv1.Resource, error) {

	<-s.initializedCh
	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	request, exists := s.uniqueAllocation[adjustRequest.PodUID]
	if !exists || request == nil {
		return tfv1.Resource{}, tfv1.Resource{}, fmt.Errorf("pod %s has not allocated GPUs", adjustRequest.PodUID)
	}

	deltaTFlopsRequest := adjustRequest.NewRequest.Tflops
	deltaTFlopsRequest.Sub(request.Request.Tflops)

	deltaVRAMRequest := adjustRequest.NewRequest.Vram
	deltaVRAMRequest.Sub(request.Request.Vram)

	deltaTFlopsLimit := adjustRequest.NewLimit.Tflops
	deltaTFlopsLimit.Sub(request.Request.Tflops)

	deltaVRAMLimit := adjustRequest.NewLimit.Vram
	deltaVRAMLimit.Sub(request.Request.Vram)

	if adjustRequest.IsScaleUp {
		for _, gpuName := range request.GPUNames {
			gpuNameNs := types.NamespacedName{Name: gpuName}
			gpu, exists := s.gpuStore[gpuNameNs]
			if !exists {
				return tfv1.Resource{}, tfv1.Resource{}, fmt.Errorf("GPU not found in allocator store %s", gpuName)
			}
			if remain, err := s.checkGPUCapacityAndQuota(gpu, request.Request, adjustRequest.NewRequest); err != nil {
				return remain, tfv1.Resource{}, err
			}
		}

		// check namespaced level quota
		if err := s.quotaStore.CheckQuotaAvailable(&tfv1.AllocRequest{
			WorkloadNameNamespace: request.WorkloadNameNamespace,

			Count: uint(len(request.GPUNames)),
			Request: tfv1.Resource{
				Tflops: deltaTFlopsRequest,
				Vram:   deltaVRAMRequest,
			},
			Limit: tfv1.Resource{
				Tflops: deltaTFlopsLimit,
				Vram:   deltaVRAMLimit,
			},
			GPUNames: request.GPUNames,
			PodMeta:  request.PodMeta,
		}); err != nil {
			return tfv1.Resource{}, tfv1.Resource{}, err
		}
	}

	// pre check passed, change GPU request and QuotaStore and markDirty to sync to Kubernetes
	if !dryRun {
		for _, gpuName := range request.GPUNames {
			gpuNameNs := types.NamespacedName{Name: gpuName}
			gpu := s.gpuStore[gpuNameNs]

			availableRes := gpu.Status.Available
			availableRes.Tflops.Sub(deltaTFlopsRequest)
			availableRes.Vram.Sub(deltaVRAMRequest)

			s.markGPUDirty(gpuNameNs)
		}

		s.quotaStore.AdjustQuota(request.PodMeta.Namespace, tfv1.Resource{
			Tflops: deltaTFlopsRequest,
			Vram:   deltaVRAMRequest,
		}, tfv1.Resource{
			Tflops: deltaTFlopsLimit,
			Vram:   deltaVRAMLimit,
		})
		request.Request = adjustRequest.NewRequest
		request.Limit = adjustRequest.NewLimit

		log.FromContext(s.ctx).Info("GPU resource allocation adjust successfully",
			"namespace", request.PodMeta.Namespace,
			"workload", request.WorkloadNameNamespace.Name,
			"pod", request.PodMeta.Name,
			"request tflops", request.Request.Tflops.String(),
			"request vram", request.Request.Vram.String(),
			"limit tflops", request.Limit.Tflops.String(),
			"limit vram", request.Limit.Vram.String())
	}
	return tfv1.Resource{}, tfv1.Resource{
		Tflops: deltaTFlopsRequest,
		Vram:   deltaVRAMRequest,
	}, nil
}

func (s *GpuAllocator) ListNonUsingNodes() sets.Set[string] {
	<-s.initializedCh
	s.storeMutex.RLock()
	defer s.storeMutex.RUnlock()

	set := sets.New[string]()
	for nodeName, podNames := range s.nodeWorkerStore {
		// If using by TF, the node can not be used by original scheduler
		// If using by other scheduler, won't record as TF worker, thus the map is empty
		// Return non using nodes can ensure original scheduler not conflict with TF
		if len(podNames) == 0 {
			set.Insert(nodeName)
		}
	}
	return set
}

func (s *GpuAllocator) DeallocByPodIdentifier(ctx context.Context, podIdentifier types.NamespacedName) {
	// Read allocation info under RLock to avoid data race on maps
	s.storeMutex.RLock()
	podUID := s.podNamespaceNsToPodUID[podIdentifier.String()]
	request, exists := s.uniqueAllocation[podUID]
	s.storeMutex.RUnlock()

	if exists && request != nil {
		s.Dealloc(request.WorkloadNameNamespace, request.GPUNames, request.PodMeta)
	}
}

func (s *GpuAllocator) GetAllocationReqByNodeName(nodeName string) []*tfv1.AllocRequest {
	s.storeMutex.RLock()
	defer s.storeMutex.RUnlock()

	allocRequests := make([]*tfv1.AllocRequest, 0, 8)
	workers, exists := s.nodeWorkerStore[nodeName]
	if !exists || workers == nil {
		return allocRequests
	}
	for workerName := range workers {
		podUID := s.podNamespaceNsToPodUID[workerName.String()]
		if podUID == "" {
			continue
		}
		if request, exists := s.uniqueAllocation[podUID]; exists {
			allocRequests = append(allocRequests, request)
		}
	}
	return allocRequests
}

func (s *GpuAllocator) checkGPUCapacityAndQuota(gpu *tfv1.GPU, oldRes, newRes tfv1.Resource) (tfv1.Resource, error) {
	if gpu.Status.Available == nil {
		return tfv1.Resource{}, fmt.Errorf("GPU available is nil, skip check")
	}
	remainTflops := gpu.Status.Available.Tflops.DeepCopy()
	remainVram := gpu.Status.Available.Vram.DeepCopy()
	remainRes := tfv1.Resource{
		Tflops: remainTflops,
		Vram:   remainVram,
	}

	// Get actual TFLOPs values, converting from ComputePercent if needed
	var oldTflops, newTflops resource.Quantity
	if !oldRes.ComputePercent.IsZero() {
		requiredTflops := utils.ComputePercentToTflops(gpu.Status.Capacity.Tflops, oldRes)
		oldTflops = *requiredTflops
	} else {
		oldTflops = oldRes.Tflops
	}
	if !newRes.ComputePercent.IsZero() {
		requiredTflops := utils.ComputePercentToTflops(gpu.Status.Capacity.Tflops, newRes)
		newTflops = *requiredTflops
	} else {
		newTflops = newRes.Tflops
	}

	remainTflops.Add(oldTflops)
	remainTflops.Sub(newTflops)
	if remainTflops.Cmp(resource.Quantity{}) < 0 {
		return remainRes, ScalingQuotaExceededError
	}

	remainVram.Add(oldRes.Vram)
	remainVram.Sub(newRes.Vram)
	if remainVram.Cmp(resource.Quantity{}) < 0 {
		return remainRes, ScalingQuotaExceededError
	}
	return remainRes, nil
}

func (s *GpuAllocator) GetQuotaStore() *quota.QuotaStore {
	return s.quotaStore
}

type scoredGPU struct {
	nodeName string
	gpuName  string
	score    int
}

func (s *GpuAllocator) GetScoringStrategy(cfg *config.GPUFitConfig, req *tfv1.AllocRequest) Strategy {
	return NewStrategy(s.getPlacementMode(s.ctx, req.PoolName), cfg, s.nodeGpuStore)
}

// First level is k8s node name, second level is GPU name, value is score
func (s *GpuAllocator) Score(
	ctx context.Context, strategy Strategy, req *tfv1.AllocRequest, nodeGPUs map[string][]*tfv1.GPU,
) map[string]map[string]int {
	result := make(map[string]map[string]int, len(nodeGPUs))
	allScores := make([]scoredGPU, 0, len(nodeGPUs))

	for nodeName, gpus := range nodeGPUs {
		for _, gpu := range gpus {
			res := strategy.Score(gpu, true)

			// making Pending GPU to lower score, prefer not scheduling to them
			if gpu.Status.Phase == tfv1.TensorFusionGPUPhasePending {
				res = res / 4
			}

			if _, exists := result[nodeName]; !exists {
				result[nodeName] = make(map[string]int, len(gpus))
			}
			result[nodeName][gpu.Name] = res
			allScores = append(allScores, scoredGPU{
				nodeName: nodeName,
				gpuName:  gpu.Name,
				score:    res,
			})
		}
	}

	if log.FromContext(ctx).V(5).Enabled() {
		log.FromContext(ctx).Info("GPU scheduler score stage completed", "pod", req.PodMeta.Name, "top score gpus", strings.Join(topScoreItems(allScores), ", "))
	}
	return result
}

func topScoreItems(allScores []scoredGPU) []string {
	sort.Slice(allScores, func(i, j int) bool {
		return allScores[i].score > allScores[j].score
	})
	// Get top N (10 at most) scored GPUs
	topN := min(len(allScores), 10)

	// Format top scores for logging
	topScores := make([]string, topN)
	for i := range topN {
		topScores[i] = fmt.Sprintf("%s/%s:%d", allScores[i].nodeName, allScores[i].gpuName, allScores[i].score)
	}
	return topScores
}

// startSyncLoop starts a goroutine that periodically syncs the in-memory store with Kubernetes
func (s *GpuAllocator) startSyncLoop(ctx context.Context) {
	log := log.FromContext(ctx)
	ticker := time.NewTicker(s.syncInterval)
	defer ticker.Stop()

	log.Info("GPU allocator sync loop started", "syncInterval", s.syncInterval)

	for {
		select {
		case <-ticker.C:
			// Sync changes back to Kubernetes
			s.syncToK8s(ctx)
		case <-ctx.Done():
			log.Info("Stopping GPU allocator sync loop")
			return
		}
	}
}

// Stop stops all background goroutines
func (s *GpuAllocator) Stop() {
	// Stop all goroutines by canceling the context
	if s.cancel != nil {
		s.cancel()
	}
}

// InitGPUAndQuotaStore initializes both GPU store and quota store from Kubernetes
func (s *GpuAllocator) InitGPUAndQuotaStore() error {
	err := error(nil)
	s.initGPUStoreOnce.Do(func() {
		err = s.initGPUAndQuotaStore()
	})
	return err
}

func (s *GpuAllocator) initGPUAndQuotaStore() error {
	log := log.FromContext(s.ctx)

	// Initialize GPU store
	log.Info("Initializing GPU store")
	gpus := &tfv1.GPUList{}
	if err := s.List(s.ctx, gpus); err != nil {
		return fmt.Errorf("list GPUs: %w", err)
	}
	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()
	s.gpuStore = make(map[types.NamespacedName]*tfv1.GPU, len(gpus.Items))
	s.nodeGpuStore = make(map[string]map[string]*tfv1.GPU, len(gpus.Items)/2)

	for i := range gpus.Items {
		gpu := gpus.Items[i].DeepCopy()
		key := types.NamespacedName{Name: gpu.Name}
		s.gpuStore[key] = gpu

		if gpu.Status.Capacity == nil {
			// not a valid GPU, skip for now until updated by informer
			continue
		}
		if s.gpuStore[key].Status.Available == nil {
			s.gpuStore[key].Status.Available = gpu.Status.Capacity.DeepCopy()
		}

		gpuNodeName := gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
		if gpuNodeName != "" {
			if _, exists := s.nodeGpuStore[gpuNodeName]; !exists {
				s.nodeGpuStore[gpuNodeName] = make(map[string]*tfv1.GPU, 4)
			}
			s.nodeGpuStore[gpuNodeName][gpu.Name] = gpu
		}

		pool := gpu.Labels[constants.GpuPoolKey]
		if pool != "" {
			if _, exists := s.poolGpuStore[pool]; !exists {
				s.poolGpuStore[pool] = make(map[string]*tfv1.GPU, 128)
			}
			s.poolGpuStore[pool][gpu.Name] = gpu
		}
	}
	log.Info("GPU store initialized", "count", len(s.gpuStore))

	// Initialize quota store
	if err := s.quotaStore.InitQuotaStore(); err != nil {
		return fmt.Errorf("initialize quota store: %w", err)
	}

	return nil
}

var indexSetupOnce sync.Once

// SetupWithManager sets up the GpuAllocator with the Manager.
func (s *GpuAllocator) SetupWithManager(ctx context.Context, mgr manager.Manager) error {
	log.FromContext(ctx).Info("Setting up GPU watches with manager")
	err := mgr.Add(manager.RunnableFunc(func(ctx context.Context) error {
		log := log.FromContext(ctx)
		// Create a context with cancel function for the sync loop
		_, cancel := context.WithCancel(ctx)
		s.cancel = cancel

		// Initialize the GPU store and quota store, list all CR to memory
		if err := s.InitGPUAndQuotaStore(); err != nil {
			log.Error(err, "Failed to initialize GPU and quota store")
			return err
		}

		// reconcile allocation state based on existing workers, run only when it's elected as leader
		// and only if it's leader, it will start allocating resources to workers, and start sync loop here
		s.ReconcileAllocationState()
		log.Info("GPU store data reconciled")

		// ensure the indexer is set up only once
		var indexErr error
		indexSetupOnce.Do(func() {
			indexErr = mgr.GetCache().IndexField(ctx, &tfv1.GPU{}, "metadata.name", func(obj client.Object) []string {
				return []string{obj.GetName()}
			})
		})
		if indexErr != nil {
			return fmt.Errorf("failed to setup indexer for field metadata.name: %w", indexErr)
		}
		err := s.StartInformerForGPU(ctx, mgr)
		if err != nil {
			return err
		}
		err = s.quotaStore.StartInformerForGPUQuota(ctx, mgr)
		if err != nil {
			return err
		}
		// unlock all pending allocation/deallocation/scale operations after first initialization
		s.SetAllocatorReady()

		// Start the background sync goroutine
		go s.startSyncLoop(ctx)
		return nil
	}))
	return err
}

func (s *GpuAllocator) SetAllocatorReady() {
	close(s.initializedCh)
}

func (s *GpuAllocator) StartInformerForGPU(ctx context.Context, mgr manager.Manager) error {
	log := log.FromContext(ctx)

	informer, err := mgr.GetCache().GetInformer(ctx, &tfv1.GPU{})
	if err != nil {
		return fmt.Errorf("failed to get GPU informer: %w", err)
	}

	// Add event handlers
	_, err = informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			gpu, ok := obj.(*tfv1.GPU)
			if !ok {
				log.Error(fmt.Errorf("unexpected type"), "expected GPU")
				return
			}
			s.handleGPUCreate(ctx, gpu)
		},
		DeleteFunc: func(obj any) {
			gpu, ok := obj.(*tfv1.GPU)
			if !ok {
				// When a delete is dropped, the relist will notice a GPU in the store not
				// in the list, leading to the insertion of a tombstone object which contains
				// the deleted key/value.
				tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					log.Error(fmt.Errorf("unexpected type"), "expected GPU or tombstone")
					return
				}
				gpu, ok = tombstone.Obj.(*tfv1.GPU)
				if !ok {
					log.Error(fmt.Errorf("unexpected type"), "expected GPU in tombstone")
					return
				}
			}
			s.handleGPUDelete(ctx, gpu)
		},
		UpdateFunc: func(oldObj, newObj any) {
			newGPU, ok := newObj.(*tfv1.GPU)
			if !ok {
				log.Error(fmt.Errorf("unexpected type"), "expected new GPU")
				return
			}
			s.handleGPUUpdate(ctx, newGPU)
		},
	})
	return err
}

// handleGPUCreate handles GPU creation events
func (s *GpuAllocator) handleGPUCreate(ctx context.Context, gpu *tfv1.GPU) {
	log := log.FromContext(ctx)
	key := types.NamespacedName{Name: gpu.Name}

	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	if s.gpuStore[key] != nil {
		if gpu.Status.GPUModel != "" {
			mu.Lock()
			if _, exists := GPUCapacityMap[gpu.Status.GPUModel]; !exists {
				GPUCapacityMap[gpu.Status.GPUModel] = *gpu.Status.Capacity
			}
			mu.Unlock()
		}
		syncGPUMetadataAndStatusFromCluster(s.gpuStore[key], gpu)
		log.V(6).Info("GPU already exists in store", "name", key.Name)
		return
	}

	// Add GPU to store
	gpuInMem := gpu.DeepCopy()
	if gpuInMem.Status.Capacity == nil {
		gpuInMem.Status.Capacity = &tfv1.Resource{}
	}
	if gpuInMem.Status.Available == nil {
		gpuInMem.Status.Available = gpuInMem.Status.Capacity.DeepCopy()
	}
	s.gpuStore[key] = gpuInMem

	s.addOrUpdateGPUMaps(gpuInMem)
	log.Info("Added GPU to store", "name", key.Name, "phase", gpu.Status.Phase)
}

// handleGPUDelete handles GPU deletion events
func (s *GpuAllocator) handleGPUDelete(ctx context.Context, gpu *tfv1.GPU) {
	log := log.FromContext(ctx)
	key := types.NamespacedName{Name: gpu.Name}

	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	// Remove GPU from store
	delete(s.gpuStore, key)

	if gpu.Status.NodeSelector != nil {
		gpuNodeName := gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
		if s.nodeGpuStore[gpuNodeName] != nil {
			delete(s.nodeGpuStore[gpuNodeName], gpu.Name)
		}
	}

	if gpu.Labels != nil {
		pool := gpu.Labels[constants.GpuPoolKey]
		if pool != "" {
			delete(s.poolGpuStore[pool], gpu.Name)
		}
	}
	log.Info("Removed GPU from store", "name", key.Name)
}

// handleGPUUpdate handles GPU update events
func (s *GpuAllocator) handleGPUUpdate(ctx context.Context, gpu *tfv1.GPU) {
	log := log.FromContext(ctx)
	key := types.NamespacedName{Name: gpu.Name}

	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	if old, ok := s.gpuStore[key]; ok && old != nil {
		s.handleGPUUpdateCapacityDiff(old, gpu)

		// should never update available and runningApps here, to avoid circular update
		syncGPUMetadataAndStatusFromCluster(old, gpu)
		log.V(6).Info("Updated GPU in store (preserve Available)", "name", key.Name, "phase", gpu.Status.Phase)
	} else {
		s.gpuStore[key] = gpu.DeepCopy()
		log.V(6).Info("Updated GPU in store (new entry)", "name", key.Name, "phase", gpu.Status.Phase)
	}

	s.addOrUpdateGPUMaps(s.gpuStore[key])
}

func (s *GpuAllocator) addOrUpdateGPUMaps(gpuInMem *tfv1.GPU) {
	if gpuInMem.Status.NodeSelector != nil {
		gpuNodeName := gpuInMem.Status.NodeSelector[constants.KubernetesHostNameLabel]
		if gpuNodeName != "" {
			if _, exists := s.nodeGpuStore[gpuNodeName]; !exists {
				s.nodeGpuStore[gpuNodeName] = make(map[string]*tfv1.GPU, 4)
			}
			s.nodeGpuStore[gpuNodeName][gpuInMem.Name] = gpuInMem
			if _, exists := s.nodeWorkerStore[gpuNodeName]; !exists {
				s.nodeWorkerStore[gpuNodeName] = make(map[types.NamespacedName]struct{}, 4)
			}
		}

	}

	if gpuInMem.Labels != nil {
		pool := gpuInMem.Labels[constants.GpuPoolKey]
		if pool != "" {
			if _, exists := s.poolGpuStore[pool]; !exists {
				s.poolGpuStore[pool] = make(map[string]*tfv1.GPU, 128)
			}
			s.poolGpuStore[pool][gpuInMem.Name] = gpuInMem
		}
	}

	if gpuInMem.Status.GPUModel != "" {
		mu.Lock()
		GPUCapacityMap[gpuInMem.Status.GPUModel] = *gpuInMem.Status.Capacity
		mu.Unlock()
	}
}

func syncGPUMetadataAndStatusFromCluster(old *tfv1.GPU, gpu *tfv1.GPU) {
	old.Annotations = gpu.Annotations
	old.Labels = gpu.Labels
	old.ResourceVersion = gpu.ResourceVersion
	old.Generation = gpu.Generation
	old.OwnerReferences = gpu.OwnerReferences
	old.Kind = gpu.Kind
	old.APIVersion = gpu.APIVersion
	old.Status.Phase = gpu.Status.Phase
	old.Status.Message = gpu.Status.Message
	old.Status.UUID = gpu.Status.UUID
	old.Status.NodeSelector = gpu.Status.NodeSelector
	old.Status.GPUModel = gpu.Status.GPUModel
	old.Status.UsedBy = gpu.Status.UsedBy
	old.Status.Vendor = gpu.Status.Vendor
	old.Status.NUMANode = gpu.Status.NUMANode
	old.Status.Index = gpu.Status.Index
}

func (s *GpuAllocator) handleGPUUpdateCapacityDiff(old, gpu *tfv1.GPU) {
	if gpu == nil || gpu.Status.Capacity == nil {
		return
	}
	if old.Status.Capacity == nil {
		old.Status.Capacity = gpu.Status.Capacity.DeepCopy()
		old.Status.Available = gpu.Status.Capacity.DeepCopy()
	}

	tflopsDiff := gpu.Status.Capacity.Tflops.DeepCopy()
	tflopsDiff.Sub(old.Status.Capacity.Tflops)
	if tflopsDiff.Value() != 0 {
		old.Status.Capacity.Tflops.Add(tflopsDiff)
		old.Status.Available.Tflops.Add(tflopsDiff)
	}
	vramDiff := gpu.Status.Capacity.Vram.DeepCopy()
	vramDiff.Sub(old.Status.Capacity.Vram)
	if vramDiff.Value() != 0 {
		old.Status.Capacity.Vram.Add(vramDiff)
		old.Status.Available.Vram.Add(vramDiff)
	}
}

// syncToK8s syncs the modified GPUs and quotas from in-memory store to Kubernetes
func (s *GpuAllocator) syncToK8s(ctx context.Context) {
	// Sync GPU status
	s.SyncGPUsToK8s()

	// Sync quota status
	s.quotaStore.SyncQuotasToK8s(ctx)
}

// SyncGPUsToK8s syncs GPU status to Kubernetes
func (s *GpuAllocator) SyncGPUsToK8s() {
	log := log.FromContext(s.ctx)
	s.dirtyQueueLock.Lock()
	// Get all dirty GPUs and clear the queue
	dirtyGPUs := make([]types.NamespacedName, 0, len(s.dirtyQueue))
	for key := range s.dirtyQueue {
		dirtyGPUs = append(dirtyGPUs, key)
	}
	s.dirtyQueue = make(map[types.NamespacedName]struct{}, 16)
	s.dirtyQueueLock.Unlock()

	// No dirty GPUs to sync
	if len(dirtyGPUs) == 0 {
		return
	}

	s.storeMutex.RLock()
	defer s.storeMutex.RUnlock()

	dirtyNodes := make(map[string]struct{})

	for _, key := range dirtyGPUs {
		gpu, exists := s.gpuStore[key]
		if !exists {
			continue
		}

		dirtyNodes[gpu.Labels[constants.LabelKeyOwner]] = struct{}{}

		// Update the GPU status in Kubernetes with retry on conflict
		// Get the latest version, update status, and retry on conflict
		if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
			// Get the latest version before attempting an update
			latest := &tfv1.GPU{}
			if err := s.Get(s.ctx, key, latest); err != nil {
				if errors.IsNotFound(err) {
					// skip not existing resource to avoid infinite loop
					log.V(6).Info("GPU not found, skipping update", "gpu", key.String())
					return nil // return nil to stop retry
				}
				return err
			}

			// Apply our status updates to the latest version
			latest.Status.Available = gpu.Status.Available
			latest.Status.RunningApps = gpu.Status.RunningApps

			// Attempt to update with the latest version
			return s.Status().Update(s.ctx, latest)
		}); err != nil {
			// If update fails after retries, put the GPU back in the dirty queue
			s.dirtyQueueLock.Lock()
			s.dirtyQueue[key] = struct{}{}
			s.dirtyQueueLock.Unlock()
			log.Error(err, "Failed to update GPU status after retries, will retry later", "gpu", key.String())
		}
	}

	for nodeName := range dirtyNodes {
		// First, get the current node to check if annotations exist
		node := &tfv1.GPUNode{}
		nodeKey := client.ObjectKey{Name: nodeName}
		if err := s.Get(s.ctx, nodeKey, node); err != nil {
			log.Error(err, "Failed to get GPU node for updating last report time", "node", nodeName)
			continue
		}

		var patch []byte
		timeValue := time.Now().Format(time.RFC3339)
		encodedKey := utils.EscapeJSONPointer(constants.LastSyncTimeAnnotationKey)

		// Check if annotations already exist
		if node.Annotations == nil {
			// Create annotations if they don't exist
			patch = []byte(`[{
			"op": "add",
				"path": "/metadata/annotations",
				"value": {
					"` + constants.LastSyncTimeAnnotationKey + `": "` + timeValue + `"
				}
			}]`)
		} else {
			// Add to existing annotations
			patch = []byte(`[{
				"op": "add",
				"path": "/metadata/annotations/` + encodedKey + `",
				"value": "` + timeValue + `"
		}]`)
		}

		err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
			return s.Patch(s.ctx, &tfv1.GPUNode{
				ObjectMeta: metav1.ObjectMeta{
					Name: nodeName,
				},
			}, client.RawPatch(types.JSONPatchType, patch))
		})
		if err != nil {
			log.Error(err, "Failed to update GPU node last report time, allocation state may be inconsistent", "node", nodeName)
		}
	}
}

// listGPUsFromPool gets GPUs from the specified pool using the in-memory store
func (s *GpuAllocator) listGPUsFromPool(poolName string) []*tfv1.GPU {
	s.storeMutex.RLock()
	defer s.storeMutex.RUnlock()
	return lo.Values(s.poolGpuStore[poolName])
}

func (s *GpuAllocator) markGPUDirty(key types.NamespacedName) {
	s.dirtyQueueLock.Lock()
	defer s.dirtyQueueLock.Unlock()
	s.dirtyQueue[key] = struct{}{}
	log.FromContext(s.ctx).V(6).Info("[SYNC] GPU marked as dirty", "gpu", key.Name, "queueSize", len(s.dirtyQueue))
}

func (s *GpuAllocator) markGPUDirtyLocked(key types.NamespacedName) {
	s.dirtyQueue[key] = struct{}{}
}

// When it's leader, should reconcile state based on existing workers
// this function is run inside storeMutex lock
func (s *GpuAllocator) ReconcileAllocationState() {
	s.reconcileWorkerOnce.Do(func() {
		s.reconcileAllocationState()
		go s.startWorkerCleanUpChecker()
	})
}

func (s *GpuAllocator) ReconcileAllocationStateForTesting() {
	s.reconcileAllocationState()
}

func (s *GpuAllocator) CheckQuotaAndFilterSingleNodePreempt(
	nodeName string, allocReq *tfv1.AllocRequest, toPreemptPods sets.Set[types.NamespacedName],
) error {
	<-s.initializedCh

	// Read allocation maps under RLock to avoid data race
	s.storeMutex.RLock()

	log.FromContext(s.ctx).V(4).Info("[PREEMPT] CheckQuotaAndFilterSingleNodePreempt start",
		"node", nodeName,
		"requiredGPUs", allocReq.Count,
		"victimsFromScheduler", toPreemptPods.Len(),
		"workersOnNode", len(s.nodeWorkerStore[nodeName]),
		"victimNames", toPreemptPods.UnsortedList())

	// Only need to check total quotas when preempting
	toPreemptUsage := &tfv1.GPUResourceUsage{
		Requests: tfv1.Resource{
			Tflops: resource.Quantity{},
			Vram:   resource.Quantity{},
		},
		Limits: tfv1.Resource{
			Tflops: resource.Quantity{},
			Vram:   resource.Quantity{},
		},
	}
	workers := s.nodeWorkerStore[nodeName]
	preemptAllocRequests := make([]*tfv1.AllocRequest, 0, len(workers))
	for workerName := range workers {
		if !toPreemptPods.Has(workerName) {
			continue
		}
		podUID := s.podNamespaceNsToPodUID[workerName.String()]
		if podUID == "" {
			continue
		}
		existingAllocation := s.uniqueAllocation[podUID]
		if existingAllocation == nil {
			continue
		}
		// Get actual TFLOPs values, converting from ComputePercent if needed
		// We need GPU capacity to convert, so we get it from the first GPU of the allocation
		var reqTflops, limitTflops resource.Quantity
		if len(existingAllocation.GPUNames) > 0 {
			gpuNameNs := types.NamespacedName{Name: existingAllocation.GPUNames[0]}
			if gpu, exists := s.gpuStore[gpuNameNs]; exists && gpu.Status.Capacity != nil {
				if !existingAllocation.Request.ComputePercent.IsZero() {
					requiredTflops := utils.ComputePercentToTflops(gpu.Status.Capacity.Tflops, existingAllocation.Request)
					reqTflops = *requiredTflops
				} else {
					reqTflops = existingAllocation.Request.Tflops
				}
				if !existingAllocation.Limit.ComputePercent.IsZero() {
					requiredTflops := utils.ComputePercentToTflops(gpu.Status.Capacity.Tflops, existingAllocation.Limit)
					limitTflops = *requiredTflops
				} else {
					limitTflops = existingAllocation.Limit.Tflops
				}
			} else {
				// Fallback to direct TFLOPs if GPU not found
				reqTflops = existingAllocation.Request.Tflops
				limitTflops = existingAllocation.Limit.Tflops
			}
		} else {
			// Fallback to direct TFLOPs if no GPUs
			reqTflops = existingAllocation.Request.Tflops
			limitTflops = existingAllocation.Limit.Tflops
		}
		// Multiply by Count because each GPU contributes to total usage
		// For multi-GPU pods, all resources (TFLOPs and VRAM) should be multiplied by GPU count
		// Note: value assignment in Go creates a copy, safe to Mul directly
		count := int64(existingAllocation.Count)
		reqTflops.Mul(count)
		toPreemptUsage.Requests.Tflops.Add(reqTflops)
		limitTflops.Mul(count)
		toPreemptUsage.Limits.Tflops.Add(limitTflops)
		// Value assignment creates a copy, safe to modify
		reqVram := existingAllocation.Request.Vram.DeepCopy()
		reqVram.Mul(count)
		toPreemptUsage.Requests.Vram.Add(reqVram)
		limitVram := existingAllocation.Limit.Vram.DeepCopy()
		limitVram.Mul(count)
		toPreemptUsage.Limits.Vram.Add(limitVram)

		preemptAllocRequests = append(preemptAllocRequests, existingAllocation)
	}

	log.FromContext(s.ctx).V(4).Info("[PREEMPT] preemptAllocRequests built",
		"node", nodeName,
		"preemptAllocRequestsCount", len(preemptAllocRequests),
		"victimsFromScheduler", toPreemptPods.Len())

	if log.FromContext(s.ctx).V(5).Enabled() {
		log.FromContext(s.ctx).V(5).Info("Preempting node and check quotas", "nodeName", nodeName, "toPreemptUsage", toPreemptUsage)
	}

	// Release RLock before calling FilterWithPreempt which acquires its own locks
	s.storeMutex.RUnlock()

	if err := s.quotaStore.CheckTotalQuotaRelaxed(allocReq, toPreemptUsage); err != nil {
		return fmt.Errorf("quota check failed during preempt: %w", err)
	}

	// Get GPUs from the pool using the in-memory store
	if allocReq.PoolName == "" {
		return fmt.Errorf("GPU Pool name is empty, can not find GPUs during preempt")
	}

	log.FromContext(s.ctx).V(4).Info("[PREEMPT] CheckQuotaAndFilterSingleNodePreempt: allocReq details",
		"node", nodeName,
		"count", allocReq.Count,
		"requestTflops", allocReq.Request.Tflops.String(),
		"requestComputePercent", allocReq.Request.ComputePercent.String(),
		"requestVram", allocReq.Request.Vram.String(),
		"limitTflops", allocReq.Limit.Tflops.String(),
		"limitVram", allocReq.Limit.Vram.String(),
		"poolName", allocReq.PoolName,
		"gpuModel", allocReq.GPUModel,
		"workload", allocReq.WorkloadNameNamespace.Name)

	// Pass nodeName to FilterWithPreempt for optimization (avoid iterating entire pool)
	filteredGPUs, _, err := s.FilterWithPreempt(allocReq, preemptAllocRequests, nodeName)
	if err != nil {
		return err
	}
	if len(filteredGPUs) < int(allocReq.Count) {
		log.FromContext(s.ctx).V(4).Info("[PREEMPT] not enough GPUs after filter during preempt",
			"node", nodeName,
			"requiredGPUs", allocReq.Count,
			"filteredGPUsCount", len(filteredGPUs),
			"preemptAllocRequestsCount", len(preemptAllocRequests))
		return fmt.Errorf("no gpus available or valid in pool %s after filtering during preempt", allocReq.PoolName)
	}
	return nil
}

func (s *GpuAllocator) reconcileAllocationState() {
	ctx := s.ctx
	logger := log.FromContext(ctx)

	workers := &v1.PodList{}
	if err := s.List(ctx, workers, client.MatchingLabels(map[string]string{
		constants.LabelComponent: constants.ComponentWorker,
	})); err != nil {
		logger.Error(err, "Failed to list Workloads to reconcile allocation state")
		return
	}

	// filter out pending workers which doesn't have nodeName or is being deleted
	workers.Items = lo.Filter(workers.Items, func(worker v1.Pod, _ int) bool {
		scheduled := worker.Spec.NodeName != ""

		deletedAndDeAllocated := !worker.DeletionTimestamp.IsZero() &&
			!controllerutil.ContainsFinalizer(&worker, constants.Finalizer)

		// Only register active pods in uniqueAllocation.
		// Pods that are deletedAndDeAllocated must NOT be registered: their resources
		// are not counted in Available, so a late Dealloc would add resources back
		// that were never subtracted, causing Available to exceed correct state.
		if scheduled && !deletedAndDeAllocated {
			allocRequest, msg, err := s.ComposeAllocationRequest(&worker)
			if err != nil {
				logger.Error(err, "Failed to compose allocation request for existing worker Pod, annotation may not be valid", "pod", worker.Name, "msg", msg)
				return false
			}
			s.uniqueAllocation[string(worker.UID)] = allocRequest
			s.podNamespaceNsToPodUID[worker.Namespace+"/"+worker.Name] = string(worker.UID)
			s.addAllocationMap(worker.Spec.NodeName, worker.ObjectMeta)
		}
		return scheduled && !deletedAndDeAllocated
	})

	actualAvailableMap := make(map[types.NamespacedName]*tfv1.Resource)
	actualRunningAppsMap := make(map[types.NamespacedName][]*tfv1.RunningAppDetail)

	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	for gpuKey, gpu := range s.gpuStore {
		if gpu.Status.Capacity != nil {
			actualAvailableMap[gpuKey] = gpu.Status.Capacity.DeepCopy()
			actualRunningAppsMap[gpuKey] = gpu.Status.RunningApps
			gpu.Status.RunningApps = []*tfv1.RunningAppDetail{}

		}

		// This is important for progressive migration mode
		// No workers, but node contains GPU, need include into nodeWorkerStore with empty map
		gpuNodeName := gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
		if _, exists := s.nodeWorkerStore[gpuNodeName]; !exists {
			s.nodeWorkerStore[gpuNodeName] = make(map[types.NamespacedName]struct{}, 4)
		}
	}

	for _, worker := range workers.Items {
		allocRequest := s.uniqueAllocation[string(worker.UID)]
		gpuIds := worker.Annotations[constants.GPUDeviceIDsAnnotation]
		gpuIdsList := strings.SplitSeq(gpuIds, ",")

		for gpuId := range gpuIdsList {
			gpuKey := types.NamespacedName{Name: gpuId}
			gpuAvailableRes, ok := actualAvailableMap[gpuKey]
			if ok {
				var reqTflops resource.Quantity
				gpu := s.gpuStore[gpuKey]
				if gpu != nil && gpu.Status.Capacity != nil && !allocRequest.Request.ComputePercent.IsZero() {
					requiredTflops := utils.ComputePercentToTflops(gpu.Status.Capacity.Tflops, allocRequest.Request)
					reqTflops = *requiredTflops
				} else {
					reqTflops = allocRequest.Request.Tflops
				}
				gpuAvailableRes.Tflops.Sub(reqTflops)
				gpuAvailableRes.Vram.Sub(allocRequest.Request.Vram)
			}
			addRunningApp(ctx, s.gpuStore[gpuKey], allocRequest)
		}
	}

	for gpuKey, gpu := range s.gpuStore {
		if gpu.Status.Capacity == nil {
			log.FromContext(ctx).Info("[Warning] GPU capacity is nil, skip reconcile", "gpu", gpuKey.Name)
			continue
		}
		sameTflops := gpu.Status.Available.Tflops.Equal(actualAvailableMap[gpuKey].Tflops)
		sameVRAM := gpu.Status.Available.Vram.Equal(actualAvailableMap[gpuKey].Vram)
		if !sameTflops || !sameVRAM {
			gpu.Status.Available.Tflops = actualAvailableMap[gpuKey].Tflops
			gpu.Status.Available.Vram = actualAvailableMap[gpuKey].Vram
			s.markGPUDirtyLocked(gpuKey)
			log.FromContext(ctx).Info("Correcting gpu available resources", "gpu", gpuKey.Name, "tflops", gpu.Status.Available.Tflops.String(), "vram", gpu.Status.Available.Vram.String())
		}

		if !equality.Semantic.DeepEqual(gpu.Status.RunningApps, actualRunningAppsMap[gpuKey]) {
			s.markGPUDirtyLocked(gpuKey)
			log.FromContext(ctx).Info("Correcting gpu running apps", "gpu", gpuKey.Name, "runningApps", len(gpu.Status.RunningApps))
		}
	}

	// reconcile quota store state
	s.quotaStore.ReconcileQuotaStore(ctx, s.uniqueAllocation)
	log.FromContext(ctx).Info("Quota store data reconciled")
}

func (s *GpuAllocator) startWorkerCleanUpChecker() {
	ticker := time.NewTicker(CleanUpCheckInterval)
	for {
		select {
		case <-ticker.C:
			// Collect candidates under RLock to avoid data race on map iteration
			type deallocCandidate struct {
				workloadNameNs tfv1.NameNamespace
				gpuNames       []string
				podMeta        metav1.ObjectMeta
			}
			var candidates []deallocCandidate

			s.storeMutex.RLock()
			for _, allocRequest := range s.uniqueAllocation {
				if allocRequest.PodMeta.Name == "" {
					continue
				}
				candidates = append(candidates, deallocCandidate{
					workloadNameNs: allocRequest.WorkloadNameNamespace,
					gpuNames:       allocRequest.GPUNames,
					podMeta:        allocRequest.PodMeta,
				})
			}
			totalWorkers := len(s.uniqueAllocation)
			s.storeMutex.RUnlock()

			cleaned := 0
			for _, c := range candidates {
				pod := &v1.Pod{}
				err := s.Get(s.ctx, types.NamespacedName{Namespace: c.podMeta.Namespace, Name: c.podMeta.Name}, pod)
				if errors.IsNotFound(err) {
					log.FromContext(s.ctx).Info("Pod has been deleted, deallocate GPU", "pod", c.podMeta.Name, "namespace", c.podMeta.Namespace)
					s.Dealloc(c.workloadNameNs, c.gpuNames, c.podMeta)
					cleaned++
				}
			}
			log.FromContext(s.ctx).Info("GPU allocation cleaned up check completed", "total workers",
				totalWorkers, "backup cleaner cleaned", cleaned)
		case <-s.ctx.Done():
			return
		}
	}
}

func addRunningApp(ctx context.Context, gpu *tfv1.GPU, allocRequest *tfv1.AllocRequest) {
	workloadNameNamespace := allocRequest.WorkloadNameNamespace
	if gpu == nil {
		log.FromContext(ctx).Info("[Warning] GPU is nil, skip adding running app", "workload", workloadNameNamespace.Name, "namespace", workloadNameNamespace.Namespace)
		return
	}
	if gpu.Status.RunningApps == nil {
		gpu.Status.RunningApps = []*tfv1.RunningAppDetail{}
	}

	item, found := lo.Find(gpu.Status.RunningApps, func(app *tfv1.RunningAppDetail) bool {
		return app.Name == workloadNameNamespace.Name && app.Namespace == workloadNameNamespace.Namespace
	})

	if found {
		item.Count++
		item.Pods = append(item.Pods, &tfv1.PodGPUInfo{
			Name:      allocRequest.PodMeta.Name,
			Namespace: allocRequest.PodMeta.Namespace,
			UID:       string(allocRequest.PodMeta.UID),
			Requests:  allocRequest.Request,
			Limits:    allocRequest.Limit,
			QoS:       allocRequest.QoS,
		})
	} else {
		gpu.Status.RunningApps = append(gpu.Status.RunningApps, &tfv1.RunningAppDetail{
			Name:      workloadNameNamespace.Name,
			Namespace: workloadNameNamespace.Namespace,
			Count:     1,
			Pods: []*tfv1.PodGPUInfo{
				{
					Name:      allocRequest.PodMeta.Name,
					Namespace: allocRequest.PodMeta.Namespace,
					UID:       string(allocRequest.PodMeta.UID),
					Requests:  allocRequest.Request,
					Limits:    allocRequest.Limit,
				},
			},
		})
	}
}

func removeRunningApp(ctx context.Context, gpu *tfv1.GPU, allocRequest *tfv1.AllocRequest) {
	workloadNameNamespace := allocRequest.WorkloadNameNamespace
	item, found := lo.Find(gpu.Status.RunningApps, func(app *tfv1.RunningAppDetail) bool {
		return app.Name == workloadNameNamespace.Name && app.Namespace == workloadNameNamespace.Namespace
	})
	if found {
		item.Count--
		if item.Count == 0 {
			// scale down to zero, not running any more
			gpu.Status.RunningApps = lo.Filter(gpu.Status.RunningApps, func(app *tfv1.RunningAppDetail, _ int) bool {
				return app.Name != workloadNameNamespace.Name && app.Namespace != workloadNameNamespace.Namespace
			})
		} else {
			item.Pods = lo.Filter(item.Pods, func(pod *tfv1.PodGPUInfo, _ int) bool {
				return pod.UID != string(allocRequest.PodMeta.UID)
			})
		}
	} else {
		// should not happen, if deallocation twice, it should be a bug
		log.FromContext(ctx).Info("[Warning] The app to remove not found, could be caused by deallocation twice bug", "gpu", gpu.Name, "namespace", gpu.Namespace, "workload", workloadNameNamespace.Name, "namespace", workloadNameNamespace.Namespace)
	}
}

func (s *GpuAllocator) ComposeAllocationRequest(pod *v1.Pod) (*tfv1.AllocRequest, string, error) {
	// allow Pods with no requests/limits to use TensorFusion, Pod webhook will ensure at least one request/limit is set
	gpuRequestResource, err := utils.GetGPUResource(pod, true)
	if err != nil {
		log.FromContext(s.ctx).Error(err, "Invalid gpu request annotation", "pod", pod.Name, "namespace", pod.Namespace)
	}
	gpuLimitResource, err := utils.GetGPUResource(pod, false)
	if err != nil {
		log.FromContext(s.ctx).Error(err, "Invalid gpu limit annotation", "pod", pod.Name, "namespace", pod.Namespace)
	}

	count := 1
	if gpuCountStr, exists := pod.Annotations[constants.GpuCountAnnotation]; exists {
		count, err = strconv.Atoi(gpuCountStr)
		if err != nil {
			return &tfv1.AllocRequest{}, "invalid gpu count annotation", err
		}
	}
	if count > MaxGPUCounterPerAllocation {
		return &tfv1.AllocRequest{}, "gpu count annotation is too large", nil
	}

	qosLevel := tfv1.QoSLevel(pod.Annotations[constants.QoSLevelAnnotation])
	if qosLevel == "" {
		qosLevel = tfv1.QoSMedium
	}

	gpuVendor := pod.Annotations[constants.GpuVendorAnnotation]

	gpuIndices, hasError := utils.ParseIndicesAnnotation(pod.Annotations[constants.GpuIndicesAnnotation])
	if hasError {
		return &tfv1.AllocRequest{}, "invalid gpu-indices annotation",
			fmt.Errorf("can not parse gpu indices annotation")
	}

	allocRequest := tfv1.AllocRequest{
		PoolName: pod.Annotations[constants.GpuPoolKey],
		Request:  gpuRequestResource,
		Limit:    gpuLimitResource,

		Count:      uint(count),
		GPUModel:   pod.Annotations[constants.GPUModelAnnotation],
		GPUIndices: gpuIndices,
		GPUVendor:  gpuVendor,
		WorkloadNameNamespace: tfv1.NameNamespace{
			Name:      pod.Labels[constants.WorkloadKey],
			Namespace: pod.Namespace,
		},
		PodMeta: pod.ObjectMeta,
		QoS:     qosLevel,
	}

	// for already allocated workers, set the GPU device IDs for further scaling and retrieval
	if gpuIdStr, exists := pod.Annotations[constants.GPUDeviceIDsAnnotation]; exists {
		gpuIds := strings.SplitSeq(gpuIdStr, ",")
		allocRequest.GPUNames = slices.Collect(gpuIds)
	}

	return &allocRequest, "", nil
}

func (s *GpuAllocator) addAllocationMap(gpuNodeName string, podMeta metav1.ObjectMeta) {
	if _, exists := s.nodeWorkerStore[gpuNodeName]; !exists {
		s.nodeWorkerStore[gpuNodeName] = make(map[types.NamespacedName]struct{}, 4)
	}
	workerPodKey := types.NamespacedName{Namespace: podMeta.Namespace, Name: podMeta.Name}
	s.nodeWorkerStore[gpuNodeName][workerPodKey] = struct{}{}
}

func (s *GpuAllocator) getPlacementMode(ctx context.Context, poolName string) tfv1.PlacementMode {
	pool := &tfv1.GPUPool{}
	if err := s.Get(ctx, client.ObjectKey{Name: poolName}, pool); err != nil {
		// if failed to get pool, default to compact first
		return tfv1.PlacementModeNodeCompactGPULowLoad
	}

	if pool.Spec.SchedulingConfigTemplate == nil || *pool.Spec.SchedulingConfigTemplate == "" {
		return tfv1.PlacementModeNodeCompactGPULowLoad
	}

	// get scheduling config template
	schedulingConfigTemplate := &tfv1.SchedulingConfigTemplate{}
	if err := s.Get(ctx, client.ObjectKey{Name: *pool.Spec.SchedulingConfigTemplate}, schedulingConfigTemplate); err != nil {
		// if failed to get scheduling config template, default to compact first
		return tfv1.PlacementModeNodeCompactGPULowLoad
	}
	return schedulingConfigTemplate.Spec.Placement.Mode
}

// normalize score to [0, 100]
func normalizeScore(cfg *config.GPUFitConfig, vramScore, tflopsScore float64) int {
	score := int(math.Round(vramScore*cfg.VramWeight + tflopsScore*cfg.TflopsWeight))
	if score < 0 {
		return 0
	}
	if score > 100 {
		return 100
	}
	return score
}
