// Package gpuallocator handles GPU allocation
package gpuallocator

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator/filter"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

type Strategy interface {
	SelectGPUs(gpus []tfv1.GPU, count uint) ([]*tfv1.GPU, error)
}

// NewStrategy creates a strategy based on the placement mode
func NewStrategy(placementMode tfv1.PlacementMode) Strategy {
	switch placementMode {
	case tfv1.PlacementModeLowLoadFirst:
		return LowLoadFirst{}
	default:
		// CompactFirst is the default strategy
		return CompactFirst{}
	}
}

type GpuAllocator struct {
	client.Client
	filterRegistry *filter.FilterRegistry

	// In-memory store of GPUs
	gpuStore     map[types.NamespacedName]*tfv1.GPU
	storeMutex   sync.RWMutex
	syncInterval time.Duration
	cancel       context.CancelFunc

	// Queue for tracking modified GPUs that need to be synced
	dirtyQueue     map[types.NamespacedName]struct{}
	dirtyQueueLock sync.Mutex
}

// AllocRequest encapsulates all parameters needed for GPU allocation
type AllocRequest struct {
	// Name of the GPU pool to allocate from
	PoolName string
	// Namespace information for the workload
	WorkloadNameNamespace tfv1.NameNamespace
	// Resource requirements for the allocation
	Request tfv1.Resource
	// Number of GPUs to allocate
	Count uint
	// Specific GPU model to allocate, empty string means any model
	GPUModel string
}

// Alloc allocates a request to a gpu or multiple gpus from the same node.
func (s *GpuAllocator) Alloc(ctx context.Context, req AllocRequest) ([]*tfv1.GPU, error) {
	// Get GPUs from the pool using the in-memory store
	poolGPUs := s.listGPUsFromPool(req.PoolName)

	// Add SameNodeFilter if count > 1 to ensure GPUs are from the same node
	filterRegistry := s.filterRegistry.With(filter.NewResourceFilter(req.Request))

	// Add GPU model filter if specified
	if req.GPUModel != "" {
		filterRegistry = filterRegistry.With(filter.NewGPUModelFilter(req.GPUModel))
	}

	if req.Count > 1 {
		filterRegistry = filterRegistry.With(filter.NewSameNodeFilter(req.Count))
	}

	// Apply the filters in sequence
	filteredGPUs, err := filterRegistry.Apply(ctx, poolGPUs)
	if err != nil {
		return nil, fmt.Errorf("apply filters: %w", err)
	}

	if len(filteredGPUs) == 0 {
		return nil, fmt.Errorf("no gpus available in pool %s after filtering", req.PoolName)
	}

	pool := &tfv1.GPUPool{}
	if err := s.Get(ctx, client.ObjectKey{Name: req.PoolName}, pool); err != nil {
		return nil, fmt.Errorf("get pool %s: %w", req.PoolName, err)
	}

	schedulingConfigTemplate := &tfv1.SchedulingConfigTemplate{}
	if pool.Spec.SchedulingConfigTemplate != nil {
		if err := s.Get(ctx, client.ObjectKey{Name: *pool.Spec.SchedulingConfigTemplate}, schedulingConfigTemplate); err != nil {
			return nil, fmt.Errorf("get scheduling config template %s: %w", *pool.Spec.SchedulingConfigTemplate, err)
		}
	}

	strategy := NewStrategy(schedulingConfigTemplate.Spec.Placement.Mode)
	selectedGPUs, err := strategy.SelectGPUs(filteredGPUs, req.Count)
	if err != nil {
		return nil, fmt.Errorf("select GPU: %w", err)
	}

	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	appAdded := false
	for _, selectedGPU := range selectedGPUs {

		// Get the GPU from the store
		key := types.NamespacedName{Name: selectedGPU.Name, Namespace: selectedGPU.Namespace}
		gpu, exists := s.gpuStore[key]
		if !exists {
			// If not in store, create a new entry
			gpu = selectedGPU.DeepCopy()
			s.gpuStore[key] = gpu
		}

		// reduce available resource on the GPU status
		gpu.Status.Available.Tflops.Sub(req.Request.Tflops)
		gpu.Status.Available.Vram.Sub(req.Request.Vram)

		if !appAdded {
			addRunningApp(ctx, gpu, req.WorkloadNameNamespace)
			appAdded = true
		}

		s.markGPUDirty(key)
	}

	// Return copies of the selected GPUs from the store
	result := make([]*tfv1.GPU, len(selectedGPUs))
	for i, gpu := range selectedGPUs {
		key := types.NamespacedName{Name: gpu.Name, Namespace: gpu.Namespace}
		result[i] = s.gpuStore[key].DeepCopy()
	}

	return result, nil
}

// Dealloc a request from gpu to release available resources on it.
func (s *GpuAllocator) Dealloc(ctx context.Context, workloadNameNamespace tfv1.NameNamespace, request tfv1.Resource, gpus []types.NamespacedName) error {
	log := log.FromContext(ctx)
	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	appRemoved := false
	for _, gpu := range gpus {
		// Get the GPU from the store
		storeGPU, exists := s.gpuStore[gpu]
		if !exists {
			log.Error(fmt.Errorf("GPU not found in store"), "Failed to deallocate GPU", "name", gpu.String())
			continue
		}

		// Add resources back to the GPU
		storeGPU.Status.Available.Tflops.Add(request.Tflops)
		storeGPU.Status.Available.Vram.Add(request.Vram)
		if !appRemoved {
			removeRunningApp(ctx, storeGPU, workloadNameNamespace)
			appRemoved = true
		}

		s.markGPUDirty(gpu)
	}

	return nil
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

	allocator := &GpuAllocator{
		Client:         client,
		filterRegistry: baseRegistry,
		gpuStore:       make(map[types.NamespacedName]*tfv1.GPU),
		syncInterval:   syncInterval,
		dirtyQueue:     make(map[types.NamespacedName]struct{}),
	}

	return allocator
}

// startSyncLoop starts a goroutine that periodically syncs the in-memory store with Kubernetes
func (s *GpuAllocator) startSyncLoop(ctx context.Context) {
	log := log.FromContext(ctx)
	ticker := time.NewTicker(s.syncInterval)
	defer ticker.Stop()

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

// initGPUStore initializes the in-memory GPU store from Kubernetes
func (s *GpuAllocator) initGPUStore(ctx context.Context) error {
	log := log.FromContext(ctx)
	log.Info("Initializing GPU store")
	gpus := &tfv1.GPUList{}
	if err := s.List(ctx, gpus); err != nil {
		return fmt.Errorf("list GPUs: %w", err)
	}
	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()
	// Initialize the store with current GPUs
	s.gpuStore = make(map[types.NamespacedName]*tfv1.GPU, len(gpus.Items))
	for i := range gpus.Items {
		gpu := &gpus.Items[i]
		key := types.NamespacedName{Name: gpu.Name, Namespace: gpu.Namespace}
		s.gpuStore[key] = gpu.DeepCopy()
	}

	log.Info("GPU store initialized", "count", len(s.gpuStore))
	return nil
}

var indexSetupOnce sync.Once

// SetupWithManager sets up the GpuAllocator with the Manager.
func (s *GpuAllocator) SetupWithManager(ctx context.Context, mgr manager.Manager) (<-chan struct{}, error) {
	log := log.FromContext(ctx)
	log.Info("Setting up GPU watches with manager")

	readyCh := make(chan struct{}, 1)

	// ensure the indexer is set up only once
	var indexErr error
	indexSetupOnce.Do(func() {
		indexErr = mgr.GetCache().IndexField(ctx, &tfv1.GPU{}, "metadata.name", func(obj client.Object) []string {
			return []string{obj.GetName()}
		})
	})
	if indexErr != nil {
		return readyCh, fmt.Errorf("failed to setup indexer for field metadata.name: %w", indexErr)
	}

	informer, err := mgr.GetCache().GetInformer(ctx, &tfv1.GPU{})
	if err != nil {
		return readyCh, fmt.Errorf("failed to get GPU informer: %w", err)
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

	if err != nil {
		return readyCh, fmt.Errorf("failed to add event handler: %w", err)
	}

	err = mgr.Add(manager.RunnableFunc(func(ctx context.Context) error {
		// Create a context with cancel function for the sync loop
		_, cancel := context.WithCancel(ctx)
		s.cancel = cancel
		// Initialize the GPU store
		if err := s.initGPUStore(ctx); err != nil {
			log.Error(err, "Failed to initialize GPU store")
			return err
		}
		readyCh <- struct{}{}
		return nil
	}))

	go func() {
		<-mgr.Elected()
		// reconcile allocation state based on existing workers, run only when it's elected as leader
		// and only if it's leader, it will start allocating resources to workers, and start sync loop here
		s.reconcileAllocationState(ctx)
		log.Info("GPU store data reconciled")
		// Start the background sync goroutine
		go s.startSyncLoop(ctx)
	}()

	return readyCh, err
}

// handleGPUCreate handles GPU creation events
func (s *GpuAllocator) handleGPUCreate(ctx context.Context, gpu *tfv1.GPU) {
	log := log.FromContext(ctx)
	key := types.NamespacedName{Name: gpu.Name, Namespace: gpu.Namespace}

	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	// Add GPU to store
	s.gpuStore[key] = gpu.DeepCopy()
	log.V(4).Info("Added GPU to store", "name", key.Name, "phase", gpu.Status.Phase)
}

// handleGPUDelete handles GPU deletion events
func (s *GpuAllocator) handleGPUDelete(ctx context.Context, gpu *tfv1.GPU) {
	log := log.FromContext(ctx)
	key := types.NamespacedName{Name: gpu.Name, Namespace: gpu.Namespace}

	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	// Remove GPU from store
	delete(s.gpuStore, key)
	log.V(4).Info("Removed GPU from store", "name", key.Name)
}

// handleGPUUpdate handles GPU update events
func (s *GpuAllocator) handleGPUUpdate(ctx context.Context, gpu *tfv1.GPU) {
	log := log.FromContext(ctx)
	key := types.NamespacedName{Name: gpu.Name, Namespace: gpu.Namespace}

	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	if old, ok := s.gpuStore[key]; ok && old != nil {
		// Keep the Available field in the store
		newGpu := gpu.DeepCopy()
		if old.Status.Available != nil {
			newGpu.Status.Available = old.Status.Available
		}
		s.gpuStore[key] = newGpu
		log.V(4).Info("Updated GPU in store (preserve Available)", "name", key.Name, "phase", gpu.Status.Phase)
	} else {
		s.gpuStore[key] = gpu.DeepCopy()
		log.V(4).Info("Updated GPU in store (new entry)", "name", key.Name, "phase", gpu.Status.Phase)
	}
}

// syncToK8s syncs the modified GPUs from in-memory store to Kubernetes
func (s *GpuAllocator) syncToK8s(ctx context.Context) {
	log := log.FromContext(ctx)
	s.dirtyQueueLock.Lock()
	// Get all dirty GPUs and clear the queue
	dirtyGPUs := make([]types.NamespacedName, 0, len(s.dirtyQueue))
	for key := range s.dirtyQueue {
		dirtyGPUs = append(dirtyGPUs, key)
	}
	s.dirtyQueue = make(map[types.NamespacedName]struct{})
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
		// Create a copy to avoid modifying the memory store directly
		gpuCopy := gpu.DeepCopy()

		dirtyNodes[gpuCopy.Labels[constants.LabelKeyOwner]] = struct{}{}

		// Update the GPU status in Kubernetes
		if err := s.Status().Update(ctx, gpuCopy); err != nil {
			// If update fails, put the GPU back in the dirty queue
			s.dirtyQueueLock.Lock()
			s.dirtyQueue[key] = struct{}{}
			s.dirtyQueueLock.Unlock()
			log.Error(err, "Failed to update GPU status, will retry later", "gpu", key.String())
		}
	}

	for nodeName := range dirtyNodes {
		// Refer https://datatracker.ietf.org/doc/html/rfc6901#section-3 encode `/` as `~1`
		patch := []byte(`[{
			"op": "add",
			"path": "/metadata/annotations/` + strings.ReplaceAll(constants.GPULastReportTimeAnnotationKey, "/", "~1") + `",
			"value": "` + time.Now().Format(time.RFC3339) + `"
		}]`)
		err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
			return s.Patch(ctx, &tfv1.GPUNode{
				ObjectMeta: metav1.ObjectMeta{
					Name: nodeName,
				},
			}, client.RawPatch(types.JSONPatchType, patch))
		})
		if err != nil {
			log.Error(err, "Failed to update GPU node last report time, will retry later", "node", nodeName)
		}
	}
}

// listGPUsFromPool gets GPUs from the specified pool using the in-memory store
func (s *GpuAllocator) listGPUsFromPool(poolName string) []tfv1.GPU {
	s.storeMutex.RLock()
	defer s.storeMutex.RUnlock()

	result := make([]tfv1.GPU, 0, len(s.gpuStore)/2)
	for _, gpu := range s.gpuStore {
		if gpu.Labels[constants.GpuPoolKey] == poolName {
			result = append(result, *gpu)
		}
	}

	return result
}

func (s *GpuAllocator) markGPUDirty(key types.NamespacedName) {
	s.dirtyQueueLock.Lock()
	defer s.dirtyQueueLock.Unlock()
	s.dirtyQueue[key] = struct{}{}
}

// When it's leader, should reconcile state based on existing workers
// this function is run inside storeMutex lock
func (s *GpuAllocator) reconcileAllocationState(ctx context.Context) {
	logger := log.FromContext(ctx)
	workers := &v1.PodList{}
	if err := s.List(ctx, workers, client.MatchingLabels(map[string]string{
		constants.LabelComponent: constants.ComponentWorker,
	})); err != nil {
		logger.Error(err, "Failed to list Workloads to reconcile allocation state")
		return
	}

	tflopsCapacityMap := make(map[types.NamespacedName]resource.Quantity)
	vramCapacityMap := make(map[types.NamespacedName]resource.Quantity)
	gpuMap := make(map[types.NamespacedName]*tfv1.GPU)

	for gpuKey, gpu := range s.gpuStore {
		if gpu.Status.Capacity != nil {
			tflopsCapacityMap[gpuKey] = gpu.Status.Capacity.Tflops
			vramCapacityMap[gpuKey] = gpu.Status.Capacity.Vram
			gpu.Status.RunningApps = []*tfv1.RunningAppDetail{}
			gpuMap[gpuKey] = gpu
		}
	}

	for _, worker := range workers.Items {
		tflopsRequest, _ := resource.ParseQuantity(worker.Annotations[constants.TFLOPSRequestAnnotation])
		vramRequest, _ := resource.ParseQuantity(worker.Annotations[constants.VRAMRequestAnnotation])
		gpuIds := worker.Annotations[constants.GpuKey]
		gpuIdsList := strings.Split(gpuIds, ",")
		appAdded := false
		for _, gpuId := range gpuIdsList {
			gpuKey := types.NamespacedName{Name: gpuId}
			gpuCapacity, ok := tflopsCapacityMap[gpuKey]
			if ok {
				gpuCapacity.Sub(tflopsRequest)
			}
			gpuCapacity, ok = vramCapacityMap[gpuKey]
			if ok {
				gpuCapacity.Sub(vramRequest)
			}
			if !appAdded {
				addRunningApp(ctx, gpuMap[gpuKey], tfv1.NameNamespace{Namespace: worker.Namespace, Name: worker.Labels[constants.WorkloadKey]})
				appAdded = true
			}
		}
	}

	for gpuKey, gpu := range s.gpuStore {
		if gpu.Status.Capacity == nil {
			log.FromContext(ctx).Info("[Warning] GPU capacity is nil, skip reconcile", "gpu", gpuKey.Name)
			continue
		}
		sameTflops := gpu.Status.Available.Tflops.Equal(tflopsCapacityMap[gpuKey])
		sameVRAM := gpu.Status.Available.Vram.Equal(vramCapacityMap[gpuKey])
		if !sameTflops || !sameVRAM {
			gpu.Status.Available.Tflops = tflopsCapacityMap[gpuKey]
			gpu.Status.Available.Vram = vramCapacityMap[gpuKey]
			s.markGPUDirty(gpuKey)
			log.FromContext(ctx).Info("Correcting gpu available resources", "gpu", gpuKey.Name, "tflops", gpu.Status.Available.Tflops.String(), "vram", gpu.Status.Available.Vram.String())
		}
	}
}

func addRunningApp(ctx context.Context, gpu *tfv1.GPU, workloadNameNamespace tfv1.NameNamespace) {
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
	} else {
		gpu.Status.RunningApps = append(gpu.Status.RunningApps, &tfv1.RunningAppDetail{
			Name:      workloadNameNamespace.Name,
			Namespace: workloadNameNamespace.Namespace,
			Count:     1,
		})
	}
}

func removeRunningApp(ctx context.Context, gpu *tfv1.GPU, workloadNameNamespace tfv1.NameNamespace) {
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
		}
	} else {
		// should not happen, if deallocation twice, it should be a bug
		log.FromContext(ctx).Info("[Warning] The app to remove not found, could be caused by deallocation twice bug", "gpu", gpu.Name, "namespace", gpu.Namespace, "workload", workloadNameNamespace.Name, "namespace", workloadNameNamespace.Namespace)
	}
}
