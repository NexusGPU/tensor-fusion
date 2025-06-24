package gpuallocator

import (
	"context"
	"fmt"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/quota"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// QuotaStore manages GPU resource quotas in memory for atomic operations
type QuotaStore struct {
	client.Client

	// In-memory quota store: namespace -> quota info
	quotaStore map[string]*QuotaStoreEntry
	storeMutex sync.RWMutex

	// Queue for tracking modified quotas that need to be synced to K8s
	dirtyQuotas    map[string]struct{}
	dirtyQuotaLock sync.Mutex

	// Calculator for shared quota computation logic
	calculator *quota.Calculator
}

// QuotaStoreEntry represents quota information in memory
type QuotaStoreEntry struct {
	// Original quota definition from K8s
	quota *tfv1.GPUResourceQuota

	// Current usage calculated in memory (authoritative)
	currentUsage *tfv1.GPUResourceUsage

	// Available quota = quota.Spec.Total - currentUsage
	available *tfv1.GPUResourceUsage
}

// NewQuotaStore creates a new quota store
func NewQuotaStore(client client.Client) *QuotaStore {
	return &QuotaStore{
		Client:      client,
		quotaStore:  make(map[string]*QuotaStoreEntry),
		dirtyQuotas: make(map[string]struct{}),
		calculator:  quota.NewCalculator(),
	}
}

// checkQuotaAvailable is the quota checking logic
// Note: This method assumes proper locking is handled by the caller
func (qs *QuotaStore) checkQuotaAvailable(namespace string, req AllocRequest) error {
	entry, exists := qs.quotaStore[namespace]
	if !exists {
		// No quota defined for this namespace, allow allocation
		return nil
	}

	replicas := int32(req.Count)

	// 1. Check single (per-workload) limits first
	if err := qs.checkSingleLimits(entry, req, replicas); err != nil {
		return err
	}

	// 2. Check total namespace limits
	return qs.checkTotalLimits(entry, req, replicas)
}

// checkSingleLimits checks per-workload limits
func (qs *QuotaStore) checkSingleLimits(entry *QuotaStoreEntry, req AllocRequest, replicas int32) error {
	single := &entry.quota.Spec.Single

	// Check maximum limits per workload
	if single.Max != nil {
		// Check single TFlops limit (per GPU)
		if single.Max.TFlops != nil && req.Request.Tflops.Cmp(*single.Max.TFlops) > 0 {
			return &QuotaExceededError{
				Namespace: entry.quota.Namespace,
				Resource:  "single.max.tflops",
				Requested: req.Request.Tflops,
				Available: *single.Max.TFlops,
				Limit:     *single.Max.TFlops,
			}
		}

		// Check single VRAM limit (per GPU)
		if single.Max.VRAM != nil && req.Request.Vram.Cmp(*single.Max.VRAM) > 0 {
			return &QuotaExceededError{
				Namespace: entry.quota.Namespace,
				Resource:  "single.max.vram",
				Requested: req.Request.Vram,
				Available: *single.Max.VRAM,
				Limit:     *single.Max.VRAM,
			}
		}

		// Check single workers limit (per workload)
		if single.Max.Workers != nil && replicas > *single.Max.Workers {
			return &QuotaExceededError{
				Namespace: entry.quota.Namespace,
				Resource:  "single.max.workers",
				Requested: *resource.NewQuantity(int64(replicas), resource.DecimalSI),
				Available: *resource.NewQuantity(int64(*single.Max.Workers), resource.DecimalSI),
				Limit:     *resource.NewQuantity(int64(*single.Max.Workers), resource.DecimalSI),
			}
		}
	}

	// Check minimum limits per workload
	if single.Min != nil {
		// Check single TFlops minimum (per GPU)
		if single.Min.TFlops != nil && req.Request.Tflops.Cmp(*single.Min.TFlops) < 0 {
			return &QuotaExceededError{
				Namespace: entry.quota.Namespace,
				Resource:  "single.min.tflops",
				Requested: req.Request.Tflops,
				Available: *single.Min.TFlops,
				Limit:     *single.Min.TFlops,
			}
		}

		// Check single VRAM minimum (per GPU)
		if single.Min.VRAM != nil && req.Request.Vram.Cmp(*single.Min.VRAM) < 0 {
			return &QuotaExceededError{
				Namespace: entry.quota.Namespace,
				Resource:  "single.min.vram",
				Requested: req.Request.Vram,
				Available: *single.Min.VRAM,
				Limit:     *single.Min.VRAM,
			}
		}

		// Check single workers minimum (per workload)
		if single.Min.Workers != nil && replicas < *single.Min.Workers {
			return &QuotaExceededError{
				Namespace: entry.quota.Namespace,
				Resource:  "single.min.workers",
				Requested: *resource.NewQuantity(int64(replicas), resource.DecimalSI),
				Available: *resource.NewQuantity(int64(*single.Min.Workers), resource.DecimalSI),
				Limit:     *resource.NewQuantity(int64(*single.Min.Workers), resource.DecimalSI),
			}
		}
	}

	return nil
}

// checkTotalLimits checks total namespace limits
func (qs *QuotaStore) checkTotalLimits(entry *QuotaStoreEntry, req AllocRequest, replicas int32) error {
	// Calculate total request for all replicas
	totalTFlopsRequest := req.Request.Tflops.DeepCopy()
	totalTFlopsRequest.Set(totalTFlopsRequest.Value() * int64(replicas))

	totalVRAMRequest := req.Request.Vram.DeepCopy()
	totalVRAMRequest.Set(totalVRAMRequest.Value() * int64(replicas))

	// Check if available quota is sufficient
	if entry.available.RequestsTFlops != nil && entry.available.RequestsTFlops.Cmp(totalTFlopsRequest) < 0 {
		return &QuotaExceededError{
			Namespace: entry.quota.Namespace,
			Resource:  "total.requests.tflops",
			Requested: totalTFlopsRequest,
			Available: *entry.available.RequestsTFlops,
			Limit:     *entry.quota.Spec.Total.RequestsTFlops,
		}
	}

	if entry.available.RequestsVRAM != nil && entry.available.RequestsVRAM.Cmp(totalVRAMRequest) < 0 {
		return &QuotaExceededError{
			Namespace: entry.quota.Namespace,
			Resource:  "total.requests.vram",
			Requested: totalVRAMRequest,
			Available: *entry.available.RequestsVRAM,
			Limit:     *entry.quota.Spec.Total.RequestsVRAM,
		}
	}

	if entry.available.Workers != nil && *entry.available.Workers < replicas {
		return &QuotaExceededError{
			Namespace: entry.quota.Namespace,
			Resource:  "total.workers",
			Requested: *resource.NewQuantity(int64(replicas), resource.DecimalSI),
			Available: *resource.NewQuantity(int64(*entry.available.Workers), resource.DecimalSI),
			Limit:     *resource.NewQuantity(int64(*entry.quota.Spec.Total.Workers), resource.DecimalSI),
		}
	}

	return nil
}

// AllocateQuota atomically allocates quota resources
// This function is called under GPU allocator's storeMutex
func (qs *QuotaStore) AllocateQuota(namespace string, req AllocRequest) {
	// Note: storeMutex is already held by the caller (GpuAllocator.Alloc)

	entry, exists := qs.quotaStore[namespace]
	if !exists {
		// No quota defined, nothing to allocate
		return
	}

	// Create resource operation using calculator
	operation := qs.calculator.NewResourceOperation(req.Request, int32(req.Count))

	// Apply allocation operation atomically
	qs.calculator.ApplyUsageOperation(entry.currentUsage, entry.available, operation, true)

	// Mark quota as dirty for sync to K8s
	qs.markQuotaDirty(namespace)
}

// DeallocateQuota atomically deallocates quota resources
// This function is called under GPU allocator's storeMutex
func (qs *QuotaStore) DeallocateQuota(namespace string, request tfv1.Resource, replicas int32) {
	// Note: storeMutex is already held by the caller (GpuAllocator.Dealloc)

	entry, exists := qs.quotaStore[namespace]
	if !exists {
		// No quota defined, nothing to deallocate
		return
	}

	// Create resource operation using calculator
	operation := qs.calculator.NewResourceOperation(request, replicas)

	// Calculate actual deallocation amounts (clamped to current usage)
	actualOperation := qs.calculator.CalculateActualDeallocation(entry.currentUsage, operation)

	// Apply deallocation operation atomically
	qs.calculator.ApplyUsageOperation(entry.currentUsage, entry.available, actualOperation, false)

	// Mark quota as dirty for sync to K8s
	qs.markQuotaDirty(namespace)
}

// initQuotaStore initializes the quota store from Kubernetes
func (qs *QuotaStore) initQuotaStore(ctx context.Context) error {
	log := log.FromContext(ctx)
	log.Info("Initializing quota store")

	// Load all GPUResourceQuotas from K8s
	quotaList := &tfv1.GPUResourceQuotaList{}
	if err := qs.List(ctx, quotaList); err != nil {
		return fmt.Errorf("failed to list GPUResourceQuotas: %w", err)
	}

	// If no GPUResourceQuotas are found, skip quota store initialization
	if len(quotaList.Items) == 0 {
		log.Info("No GPUResourceQuotas found, skipping quota store initialization")
		return nil
	}

	// Initialize quota store
	qs.quotaStore = make(map[string]*QuotaStoreEntry)

	for i := range quotaList.Items {
		quota := &quotaList.Items[i]
		namespace := quota.Namespace

		// Validate quota configuration
		if err := qs.validateQuotaConfig(quota); err != nil {
			log.Error(err, "Invalid quota configuration, skipping", "namespace", namespace)
			continue
		}

		// Initialize current usage to zero using calculator
		currentUsage := qs.calculator.CreateZeroUsage()

		// Initialize available quota to total quota using calculator
		available := qs.calculator.CopyUsageToTotal(quota)

		qs.quotaStore[namespace] = &QuotaStoreEntry{
			quota:        quota.DeepCopy(),
			currentUsage: currentUsage,
			available:    available,
		}
	}

	log.Info("Quota store initialized", "count", len(qs.quotaStore))
	return nil
}

// validateQuotaConfig validates quota configuration
func (qs *QuotaStore) validateQuotaConfig(quota *tfv1.GPUResourceQuota) error {
	// Check for negative values
	if quota.Spec.Total.RequestsTFlops != nil && quota.Spec.Total.RequestsTFlops.Sign() < 0 {
		return fmt.Errorf("requests.tflops cannot be negative")
	}
	if quota.Spec.Total.RequestsVRAM != nil && quota.Spec.Total.RequestsVRAM.Sign() < 0 {
		return fmt.Errorf("requests.vram cannot be negative")
	}
	if quota.Spec.Total.LimitsTFlops != nil && quota.Spec.Total.LimitsTFlops.Sign() < 0 {
		return fmt.Errorf("limits.tflops cannot be negative")
	}
	if quota.Spec.Total.LimitsVRAM != nil && quota.Spec.Total.LimitsVRAM.Sign() < 0 {
		return fmt.Errorf("limits.vram cannot be negative")
	}
	if quota.Spec.Total.Workers != nil && *quota.Spec.Total.Workers < 0 {
		return fmt.Errorf("workers cannot be negative")
	}

	// Validate limits >= requests
	if quota.Spec.Total.RequestsTFlops != nil && quota.Spec.Total.LimitsTFlops != nil {
		if quota.Spec.Total.LimitsTFlops.Cmp(*quota.Spec.Total.RequestsTFlops) < 0 {
			return fmt.Errorf("limits.tflops cannot be less than requests.tflops")
		}
	}
	if quota.Spec.Total.RequestsVRAM != nil && quota.Spec.Total.LimitsVRAM != nil {
		if quota.Spec.Total.LimitsVRAM.Cmp(*quota.Spec.Total.RequestsVRAM) < 0 {
			return fmt.Errorf("limits.vram cannot be less than requests.vram")
		}
	}

	// Validate alert threshold percentage range
	if quota.Spec.Total.AlertThresholdPercent != nil {
		threshold := *quota.Spec.Total.AlertThresholdPercent
		if threshold < 0 || threshold > 100 {
			return fmt.Errorf("alertThresholdPercent must be between 0 and 100, got %d", threshold)
		}
	}

	return nil
}

// reconcileQuotaStore rebuilds quota usage from actual worker pods
func (qs *QuotaStore) reconcileQuotaStore(ctx context.Context, workerPods []v1.Pod) {
	log := log.FromContext(ctx)
	log.Info("Reconciling quota store")

	// Check QuotaStore state before reconciling
	if len(qs.quotaStore) == 0 {
		log.Info("QuotaStore is empty, no quotas to reconcile")
		return
	}

	// Reset all current usage to zero
	qs.storeMutex.Lock()
	defer qs.storeMutex.Unlock()

	for namespace, entry := range qs.quotaStore {
		// Reset current usage using calculator
		entry.currentUsage = qs.calculator.CreateZeroUsage()

		// Reset available to total using calculator
		entry.available = qs.calculator.CopyUsageToTotal(entry.quota)

		qs.markQuotaDirty(namespace)
	}

	// Process worker pods to rebuild quota usage
	for _, pod := range workerPods {
		// Skip pods that are not running or pending
		if pod.Status.Phase != v1.PodRunning && pod.Status.Phase != v1.PodPending {
			continue
		}

		// Skip pods with deletion timestamp
		if !pod.DeletionTimestamp.IsZero() {
			continue
		}

		namespace := pod.Namespace
		entry, exists := qs.quotaStore[namespace]
		if !exists {
			continue // No quota defined for this namespace
		}

		// Add pod resources to current usage
		qs.addPodToUsage(entry, &pod)
	}

	log.Info("Quota store reconcile completed", "quota_count", len(qs.quotaStore), "processed_pods", len(workerPods))
}

// addPodToUsage adds pod resources to current usage and updates available
func (qs *QuotaStore) addPodToUsage(entry *QuotaStoreEntry, pod *v1.Pod) {
	// Get resource information from pod annotations
	tflopsAnnotation, exists := pod.Annotations[constants.TFLOPSRequestAnnotation]
	if !exists {
		return // Skip pods without resource annotations
	}

	vramAnnotation, exists := pod.Annotations[constants.VRAMRequestAnnotation]
	if !exists {
		return // Skip pods without resource annotations
	}

	// Parse resource quantities from annotations
	tflopsRequest, err := resource.ParseQuantity(tflopsAnnotation)
	if err != nil {
		return // Skip pods with invalid resource annotations
	}

	vramRequest, err := resource.ParseQuantity(vramAnnotation)
	if err != nil {
		return // Skip pods with invalid resource annotations
	}

	// Create resource operation from pod resource
	podResource := tfv1.Resource{
		Tflops: tflopsRequest,
		Vram:   vramRequest,
	}
	operation := qs.calculator.NewResourceOperation(podResource, 1)

	// Apply allocation operation for this pod
	qs.calculator.ApplyUsageOperation(entry.currentUsage, entry.available, operation, true)
}

// markQuotaDirty marks a quota as dirty for sync to K8s
func (qs *QuotaStore) markQuotaDirty(namespace string) {
	qs.dirtyQuotaLock.Lock()
	defer qs.dirtyQuotaLock.Unlock()
	qs.dirtyQuotas[namespace] = struct{}{}
}

// clearQuotaDirty clears a quota from dirty queue after successful sync
func (qs *QuotaStore) clearQuotaDirty(namespace string) {
	qs.dirtyQuotaLock.Lock()
	defer qs.dirtyQuotaLock.Unlock()
	delete(qs.dirtyQuotas, namespace)
}

// GetQuotaStatus returns current quota status for a namespace
func (qs *QuotaStore) GetQuotaStatus(namespace string) (*tfv1.GPUResourceUsage, *tfv1.GPUResourceUsage, bool) {
	qs.storeMutex.RLock()
	defer qs.storeMutex.RUnlock()

	entry, exists := qs.quotaStore[namespace]
	if !exists {
		return nil, nil, false
	}

	return entry.currentUsage.DeepCopy(), entry.available.DeepCopy(), true
}

// syncQuotasToK8s syncs dirty quotas to Kubernetes
func (qs *QuotaStore) syncQuotasToK8s(ctx context.Context) {
	log := log.FromContext(ctx)

	// Get dirty quotas
	qs.dirtyQuotaLock.Lock()
	dirtyNamespaces := make([]string, 0, len(qs.dirtyQuotas))
	for namespace := range qs.dirtyQuotas {
		dirtyNamespaces = append(dirtyNamespaces, namespace)
	}
	// Don't clear dirty quotas here - clear only after successful sync
	qs.dirtyQuotaLock.Unlock()

	if len(dirtyNamespaces) == 0 {
		return
	}

	qs.storeMutex.RLock()
	defer qs.storeMutex.RUnlock()

	for _, namespace := range dirtyNamespaces {
		entry, exists := qs.quotaStore[namespace]
		if !exists {
			continue
		}

		// Calculate available percentages using calculator
		availablePercent := qs.calculator.CalculateAvailablePercent(entry.quota, entry.currentUsage)

		// Update the quota status
		quotaCopy := entry.quota.DeepCopy()
		quotaCopy.Status.Used = *entry.currentUsage.DeepCopy()
		quotaCopy.Status.AvailablePercent = *availablePercent
		now := metav1.Now()
		quotaCopy.Status.LastUpdateTime = &now

		// Update conditions using calculator
		qs.updateQuotaConditions(quotaCopy, entry)

		// Sync to Kubernetes
		if err := qs.Status().Update(ctx, quotaCopy); err != nil {
			log.Error(err, "Failed to update quota status", "namespace", namespace)
			// Keep in dirty queue for retry (already marked as dirty)
		} else {
			log.V(2).Info("Quota status synced to K8s", "namespace", namespace)
			// Clear from dirty queue only on successful sync
			qs.clearQuotaDirty(namespace)
		}
	}
}

// updateQuotaConditions updates the quota conditions based on current usage
func (qs *QuotaStore) updateQuotaConditions(quota *tfv1.GPUResourceQuota, entry *QuotaStoreEntry) {
	// Get default alert threshold
	alertThreshold := int32(95)
	if entry.quota.Spec.Total.AlertThresholdPercent != nil {
		alertThreshold = *entry.quota.Spec.Total.AlertThresholdPercent
	}

	// Use calculator to check conditions
	exceeded := qs.calculator.IsQuotaExceeded(entry.quota, entry.currentUsage)
	availablePercent := qs.calculator.CalculateAvailablePercent(entry.quota, entry.currentUsage)
	alertReached := qs.calculator.IsAlertThresholdReached(availablePercent, alertThreshold)

	// Use calculator to create standard conditions
	quota.Status.Conditions = qs.calculator.CreateStandardConditions(exceeded, alertReached, alertThreshold)
}

// QuotaExceededError represents a quota exceeded error with detailed information
type QuotaExceededError struct {
	Namespace string
	Resource  string
	Requested resource.Quantity
	Available resource.Quantity
	Limit     resource.Quantity
}

func (e *QuotaExceededError) Error() string {
	return fmt.Sprintf("quota exceeded in namespace %s for %s: requested %s, available %s, limit %s",
		e.Namespace, e.Resource, e.Requested.String(), e.Available.String(), e.Limit.String())
}

// IsQuotaError checks if the error is a quota-related error
func IsQuotaError(err error) bool {
	_, ok := err.(*QuotaExceededError)
	return ok
}
