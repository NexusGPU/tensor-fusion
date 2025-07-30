package quota

import (
	"context"
	"fmt"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

const (
	MaxTFlopsLimitResource   = "single.max.tflops.limit"
	MaxVRAMLimitResource     = "single.max.vram.limit"
	MaxTFlopsRequestResource = "single.max.tflops.request"
	MaxVRAMRequestResource   = "single.max.vram.request"
	MaxGPULimitResource      = "single.max.gpuCount.limit"

	TotalMaxTFlopsLimitResource   = "total.max.tflops.limit"
	TotalMaxVRAMLimitResource     = "total.max.vram.limit"
	TotalMaxTFlopsRequestResource = "total.max.tflops.request"
	TotalMaxVRAMRequestResource   = "total.max.vram.request"

	TotalMaxWorkersLimitResource = "total.max.workers.limit"
	DefaultAlertThresholdPercent = int32(95)
)

// QuotaStore manages GPU resource quotas in memory for atomic operations
type QuotaStore struct {
	client.Client

	// In-memory quota store: namespace -> quota info
	QuotaStore map[string]*QuotaStoreEntry
	StoreMutex sync.RWMutex

	// Calculator for shared quota computation logic
	Calculator *Calculator

	// Queue for tracking modified quotas that need to be synced to K8s
	dirtyQuotas    map[string]struct{}
	dirtyQuotaLock sync.Mutex

	ctx context.Context
}

// QuotaStoreEntry represents quota information in memory
type QuotaStoreEntry struct {
	// Original Quota definition from K8s
	Quota *tfv1.GPUResourceQuota

	// Current usage calculated in memory (authoritative)
	CurrentUsage *tfv1.GPUResourceUsage
}

// NewQuotaStore creates a new quota store
func NewQuotaStore(client client.Client, ctx context.Context) *QuotaStore {
	return &QuotaStore{
		Client:      client,
		QuotaStore:  make(map[string]*QuotaStoreEntry),
		dirtyQuotas: make(map[string]struct{}),
		Calculator:  NewCalculator(),
		ctx:         ctx,
	}
}

// CheckQuotaAvailable is the quota checking logic
// Note: This method assumes proper locking is handled by the caller
func (qs *QuotaStore) CheckQuotaAvailable(namespace string, req *tfv1.AllocRequest) error {
	entry, exists := qs.QuotaStore[namespace]
	if !exists {
		// No quota defined for this namespace, allow allocation
		return nil
	}
	if err := qs.checkSingleQuotas(entry, req); err != nil {
		return err
	}
	return qs.checkTotalQuotas(entry, req)
}

func (qs *QuotaStore) AdjustQuota(namespace string, reqDelta tfv1.Resource, limitDelta tfv1.Resource) {
	qs.StoreMutex.Lock()
	defer qs.StoreMutex.Unlock()

	entry, exists := qs.QuotaStore[namespace]
	if !exists {
		return
	}
	entry.CurrentUsage.Requests.Tflops.Add(reqDelta.Tflops)
	entry.CurrentUsage.Requests.Vram.Add(reqDelta.Vram)
	entry.CurrentUsage.Limits.Tflops.Add(limitDelta.Tflops)
	entry.CurrentUsage.Limits.Vram.Add(limitDelta.Vram)
	qs.markQuotaDirty(namespace)
}

// checkSingleQuotas checks per-workload limits
func (qs *QuotaStore) checkSingleQuotas(entry *QuotaStoreEntry, req *tfv1.AllocRequest) error {
	single := &entry.Quota.Spec.Single
	if single.MaxLimits != nil {
		if !single.MaxLimits.Tflops.IsZero() && req.Limit.Tflops.Cmp(single.MaxLimits.Tflops) > 0 {
			return &QuotaExceededError{
				Namespace: entry.Quota.Namespace,
				Resource:  MaxTFlopsLimitResource,
				Requested: req.Limit.Tflops,
				Limit:     single.MaxLimits.Tflops,
			}
		}

		// Check single VRAM limit (per GPU)
		if !single.MaxLimits.Vram.IsZero() && req.Request.Vram.Cmp(single.MaxLimits.Vram) > 0 {
			return &QuotaExceededError{
				Namespace: entry.Quota.Namespace,
				Resource:  MaxVRAMLimitResource,
				Requested: req.Request.Vram,
				Limit:     single.MaxLimits.Vram,
			}
		}

		// Check single GPU count limit (per worker)
		if single.MaxGPUCount != nil && int32(req.Count) > *single.MaxGPUCount {
			return &QuotaExceededError{
				Namespace: entry.Quota.Namespace,
				Resource:  MaxGPULimitResource,
				Requested: *resource.NewQuantity(int64(req.Count), resource.DecimalSI),
				Limit:     *resource.NewQuantity(int64(*single.MaxGPUCount), resource.DecimalSI),
			}
		}
	}
	return nil
}

func (qs *QuotaStore) checkTotalQuotas(entry *QuotaStoreEntry, req *tfv1.AllocRequest) error {
	quotaNs := entry.Quota.Namespace
	if entry.Quota.Spec.Total.Requests != nil {
		total := entry.Quota.Spec.Total.Requests
		current := entry.CurrentUsage.Requests
		err := checkTotalExceeded(req, total, current, quotaNs, true)
		if err != nil {
			return err
		}
	}

	// Check total limits
	if entry.Quota.Spec.Total.Limits != nil {
		total := entry.Quota.Spec.Total.Limits
		usage := entry.CurrentUsage.Limits
		err := checkTotalExceeded(req, total, usage, quotaNs, false)
		if err != nil {
			return err
		}
	}

	// Check total workers, each allocation will create one worker instance
	if entry.Quota.Spec.Total.MaxWorkers != nil {
		if entry.CurrentUsage.Workers >= *entry.Quota.Spec.Total.MaxWorkers {
			return &QuotaExceededError{
				Namespace: quotaNs,
				Resource:  TotalMaxWorkersLimitResource,
				Requested: *resource.NewQuantity(1, resource.DecimalSI),
				Limit:     *resource.NewQuantity(int64(*entry.Quota.Spec.Total.MaxWorkers), resource.DecimalSI),
			}
		}
	}
	return nil
}

func checkTotalExceeded(req *tfv1.AllocRequest, totalQuota *tfv1.Resource, current tfv1.Resource, quotaNs string, isRequest bool) error {
	reqGPUNum := int64(req.Count)
	var tflops, vram int64
	if isRequest {
		tflops, _ = req.Request.Tflops.AsInt64()
		vram, _ = req.Request.Vram.AsInt64()
		tflops *= reqGPUNum
		vram *= reqGPUNum
	} else {
		tflops, _ = req.Limit.Tflops.AsInt64()
		vram, _ = req.Limit.Vram.AsInt64()
		tflops *= reqGPUNum
		vram *= reqGPUNum
	}

	tflopsQuota, _ := totalQuota.Tflops.AsInt64()
	tflopsCurrent, _ := current.Tflops.AsInt64()
	if !totalQuota.Tflops.IsZero() &&
		tflopsQuota < (tflopsCurrent+tflops) {
		var exceededMsg string
		if isRequest {
			exceededMsg = TotalMaxTFlopsRequestResource
		} else {
			exceededMsg = TotalMaxTFlopsLimitResource
		}
		return &QuotaExceededError{
			Namespace: quotaNs,
			Resource:  exceededMsg,
			Requested: *resource.NewQuantity(tflops, resource.DecimalSI),
			Limit:     totalQuota.Tflops,
		}
	}

	vramQuota, _ := totalQuota.Vram.AsInt64()
	vramCurrent, _ := current.Vram.AsInt64()
	if !totalQuota.Vram.IsZero() && vramQuota < (vramCurrent+vram) {
		var exceededMsg string
		if isRequest {
			exceededMsg = TotalMaxVRAMRequestResource
		} else {
			exceededMsg = TotalMaxVRAMLimitResource
		}
		return &QuotaExceededError{
			Namespace: quotaNs,
			Resource:  exceededMsg,
			Requested: *resource.NewQuantity(vram, resource.DecimalSI),
			Limit:     totalQuota.Vram,
		}
	}
	return nil
}

// AllocateQuota atomically allocates quota resources
// This function is called under GPU allocator's storeMutex
func (qs *QuotaStore) AllocateQuota(namespace string, req *tfv1.AllocRequest) {
	entry, exists := qs.QuotaStore[namespace]
	if !exists {
		return
	}

	qs.StoreMutex.Lock()
	defer qs.StoreMutex.Unlock()

	qs.Calculator.ApplyUsageOperation(entry.CurrentUsage, req, true)
	qs.markQuotaDirty(namespace)
}

// DeallocateQuota atomically deallocate quota resources
// This function is called under GPU allocator's storeMutex
func (qs *QuotaStore) DeallocateQuota(namespace string, allocation *tfv1.AllocRequest) {
	entry, exists := qs.QuotaStore[namespace]
	if !exists {
		return
	}

	qs.StoreMutex.Lock()
	defer qs.StoreMutex.Unlock()

	qs.Calculator.ApplyUsageOperation(entry.CurrentUsage, allocation, false)
	qs.markQuotaDirty(namespace)
}

// initQuotaStore initializes the quota store from Kubernetes
func (qs *QuotaStore) InitQuotaStore() error {
	log := log.FromContext(qs.ctx)
	log.Info("Initializing quota store")

	quotaList := &tfv1.GPUResourceQuotaList{}
	if err := qs.List(qs.ctx, quotaList); err != nil {
		return fmt.Errorf("failed to list GPUResourceQuotas: %w", err)
	}

	if len(quotaList.Items) == 0 {
		log.Info("No GPUResourceQuotas found, skipping quota store initialization")
		return nil
	}
	qs.QuotaStore = make(map[string]*QuotaStoreEntry)

	for i := range quotaList.Items {
		quota := &quotaList.Items[i]
		namespace := quota.Namespace

		// Validate quota configuration
		if err := qs.validateQuotaConfig(quota); err != nil {
			log.Error(err, "Invalid quota configuration, skipping", "namespace", namespace)
			continue
		}
		currentUsage := qs.Calculator.CreateZeroUsage()

		qs.QuotaStore[namespace] = &QuotaStoreEntry{
			Quota:        quota.DeepCopy(),
			CurrentUsage: currentUsage,
		}
	}

	log.Info("Quota store initialized", "count", len(qs.QuotaStore))
	return nil
}

// ValidateQuotaConfig validates quota configuration
func (qs *QuotaStore) validateQuotaConfig(quota *tfv1.GPUResourceQuota) error {
	// Check for negative values
	if !quota.Spec.Total.Requests.Tflops.IsZero() && quota.Spec.Total.Requests.Tflops.Cmp(resource.Quantity{}) < 0 {
		return fmt.Errorf("requests.tflops cannot be negative")
	}
	if !quota.Spec.Total.Requests.Vram.IsZero() && quota.Spec.Total.Requests.Vram.Cmp(resource.Quantity{}) < 0 {
		return fmt.Errorf("requests.vram cannot be negative")
	}
	if !quota.Spec.Total.Limits.Tflops.IsZero() && quota.Spec.Total.Limits.Tflops.Cmp(resource.Quantity{}) < 0 {
		return fmt.Errorf("limits.tflops cannot be negative")
	}
	if !quota.Spec.Total.Limits.Vram.IsZero() && quota.Spec.Total.Limits.Vram.Cmp(resource.Quantity{}) < 0 {
		return fmt.Errorf("limits.vram cannot be negative")
	}
	if quota.Spec.Total.MaxWorkers != nil && *quota.Spec.Total.MaxWorkers < 0 {
		return fmt.Errorf("workers cannot be negative")
	}

	// Validate limits >= requests
	if !quota.Spec.Total.Requests.Tflops.IsZero() && !quota.Spec.Total.Limits.Tflops.IsZero() {
		if quota.Spec.Total.Limits.Tflops.Cmp(quota.Spec.Total.Requests.Tflops) < 0 {
			return fmt.Errorf("limits.tflops cannot be less than requests.tflops")
		}
	}
	if !quota.Spec.Total.Requests.Vram.IsZero() && !quota.Spec.Total.Limits.Vram.IsZero() {
		if quota.Spec.Total.Limits.Vram.Cmp(quota.Spec.Total.Requests.Vram) < 0 {
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

// ReconcileQuotaStore rebuilds quota usage from actual worker pods
func (qs *QuotaStore) ReconcileQuotaStore(ctx context.Context, namespacedAllocations map[string]*tfv1.AllocRequest) {
	log := log.FromContext(ctx)
	if len(qs.QuotaStore) == 0 {
		return
	}
	log.Info("Reconciling namespace level quota store")

	// Reset all current usage to zero
	qs.StoreMutex.Lock()
	defer qs.StoreMutex.Unlock()

	for _, entry := range qs.QuotaStore {
		entry.CurrentUsage = qs.Calculator.CreateZeroUsage()
	}

	// Process worker pods to rebuild quota usage
	for _, allocation := range namespacedAllocations {
		ns := allocation.WorkloadNameNamespace.Namespace
		entry, exists := qs.QuotaStore[ns]
		if !exists {
			continue // No quota defined for this namespace
		}
		qs.Calculator.ApplyUsageOperation(entry.CurrentUsage, allocation, true)
	}

	log.Info("Quota store reconcile completed", "quota_count", len(qs.QuotaStore), "processed_pods", len(namespacedAllocations))
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
func (qs *QuotaStore) GetQuotaStatus(namespace string) (*tfv1.GPUResourceUsage, bool) {
	qs.StoreMutex.RLock()
	defer qs.StoreMutex.RUnlock()

	entry, exists := qs.QuotaStore[namespace]
	if !exists {
		return nil, false
	}
	return entry.CurrentUsage.DeepCopy(), true
}

// syncQuotasToK8s syncs dirty quotas to Kubernetes
func (qs *QuotaStore) SyncQuotasToK8s(ctx context.Context) {
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

	qs.StoreMutex.RLock()
	defer qs.StoreMutex.RUnlock()

	for _, namespace := range dirtyNamespaces {
		entry, exists := qs.QuotaStore[namespace]
		if !exists {
			continue
		}

		// Calculate available percentages using calculator
		availablePercent := qs.Calculator.CalculateAvailablePercent(entry.Quota, entry.CurrentUsage)

		// Update the quota status
		toPatch := &tfv1.GPUResourceQuota{
			ObjectMeta: metav1.ObjectMeta{
				Name:      entry.Quota.Name,
				Namespace: namespace,
			},
		}
		toPatch.Status.Used = *entry.CurrentUsage
		toPatch.Status.AvailablePercent = *availablePercent
		now := metav1.Now()
		toPatch.Status.LastUpdateTime = &now

		// calculate if alert conditions are met
		alertThreshold := DefaultAlertThresholdPercent
		if entry.Quota.Spec.Total.AlertThresholdPercent != nil {
			alertThreshold = *entry.Quota.Spec.Total.AlertThresholdPercent
		}
		previousMsg := ""
		if len(entry.Quota.Status.Conditions) > 0 {
			previousMsg = entry.Quota.Status.Conditions[0].Message
		}
		newConditions := qs.Calculator.CalculateConditions(availablePercent, alertThreshold, previousMsg)
		if newConditions != nil {
			toPatch.Status.Conditions = newConditions
		}

		// Use patch to update the quota status, avoid potential conflicts
		if err := qs.Status().Patch(ctx, toPatch, client.MergeFrom(&tfv1.GPUResourceQuota{})); err != nil {
			log.Error(err, "Failed to sync quota status to K8s", "namespace", namespace)
		} else {
			log.V(6).Info("Quota status synced to K8s", "namespace", namespace)
			qs.clearQuotaDirty(namespace)
		}
	}
}

// QuotaExceededError represents a quota exceeded error with detailed information
type QuotaExceededError struct {
	Namespace string
	Resource  string
	Requested resource.Quantity
	Limit     resource.Quantity
}

func (e *QuotaExceededError) Error() string {
	return fmt.Sprintf("quota exceeded in namespace %s for %s: requested %s, limit %s",
		e.Namespace, e.Resource, e.Requested.String(), e.Limit.String())
}

// handleGPUQuotaCreate handles GPU quota creation events
func (qs *QuotaStore) handleGPUQuotaCreate(ctx context.Context, gpuQuota *tfv1.GPUResourceQuota) {
	log := log.FromContext(ctx)
	key := gpuQuota.Namespace

	qs.StoreMutex.Lock()
	defer qs.StoreMutex.Unlock()

	qs.QuotaStore[key].Quota = gpuQuota.DeepCopy()
	log.V(4).Info("Added GPU quota to store", "namespace", key)
}

// handleGPUQuotaDelete handles GPU quota deletion events
func (qs *QuotaStore) handleGPUQuotaDelete(ctx context.Context, gpuQuota *tfv1.GPUResourceQuota) {
	log := log.FromContext(ctx)
	key := gpuQuota.Namespace

	qs.StoreMutex.Lock()
	defer qs.StoreMutex.Unlock()

	delete(qs.QuotaStore, key)
	log.V(4).Info("Removed GPU quota from store", "namespace", key)
}

// handleGPUQuotaUpdate handles GPU quota update events
func (qs *QuotaStore) handleGPUQuotaUpdate(ctx context.Context, gpuQuota *tfv1.GPUResourceQuota) {
	log := log.FromContext(ctx)
	key := gpuQuota.Namespace

	qs.StoreMutex.Lock()
	defer qs.StoreMutex.Unlock()

	qs.QuotaStore[key].Quota = gpuQuota.DeepCopy()
	log.V(4).Info("Updated GPU quota in store (preserve Used)", "namespace", key)
}

func (qs *QuotaStore) StartInformerForGPUQuota(ctx context.Context, mgr manager.Manager) error {
	log := log.FromContext(ctx)

	informer, err := mgr.GetCache().GetInformer(ctx, &tfv1.GPUResourceQuota{})
	if err != nil {
		return fmt.Errorf("failed to get GPUResourceQuota informer: %w", err)
	}

	_, err = informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			gpuQuota, ok := obj.(*tfv1.GPUResourceQuota)
			if !ok {
				log.Error(fmt.Errorf("unexpected type"), "expected GPUResourceQuota")
				return
			}
			qs.handleGPUQuotaCreate(ctx, gpuQuota)
		},
		DeleteFunc: func(obj any) {
			gpuQuota, ok := obj.(*tfv1.GPUResourceQuota)
			if !ok {
				tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					log.Error(fmt.Errorf("unexpected type"), "expected GPUResourceQuota or tombstone")
					return
				}
				gpuQuota, ok = tombstone.Obj.(*tfv1.GPUResourceQuota)
				if !ok {
					log.Error(fmt.Errorf("unexpected type"), "expected GPUResourceQuota in tombstone")
					return
				}
			}
			qs.handleGPUQuotaDelete(ctx, gpuQuota)
		},
		UpdateFunc: func(oldObj, newObj any) {
			newGPUQuota, ok := newObj.(*tfv1.GPUResourceQuota)
			if !ok {
				log.Error(fmt.Errorf("unexpected type"), "expected new GPUResourceQuota")
				return
			}
			qs.handleGPUQuotaUpdate(ctx, newGPUQuota)
		},
	})
	return err
}
