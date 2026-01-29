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

package provider

import (
	"context"
	"fmt"
	"maps"
	"strconv"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

// Manager handles ProviderConfig resources in memory with hot-reload support
type Manager struct {
	client client.Client

	// providers maps vendor name to ProviderConfig
	providers map[string]*tfv1.ProviderConfig
	mu        sync.RWMutex

	// gpuInfoCache caches GpuInfo converted from ProviderConfig
	gpuInfoCache map[string][]config.GpuInfo
	gpuInfoMu    sync.RWMutex

	// inUseResources maps vendor to resource names that should be removed
	inUseResources map[string][]corev1.ResourceName
	inUseMu        sync.RWMutex

	// resourceNameToVendor maps resource name to vendor for fast lookup
	resourceNameToVendor map[corev1.ResourceName]string
	resourceNameMu       sync.RWMutex
}

var (
	globalManager     *Manager
	globalManagerOnce sync.Once
)

// GetManager returns the global provider manager instance
func GetManager() *Manager {
	return globalManager
}

// NewManager creates a new Provider Manager
func NewManager(client client.Client) *Manager {
	return &Manager{
		client:               client,
		providers:            make(map[string]*tfv1.ProviderConfig),
		gpuInfoCache:         make(map[string][]config.GpuInfo),
		inUseResources:       make(map[string][]corev1.ResourceName),
		resourceNameToVendor: make(map[corev1.ResourceName]string),
	}
}

// InitGlobalManager initializes the global provider manager
func InitGlobalManager(client client.Client) *Manager {
	globalManagerOnce.Do(func() {
		globalManager = NewManager(client)
	})
	return globalManager
}

// SetGlobalManagerForTesting sets a custom manager for testing purposes
// This bypasses the sync.Once and allows tests to reset the global manager
func SetGlobalManagerForTesting(mgr *Manager) {
	globalManager = mgr
}

// SetupWithManager sets up the provider manager with the controller-runtime manager
func (m *Manager) SetupWithManager(ctx context.Context, mgr manager.Manager) error {
	// Load all existing ProviderConfigs at startup
	if err := m.loadAllProviders(ctx); err != nil {
		return fmt.Errorf("failed to load providers: %w", err)
	}
	return nil
}

// loadAllProviders loads all ProviderConfig resources from the cluster
func (m *Manager) loadAllProviders(ctx context.Context) error {
	logger := log.FromContext(ctx)

	var providerList tfv1.ProviderConfigList
	if err := m.client.List(ctx, &providerList); err != nil {
		return fmt.Errorf("failed to list ProviderConfigs: %w", err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for i := range providerList.Items {
		provider := &providerList.Items[i]
		m.providers[provider.Spec.Vendor] = provider
		m.updateCaches(provider)
		logger.Info("loaded ProviderConfig", "vendor", provider.Spec.Vendor, "name", provider.Name)
	}

	logger.Info("loaded all ProviderConfigs", "count", len(providerList.Items))
	return nil
}

// UpdateProvider updates or adds a ProviderConfig in the cache
func (m *Manager) UpdateProvider(provider *tfv1.ProviderConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.providers[provider.Spec.Vendor] = provider.DeepCopy()
	m.updateCaches(provider)
}

// DeleteProvider removes a ProviderConfig from the cache
func (m *Manager) DeleteProvider(vendor string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.providers, vendor)

	m.gpuInfoMu.Lock()
	delete(m.gpuInfoCache, vendor)
	m.gpuInfoMu.Unlock()

	m.inUseMu.Lock()
	delete(m.inUseResources, vendor)
	m.inUseMu.Unlock()

	// Remove resource name mappings for this vendor
	m.resourceNameMu.Lock()
	for resourceName, v := range m.resourceNameToVendor {
		if v == vendor {
			delete(m.resourceNameToVendor, resourceName)
		}
	}
	m.resourceNameMu.Unlock()
}

// updateCaches updates the derived caches from provider config
// Must be called with m.mu held
func (m *Manager) updateCaches(provider *tfv1.ProviderConfig) {
	// Update GPU info cache
	gpuInfos := m.convertToGpuInfos(provider)
	m.gpuInfoMu.Lock()
	m.gpuInfoCache[provider.Spec.Vendor] = gpuInfos
	m.gpuInfoMu.Unlock()

	// Update in-use resources cache
	resources := make([]corev1.ResourceName, 0, len(provider.Spec.InUseResourceNames))
	for _, name := range provider.Spec.InUseResourceNames {
		resources = append(resources, corev1.ResourceName(name))
	}
	m.inUseMu.Lock()
	m.inUseResources[provider.Spec.Vendor] = resources
	m.inUseMu.Unlock()

	// Update resource name to vendor reverse map
	m.resourceNameMu.Lock()
	// Remove old mappings for this vendor first
	for resourceName, vendor := range m.resourceNameToVendor {
		if vendor == provider.Spec.Vendor {
			delete(m.resourceNameToVendor, resourceName)
		}
	}
	// Add new mappings
	for _, name := range provider.Spec.InUseResourceNames {
		m.resourceNameToVendor[corev1.ResourceName(name)] = provider.Spec.Vendor
	}
	m.resourceNameMu.Unlock()
}

// convertToGpuInfos converts ProviderConfig hardware metadata to GpuInfo slice
func (m *Manager) convertToGpuInfos(provider *tfv1.ProviderConfig) []config.GpuInfo {
	// Build template lookup map
	templateMap := make(map[string]*tfv1.VirtualizationTemplate)
	for i := range provider.Spec.VirtualizationTemplates {
		tmpl := &provider.Spec.VirtualizationTemplates[i]
		templateMap[tmpl.ID] = tmpl
	}

	gpuInfos := make([]config.GpuInfo, 0, len(provider.Spec.HardwareMetadata))
	for _, hw := range provider.Spec.HardwareMetadata {
		costPerHour := 0.0
		if hw.CostPerHour != "" {
			if parsed, err := strconv.ParseFloat(hw.CostPerHour, 64); err == nil {
				costPerHour = parsed
			}
		}

		gpuInfo := config.GpuInfo{
			Model:                  hw.Model,
			Vendor:                 provider.Spec.Vendor,
			CostPerHour:            costPerHour,
			Fp16TFlops:             hw.Fp16TFlops,
			FullModelName:          hw.FullModelName,
			MaxPartitions:          hw.MaxPartitions,
			MaxPlacementSlots:      hw.MaxPlacementSlots,
			MaxIsolationGroups:     hw.MaxIsolationGroups,
			TotalExtendedResources: hw.TotalExtendedResources,
		}

		// Convert partition template references to PartitionTemplateInfo
		for _, ref := range hw.PartitionTemplateRefs {
			if tmpl, ok := templateMap[ref]; ok {
				computePercent := 0.0
				if tmpl.ComputePercent != "" {
					if parsed, err := strconv.ParseFloat(tmpl.ComputePercent, 64); err == nil {
						computePercent = parsed
					}
				}
				partitionInfo := config.PartitionTemplateInfo{
					TemplateID:                     tmpl.ID,
					Name:                           tmpl.Name,
					MemoryGigabytes:                tmpl.MemoryGigabytes,
					ComputePercent:                 computePercent,
					Description:                    tmpl.Description,
					MaxPartition:                   tmpl.MaxInstances,
					PlacementLimit:                 tmpl.PlacementLimit,
					PlacementOffSet:                tmpl.PlacementOffset,
					ExtendedResources:              tmpl.ExtendedResources,
					IsolationGroupSharing:          tmpl.IsolationGroupSharing,
					MaxPartitionsPerIsolationGroup: tmpl.MaxPartitionsPerIsolationGroup,
					IsolationGroupSlots:            tmpl.IsolationGroupSlots,
				}
				gpuInfo.PartitionTemplates = append(gpuInfo.PartitionTemplates, partitionInfo)
			}
		}

		gpuInfos = append(gpuInfos, gpuInfo)
	}

	return gpuInfos
}

// GetProvider returns the ProviderConfig for a vendor
func (m *Manager) GetProvider(vendor string) (*tfv1.ProviderConfig, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	provider, ok := m.providers[vendor]
	if !ok {
		return nil, false
	}
	return provider.DeepCopy(), true
}

// GetProviderOrDefault returns the ProviderConfig for a vendor, or default NVIDIA if not found
func (m *Manager) GetProviderOrDefault(vendor string) *tfv1.ProviderConfig {
	if provider, ok := m.GetProvider(vendor); ok {
		return provider
	}
	// Fall back to NVIDIA as default
	if provider, ok := m.GetProvider(constants.AcceleratorVendorNvidia); ok {
		return provider
	}
	return nil
}

// GetAllProviders returns all cached ProviderConfigs
// Note: Returns a shallow copy of the map. The map itself is new, but values are shared references.
// Callers should not modify the returned ProviderConfig objects.
func (m *Manager) GetAllProviders() map[string]*tfv1.ProviderConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]*tfv1.ProviderConfig, len(m.providers))
	maps.Copy(result, m.providers)
	return result
}

// GetMiddlewareImage returns the middleware (hypervisor) image for a vendor
func (m *Manager) GetMiddlewareImage(vendor string, defaultImage string) string {
	if provider, ok := m.GetProvider(vendor); ok && provider.Spec.Images.Middleware != "" {
		return provider.Spec.Images.Middleware
	}
	return defaultImage
}

// GetRemoteClientImage returns the remote client image for a vendor
func (m *Manager) GetRemoteClientImage(vendor string, defaultImage string) string {
	if provider, ok := m.GetProvider(vendor); ok && provider.Spec.Images.RemoteClient != "" {
		return provider.Spec.Images.RemoteClient
	}
	return defaultImage
}

// GetRemoteWorkerImage returns the remote worker image for a vendor
func (m *Manager) GetRemoteWorkerImage(vendor string, defaultImage string) string {
	if provider, ok := m.GetProvider(vendor); ok && provider.Spec.Images.RemoteWorker != "" {
		return provider.Spec.Images.RemoteWorker
	}
	return defaultImage
}

// GetGpuInfos returns GpuInfo slice for a vendor
func (m *Manager) GetGpuInfos(vendor string) []config.GpuInfo {
	m.gpuInfoMu.RLock()
	defer m.gpuInfoMu.RUnlock()

	if infos, ok := m.gpuInfoCache[vendor]; ok {
		return infos
	}
	return nil
}

// GetAllGpuInfos returns all GpuInfo from all providers
func (m *Manager) GetAllGpuInfos() []config.GpuInfo {
	m.gpuInfoMu.RLock()
	defer m.gpuInfoMu.RUnlock()

	totalLen := 0
	for _, infos := range m.gpuInfoCache {
		totalLen += len(infos)
	}
	all := make([]config.GpuInfo, 0, totalLen)
	for _, infos := range m.gpuInfoCache {
		all = append(all, infos...)
	}
	return all
}

// GetGpuInfoByModel returns GpuInfo for a specific model across all vendors
func (m *Manager) GetGpuInfoByModel(model string) *config.GpuInfo {
	m.gpuInfoMu.RLock()
	defer m.gpuInfoMu.RUnlock()

	for _, infos := range m.gpuInfoCache {
		for i := range infos {
			if infos[i].Model == model || infos[i].FullModelName == model {
				return &infos[i]
			}
		}
	}
	return nil
}

// GetGpuInfoByFullModelName returns GpuInfo for a specific full model name
func (m *Manager) GetGpuInfoByFullModelName(fullModelName string) *config.GpuInfo {
	m.gpuInfoMu.RLock()
	defer m.gpuInfoMu.RUnlock()

	for _, infos := range m.gpuInfoCache {
		for i := range infos {
			if infos[i].FullModelName == fullModelName {
				return &infos[i]
			}
		}
	}
	return nil
}

// GetInUseResourceNames returns resource names that should be removed for a vendor
func (m *Manager) GetInUseResourceNames(vendor string) []corev1.ResourceName {
	m.inUseMu.RLock()
	defer m.inUseMu.RUnlock()

	if resources, ok := m.inUseResources[vendor]; ok {
		return resources
	}
	return nil
}

// GetAllInUseResourceNames returns all in-use resource names from all providers
func (m *Manager) GetAllInUseResourceNames() []corev1.ResourceName {
	m.inUseMu.RLock()
	defer m.inUseMu.RUnlock()

	var all []corev1.ResourceName
	seen := make(map[corev1.ResourceName]bool)
	for _, resources := range m.inUseResources {
		for _, r := range resources {
			if !seen[r] {
				all = append(all, r)
				seen[r] = true
			}
		}
	}
	return all
}

// GetGPUPricingMap returns a map of full model name to cost per hour
func (m *Manager) GetGPUPricingMap() map[string]float64 {
	m.gpuInfoMu.RLock()
	defer m.gpuInfoMu.RUnlock()

	pricingMap := make(map[string]float64)
	for _, infos := range m.gpuInfoCache {
		for _, info := range infos {
			pricingMap[info.FullModelName] = info.CostPerHour
		}
	}
	return pricingMap
}

// GetFp16TFlops returns the FP16 TFlops for a model
func (m *Manager) GetFp16TFlops(fullModelName string) resource.Quantity {
	if info := m.GetGpuInfoByFullModelName(fullModelName); info != nil {
		return info.Fp16TFlops
	}
	return resource.Quantity{}
}

// HasProvider checks if a provider exists for a vendor
func (m *Manager) HasProvider(vendor string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.providers[vendor]
	return ok
}

// ProviderCount returns the number of loaded providers
func (m *Manager) ProviderCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.providers)
}

// GetVendorFromResourceName returns the vendor that owns a given resource name
func (m *Manager) GetVendorFromResourceName(resourceName corev1.ResourceName) string {
	m.resourceNameMu.RLock()
	defer m.resourceNameMu.RUnlock()

	return m.resourceNameToVendor[resourceName]
}

// Reload reloads all providers from the cluster
func (m *Manager) Reload(ctx context.Context) error {
	// Clear existing caches
	m.mu.Lock()
	m.providers = make(map[string]*tfv1.ProviderConfig)
	m.mu.Unlock()

	m.gpuInfoMu.Lock()
	m.gpuInfoCache = make(map[string][]config.GpuInfo)
	m.gpuInfoMu.Unlock()

	m.inUseMu.Lock()
	m.inUseResources = make(map[string][]corev1.ResourceName)
	m.inUseMu.Unlock()

	m.resourceNameMu.Lock()
	m.resourceNameToVendor = make(map[corev1.ResourceName]string)
	m.resourceNameMu.Unlock()

	return m.loadAllProviders(ctx)
}
