package external_dp

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/fsnotify/fsnotify"
	"k8s.io/klog/v2"
)

const (
	// Default kubelet checkpoint file path
	defaultKubeletCheckpointPath = "/var/lib/kubelet/device-plugins/kubelet_internal_checkpoint"

	// Polling intervals
	defaultPollInterval     = 30 * time.Second
	defaultPatchAllInterval = 120 * time.Second
	patchAllIntervalJitter  = 0.15 // ±15% jitter
)

// KubeletCheckpoint represents the structure of kubelet device checkpoint file
type KubeletCheckpoint struct {
	Data CheckpointData `json:"Data"`
}

type CheckpointData struct {
	PodDeviceEntries  []PodDeviceEntry    `json:"PodDeviceEntries,omitempty"`
	RegisteredDevices map[string][]string `json:"RegisteredDevices,omitempty"`
}

type PodDeviceEntry struct {
	PodUID        string              `json:"PodUID"`
	ContainerName string              `json:"ContainerName"`
	ResourceName  string              `json:"ResourceName"`
	DeviceIDs     map[string][]string `json:"DeviceIDs"`
}

// VendorDetector interface for vendor-specific device plugin detectors
type VendorDetector interface {
	// GetResourceName returns the resource name this detector handles (e.g., "nvidia.com/gpu")
	GetResourceName() string
	// GetUsedBySystem returns the UsedBy system name for this vendor
	GetUsedBySystem() string
}

// APIServerInterface defines the interface for GPU API operations
type APIServerInterface interface {
	GetGPU(uuid string) (*tfv1.GPU, error)
	UpdateGPUStatus(gpu *tfv1.GPU) error
}

// KubeletClientInterface defines the interface for pod listing
type KubeletClientInterface interface {
	GetAllPods() map[string]interface{} // Returns map of pod UID to pod (can be *corev1.Pod)
}

// DevicePluginDetector watches kubelet device checkpoint and manages GPU resource patching
type DevicePluginDetector struct {
	ctx               context.Context
	checkpointPath    string
	apiServer         APIServerInterface
	kubeletClient     KubeletClientInterface
	vendorDetectors   map[string]VendorDetector // key: resource name
	previousDeviceIDs map[string]bool
	mu                sync.RWMutex
	watcher           *fsnotify.Watcher
	stopCh            chan struct{}
}

// NewDevicePluginDetector creates a new device plugin detector
func NewDevicePluginDetector(
	ctx context.Context,
	checkpointPath string,
	apiServer APIServerInterface,
	kubeletClient KubeletClientInterface,
) (*DevicePluginDetector, error) {
	if checkpointPath == "" {
		checkpointPath = defaultKubeletCheckpointPath
	}

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("failed to create filesystem watcher: %w", err)
	}

	detector := &DevicePluginDetector{
		ctx:               ctx,
		checkpointPath:    checkpointPath,
		apiServer:         apiServer,
		kubeletClient:     kubeletClient,
		vendorDetectors:   make(map[string]VendorDetector),
		previousDeviceIDs: make(map[string]bool),
		watcher:           watcher,
		stopCh:            make(chan struct{}),
	}

	// Register vendor-specific detectors
	detector.registerVendorDetectors()

	return detector, nil
}

// registerVendorDetectors registers all vendor-specific detectors
func (d *DevicePluginDetector) registerVendorDetectors() {
	// Register NVIDIA detector
	nvdpDetector := NewNvidiaDevicePluginDetector()
	d.vendorDetectors[nvdpDetector.GetResourceName()] = nvdpDetector

	// Add more vendor detectors here as needed
	// amdDetector := NewAMDDevicePluginDetector()
	// d.vendorDetectors[amdDetector.GetResourceName()] = amdDetector
}

// Start starts watching the checkpoint file and processing device allocations
func (d *DevicePluginDetector) Start() error {
	klog.Info("Starting device plugin detector", "checkpointPath", d.checkpointPath)

	// Setup filesystem watcher
	if err := d.setupFilesystemWatcher(); err != nil {
		klog.Warningf("Failed to setup filesystem watcher, falling back to polling only: %v", err)
	}

	// Start processing loop
	go d.run()

	return nil
}

// Stop stops the detector
func (d *DevicePluginDetector) Stop() {
	close(d.stopCh)
	if d.watcher != nil {
		d.watcher.Close()
	}
}

// setupFilesystemWatcher sets up filesystem watcher for the checkpoint file
func (d *DevicePluginDetector) setupFilesystemWatcher() error {
	// Watch the directory containing the checkpoint file
	dir := filepath.Dir(d.checkpointPath)
	if err := d.watcher.Add(dir); err != nil {
		return fmt.Errorf("failed to watch directory %s: %w", dir, err)
	}

	// Also watch the file itself if it exists
	if _, err := os.Stat(d.checkpointPath); err == nil {
		if err := d.watcher.Add(d.checkpointPath); err != nil {
			klog.Warningf("Failed to watch checkpoint file directly: %v", err)
		}
	}

	klog.Infof("Filesystem watcher enabled for checkpoint file: %s", d.checkpointPath)
	return nil
}

// run is the main processing loop
func (d *DevicePluginDetector) run() {
	// Create tickers for periodic polling
	pollTicker := time.NewTicker(defaultPollInterval)
	defer pollTicker.Stop()

	patchAllInterval := d.durationWithJitter(defaultPatchAllInterval, patchAllIntervalJitter)
	patchAllTicker := time.NewTicker(patchAllInterval)
	defer patchAllTicker.Stop()

	// Process initial state
	if err := d.processDeviceState(false); err != nil {
		klog.Errorf("Failed to process initial device state: %v", err)
	}

	for {
		select {
		case <-d.ctx.Done():
			klog.Info("Device plugin detector shutdown requested")
			return

		case <-d.stopCh:
			klog.Info("Device plugin detector stopped")
			return

		case event, ok := <-d.watcher.Events:
			if !ok {
				klog.Warning("Filesystem watcher channel closed, restarting watcher")
				// Try to restart watcher
				if err := d.setupFilesystemWatcher(); err != nil {
					klog.Errorf("Failed to restart filesystem watcher: %v", err)
				}
				continue
			}

			// Process checkpoint file changes
			if event.Op&(fsnotify.Write|fsnotify.Create) != 0 &&
				(event.Name == d.checkpointPath || strings.HasSuffix(event.Name, filepath.Base(d.checkpointPath))) {
				klog.V(4).Infof("Checkpoint file changed: %s", event.Name)
				if err := d.processDeviceState(false); err != nil {
					klog.Errorf("Failed to process device state after filesystem event: %v", err)
				}
			}

		case err := <-d.watcher.Errors:
			if err != nil {
				klog.Errorf("Filesystem watcher error: %v", err)
			}

		case <-pollTicker.C:
			// Periodic polling fallback
			klog.V(4).Info("Periodic polling check")
			if err := d.processDeviceState(false); err != nil {
				klog.Errorf("Failed to process device state during periodic check: %v", err)
			}

		case <-patchAllTicker.C:
			// Periodic full patch check to handle deleted pods
			klog.V(4).Info("Checking all devices for deleted pods")
			if err := d.processDeviceState(true); err != nil {
				klog.Errorf("Failed to process device state during patch all check: %v", err)
			}
			// Reset ticker with new jitter
			patchAllTicker.Reset(d.durationWithJitter(defaultPatchAllInterval, patchAllIntervalJitter))
		}
	}
}

// processDeviceState reads and processes the device checkpoint state
func (d *DevicePluginDetector) processDeviceState(patchAllDevices bool) error {
	// Read checkpoint file
	checkpoint, err := d.readCheckpointFile()
	if err != nil {
		return fmt.Errorf("failed to read checkpoint file: %w", err)
	}

	// Extract registered device IDs (for comparison)
	_, registeredDeviceIDs, err := d.extractDeviceIDs(checkpoint)
	if err != nil {
		return fmt.Errorf("failed to extract device IDs: %w", err)
	}

	// Get current pods to check for deleted pods
	currentPods := d.kubeletClient.GetAllPods()
	currentPodUIDs := make(map[string]bool, len(currentPods))
	for uid := range currentPods {
		currentPodUIDs[uid] = true
	}

	// Build device ID to entry mapping for vendor-specific processing
	deviceToEntry := make(map[string]PodDeviceEntry)

	// Filter allocated devices by checking if pods still exist
	// This handles the case where pods are deleted but checkpoint isn't updated
	validAllocatedDeviceIDs := make(map[string]bool)

	if checkpoint.Data.PodDeviceEntries != nil {
		for _, entry := range checkpoint.Data.PodDeviceEntries {
			// Check if we have a detector for this resource
			if _, hasDetector := d.vendorDetectors[entry.ResourceName]; !hasDetector {
				continue
			}

			// Check if pod still exists
			if !currentPodUIDs[entry.PodUID] {
				// Pod was deleted, but checkpoint may still have it
				// We'll handle this in the removed devices logic
				continue
			}

			// Extract device IDs from this entry
			for _, deviceList := range entry.DeviceIDs {
				for _, deviceID := range deviceList {
					deviceIDLower := strings.ToLower(deviceID)
					validAllocatedDeviceIDs[deviceIDLower] = true
					deviceToEntry[deviceIDLower] = entry
				}
			}
		}
	}

	// Determine added and removed devices
	d.mu.Lock()
	previousDeviceIDs := make(map[string]bool, len(d.previousDeviceIDs))
	for k, v := range d.previousDeviceIDs {
		previousDeviceIDs[k] = v
	}
	d.mu.Unlock()

	var addedDevices, removedDevices map[string]bool

	if patchAllDevices {
		// Patch all devices: treat all allocated as added, and all registered but not allocated as removed
		addedDevices = validAllocatedDeviceIDs
		removedDevices = make(map[string]bool)
		for deviceID := range registeredDeviceIDs {
			if !validAllocatedDeviceIDs[deviceID] {
				removedDevices[deviceID] = true
			}
		}
	} else {
		// Only process changes
		addedDevices = make(map[string]bool)
		removedDevices = make(map[string]bool)

		for deviceID := range validAllocatedDeviceIDs {
			if !previousDeviceIDs[deviceID] {
				addedDevices[deviceID] = true
			}
		}

		for deviceID := range previousDeviceIDs {
			if !validAllocatedDeviceIDs[deviceID] {
				removedDevices[deviceID] = true
			}
		}
	}

	// Process added devices using vendor-specific detectors
	hasError := false
	for deviceID := range addedDevices {
		entry, exists := deviceToEntry[deviceID]
		if !exists {
			// Try to find entry from checkpoint
			entry = d.findEntryForDevice(checkpoint, deviceID)
		}

		detector, hasDetector := d.vendorDetectors[entry.ResourceName]
		if !hasDetector {
			klog.Warningf("No detector found for resource %s, device %s", entry.ResourceName, deviceID)
			continue
		}

		usedBySystem := detector.GetUsedBySystem()
		klog.Infof("Device added: %s, resource: %s, patching with usedBy: %s", deviceID, entry.ResourceName, usedBySystem)
		if err := d.patchGPUResource(deviceID, usedBySystem); err != nil {
			klog.Errorf("Failed to patch GPU resource for added device %s: %v", deviceID, err)
			hasError = true
		}
	}

	// Process removed devices
	for deviceID := range removedDevices {
		// Find which resource this device belongs to
		entry := d.findEntryForDevice(checkpoint, deviceID)
		if entry.ResourceName == "" {
			// Try to find from previous state - use NVIDIA as default
			entry.ResourceName = "nvidia.com/gpu"
		}

		usedBySystem := string(tfv1.UsedByTensorFusion)
		klog.Infof("Device removed: %s, patching with usedBy: %s", deviceID, usedBySystem)
		if err := d.patchGPUResource(deviceID, usedBySystem); err != nil {
			klog.Errorf("Failed to patch GPU resource for removed device %s: %v", deviceID, err)
			hasError = true
		}
	}

	// Update previous state only if no errors occurred
	if !hasError {
		d.mu.Lock()
		d.previousDeviceIDs = validAllocatedDeviceIDs
		d.mu.Unlock()
	}

	return nil
}

// patchGPUResource patches a GPU resource with the specified usedBy value
func (d *DevicePluginDetector) patchGPUResource(deviceID, usedBySystem string) error {
	const maxRetries = 3

	for i := 0; i < maxRetries; i++ {
		// Get current GPU resource
		gpu, err := d.apiServer.GetGPU(deviceID)
		if err != nil {
			if i < maxRetries-1 {
				backoff := time.Duration(200*(1<<i)) * time.Millisecond
				time.Sleep(backoff)
				continue
			}
			return fmt.Errorf("failed to get GPU resource: %w", err)
		}

		// Check if already set to desired value
		if gpu.Status.UsedBy == tfv1.UsedBySystem(usedBySystem) {
			return nil
		}

		// Patch the GPU status
		gpu.Status.UsedBy = tfv1.UsedBySystem(usedBySystem)
		if err := d.apiServer.UpdateGPUStatus(gpu); err != nil {
			if i < maxRetries-1 {
				backoff := time.Duration(200*(1<<i)) * time.Millisecond
				time.Sleep(backoff)
				continue
			}
			return fmt.Errorf("failed to patch GPU resource status: %w", err)
		}

		klog.V(4).Infof("Successfully patched GPU resource %s with usedBy: %s", deviceID, usedBySystem)
		return nil
	}

	return fmt.Errorf("failed to patch GPU resource after %d retries", maxRetries)
}

// readCheckpointFile reads and parses the kubelet checkpoint file
func (d *DevicePluginDetector) readCheckpointFile() (*KubeletCheckpoint, error) {
	data, err := os.ReadFile(d.checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint file: %w", err)
	}

	var checkpoint KubeletCheckpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return nil, fmt.Errorf("failed to parse checkpoint JSON: %w", err)
	}

	return &checkpoint, nil
}

// extractDeviceIDs extracts allocated and registered device IDs from checkpoint
func (d *DevicePluginDetector) extractDeviceIDs(checkpoint *KubeletCheckpoint) (allocated, registered map[string]bool, err error) {
	allocated = make(map[string]bool)
	registered = make(map[string]bool)

	// Extract allocated devices from pod device entries
	if checkpoint.Data.PodDeviceEntries != nil {
		for _, entry := range checkpoint.Data.PodDeviceEntries {
			// Only process resources we have detectors for
			if _, hasDetector := d.vendorDetectors[entry.ResourceName]; !hasDetector {
				continue
			}

			for _, deviceList := range entry.DeviceIDs {
				for _, deviceID := range deviceList {
					allocated[strings.ToLower(deviceID)] = true
				}
			}
		}
	}

	// Extract registered devices
	if checkpoint.Data.RegisteredDevices != nil {
		for resourceName, deviceIDs := range checkpoint.Data.RegisteredDevices {
			if _, hasDetector := d.vendorDetectors[resourceName]; hasDetector {
				for _, deviceID := range deviceIDs {
					registered[strings.ToLower(deviceID)] = true
				}
			}
		}
	}

	return allocated, registered, nil
}

// findEntryForDevice finds the pod device entry for a given device ID
func (d *DevicePluginDetector) findEntryForDevice(checkpoint *KubeletCheckpoint, deviceID string) PodDeviceEntry {
	deviceIDLower := strings.ToLower(deviceID)

	if checkpoint.Data.PodDeviceEntries != nil {
		for _, entry := range checkpoint.Data.PodDeviceEntries {
			for _, deviceList := range entry.DeviceIDs {
				for _, id := range deviceList {
					if strings.ToLower(id) == deviceIDLower {
						return entry
					}
				}
			}
		}
	}

	return PodDeviceEntry{}
}

// durationWithJitter creates a duration with jitter to avoid thundering herd problems
func (d *DevicePluginDetector) durationWithJitter(baseDuration time.Duration, jitterPercent float64) time.Duration {
	jitterRange := float64(baseDuration) * jitterPercent
	jitterOffset := (rand.Float64()*2 - 1) * jitterRange // -1 to 1
	return baseDuration + time.Duration(jitterOffset)
}
