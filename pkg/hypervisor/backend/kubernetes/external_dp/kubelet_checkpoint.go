package external_dp

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/fsnotify/fsnotify"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	// Default kubelet checkpoint file path
	defaultKubeletCheckpointPath = "/var/lib/kubelet/device-plugins/kubelet_internal_checkpoint"

	// Default kubelet pod-resources socket path
	defaultKubeletPodResourcesSocket = "/var/lib/kubelet/pod-resources/kubelet.sock"

	// Polling intervals
	defaultPollInterval     = 30 * time.Second
	defaultPatchAllInterval = 120 * time.Second
	patchAllIntervalJitter  = 0.15 // ±15% jitter
)

var (
	scheme = runtime.NewScheme()
)

func init() {
	utilruntime.Must(tfv1.AddToScheme(scheme))
}

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
	GetResourceNamePrefixes() []string
	// GetUsedBySystem returns the UsedBy system name for this vendor
	GetUsedBySystemAndRealDeviceID(deviceID, resourceName string) (system string, realDeviceID string)
}

// APIClientInterface defines the interface for GPU API operations
type APIClientInterface interface {
	GetGPU(uuid string) (*tfv1.GPU, error)
	UpdateGPUStatus(gpu *tfv1.GPU) error
}

// DevicePluginDetector watches kubelet device checkpoint and manages GPU resource patching
type DevicePluginDetector struct {
	ctx               context.Context
	checkpointPath    string
	apiClient         APIClientInterface
	vendorDetectors   map[string]VendorDetector // key: resource name
	previousDeviceIDs map[string]string
	mu                sync.RWMutex
	watcher           *fsnotify.Watcher
	stopCh            chan struct{}

	k8sClient client.Client
}

// NewDevicePluginDetector creates a new device plugin detector
func NewDevicePluginDetector(
	ctx context.Context,
	checkpointPath string,
	apiClient APIClientInterface,
	restConfig *rest.Config,
) (*DevicePluginDetector, error) {
	k8sClient, err := client.New(restConfig, client.Options{
		Scheme: scheme,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %w", err)
	}
	if checkpointPath == "" {
		checkpointPath = defaultKubeletCheckpointPath
	}

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		klog.Errorf("failed to create filesystem watcher for kubelet CDI checkpoint file: %v", err)
	}

	detector := &DevicePluginDetector{
		ctx:               ctx,
		checkpointPath:    checkpointPath,
		apiClient:         apiClient,
		vendorDetectors:   make(map[string]VendorDetector),
		previousDeviceIDs: make(map[string]string),
		watcher:           watcher,
		k8sClient:         k8sClient,
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
	resourceNamePrefixes := nvdpDetector.GetResourceNamePrefixes()
	for _, resourceNamePrefix := range resourceNamePrefixes {
		d.vendorDetectors[resourceNamePrefix] = nvdpDetector
	}

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
		_ = d.watcher.Close()
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
	d.mu.Lock()
	defer d.mu.Unlock()
	// Read checkpoint file
	checkpoint, err := d.readCheckpointFile()
	if err != nil {
		return fmt.Errorf("failed to read checkpoint file: %w", err)
	}

	// Extract registered device IDs (for comparison)
	allocated, registeredDeviceIDs := d.extractDeviceIDs(checkpoint)
	if d.grpcEndpointAvailable() {
		// Use kubelet pod-resources gRPC API as SSoT if available, otherwise fallback to checkpoint
		allocatedDevices, err := d.getAllocatedDevices()
		if err != nil {
			klog.Errorf("Failed to get allocated devices from gRPC: %v", err)
		} else {
			allocated = allocatedDevices
		}
	}

	// Determine added and removed devices
	previousDeviceIDs := make(map[string]string, len(d.previousDeviceIDs))
	maps.Copy(previousDeviceIDs, d.previousDeviceIDs)

	var addedDevices, removedDevices map[string]string

	if patchAllDevices {
		// Patch all devices: treat all allocated as added, and all registered but not allocated as removed
		addedDevices = allocated
		removedDevices = make(map[string]string)
		for deviceID := range registeredDeviceIDs {
			if resName, exists := allocated[deviceID]; !exists {
				removedDevices[deviceID] = resName
			}
		}
	} else {
		// Only process changes
		addedDevices = make(map[string]string)
		removedDevices = make(map[string]string)

		for deviceID, resName := range allocated {
			if _, exists := previousDeviceIDs[deviceID]; !exists {
				addedDevices[deviceID] = resName
			}
		}

		for deviceID, resName := range previousDeviceIDs {
			if _, exists := allocated[deviceID]; !exists {
				removedDevices[deviceID] = resName
			}
		}
	}

	// Process added devices using vendor-specific detectors
	hasError := false
	for deviceID, resName := range addedDevices {
		for _, detector := range d.vendorDetectors {
			resourceNamePrefixes := detector.GetResourceNamePrefixes()
			if slices.Contains(resourceNamePrefixes, resName) {
				usedBySystem, realDeviceID := detector.GetUsedBySystemAndRealDeviceID(deviceID, resName)
				klog.V(4).Infof(
					"Device added: %s, resource: %s, patching with usedBy: %s, realDeviceID: %s",
					deviceID,
					resName,
					usedBySystem,
					realDeviceID,
				)
				if err := d.patchGPUResource(realDeviceID, usedBySystem); err != nil {
					klog.Errorf("Failed to patch GPU resource for added device %s: %v", deviceID, err)
					hasError = true
				}
			}
		}
	}

	// Process removed devices
	for deviceID, resName := range removedDevices {
		for _, detector := range d.vendorDetectors {
			resourceNamePrefixes := detector.GetResourceNamePrefixes()
			if slices.Contains(resourceNamePrefixes, resName) {
				klog.V(4).Infof(
					"Device plugin allocated container removed: %s, resource: %s, patching usedBy field to tensor fusion",
					deviceID,
					resName,
				)
				if err := d.patchGPUResource(deviceID, string(tfv1.UsedByTensorFusion)); err != nil {
					klog.Errorf("Failed to patch GPU resource usedBy field to tensor fusion for removed device %s: %v", deviceID, err)
					hasError = true
				}
			}
		}
	}

	// Update previous state only if no errors occurred
	if !hasError {
		d.previousDeviceIDs = allocated
	}
	return nil
}

// patchGPUResource patches a GPU resource with the specified usedBy value
func (d *DevicePluginDetector) patchGPUResource(deviceID, usedBySystem string) error {
	const maxRetries = 3

	for i := range maxRetries {
		// Get current GPU resource
		gpu, err := d.apiClient.GetGPU(deviceID)
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
		if err := d.apiClient.UpdateGPUStatus(gpu); err != nil {
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
func (d *DevicePluginDetector) extractDeviceIDs(
	checkpoint *KubeletCheckpoint,
) (allocated, registered map[string]string) {
	allocated = make(map[string]string)
	registered = make(map[string]string)

	// Extract allocated devices from pod device entries
	if checkpoint.Data.PodDeviceEntries != nil {
		for _, entry := range checkpoint.Data.PodDeviceEntries {
			if strings.HasPrefix(entry.ResourceName, constants.PodIndexAnnotation) {
				continue
			}
			for _, deviceList := range entry.DeviceIDs {
				for _, deviceID := range deviceList {
					allocated[strings.ToLower(deviceID)] = entry.ResourceName
				}
			}
		}
	}

	// Extract registered devices
	if checkpoint.Data.RegisteredDevices != nil {
		for resourceName, deviceIDs := range checkpoint.Data.RegisteredDevices {
			if strings.HasPrefix(resourceName, constants.PodIndexAnnotation) {
				continue
			}
			for _, deviceID := range deviceIDs {
				registered[strings.ToLower(deviceID)] = resourceName
			}
		}
	}

	return allocated, registered
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

// grpcEndpointAvailable checks if the kubelet pod-resources gRPC socket is accessible
func (d *DevicePluginDetector) grpcEndpointAvailable() bool {
	socketPath := defaultKubeletPodResourcesSocket
	if _, err := os.Stat(socketPath); err != nil {
		return false
	}
	return true
}

// getAllocatedDevices queries the kubelet pod-resources gRPC API to get allocated device IDs
// Returns a map of lowercase device IDs that are currently allocated to pods
func (d *DevicePluginDetector) getAllocatedDevices() (map[string]string, error) {
	conn, err := d.dialPodResourcesSocket(defaultKubeletPodResourcesSocket, 5*time.Second)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to pod-resources socket: %w", err)
	}
	defer func() {
		if err := conn.Close(); err != nil {
			klog.Errorf("failed to close pod-resources socket: %v", err)
		}
	}()
	// Note: pod-resources API types are not exported from k8s.io/kubernetes and not in vendor.
	// Using gRPC Invoke directly with minimal types matching the API structure.
	ctx, cancel := context.WithTimeout(d.ctx, 10*time.Second)
	defer cancel()

	var resp podResourcesResponse
	if err := conn.Invoke(ctx, "/v1.PodResourcesLister/List", &emptypb.Empty{}, &resp); err != nil {
		return nil, fmt.Errorf("failed to list pod resources: %w", err)
	}

	allocatedDevices := make(map[string]string)

	for _, podResource := range resp.PodResources {
		for _, container := range podResource.Containers {
			for _, device := range container.Devices {
				for _, deviceID := range device.DeviceIds {
					allocatedDevices[strings.ToLower(deviceID)] = device.ResourceName
				}
			}
		}
	}

	klog.V(4).Infof("Retrieved %d allocated devices from pod-resources API", len(allocatedDevices))
	return allocatedDevices, nil
}

// podResourcesResponse matches the pod-resources API response structure
// These types are manually defined because k8s.io/kubernetes/pkg/kubelet/apis/podresources/v1
// is not exported and not available in vendor directory
type podResourcesResponse struct {
	PodResources []*podResource `json:"pod_resources"`
}

func (m *podResourcesResponse) Reset()         { *m = podResourcesResponse{} }
func (m *podResourcesResponse) String() string { return "podResourcesResponse" }
func (*podResourcesResponse) ProtoMessage()    {}

type podResource struct {
	Name       string       `json:"name"`
	Namespace  string       `json:"namespace"`
	Containers []*container `json:"containers"`
}

func (m *podResource) Reset()         { *m = podResource{} }
func (m *podResource) String() string { return "podResource" }
func (*podResource) ProtoMessage()    {}

type container struct {
	Name    string    `json:"name"`
	Devices []*device `json:"devices"`
}

func (m *container) Reset()         { *m = container{} }
func (m *container) String() string { return "container" }
func (*container) ProtoMessage()    {}

type device struct {
	ResourceName string   `json:"resource_name"`
	DeviceIds    []string `json:"device_ids"`
}

func (m *device) Reset()         { *m = device{} }
func (m *device) String() string { return "device" }
func (*device) ProtoMessage()    {}

// dialPodResourcesSocket establishes a gRPC connection to the kubelet pod-resources socket
func (d *DevicePluginDetector) dialPodResourcesSocket(
	socketPath string,
	timeout time.Duration,
) (*grpc.ClientConn, error) {
	target := "unix://" + socketPath
	conn, err := grpc.NewClient(target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, addr string) (net.Conn, error) {
			socketPath := addr
			if len(addr) > 7 && addr[:7] == "unix://" {
				socketPath = addr[7:]
			}
			return net.DialTimeout("unix", socketPath, timeout)
		}),
	)
	return conn, err
}
