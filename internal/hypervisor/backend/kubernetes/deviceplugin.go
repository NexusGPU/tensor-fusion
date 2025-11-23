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

package kubernetes

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const (
	// DevicePluginPath is the path where device plugins should register
	DevicePluginPath = "/var/lib/kubelet/device-plugins"
	// KubeletSocket is the kubelet registration socket
	KubeletSocket = "kubelet.sock"
	// ResourceName is the resource name advertised to kubelet
	ResourceName = "tensor-fusion.ai/index"
	// DevicePluginEndpoint is the endpoint name for this device plugin
	DevicePluginEndpoint = "tensor-fusion-index.sock"
)

// DevicePlugin implements the Kubernetes device plugin interface
type DevicePlugin struct {
	pluginapi.UnimplementedDevicePluginServer

	ctx              context.Context
	deviceController framework.DeviceController
	workerController framework.WorkerController
	kubeletClient    *PodCacheManager

	server       *grpc.Server
	socketPath   string
	resourceName string

	mu       sync.RWMutex
	devices  []*pluginapi.Device
	stopCh   chan struct{}
	updateCh chan []*pluginapi.Device
}

// NewDevicePlugin creates a new device plugin instance
func NewDevicePlugin(ctx context.Context, deviceController framework.DeviceController, workerController framework.WorkerController, kubeletClient *PodCacheManager) *DevicePlugin {
	return &DevicePlugin{
		ctx:              ctx,
		deviceController: deviceController,
		workerController: workerController,
		kubeletClient:    kubeletClient,
		socketPath:       filepath.Join(DevicePluginPath, DevicePluginEndpoint),
		resourceName:     ResourceName,
		stopCh:           make(chan struct{}),
		updateCh:         make(chan []*pluginapi.Device, 1),
	}
}

// Start starts the device plugin gRPC server and registers with kubelet
func (dp *DevicePlugin) Start() error {
	// Clean up any existing socket
	// Check if file exists first to avoid permission errors on non-existent files
	if _, err := os.Stat(dp.socketPath); err == nil {
		// File exists, try to remove it
		if err := os.Remove(dp.socketPath); err != nil {
			return fmt.Errorf("failed to remove existing socket: %w", err)
		}
	} else if !os.IsNotExist(err) {
		// Some other error checking file existence (e.g., permission denied on parent directory)
		// Log warning but continue - net.Listen will handle it
		klog.Warningf("Could not check socket file existence: %v", err)
	}

	// Create directory if it doesn't exist
	if err := os.MkdirAll(DevicePluginPath, 0750); err != nil {
		return fmt.Errorf("failed to create device plugin directory: %w", err)
	}

	// Create Unix socket listener
	listener, err := net.Listen("unix", dp.socketPath)
	if err != nil {
		return fmt.Errorf("failed to create listener: %w", err)
	}

	// Create gRPC server
	dp.server = grpc.NewServer()
	pluginapi.RegisterDevicePluginServer(dp.server, dp)

	// Start gRPC server
	go func() {
		klog.Infof("Starting device plugin gRPC server on %s", dp.socketPath)
		if err := dp.server.Serve(listener); err != nil {
			klog.Errorf("Device plugin gRPC server error: %v", err)
		}
	}()

	// Wait for server to be ready
	conn, err := dp.dial(dp.socketPath, 5*time.Second)
	if err != nil {
		return fmt.Errorf("failed to dial device plugin socket: %w", err)
	}
	_ = conn.Close()

	// Register with kubelet
	if err := dp.register(); err != nil {
		return fmt.Errorf("failed to register with kubelet: %w", err)
	}

	// Initialize device list with dummy index devices (1-512)
	dp.updateDeviceList()

	// Start device monitoring
	go dp.monitorDevices()

	return nil
}

// Stop stops the device plugin
func (dp *DevicePlugin) Stop() error {
	close(dp.stopCh)
	if dp.server != nil {
		dp.server.Stop()
	}
	return os.Remove(dp.socketPath)
}

// register registers the device plugin with kubelet
func (dp *DevicePlugin) register() error {
	kubeletSocketPath := filepath.Join(DevicePluginPath, KubeletSocket)

	// Check if kubelet socket exists
	if _, err := os.Stat(kubeletSocketPath); os.IsNotExist(err) {
		return fmt.Errorf("kubelet socket does not exist at %s (kubelet may not be running or device plugin support not enabled)", kubeletSocketPath)
	} else if err != nil {
		return fmt.Errorf("failed to check kubelet socket: %w", err)
	}

	conn, err := dp.dial(kubeletSocketPath, 5*time.Second)
	if err != nil {
		return fmt.Errorf("failed to dial kubelet: %w", err)
	}
	defer func() {
		_ = conn.Close()
	}()

	client := pluginapi.NewRegistrationClient(conn)
	req := &pluginapi.RegisterRequest{
		Version:      pluginapi.Version,
		Endpoint:     DevicePluginEndpoint,
		ResourceName: dp.resourceName,
		Options: &pluginapi.DevicePluginOptions{
			PreStartRequired:                false,
			GetPreferredAllocationAvailable: false,
		},
	}

	_, err = client.Register(context.Background(), req)
	if err != nil {
		return fmt.Errorf("failed to register: %w", err)
	}

	klog.Infof("Successfully registered device plugin with kubelet: %s", dp.resourceName)
	return nil
}

// dial establishes a connection to a Unix socket
func (dp *DevicePlugin) dial(unixSocketPath string, timeout time.Duration) (*grpc.ClientConn, error) {
	// Use unix:// prefix for gRPC to recognize it as a Unix socket
	// The dialer will receive the full address, so we need to strip the prefix
	target := "unix://" + unixSocketPath
	conn, err := grpc.NewClient(target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, addr string) (net.Conn, error) {
			// Strip unix:// prefix to get the actual socket path
			socketPath := addr
			if len(addr) > 7 && addr[:7] == "unix://" {
				socketPath = addr[7:]
			}
			return net.DialTimeout("unix", socketPath, timeout)
		}),
	)
	return conn, err
}

// monitorDevices periodically updates the device list
func (dp *DevicePlugin) monitorDevices() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-dp.ctx.Done():
			return
		case <-dp.stopCh:
			return
		case <-ticker.C:
			dp.updateDeviceList()
		case devices := <-dp.updateCh:
			dp.mu.Lock()
			dp.devices = devices
			dp.mu.Unlock()
		}
	}
}

// updateDeviceList updates the list of available dummy index devices
// This device plugin registers tensor-fusion.ai/index resource, not real GPU devices.
// We advertise 512 dummy devices (indices 1-512) for pod identification.
// Real GPU devices are allocated by scheduler and set in pod annotations.
func (dp *DevicePlugin) updateDeviceList() {
	dp.mu.Lock()
	defer dp.mu.Unlock()

	// Advertise 512 dummy index devices (1-512) for pod identification
	// These are NOT real GPU devices - they're just used to match pods by index
	pluginDevices := make([]*pluginapi.Device, 0, 512)
	for i := 1; i <= 512; i++ {
		pluginDevices = append(pluginDevices, &pluginapi.Device{
			ID:     fmt.Sprintf("%d", i), // Index as device ID
			Health: pluginapi.Healthy,
		})
	}

	dp.devices = pluginDevices
	select {
	case dp.updateCh <- pluginDevices:
	default:
	}
}

// GetDevicePluginOptions returns options for the device plugin
func (dp *DevicePlugin) GetDevicePluginOptions(ctx context.Context, req *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	return &pluginapi.DevicePluginOptions{
		PreStartRequired:                false,
		GetPreferredAllocationAvailable: false,
	}, nil
}

// ListAndWatch streams device list and health updates
func (dp *DevicePlugin) ListAndWatch(req *pluginapi.Empty, stream pluginapi.DevicePlugin_ListAndWatchServer) error {
	klog.Info("ListAndWatch called")

	// Send initial device list
	dp.updateDeviceList()
	dp.mu.RLock()
	devices := make([]*pluginapi.Device, len(dp.devices))
	copy(devices, dp.devices)
	dp.mu.RUnlock()

	if err := stream.Send(&pluginapi.ListAndWatchResponse{Devices: devices}); err != nil {
		return fmt.Errorf("failed to send device list: %w", err)
	}

	// Watch for updates
	for {
		select {
		case <-dp.ctx.Done():
			return nil
		case <-dp.stopCh:
			return nil
		case devices := <-dp.updateCh:
			if err := stream.Send(&pluginapi.ListAndWatchResponse{Devices: devices}); err != nil {
				return fmt.Errorf("failed to send device update: %w", err)
			}
		}
	}
}

// Allocate handles device allocation requests from kubelet
// IMPORTANT: This device plugin registers tensor-fusion.ai/index as a dummy resource.
// The pod index (1-512) is used to identify which pod is requesting allocation.
// The actual GPU device UUIDs are already set by the centralized scheduler in pod annotations:
//   - tensor-fusion.ai/gpu-ids: comma-separated GPU UUIDs (for all isolation modes)
//   - tensor-fusion.ai/partition: partition template ID (only for partitioned isolation mode)
//
// The len(req.ContainerRequests) is just the number of containers in the pod requesting
// tensor-fusion.ai/index resource - it's NOT the pod index. The pod index comes from
// DevicesIds[0] which contains the index value from resource limits.
//
// We do NOT allocate the fake tensor-fusion.ai/index device - it's only used for pod identification.
// CDIDevices in the response is kept empty to prevent kubelet from allocating the dummy device.
func (dp *DevicePlugin) Allocate(ctx context.Context, req *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	// len(req.ContainerRequests) identifies how many containers in the pod are requesting
	// tensor-fusion.ai/index resource - this is for logging/identification only
	klog.Infof("Allocate called with %d container requests (pod may have multiple containers)", len(req.ContainerRequests))

	responses := make([]*pluginapi.ContainerAllocateResponse, 0, len(req.ContainerRequests))

	for containerIdx, containerReq := range req.ContainerRequests {
		// Extract pod index from DevicesIds - this contains the index value (1-512) from resource limits
		// Resource limit: tensor-fusion.ai/index: 3 -> DevicesIds: ["3"]
		// This is the actual pod index used to match the pod in the pod cache
		podIndex := len(containerReq.DevicesIds)
		if podIndex == 0 {
			return nil, fmt.Errorf("container request %d has no DevicesIds (expected pod index value 1-512)", containerIdx)
		}

		if podIndex < constants.IndexRangeStart || podIndex > constants.IndexRangeEnd {
			return nil, fmt.Errorf("container request %d has index out of range: %d (expected 1-512)", containerIdx, podIndex)
		}

		klog.V(4).Infof("Processing allocation for container index %d, pod index %d (from DevicesIds)", containerIdx, podIndex)

		// Get worker info from kubelet client using pod index
		// This will automatically check for duplicate indices and fail fast if found
		workerInfo, err := dp.kubeletClient.GetWorkerInfoForAllocationByIndex(ctx, podIndex)
		if err != nil {
			klog.Errorf("Failed to get worker info for pod index %d: %v", podIndex, err)
			return nil, fmt.Errorf("failed to get worker info for pod index %d: %w", podIndex, err)
		}

		if workerInfo == nil {
			return nil, fmt.Errorf("worker info not found for pod index %d", podIndex)
		}

		// Device UUIDs are already set by scheduler in annotations, not from DevicesIds
		deviceUUIDs := workerInfo.AllocatedDevices
		if len(deviceUUIDs) == 0 {
			return nil, fmt.Errorf("no device UUIDs found in pod annotations for pod %s/%s", workerInfo.Namespace, workerInfo.PodName)
		}

		// Call worker controller to allocate
		allocResp, err := dp.workerController.AllocateWorker(workerInfo)
		if err != nil {
			return nil, fmt.Errorf("failed to allocate device: %w", err)
		}

		// WorkerAllocation doesn't need Success/ErrMsg check - if no error, allocation succeeded

		// Build container response - create minimal response since allocation details are tracked separately
		// IMPORTANT: CdiDevices must be empty to prevent dummy tensor-fusion.ai/index device
		// from being allocated by kubelet
		containerResp := &pluginapi.ContainerAllocateResponse{
			Envs:       make(map[string]string),
			Mounts:     []*pluginapi.Mount{},
			Devices:    []*pluginapi.DeviceSpec{},
			CdiDevices: []*pluginapi.CDIDevice{}, // Empty to prevent dummy device allocation
		}

		// Add basic environment variables for worker info
		if allocResp.WorkerInfo != nil {
			containerResp.Envs["TF_WORKER_UID"] = allocResp.WorkerInfo.WorkerUID
			containerResp.Envs["TF_POD_UID"] = allocResp.WorkerInfo.PodUID
			
			// Add device UUIDs as environment variable
			if len(allocResp.DeviceInfos) > 0 {
				deviceUUIDs := make([]string, 0, len(allocResp.DeviceInfos))
				for _, device := range allocResp.DeviceInfos {
					deviceUUIDs = append(deviceUUIDs, device.UUID)
				}
				containerResp.Envs["TF_DEVICE_UUIDS"] = fmt.Sprintf("%v", deviceUUIDs)
			}
		}

		// Get pod to extract labels and annotations
		pod := dp.kubeletClient.GetPodByUID(workerInfo.PodUID)
		labels := make(map[string]string)
		annotations := make(map[string]string)
		if pod != nil {
			if pod.Labels != nil {
				labels = pod.Labels
			}
			if pod.Annotations != nil {
				annotations = pod.Annotations
			}
		}

		// Update allocation in device controller with labels and annotations
		// Use type assertion to access the concrete implementation
		if deviceCtrl, ok := dp.deviceController.(interface {
			UpdateAllocationLabelsAndAnnotations(workerUID string, labels, annotations map[string]string)
		}); ok {
			deviceCtrl.UpdateAllocationLabelsAndAnnotations(workerInfo.PodUID, labels, annotations)
		}

		// Store allocation info in kubelet client (for backward compatibility)
		workerDetail := &api.WorkerDetail{
			WorkerUID:  workerInfo.WorkerUID,
			Allocation: allocResp,
		}

		if err := dp.kubeletClient.StoreAllocation(workerInfo.PodUID, workerDetail); err != nil {
			klog.Warningf("Failed to store allocation: %v", err)
		}

		// Remove PodIndexAnnotation after successful allocation to release the index
		// This prevents the index from being matched to this pod in future allocation cycles
		if err := dp.kubeletClient.RemovePodIndexAnnotation(ctx, workerInfo.PodUID, workerInfo.Namespace, workerInfo.PodName); err != nil {
			klog.Warningf("Failed to remove pod index annotation for pod %s/%s: %v", workerInfo.Namespace, workerInfo.PodName, err)
			// Don't fail allocation if annotation removal fails
		}

		responses = append(responses, containerResp)
	}

	return &pluginapi.AllocateResponse{
		ContainerResponses: responses,
	}, nil
}

// PreStartContainer is called before container start (optional)
func (dp *DevicePlugin) PreStartContainer(ctx context.Context, req *pluginapi.PreStartContainerRequest) (*pluginapi.PreStartContainerResponse, error) {
	return &pluginapi.PreStartContainerResponse{}, nil
}

// GetPreferredAllocation returns preferred device allocation (optional)
func (dp *DevicePlugin) GetPreferredAllocation(ctx context.Context, req *pluginapi.PreferredAllocationRequest) (*pluginapi.PreferredAllocationResponse, error) {
	return &pluginapi.PreferredAllocationResponse{
		ContainerResponses: []*pluginapi.ContainerPreferredAllocationResponse{},
	}, nil
}
