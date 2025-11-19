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
	kubeletClient    *KubeletClient

	server       *grpc.Server
	socketPath   string
	resourceName string

	mu       sync.RWMutex
	devices  []*pluginapi.Device
	stopCh   chan struct{}
	updateCh chan []*pluginapi.Device
}

// NewDevicePlugin creates a new device plugin instance
func NewDevicePlugin(ctx context.Context, deviceController framework.DeviceController, kubeletClient *KubeletClient) *DevicePlugin {
	return &DevicePlugin{
		ctx:              ctx,
		deviceController: deviceController,
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
	if err := os.Remove(dp.socketPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove existing socket: %w", err)
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
	conn.Close()

	// Register with kubelet
	if err := dp.register(); err != nil {
		return fmt.Errorf("failed to register with kubelet: %w", err)
	}

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
	conn, err := dp.dial(filepath.Join(DevicePluginPath, KubeletSocket), 5*time.Second)
	if err != nil {
		return fmt.Errorf("failed to dial kubelet: %w", err)
	}
	defer conn.Close()

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
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	conn, err := grpc.DialContext(ctx, unixSocketPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
		grpc.WithContextDialer(func(ctx context.Context, addr string) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
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

// updateDeviceList updates the list of available devices
func (dp *DevicePlugin) updateDeviceList() {
	devices, err := dp.deviceController.ListDevices(dp.ctx)
	if err != nil {
		klog.Errorf("Failed to list devices: %v", err)
		return
	}

	dp.mu.Lock()
	defer dp.mu.Unlock()

	pluginDevices := make([]*pluginapi.Device, 0, len(devices))
	for _, device := range devices {
		pluginDevices = append(pluginDevices, &pluginapi.Device{
			ID:     device.UUID,
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
func (dp *DevicePlugin) Allocate(ctx context.Context, req *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	klog.Infof("Allocate called with %d container requests", len(req.ContainerRequests))

	responses := make([]*pluginapi.ContainerAllocateResponse, 0, len(req.ContainerRequests))

	for _, containerReq := range req.ContainerRequests {
		// Extract pod UID and namespace from environment variables or annotations
		// The kubelet passes these in the container request
		podUID := ""
		podName := ""
		namespace := ""

		// Get worker info from kubelet client
		workerInfo, err := dp.kubeletClient.GetWorkerInfoForAllocation(ctx, containerReq)
		if err != nil {
			klog.Errorf("Failed to get worker info: %v", err)
			return nil, fmt.Errorf("failed to get worker info: %w", err)
		}

		if workerInfo == nil {
			return nil, fmt.Errorf("worker info not found for allocation request")
		}

		podUID = workerInfo.PodUID
		podName = workerInfo.PodName
		namespace = workerInfo.Namespace

		// Compose allocation request
		deviceUUIDs := make([]string, 0, len(containerReq.DevicesIds))
		for _, deviceID := range containerReq.DevicesIds {
			deviceUUIDs = append(deviceUUIDs, deviceID)
		}

		allocReq := &api.DeviceAllocateRequest{
			WorkerUID:         podUID,
			DeviceUUIDs:       deviceUUIDs,
			IsolationMode:     workerInfo.IsolationMode,
			MemoryLimitBytes:  workerInfo.MemoryLimitBytes,
			ComputeLimitUnits: workerInfo.ComputeLimitUnits,
			TemplateID:        workerInfo.TemplateID,
		}

		// Call device controller to allocate
		allocResp, err := dp.deviceController.AllocateDevice(allocReq)
		if err != nil {
			return nil, fmt.Errorf("failed to allocate device: %w", err)
		}

		if !allocResp.Success {
			return nil, fmt.Errorf("device allocation failed: %s", allocResp.ErrMsg)
		}

		// Build container response
		containerResp := &pluginapi.ContainerAllocateResponse{
			Envs:    allocResp.EnvVars,
			Mounts:  make([]*pluginapi.Mount, 0),
			Devices: make([]*pluginapi.DeviceSpec, 0),
		}

		// Add device nodes
		for _, deviceNode := range allocResp.DeviceNodes {
			containerResp.Devices = append(containerResp.Devices, &pluginapi.DeviceSpec{
				ContainerPath: deviceNode,
				HostPath:      deviceNode,
				Permissions:   "rw",
			})
		}

		// Add mounts
		for hostPath, containerPath := range allocResp.Mounts {
			containerResp.Mounts = append(containerResp.Mounts, &pluginapi.Mount{
				ContainerPath: containerPath,
				HostPath:      hostPath,
				ReadOnly:      false,
			})
		}

		// Add annotations as environment variables
		for key, value := range allocResp.Annotations {
			containerResp.Envs[key] = value
		}

		// Store allocation info in kubelet client
		allocation := &api.DeviceAllocation{
			DeviceUUID:    deviceUUIDs[0], // Assuming single device for now
			PodUID:        podUID,
			PodName:       podName,
			Namespace:     namespace,
			IsolationMode: workerInfo.IsolationMode,
			TemplateID:    workerInfo.TemplateID,
			MemoryLimit:   workerInfo.MemoryLimitBytes,
			ComputeLimit:  workerInfo.ComputeLimitUnits,
			WorkerID:      podUID,
			AllocatedAt:   time.Now(),
		}

		if err := dp.kubeletClient.StoreAllocation(podUID, allocation); err != nil {
			klog.Warningf("Failed to store allocation: %v", err)
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
