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
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/samber/lo"
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
	// DevicePluginEndpoint is the endpoint name for this device plugin
	DevicePluginEndpoint = "tensor-fusion-index-%d.sock"
)

// DevicePlugin implements the Kubernetes device plugin interface
type DevicePlugin struct {
	pluginapi.UnimplementedDevicePluginServer

	ctx                  context.Context
	deviceController     framework.DeviceController
	allocationController framework.WorkerAllocationController
	kubeletClient        *PodCacheManager

	server            *grpc.Server
	socketPath        string
	resourceNameIndex int
}

// NewDevicePlugins creates a new device plugin instance
func NewDevicePlugins(ctx context.Context, deviceController framework.DeviceController, allocationController framework.WorkerAllocationController, kubeletClient *PodCacheManager) []*DevicePlugin {
	devicePlugins := make([]*DevicePlugin, constants.IndexKeyLength)
	for i := range constants.IndexKeyLength {
		devicePlugins[i] = &DevicePlugin{
			ctx:                  ctx,
			deviceController:     deviceController,
			allocationController: allocationController,
			kubeletClient:        kubeletClient,
			socketPath:           filepath.Join(DevicePluginPath, fmt.Sprintf(DevicePluginEndpoint, i)),
			resourceNameIndex:    i,
		}
	}
	return devicePlugins
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
		klog.V(4).Infof("Starting device plugin gRPC server on %s", dp.socketPath)
		if err := dp.server.Serve(listener); err != nil {
			klog.Errorf("Device plugin gRPC server error: %v", err)
		}
	}()

	// Wait for server to be ready
	conn, err := dp.dial(dp.socketPath, 3*time.Second)
	if err != nil {
		return fmt.Errorf("failed to dial device plugin socket: %w", err)
	}
	_ = conn.Close()

	// Register with kubelet
	if err := dp.register(); err != nil {
		return fmt.Errorf("failed to register with kubelet: %w", err)
	}
	return nil
}

// Stop stops the device plugin
func (dp *DevicePlugin) Stop() error {
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
		Endpoint:     fmt.Sprintf(DevicePluginEndpoint, dp.resourceNameIndex),
		ResourceName: fmt.Sprintf("%s%s%x", constants.PodIndexAnnotation, constants.PodIndexDelimiter, dp.resourceNameIndex),
		Options: &pluginapi.DevicePluginOptions{
			PreStartRequired:                false,
			GetPreferredAllocationAvailable: false,
		},
	}

	_, err = client.Register(context.Background(), req)
	if err != nil {
		return fmt.Errorf("failed to register: %w", err)
	}

	klog.V(4).Infof("Successfully registered device plugin with kubelet: %s", fmt.Sprintf("%s%s%x", constants.PodIndexAnnotation, constants.PodIndexDelimiter, dp.resourceNameIndex))
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

// GetDevicePluginOptions returns options for the device plugin
func (dp *DevicePlugin) GetDevicePluginOptions(ctx context.Context, req *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	return &pluginapi.DevicePluginOptions{
		PreStartRequired:                false,
		GetPreferredAllocationAvailable: false,
	}, nil
}

// ListAndWatch streams device list and health updates
func (dp *DevicePlugin) ListAndWatch(req *pluginapi.Empty, stream pluginapi.DevicePlugin_ListAndWatchServer) error {
	klog.V(4).Infof("ListAndWatch called for device plugin index %d", dp.resourceNameIndex)

	// Build initial device list
	total := constants.IndexModLength * (constants.IndexModLength + 1) / 2
	devices := make([]*pluginapi.Device, total)
	for i := range total {
		devices[i] = &pluginapi.Device{
			ID:     fmt.Sprintf("%d-%d", dp.resourceNameIndex, i+1),
			Health: pluginapi.Healthy,
		}
	}

	// Send initial device list
	if err := stream.Send(&pluginapi.ListAndWatchResponse{Devices: devices}); err != nil {
		return fmt.Errorf("failed to send initial device list: %w", err)
	}

	// Keep the stream alive by blocking until context is cancelled or connection is closed
	// This is required by Kubernetes device plugin API - the stream must remain open
	// to allow kubelet to receive device health updates
	<-stream.Context().Done()

	// Check if context was cancelled due to error or normal shutdown
	if err := stream.Context().Err(); err != nil {
		klog.Infof("ListAndWatch stream ended for device plugin index %d: %v", dp.resourceNameIndex, err)
		return err
	}

	klog.Infof("ListAndWatch stream closed normally for device plugin index %d", dp.resourceNameIndex)
	return nil
}

// Allocate handles device allocation requests from kubelet
func (dp *DevicePlugin) Allocate(ctx context.Context, req *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	responses := make([]*pluginapi.ContainerAllocateResponse, 0, len(req.ContainerRequests))
	klog.Infof("Allocate called for device plugin index %d, container requests: %d", dp.resourceNameIndex, len(req.ContainerRequests))

	for containerIdx, containerReq := range req.ContainerRequests {
		podIndex := len(containerReq.DevicesIds)
		if podIndex <= 0 || podIndex > constants.IndexModLength {
			return nil, fmt.Errorf("container request %d dummy device requests is not valid: (expected index value 1-%d)", containerIdx, constants.IndexModLength)
		}

		podIndexFull := podIndex + (dp.resourceNameIndex * constants.IndexModLength)

		klog.V(4).Infof("Processing allocation for container index %d, pod index %d (from DevicesIds)", containerIdx, podIndexFull)
		// Get worker info from kubelet client using pod index
		// This will automatically check for duplicate indices and fail fast if found
		workerInfo, err := dp.kubeletClient.GetWorkerInfoForAllocationByIndex(podIndexFull)
		if err != nil {
			klog.Errorf("Failed to get worker info for pod index %d: %v", podIndexFull, err)
			return nil, fmt.Errorf("failed to get worker info for pod index %d: %w", podIndexFull, err)
		}
		if workerInfo == nil {
			return nil, fmt.Errorf("worker info not found for pod index %d", podIndexFull)
		}
		// Call allocation controller to allocate
		allocResp, err := dp.allocationController.AllocateWorkerDevices(workerInfo)
		if err != nil {
			return nil, fmt.Errorf("failed to allocate devices for worker %s %s: %w", workerInfo.WorkerName, workerInfo.WorkerUID, err)
		}

		containerResp := &pluginapi.ContainerAllocateResponse{
			Envs: allocResp.Envs,
			Mounts: lo.Map(allocResp.Mounts, func(mount *api.Mount, _ int) *pluginapi.Mount {
				return &pluginapi.Mount{
					ContainerPath: mount.GuestPath,
					HostPath:      mount.HostPath,
				}
			}),
			Devices: lo.Map(allocResp.Devices, func(device *api.DeviceSpec, _ int) *pluginapi.DeviceSpec {
				return &pluginapi.DeviceSpec{
					ContainerPath: device.GuestPath,
					HostPath:      device.HostPath,
					Permissions:   device.Permissions,
				}
			}),
			CdiDevices: []*pluginapi.CDIDevice{},
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
