package kubernetes

import (
	"context"
	"fmt"
	"os"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/backend/kubernetes/external_dp"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

const watcherName = "backend_watcher"

type KubeletBackend struct {
	ctx context.Context

	deviceController framework.DeviceController
	workerController framework.WorkerController

	apiClient      *APIClient
	podCacher      *PodCacheManager
	devicePlugins  []*DevicePlugin
	deviceDetector *external_dp.DevicePluginDetector

	workers       map[string]*api.WorkerInfo
	workerChanged chan *api.WorkerInfo
}

var k8sBackend framework.Backend = &KubeletBackend{}

func NewKubeletBackend(ctx context.Context, deviceController framework.DeviceController, workerController framework.WorkerController, restConfig *rest.Config) (*KubeletBackend, error) {
	// Get node name from environment or config
	nodeName := os.Getenv(constants.HypervisorGPUNodeNameEnv)
	if nodeName == "" {
		return nil, fmt.Errorf("node name env var 'GPU_NODE_NAME' for this hypervisor not set")
	}

	// Create kubelet client
	podCacher, err := NewPodCacheManager(ctx, restConfig, nodeName)
	if err != nil {
		return nil, err
	}

	// Create API server for device detector
	apiClient, err := NewAPIClientFromConfig(ctx, restConfig)
	if err != nil {
		return nil, err
	}

	// Create device plugin detector
	var deviceDetector *external_dp.DevicePluginDetector
	if os.Getenv(constants.HypervisorDetectUsedGPUEnv) == constants.TrueStringValue {
		checkpointPath := os.Getenv(constants.HypervisorKubeletCheckpointPathEnv)
		// Create adapter for kubelet client to match interface
		deviceDetector, err = external_dp.NewDevicePluginDetector(ctx, checkpointPath, apiClient, restConfig)
		if err != nil {
			return nil, err
		}
	}

	return &KubeletBackend{
		ctx:              ctx,
		deviceController: deviceController,
		workerController: workerController,
		podCacher:        podCacher,
		deviceDetector:   deviceDetector,
		apiClient:        apiClient,
		workerChanged:    make(chan *api.WorkerInfo),
	}, nil
}

func (b *KubeletBackend) Start() error {
	// Start kubelet client to watch pods
	b.podCacher.RegisterWorkerInfoSubscriber(watcherName, b.workerChanged)
	if err := b.podCacher.Start(); err != nil {
		return err
	}
	klog.Info("Kubelet client started, watching pods")

	// Create and start device plugin
	b.devicePlugins = NewDevicePlugins(b.ctx, b.deviceController, b.workerController, b.podCacher)
	for _, devicePlugin := range b.devicePlugins {
		if err := devicePlugin.Start(); err != nil {
			return err
		}
	}
	klog.Infof("Device plugins started and registered with kubelet")

	// Start device plugin detector to watch external device plugins
	if b.deviceDetector != nil {
		if err := b.deviceDetector.Start(); err != nil {
			klog.Warningf("Failed to start device plugin detector: %v", err)
		} else {
			klog.Info("Device plugin detector started")
		}
	}
	return nil
}

func (b *KubeletBackend) Stop() error {
	if b.devicePlugins != nil {
		for i, devicePlugin := range b.devicePlugins {
			if err := devicePlugin.Stop(); err != nil {
				klog.Errorf("Failed to stop device plugin %d: %v", i, err)
			}
		}
	}

	if b.deviceDetector != nil {
		b.deviceDetector.Stop()
	}

	if b.podCacher != nil {
		b.podCacher.UnregisterWorkerInfoSubscriber(watcherName)
		b.podCacher.Stop()
	}

	return nil
}

// Returns data channel and stop channel
func (b *KubeletBackend) ListAndWatchWorkers() (initList []*api.WorkerInfo, changedWorker chan *api.WorkerInfo, err error) {
	// Initialize channels if not already created

	return b.workers, dataChan, nil
}

func (b *KubeletBackend) StartWorker(workerUID string) error {
	klog.Warningf("StartWorker not implemented, should be managed by operator")
	return nil
}

func (b *KubeletBackend) StopWorker(workerUID string) error {
	klog.Warningf("StopWorker not implemented, should be managed by operator")
	return nil
}

func (b *KubeletBackend) GetProcessMappingInfo(workerUID string, hostPID uint32) (*framework.ProcessMappingInfo, error) {
	return GetWorkerInfoFromHostPID(hostPID, workerUID)
}
