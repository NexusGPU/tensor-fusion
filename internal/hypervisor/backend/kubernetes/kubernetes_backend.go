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
	podCacher        *PodCacheManager
	devicePlugins    []*DevicePlugin
	deviceDetector   *external_dp.DevicePluginDetector

	workerChanged chan<- *api.WorkerInfo
	workerCh      chan []*api.WorkerInfo
	workerStopCh  chan struct{}
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
		workerChanged:    make(chan<- *api.WorkerInfo),
	}, nil
}

func (b *KubeletBackend) Start() error {
	// Start kubelet client to watch pods
	if err := b.podCacher.Start(); err != nil {
		return err
	}
	b.podCacher.RegisterWorkerInfoSubscriber(watcherName, b.workerChanged)
	klog.Info("Kubelet client started, watching pods")

	// Create and start device plugin
	b.devicePlugins = NewDevicePlugins(b.ctx, b.deviceController, b.workerController, b.podCacher)
	for _, devicePlugin := range b.devicePlugins {
		if err := devicePlugin.Start(); err != nil {
			return err
		}
		klog.Infof("Device plugin %d started and registered with kubelet", devicePlugin.resourceNameIndex)
	}

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
	// Close worker watch stop channel (safe to close even if nil)
	if b.workerStopCh != nil {
		select {
		case <-b.workerStopCh:
			// Already closed
		default:
			close(b.workerStopCh)
		}
	}

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

func (b *KubeletBackend) ListAndWatchWorkers() (<-chan []*api.WorkerInfo, <-chan struct{}, error) {
	// Initialize channels if not already created
	if b.workerCh == nil {
		b.workerCh = make(chan []*api.WorkerInfo, 1)
		b.workerStopCh = make(chan struct{})
	}

	// Send initial worker list and start watching
	go func() {
		defer close(b.workerCh)

		// Send initial list
		if b.podCacher != nil {
			b.podCacher.mu.RLock()
			workers := make([]string, 0, len(b.podCacher.cachedPod))
			for podUID := range b.podCacher.cachedPod {
				workers = append(workers, podUID)
			}
			b.podCacher.mu.RUnlock()

			select {
			case b.workerCh <- workers:
			case <-b.ctx.Done():
				return
			case <-b.workerStopCh:
				return
			}
		}

		// Watch for worker changes
		// TODO
		for {
			select {
			case <-b.ctx.Done():
				return
			case <-b.workerStopCh:
				return
			case <-workerChangedCh:
				if b.podCacher != nil {
					b.podCacher.mu.RLock()
					workers := make([]string, 0, len(b.podCacher.cachedPod))
					for podUID := range b.podCacher.cachedPod {
						workers = append(workers, podUID)
					}
					b.podCacher.mu.RUnlock()

					select {
					case b.workerCh <- workers:
					case <-b.ctx.Done():
						return
					case <-b.workerStopCh:
						return
					}
				}
			}
		}
	}()

	return b.workerCh, b.workerStopCh, nil
}

// TODO use ns_mapper to impl this
func (b *KubeletBackend) GetWorkerToProcessMap() (map[string][]string, error) {
	return make(map[string][]string), nil
}

func (b *KubeletBackend) StartWorker(workerUID string) error {
	klog.Warningf("StartWorker not implemented, should be managed by operator")
	return nil
}

func (b *KubeletBackend) StopWorker(workerUID string) error {
	klog.Warningf("StopWorker not implemented, should be managed by operator")
	return nil
}

func (b *KubeletBackend) ReconcileDevices(devices []string) error {
	return nil
}
