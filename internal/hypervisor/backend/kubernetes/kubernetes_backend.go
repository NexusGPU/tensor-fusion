package kubernetes

import (
	"context"
	"os"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/backend/kubernetes/external_dp"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

type KubeletBackend struct {
	ctx context.Context

	deviceController framework.DeviceController
	kubeletClient    *KubeletClient
	devicePlugin     *DevicePlugin
	deviceDetector   *external_dp.DevicePluginDetector

	workerChanged chan struct{}
	workerCh      chan []string
	workerStopCh  chan struct{}
}

func NewKubeletBackend(ctx context.Context, deviceController framework.DeviceController, restConfig *rest.Config) (*KubeletBackend, error) {
	// Get node name from environment or config
	nodeName := os.Getenv("NODE_NAME")
	if nodeName == "" {
		nodeName = os.Getenv("HOSTNAME")
	}

	// Create kubelet client
	kubeletClient, err := NewKubeletClient(ctx, restConfig, nodeName)
	if err != nil {
		return nil, err
	}

	// Create API server for device detector
	apiServer, err := NewAPIServerFromConfig(ctx, restConfig)
	if err != nil {
		return nil, err
	}

	// Create device plugin detector
	checkpointPath := os.Getenv("KUBELET_CHECKPOINT_PATH")
	// Create adapter for kubelet client to match interface
	kubeletAdapter := &kubeletClientAdapter{kubeletClient: kubeletClient}
	deviceDetector, err := external_dp.NewDevicePluginDetector(ctx, checkpointPath, apiServer, kubeletAdapter)
	if err != nil {
		return nil, err
	}

	return &KubeletBackend{
		ctx:              ctx,
		deviceController: deviceController,
		kubeletClient:    kubeletClient,
		deviceDetector:   deviceDetector,
		workerChanged:    make(chan struct{}),
	}, nil
}

func (b *KubeletBackend) Start() error {
	// Start kubelet client to watch pods
	if err := b.kubeletClient.Start(); err != nil {
		return err
	}
	klog.Info("Kubelet client started, watching pods")

	// Create and start device plugin
	b.devicePlugin = NewDevicePlugin(b.ctx, b.deviceController, b.kubeletClient)
	if err := b.devicePlugin.Start(); err != nil {
		return err
	}
	klog.Info("Device plugin started and registered with kubelet")

	// Start device plugin detector to watch external device plugins
	if b.deviceDetector != nil {
		if err := b.deviceDetector.Start(); err != nil {
			klog.Warningf("Failed to start device plugin detector: %v", err)
		} else {
			klog.Info("Device plugin detector started")
		}
	}

	// Start worker change watcher
	go b.watchWorkerChanges()

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

	if b.devicePlugin != nil {
		if err := b.devicePlugin.Stop(); err != nil {
			klog.Errorf("Failed to stop device plugin: %v", err)
		}
	}

	if b.deviceDetector != nil {
		b.deviceDetector.Stop()
	}

	if b.kubeletClient != nil {
		b.kubeletClient.Stop()
	}

	return nil
}

// watchWorkerChanges watches for worker changes and notifies
func (b *KubeletBackend) watchWorkerChanges() {
	workerChangedCh := b.kubeletClient.GetWorkerChangedChan()
	for {
		select {
		case <-b.ctx.Done():
			return
		case <-workerChangedCh:
			select {
			case b.workerChanged <- struct{}{}:
			default:
			}
		}
	}
}

func (b *KubeletBackend) ListAndWatchWorkers() (<-chan []string, <-chan struct{}, error) {
	// Initialize channels if not already created
	if b.workerCh == nil {
		b.workerCh = make(chan []string, 1)
		b.workerStopCh = make(chan struct{})
	}

	// Send initial worker list and start watching
	go func() {
		defer close(b.workerCh)

		// Send initial list
		if b.kubeletClient != nil {
			b.kubeletClient.mu.RLock()
			workers := make([]string, 0, len(b.kubeletClient.podCache))
			for podUID := range b.kubeletClient.podCache {
				workers = append(workers, podUID)
			}
			b.kubeletClient.mu.RUnlock()

			select {
			case b.workerCh <- workers:
			case <-b.ctx.Done():
				return
			case <-b.workerStopCh:
				return
			}
		}

		// Watch for worker changes
		workerChangedCh := b.kubeletClient.GetWorkerChangedChan()
		for {
			select {
			case <-b.ctx.Done():
				return
			case <-b.workerStopCh:
				return
			case <-workerChangedCh:
				if b.kubeletClient != nil {
					b.kubeletClient.mu.RLock()
					workers := make([]string, 0, len(b.kubeletClient.podCache))
					for podUID := range b.kubeletClient.podCache {
						workers = append(workers, podUID)
					}
					b.kubeletClient.mu.RUnlock()

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

func (b *KubeletBackend) GetWorkerToProcessMap() (map[string][]string, error) {
	return make(map[string][]string), nil
}

func (b *KubeletBackend) StartWorker(workerUID string) error {
	return nil
}

func (b *KubeletBackend) StopWorker(workerUID string) error {
	return nil
}

func (b *KubeletBackend) ReconcileDevices(devices []string) error {
	return nil
}

func (b *KubeletBackend) GetWorkerChangedChan(ctx context.Context) <-chan struct{} {
	return b.workerChanged
}

// kubeletClientAdapter adapts KubeletClient to external_dp.KubeletClientInterface
type kubeletClientAdapter struct {
	kubeletClient *KubeletClient
}

func (k *kubeletClientAdapter) GetAllPods() map[string]interface{} {
	pods := k.kubeletClient.GetAllPods()
	result := make(map[string]interface{}, len(pods))
	for k, v := range pods {
		result[k] = v
	}
	return result
}
