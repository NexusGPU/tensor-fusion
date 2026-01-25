package kubernetes

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/backend/kubernetes/external_dp"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/google/uuid"
	"github.com/samber/lo"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/apiutil"
)

type KubeletBackend struct {
	ctx context.Context

	deviceController     framework.DeviceController
	allocationController framework.WorkerAllocationController

	apiClient      *APIClient
	podCacher      *PodCacheManager
	devicePlugins  []*DevicePlugin
	deviceDetector *external_dp.DevicePluginDetector

	nodeName string

	workers   map[string]*api.WorkerInfo
	workersMu sync.RWMutex

	subscribers   map[string]struct{}
	workerHandler *framework.WorkerChangeHandler
}

var _ framework.Backend = &KubeletBackend{}

func NewKubeletBackend(ctx context.Context, deviceController framework.DeviceController, allocationController framework.WorkerAllocationController, restConfig *rest.Config) (*KubeletBackend, error) {
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
	apiClient, err := NewAPIClientFromConfig(ctx, restConfig, nodeName)
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
		ctx:                  ctx,
		deviceController:     deviceController,
		allocationController: allocationController,
		podCacher:            podCacher,
		deviceDetector:       deviceDetector,
		apiClient:            apiClient,
		nodeName:             nodeName,
		workers:              make(map[string]*api.WorkerInfo),
		subscribers:          make(map[string]struct{}),
	}, nil
}

func (b *KubeletBackend) Start() error {
	if err := b.podCacher.Start(); err != nil {
		return err
	}
	klog.Info("Kubelet client started, watching pods")

	// Create and start device plugin
	b.devicePlugins = NewDevicePlugins(b.ctx, b.deviceController, b.allocationController, b.podCacher)
	for _, devicePlugin := range b.devicePlugins {
		if err := devicePlugin.Start(); err != nil {
			return err
		}
		time.Sleep(100 * time.Millisecond)
	}
	klog.Infof("All device plugins started and registered with kubelet")

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
		for subscriberID := range b.subscribers {
			b.podCacher.UnregisterWorkerInfoSubscriber(subscriberID)
		}
		b.subscribers = make(map[string]struct{})
		b.podCacher.Stop()
	}

	return nil
}

// RegisterWorkerUpdateHandler registers a handler for worker updates
func (b *KubeletBackend) RegisterWorkerUpdateHandler(handler framework.WorkerChangeHandler) error {
	b.workerHandler = &handler

	// Create a channel bridge to convert channel messages to handler calls
	workerCh := make(chan *api.WorkerInfo, 16)
	subscriberID := uuid.NewString()
	b.podCacher.RegisterWorkerInfoSubscriber(subscriberID, workerCh)
	b.subscribers[subscriberID] = struct{}{}

	// Start bridge goroutine
	go func() {
		defer func() {
			b.podCacher.UnregisterWorkerInfoSubscriber(subscriberID)
			delete(b.subscribers, subscriberID)
		}()

		for {
			select {
			case <-b.ctx.Done():
				return
			case worker, ok := <-workerCh:
				if !ok {
					return
				}
				if worker == nil {
					continue
				}

				// Determine if this is add, update, or remove
				b.workersMu.Lock()
				oldWorker, exists := b.workers[worker.WorkerUID]

				if worker.DeletedAt > 0 {
					// Worker was deleted
					if exists && handler.OnRemove != nil {
						handler.OnRemove(worker)
					}
					delete(b.workers, worker.WorkerUID)
				} else if !exists {
					// New worker
					b.workers[worker.WorkerUID] = worker
					if handler.OnAdd != nil {
						handler.OnAdd(worker)
					}
				} else {
					// Updated worker
					b.workers[worker.WorkerUID] = worker
					if handler.OnUpdate != nil {
						handler.OnUpdate(oldWorker, worker)
					}
				}
				b.workersMu.Unlock()
			}
		}
	}()
	return nil
}

func (b *KubeletBackend) StartWorker(worker *api.WorkerInfo) error {
	klog.Warningf("StartWorker not implemented, should be managed by operator")
	return nil
}

func (b *KubeletBackend) StopWorker(workerUID string) error {
	klog.Warningf("StopWorker not implemented, should be managed by operator")
	return nil
}

func (b *KubeletBackend) GetProcessMappingInfo(hostPID uint32) (*framework.ProcessMappingInfo, error) {
	return GetWorkerInfoFromHostPID(hostPID)
}

func (b *KubeletBackend) GetDeviceChangeHandler() framework.DeviceChangeHandler {
	return framework.DeviceChangeHandler{
		OnAdd: func(device *api.DeviceInfo) {
			if err := b.apiClient.CreateOrUpdateGPU(b.nodeName, device.UUID,
				func(gpuNode *tfv1.GPUNode, gpu *tfv1.GPU) error {

					return b.mutateGPUResourceState(device, gpuNode, gpu)
				}); err != nil {
				klog.Errorf("Failed to create or update GPU when device added: %v", err)
			} else {
				klog.Infof("Device added: %s", device.UUID)
			}
		},
		OnRemove: func(device *api.DeviceInfo) {
			if err := b.apiClient.DeleteGPU(device.UUID); err != nil {
				klog.Errorf("Failed to delete GPU when device removed: %v", err)
			} else {
				klog.Infof("Device removed: %s", device.UUID)
			}
		},
		OnUpdate: func(oldDevice, newDevice *api.DeviceInfo) {
			if err := b.apiClient.CreateOrUpdateGPU(b.nodeName, newDevice.UUID,
				func(gpuNode *tfv1.GPUNode, gpu *tfv1.GPU) error {
					return b.mutateGPUResourceState(newDevice, gpuNode, gpu)
				}); err != nil {
				klog.Errorf("Failed to update GPU when device updated: %v", err)
			} else {
				klog.Infof("Device updated: %s", newDevice.UUID)
			}
		},
		OnDiscoveryComplete: func(nodeInfo *api.NodeInfo) {
			if err := b.apiClient.UpdateGPUNodeStatus(b.nodeName, nodeInfo); err != nil {
				klog.Errorf("Failed to update GPUNode status: %v", err)
			} else {
				klog.Infof("GPUNode status updated: %s", b.nodeName)
			}
		},
	}
}

func (b *KubeletBackend) ListWorkers() []*api.WorkerInfo {
	b.workersMu.RLock()
	defer b.workersMu.RUnlock()
	return lo.Values(b.workers)
}

func (b *KubeletBackend) mutateGPUResourceState(
	device *api.DeviceInfo, gpuNode *tfv1.GPUNode, gpu *tfv1.GPU,
) error {
	// Set metadata fields
	gpu.Labels = map[string]string{
		constants.LabelKeyOwner: gpuNode.Name,
		constants.GpuPoolKey:    gpuNode.OwnerReferences[0].Name,
	}
	gpu.Annotations = map[string]string{
		constants.LastSyncTimeAnnotationKey: time.Now().Format(time.RFC3339),
	}

	if !metav1.IsControlledBy(gpu, gpuNode) {
		// Create a new controller ref.
		gvk, err := apiutil.GVKForObject(gpuNode, scheme)
		if err != nil {
			return err
		}
		ref := metav1.OwnerReference{
			APIVersion:         gvk.GroupVersion().String(),
			Kind:               gvk.Kind,
			Name:               gpuNode.GetName(),
			UID:                gpuNode.GetUID(),
			BlockOwnerDeletion: ptr.To(true),
			Controller:         ptr.To(true),
		}
		gpu.OwnerReferences = []metav1.OwnerReference{ref}
	}

	// Set status fields
	gpu.Status.Capacity = &tfv1.Resource{
		Vram:   resource.MustParse(fmt.Sprintf("%dMi", device.TotalMemoryBytes/1024/1024)),
		Tflops: resource.MustParse(fmt.Sprintf("%f", device.MaxTflops)),
	}
	gpu.Status.UUID = device.UUID
	gpu.Status.GPUModel = device.Model
	gpu.Status.Index = ptr.To(device.Index)
	gpu.Status.Vendor = device.Vendor
	gpu.Status.NUMANode = ptr.To(device.NUMANode)
	gpu.Status.IsolationMode = device.IsolationMode
	gpu.Status.NodeSelector = map[string]string{
		constants.KubernetesHostNameLabel: b.nodeName,
	}
	if gpu.Status.Available == nil {
		gpu.Status.Available = gpu.Status.Capacity.DeepCopy()
	}
	if gpu.Status.UsedBy == "" {
		gpu.Status.UsedBy = tfv1.UsedByTensorFusion
	}
	if gpu.Status.Phase == "" {
		gpu.Status.Phase = tfv1.TensorFusionGPUPhasePending
	}
	gpu.Status.Message = "managed"
	return nil
}
