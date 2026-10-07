package kubernetes

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/backend/kubernetes/external_dp"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/google/uuid"
	"github.com/samber/lo"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/apiutil"
)

type KubeletBackend struct {
	ctx context.Context

	deviceController     framework.DeviceController
	allocationController framework.WorkerAllocationController

	apiClient         *APIClient
	podCacher         *PodCacheManager
	devicePlugins     []*DevicePlugin
	deviceDetector    *external_dp.DevicePluginDetector
	podResourcesProxy *PodResourcesProxy

	nodeName       string
	checkpointPath string

	workers   map[string]*api.WorkerInfo
	workersMu sync.RWMutex

	deviceTflops   map[string]resource.Quantity
	deviceTflopsMu sync.RWMutex

	subscribers   map[string]struct{}
	subscribersMu sync.Mutex
	stopped       bool
	workerHandler *framework.WorkerChangeHandler
}

var _ framework.Backend = &KubeletBackend{}

func NewKubeletBackend(
	ctx context.Context,
	deviceController framework.DeviceController,
	allocationController framework.WorkerAllocationController,
	restConfig *rest.Config,
) (*KubeletBackend, error) {
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
	checkpointPath := os.Getenv(constants.HypervisorKubeletCheckpointPathEnv)
	if checkpointPath == "" {
		checkpointPath = filepath.Join(DevicePluginPath, "kubelet_internal_checkpoint")
	}
	if os.Getenv(constants.HypervisorDetectUsedGPUEnv) == constants.TrueStringValue {
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
		checkpointPath:       checkpointPath,
		workers:              make(map[string]*api.WorkerInfo),
		deviceTflops:         make(map[string]resource.Quantity),
		subscribers:          make(map[string]struct{}),
	}, nil
}

func (b *KubeletBackend) Start() error {
	if err := b.podCacher.Start(); err != nil {
		return err
	}
	klog.Info("Kubelet client started, watching pods")
	// Restore checkpointed workers before advertising devices. Kubelet may
	// start these containers without another Allocate RPC after registration.
	b.subscribersMu.Lock()
	handler := b.workerHandler
	b.subscribersMu.Unlock()
	if handler != nil {
		b.reconcileWorkers(*handler)
	}

	// Create and start device plugin
	b.devicePlugins = NewDevicePlugins(b.ctx, b.deviceController, b.allocateWorkerDevices, b.podCacher)
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

	// Start the kubelet pod-resources gRPC proxy that exposes TF workers to
	// the node vendor's metrics exporter (DCGM exporter and equivalents) with
	// real device UUIDs under the vendor's resource name. Opt-in: only runs
	// when the operator has both injected the pod-resources-tf hostPath mount
	// and set ENABLE_POD_RESOURCES_PROXY=true on this container. Failure here
	// is non-fatal: the rest of the hypervisor must keep running.
	if os.Getenv(constants.HypervisorPodResourcesProxyEnabledEnv) == constants.TrueStringValue {
		proxy, err := StartPodResourcesProxy(b.podCacher, b.deviceController.GetAcceleratorVendor())
		if err != nil {
			klog.Warningf("Failed to start pod-resources proxy (exporter pod labels will be missing): %v", err)
		} else {
			b.podResourcesProxy = proxy
			klog.Info("Pod-resources proxy started")
		}
	} else {
		klog.Info("Pod-resources proxy disabled (set ENABLE_POD_RESOURCES_PROXY=true to enable)")
	}
	return nil
}

func (b *KubeletBackend) Stop() error {
	b.subscribersMu.Lock()
	if b.stopped {
		b.subscribersMu.Unlock()
		return nil
	}
	b.stopped = true
	subscriberIDs := make([]string, 0, len(b.subscribers))
	for subscriberID := range b.subscribers {
		subscriberIDs = append(subscriberIDs, subscriberID)
	}
	clear(b.subscribers)
	b.subscribersMu.Unlock()

	if b.devicePlugins != nil {
		for i, devicePlugin := range b.devicePlugins {
			if err := devicePlugin.Stop(); err != nil {
				klog.Errorf("Failed to stop device plugin %d: %v", i, err)
			}
		}
	}

	if b.podResourcesProxy != nil {
		b.podResourcesProxy.Stop()
	}

	if b.deviceDetector != nil {
		b.deviceDetector.Stop()
	}

	if b.podCacher != nil {
		for _, subscriberID := range subscriberIDs {
			b.podCacher.UnregisterWorkerInfoSubscriber(subscriberID)
		}
		b.podCacher.Stop()
	}

	return nil
}

// RegisterWorkerUpdateHandler registers a handler for worker updates
func (b *KubeletBackend) RegisterWorkerUpdateHandler(handler framework.WorkerChangeHandler) error {
	b.subscribersMu.Lock()
	defer b.subscribersMu.Unlock()
	if b.stopped {
		return fmt.Errorf("kubelet backend is stopped")
	}
	b.workerHandler = &handler

	// Notifications are wakeups, not a journal. A single queued wakeup is
	// enough: reconciliation reads the authoritative Pod cache, so bursts
	// cannot discard the final update or deletion of a worker.
	workerCh := make(chan *api.WorkerInfo, 1)
	subscriberID := uuid.NewString()
	b.podCacher.RegisterWorkerInfoSubscriber(subscriberID, workerCh)
	b.subscribers[subscriberID] = struct{}{}

	// Start bridge goroutine
	go func() {
		defer func() {
			b.podCacher.UnregisterWorkerInfoSubscriber(subscriberID)
			b.subscribersMu.Lock()
			delete(b.subscribers, subscriberID)
			b.subscribersMu.Unlock()
		}()

		b.reconcileWorkers(handler)
		for {
			select {
			case <-b.ctx.Done():
				return
			case <-b.podCacher.stopCh:
				return
			case _, ok := <-workerCh:
				if !ok {
					return
				}
				b.reconcileWorkers(handler)
			}
		}
	}()
	return nil
}

// reconcileWorkers serializes the snapshot with callbacks. Deletions run before
// additions so a same-name replacement cannot inherit the previous worker.
func (b *KubeletBackend) reconcileWorkers(handler framework.WorkerChangeHandler) {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	pods := b.podCacher.GetAllPods()
	confirmed := b.checkpointedWorkers()
	removed := make(map[string]*api.WorkerInfo)
	for uid, info := range b.workers {
		if pods[uid] == nil {
			removed[uid] = info
		}
	}
	// Allocate can complete before the worker-change subscriber handles Add.
	// If Add and Delete were coalesced, the allocation still needs cleanup.
	for _, allocations := range b.allocationController.GetDeviceAllocations() {
		for _, allocation := range allocations {
			if allocation != nil && allocation.WorkerInfo != nil {
				info := allocation.WorkerInfo
				if pods[info.WorkerUID] == nil {
					removed[info.WorkerUID] = info
				}
			}
		}
	}
	for uid, info := range removed {
		// Allocation may have raced with this snapshot after a new Pod arrived.
		if b.podCacher.GetPodByUID(uid) != nil {
			continue
		}
		if handler.OnRemove != nil {
			handler.OnRemove(info)
		}
		delete(b.workers, uid)
	}
	for uid, pod := range pods {
		old, exists := b.workers[uid]
		info, _, err := b.podCacher.extractWorkerInfo(pod)
		if err != nil {
			// Pod phase is sufficient evidence of termination even when its
			// resource annotations no longer parse. Preserve allocation metadata
			// for cleanup; malformed active updates do not prove termination.
			if old == nil || !utils.IsPodStopped(pod) {
				continue
			}
			info = old.DeepCopy()
			info.Status = api.WorkerStatusTerminated
		}
		restored := confirmed[uid]
		info.AllocationConfirmed = restored != nil || (old != nil && old.AllocationConfirmed)
		partitionUUIDs := ""
		if restored != nil {
			partitionUUIDs = strings.Join(sets.List(restored.partitions), ",")
		}
		if partitionUUIDs == "" && old != nil {
			partitionUUIDs = old.Annotations[constants.PartitionUUIDsAnnotation]
		}
		if partitionUUIDs != "" {
			// extractWorkerInfo shares the informer Pod's annotations. Enrich
			// only this worker snapshot and keep metadata until UID removal.
			info.Annotations = maps.Clone(info.Annotations)
			if info.Annotations == nil {
				info.Annotations = make(map[string]string)
			}
			info.Annotations[constants.PartitionUUIDsAnnotation] = partitionUUIDs
		}
		b.workers[uid] = info
		if !exists && handler.OnAdd != nil {
			handler.OnAdd(info)
		} else if exists && !reflect.DeepEqual(old, info) && handler.OnUpdate != nil {
			handler.OnUpdate(old, info)
		}
	}
}

// allocateWorkerDevices completes shared-memory initialization before returning
// the device-plugin response. A soft limiter can open TF_SHM_PATH as soon as the
// process starts, before the asynchronous worker sync or any HTTP handshake.
func (b *KubeletBackend) allocateWorkerDevices(
	ctx context.Context, requested *api.WorkerInfo,
) (*api.WorkerAllocation, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.subscribersMu.Lock()
	handler, stopped := b.workerHandler, b.stopped
	b.subscribersMu.Unlock()
	if stopped || handler == nil || handler.OnPrepare == nil {
		return nil, fmt.Errorf("worker controller is not ready")
	}
	// The Pod cache can be ahead of the asynchronous notification consumer.
	// Reconcile synchronously so preparation never depends on that timing.
	b.reconcileWorkers(*handler)
	pod := b.podCacher.GetPodByUID(requested.WorkerUID)
	if pod == nil || utils.IsPodStopped(pod) || !pod.DeletionTimestamp.IsZero() {
		return nil, fmt.Errorf("worker %s is no longer active", requested.WorkerUID)
	}
	worker, _, err := b.podCacher.extractWorkerInfo(pod)
	if err != nil {
		return nil, err
	}
	allocation, err := b.allocationController.AllocateWorkerDevices(worker)
	if err != nil {
		return nil, err
	}
	if worker.IsolationMode == tfv1.IsolationModeSoft || worker.IsolationMode == tfv1.IsolationModeHard {
		err = handler.OnPrepare(worker.WorkerUID)
	}
	// Deletion can race allocation, including an OnRemove that ran just before
	// AllocateWorkerDevices. Do not strand that late allocation without another event.
	pod = b.podCacher.GetPodByUID(worker.WorkerUID)
	if pod == nil || utils.IsPodStopped(pod) {
		if cleanupErr := b.allocationController.DeallocateWorker(worker.WorkerUID); cleanupErr != nil {
			klog.Errorf("Failed to clean up stopped worker %s after allocation: %v", worker.WorkerUID, cleanupErr)
		}
		return nil, fmt.Errorf("worker %s stopped during allocation", worker.WorkerUID)
	}
	if !pod.DeletionTimestamp.IsZero() {
		// Terminating containers may still use the GPU. Block startup without
		// releasing their existing allocation before termination or deletion.
		return nil, fmt.Errorf("worker %s is terminating", worker.WorkerUID)
	}
	if err != nil {
		return nil, fmt.Errorf("prepare worker %s: %w", worker.WorkerUID, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return allocation, nil
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
			if device != nil {
				b.deleteDeviceTflops(device.UUID)
			}
			// Delete only when idle; a missing GPU still referenced by running
			// apps is marked (Unknown phase + missing-since annotation) so a
			// transient NVML/driver hiccup cannot wipe allocation accounting.
			if err := b.apiClient.DeleteOrMarkMissingGPU(device.UUID); err != nil {
				klog.Errorf("Failed to clean up GPU when device removed: %v", err)
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
			if nodeInfo != nil {
				if totalTflops, ok := b.getTotalDeviceTflops(); ok {
					nodeInfo.TotalTFlops = totalTflops
				}
				// GPU CRs owned by this node but absent from the current
				// enumeration (e.g. card removed while the hypervisor was down,
				// so no OnRemove ever fired) must be cleaned up, otherwise stale
				// GPU CRs keep polluting scheduling and capacity statistics forever.
				if err := b.apiClient.CleanupStaleGPUs(b.nodeName, nodeInfo.DeviceIDs); err != nil {
					klog.Errorf("Failed to clean up stale GPUs: %v", err)
				}
			}
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
	if gpuNode == nil || gpu == nil {
		return fmt.Errorf("GPU node and GPU are required")
	}
	controllerRef := metav1.GetControllerOf(gpuNode)
	poolName := ""
	if controllerRef != nil {
		poolName = controllerRef.Name
	} else if len(gpuNode.OwnerReferences) > 0 {
		// Older GPUNode objects may have an owner reference without the
		// controller bit set. Keep accepting those objects while avoiding an
		// unchecked index into an empty slice.
		poolName = gpuNode.OwnerReferences[0].Name
	}
	if poolName == "" {
		return fmt.Errorf("GPU node %s has no controller owner reference", gpuNode.Name)
	}
	// Set metadata fields
	gpu.Labels = map[string]string{
		constants.LabelKeyOwner: gpuNode.Name,
		constants.GpuPoolKey:    poolName,
	}
	gpu.Annotations = map[string]string{
		constants.LastSyncTimeAnnotationKey:               time.Now().Format(time.RFC3339),
		constants.GPUVirtualizationCapabilitiesAnnotation: buildVirtualizationCapabilitiesAnnotation(device),
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
	// Prefer ProviderConfig Fp16TFlops if configured, fall back to device-reported value
	var tflops resource.Quantity
	if resolved, ok := b.resolveDeviceTflopsFromProviderConfig(device); ok {
		tflops = resolved
	} else {
		tflops = resource.MustParse(fmt.Sprintf("%f", device.MaxTflops))
	}
	b.setDeviceTflops(device.UUID, tflops)
	gpu.Status.Capacity = &tfv1.Resource{
		Vram:   resource.MustParse(fmt.Sprintf("%dMi", device.TotalMemoryBytes/1024/1024)),
		Tflops: tflops,
	}
	gpu.Status.UUID = device.UUID
	gpu.Status.GPUModel = device.Model
	gpu.Status.Index = ptr.To(device.Index)
	gpu.Status.Vendor = device.Vendor
	gpu.Status.NUMANode = ptr.To(device.NUMANode)
	gpu.Status.Topology = convertDeviceTopologyToStatus(device.Topology)
	gpu.Status.IsolationMode = device.IsolationMode
	gpu.Status.IsolationPolicy = device.IsolationPolicy
	gpu.Status.NodeSelector = map[string]string{
		constants.KubernetesHostNameLabel: b.nodeName,
	}
	if shouldResetAvailable(gpu) {
		gpu.Status.Available = gpu.Status.Capacity.DeepCopy()
	}
	if gpu.Status.UsedBy == "" {
		gpu.Status.UsedBy = tfv1.UsedByTensorFusion
	}
	if gpu.Status.Phase == "" || gpu.Status.Phase == tfv1.TensorFusionGPUPhaseUnknown {
		// Unknown means the device was previously marked missing by discovery,
		// it has recovered now (the missing-since annotation is cleared by the
		// full annotation rewrite above), reset to Pending for the controller
		// to promote.
		gpu.Status.Phase = tfv1.TensorFusionGPUPhasePending
	}
	gpu.Status.Message = "managed"
	return nil
}

func buildVirtualizationCapabilitiesAnnotation(device *api.DeviceInfo) string {
	if device == nil {
		return ""
	}
	payload, err := json.Marshal(device.VirtualizationCapabilities)
	if err != nil {
		return ""
	}
	return string(payload)
}

func (b *KubeletBackend) resolveDeviceTflopsFromProviderConfig(device *api.DeviceInfo) (resource.Quantity, bool) {
	resolved, ok, err := b.apiClient.ResolveDeviceFp16TFlops(device.Vendor, device.Model)
	if err != nil {
		klog.Warningf("Failed to resolve tflops from ProviderConfig: vendor=%s model=%s err=%v",
			device.Vendor, device.Model, err)
		return resource.Quantity{}, false
	}
	return resolved, ok
}

func shouldResetAvailable(gpu *tfv1.GPU) bool {
	if gpu == nil || gpu.Status.Capacity == nil {
		return false
	}
	if gpu.Status.Available == nil {
		return true
	}
	if gpu.Status.Available.Tflops.IsZero() &&
		gpu.Status.Available.Vram.IsZero() &&
		gpu.Status.Available.ComputePercent.IsZero() &&
		len(gpu.Status.RunningApps) == 0 &&
		len(gpu.Status.AllocatedPartitions) == 0 {
		return true
	}
	return false
}

func (b *KubeletBackend) setDeviceTflops(uuid string, tflops resource.Quantity) {
	if uuid == "" {
		return
	}
	b.deviceTflopsMu.Lock()
	defer b.deviceTflopsMu.Unlock()
	b.deviceTflops[uuid] = tflops
}

func (b *KubeletBackend) deleteDeviceTflops(uuid string) {
	if uuid == "" {
		return
	}
	b.deviceTflopsMu.Lock()
	defer b.deviceTflopsMu.Unlock()
	delete(b.deviceTflops, uuid)
}

func (b *KubeletBackend) getTotalDeviceTflops() (float64, bool) {
	b.deviceTflopsMu.RLock()
	defer b.deviceTflopsMu.RUnlock()
	total := 0.0
	found := false
	for _, tflops := range b.deviceTflops {
		if tflops.IsZero() {
			continue
		}
		total += tflops.AsApproximateFloat64()
		found = true
	}
	return total, found
}

func convertDeviceTopologyToStatus(topology *api.DeviceTopology) *tfv1.GPUTopologyStatus {
	if topology == nil {
		return nil
	}
	peers := make([]tfv1.GPUPeerLinkStatus, 0, len(topology.Peers))
	for _, peer := range topology.Peers {
		if peer.PeerUUID == "" {
			continue
		}
		peers = append(peers, tfv1.GPUPeerLinkStatus{
			PeerGPUUUID: peer.PeerUUID,
			Tier:        peer.Tier,
			LinkType:    peer.LinkType,
			Bandwidth:   peer.Bandwidth,
		})
	}
	return &tfv1.GPUTopologyStatus{
		Peers: peers,
	}
}
