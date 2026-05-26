package worker

import (
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/device"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/samber/lo"
	"k8s.io/klog/v2"
)

// AllocationController manages worker device allocations
// This is a shared dependency between DeviceController, WorkerController, and Backend
type AllocationController struct {
	deviceController framework.DeviceController

	mu                sync.RWMutex
	workerAllocations map[string]*api.WorkerAllocation
	deviceAllocations map[string][]*api.WorkerAllocation
}

var _ framework.WorkerAllocationController = &AllocationController{}

type visibleDeviceRef struct {
	index int32
	uuid  string
}

// NewAllocationController creates a new AllocationController
func NewAllocationController(deviceController framework.DeviceController) *AllocationController {
	return &AllocationController{
		deviceController:  deviceController,
		workerAllocations: make(map[string]*api.WorkerAllocation, 32),
		deviceAllocations: make(map[string][]*api.WorkerAllocation, 32),
	}
}

// AllocateWorkerDevices allocates devices for a worker request
func (a *AllocationController) AllocateWorkerDevices(request *api.WorkerInfo) (*api.WorkerAllocation, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if len(request.AllocatedDevices) == 0 {
		return nil, fmt.Errorf("worker %s has no allocated devices", request.WorkerUID)
	}

	// idempotency check
	if a.workerAllocations[request.WorkerUID] != nil {
		klog.Infof("worker %s already allocated, skipping", request.WorkerUID)
		return a.workerAllocations[request.WorkerUID], nil
	}

	deviceInfos := make([]*api.DeviceInfo, 0, len(request.AllocatedDevices))

	// partitioned mode, call split device
	isPartitioned := request.IsolationMode == tfv1.IsolationModePartitioned && request.PartitionTemplateID != ""

	// Track partitions created in this call so we can release them on any
	// subsequent failure. Without this, mid-loop / post-loop errors leave
	// hardware partitions allocated with no entry in workerAllocations,
	// so DeallocateWorker (triggered by pod deletion) finds nothing to clean
	// and the partition stays orphaned until manual intervention.
	var splittedPartitions []*api.DeviceInfo

	for _, deviceUUID := range request.AllocatedDevices {
		device, exists := a.deviceController.GetDevice(deviceUUID)
		if !exists {
			klog.Errorf("worker %s requested device %s not found in device controller, skipping", request.WorkerUID, deviceUUID)
			continue
		}
		if isPartitioned {
			deviceInfo, err := a.deviceController.SplitDevice(deviceUUID, request.PartitionTemplateID)
			if err != nil {
				a.rollbackSplits(splittedPartitions)
				return nil, err
			}
			splittedPartitions = append(splittedPartitions, deviceInfo)
			deviceInfos = append(deviceInfos, deviceInfo)
		} else {
			deviceInfos = append(deviceInfos, device)
		}
	}

	mounts, err := a.deviceController.GetVendorMountLibs()
	if err != nil {
		if device.IsNonFatalAcceleratorError(err) {
			// NOT_SUPPORTED or NOT_FOUND are expected for some vendors (e.g., Nvidia)
			klog.V(4).Infof("vendor mount libs not available for worker %s: %v", request.WorkerUID, err)
			mounts = []*api.Mount{}
		} else {
			// Fatal errors (INTERNAL, OPERATION_FAILED, etc.) should block allocation
			klog.Errorf("failed to get vendor mount libs for worker %s: %v", request.WorkerUID, err)
			a.rollbackSplits(splittedPartitions)
			return nil, err
		}
	}

	envs := make(map[string]string, 8)
	devices := make(map[string]*api.DeviceSpec, 8)
	for _, deviceInfo := range deviceInfos {
		maps.Copy(envs, deviceInfo.DeviceEnv)
		for devNode, guestPath := range deviceInfo.DeviceNode {
			if _, exists := devices[devNode]; exists {
				continue
			}
			devices[devNode] = &api.DeviceSpec{
				HostPath:    devNode,
				GuestPath:   guestPath,
				Permissions: "rwm",
			}
		}
	}
	if !isPartitioned {
		// Non-partitioned modes (shared/soft/hard): pin the vendor's
		// *_VISIBLE_DEVICES env to the devices this worker actually got. The
		// matching OCI runtime hook (nvidia-container-toolkit /
		// mthreads-container-runtime / ascend-docker-runtime) reads this env
		// as its authoritative request channel — without it, hooks fall back
		// to the image-baked default (typically `=all`), which on multi-card
		// hosts leaks every card into the pod and silently breaks isolation.
		//
		// Partitioned (MIG / MUSA template) mode is intentionally skipped:
		// the provider's AccelAssignPartition already wrote the partition-
		// scoped env (NVIDIA_VISIBLE_DEVICES=MIG-<uuid>,
		// MTHREADS_VISIBLE_DEVICES=<idx>+MUSA_VGPU_PARTITION_UUID, ...) into
		// deviceInfo.DeviceEnv and that was copied into envs above.
		// Overwriting it with the parent device identifier would expose the
		// whole card.
		if isNvidiaVendor(deviceInfos) {
			// NVIDIA canonicalizes to NVML "GPU-<hex>" form. CDIDevices is
			// never emitted here — see deviceplugin.go.
			names := buildPinnedNvidiaDeviceNames(deviceInfos)
			if len(names) > 0 {
				envs[constants.NvidiaVisibleAllDeviceEnv] = strings.Join(names, ",")
			}
		} else if envName, ok := indexBasedVisibleDevicesEnvName(deviceInfos); ok {
			// MThreads / Ascend: mthreads-container-runtime and
			// ascend-docker-runtime both accept comma-separated device
			// indices — matches what vgpu-provider-internal's MUSA
			// AccelAssignPartition emits in partitioned mode
			// (`MTHREADS_VISIBLE_DEVICES=%u`).
			indices := buildPinnedDeviceIndices(deviceInfos)
			if len(indices) > 0 {
				envs[envName] = strings.Join(indices, ",")
			}
		}
	}

	allocation := &api.WorkerAllocation{
		WorkerInfo:  request,
		DeviceInfos: deviceInfos,
		Envs:        envs,
		Mounts:      mounts,
		Devices:     lo.Values(devices),
	}

	a.workerAllocations[request.WorkerUID] = allocation
	for _, deviceUUID := range request.AllocatedDevices {
		a.addDeviceAllocation(deviceUUID, allocation)
	}
	return allocation, nil
}

// rollbackSplits releases partitions that were created during a failed
// allocation. Errors are logged best-effort; the caller is already returning
// the original allocation error.
func (a *AllocationController) rollbackSplits(splits []*api.DeviceInfo) {
	for _, split := range splits {
		if split == nil || split.ParentUUID == "" {
			continue
		}
		if err := a.deviceController.RemovePartitionedDevice(split.UUID, split.ParentUUID); err != nil {
			klog.Errorf("rollback partition %s on parent %s failed: %v",
				split.UUID, split.ParentUUID, err)
		}
	}
}

// DeallocateWorker deallocates devices for a worker
// For partitioned devices, this also calls RemovePartitionedDevice to release the partition
func (a *AllocationController) DeallocateWorker(workerUID string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	allocation, exists := a.workerAllocations[workerUID]
	if !exists {
		klog.V(4).Infof("worker allocation not found for worker %s, may have already been deallocated", workerUID)
		return nil
	}
	delete(a.workerAllocations, workerUID)
	for _, deviceUUID := range allocation.WorkerInfo.AllocatedDevices {
		a.removeDeviceAllocation(deviceUUID, allocation)
	}

	// For partitioned devices, release the partition via device controller
	for _, deviceInfo := range allocation.DeviceInfos {
		if deviceInfo.ParentUUID != "" {
			// This is a partitioned device, release the partition
			if err := a.deviceController.RemovePartitionedDevice(deviceInfo.UUID, deviceInfo.ParentUUID); err != nil {
				klog.Errorf("failed to remove partition %s from device %s for worker %s: %v",
					deviceInfo.UUID, deviceInfo.ParentUUID, workerUID, err)
				// Continue deallocating other resources even if partition removal fails
			}
		}
	}

	klog.Infof("worker %s deallocated", workerUID)
	return nil
}

// RecoverPartitionedWorker rebuilds allocation state for an existing partitioned worker
// after hypervisor restart. partitionUUIDs is a comma-separated string of "partitionUUID:parentGPU" pairs.
func (a *AllocationController) RecoverPartitionedWorker(request *api.WorkerInfo, partitionUUIDs string) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.workerAllocations[request.WorkerUID] != nil {
		return // already allocated, skip
	}

	var deviceInfos []*api.DeviceInfo
	for _, pair := range strings.Split(partitionUUIDs, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		parts := strings.SplitN(pair, ":", 2)
		if len(parts) != 2 {
			klog.Warningf("invalid partition UUID pair %q for worker %s, skipping", pair, request.WorkerUID)
			continue
		}
		partUUID, parentUUID := parts[0], parts[1]

		// Build a minimal DeviceInfo with partition and parent UUID for deallocation
		deviceInfo := &api.DeviceInfo{
			UUID:       partUUID,
			ParentUUID: parentUUID,
		}
		// Enrich from the parent device if available
		if parent, exists := a.deviceController.GetDevice(parentUUID); exists {
			deviceInfo.Vendor = parent.Vendor
			deviceInfo.Model = parent.Model
		}
		deviceInfos = append(deviceInfos, deviceInfo)
	}

	if len(deviceInfos) == 0 {
		klog.Warningf("no valid partition UUIDs to recover for worker %s", request.WorkerUID)
		return
	}

	allocation := &api.WorkerAllocation{
		WorkerInfo:  request,
		DeviceInfos: deviceInfos,
	}
	a.workerAllocations[request.WorkerUID] = allocation
	for _, deviceUUID := range request.AllocatedDevices {
		a.addDeviceAllocation(deviceUUID, allocation)
	}
	klog.Infof("recovered partitioned worker %s with %d partition(s)", request.WorkerUID, len(deviceInfos))
}

// GetWorkerAllocation returns the allocation for a specific worker
func (a *AllocationController) GetWorkerAllocation(workerUID string) (*api.WorkerAllocation, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	allocation, exists := a.workerAllocations[workerUID]
	return allocation, exists
}

// GetDeviceAllocations returns all device allocations keyed by device UUID
func (a *AllocationController) GetDeviceAllocations() map[string][]*api.WorkerAllocation {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return maps.Clone(a.deviceAllocations)
}

// addDeviceAllocation adds an allocation to a device (internal, must be called with lock held)
func (a *AllocationController) addDeviceAllocation(deviceUUID string, allocation *api.WorkerAllocation) {
	if _, exists := a.deviceAllocations[deviceUUID]; !exists {
		a.deviceAllocations[deviceUUID] = make([]*api.WorkerAllocation, 0, 8)
	}
	a.deviceAllocations[deviceUUID] = append(a.deviceAllocations[deviceUUID], allocation)
}

// removeDeviceAllocation removes an allocation from a device (internal, must be called with lock held)
func (a *AllocationController) removeDeviceAllocation(deviceUUID string, allocation *api.WorkerAllocation) {
	if _, exists := a.deviceAllocations[deviceUUID]; !exists {
		return
	}
	a.deviceAllocations[deviceUUID] = lo.Filter(
		a.deviceAllocations[deviceUUID],
		func(wa *api.WorkerAllocation, _ int) bool {
			return wa.WorkerInfo.WorkerUID != allocation.WorkerInfo.WorkerUID
		},
	)
}

func isNvidiaVendor(deviceInfos []*api.DeviceInfo) bool {
	if len(deviceInfos) == 0 {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(deviceInfos[0].Vendor), constants.AcceleratorVendorNvidia)
}

// buildPinnedNvidiaDeviceNames returns canonical NVIDIA device names for
// NVIDIA_VISIBLE_DEVICES and CDI device requests. The returned strings match
// what nvidia-container-toolkit registers in the CDI spec (and what NVML
// returns) byte-for-byte: "GPU-<hex>" for full GPUs, "MIG-<...>" for MIG
// partitions. Upstream lowercasing of the "GPU-"/"MIG-" prefix is repaired
// here; the hex payload is preserved as-is to stay compatible with both
// default-CDI and legacy device-plugin modes.
func buildPinnedNvidiaDeviceNames(deviceInfos []*api.DeviceInfo) []string {
	visibleDevices := make([]visibleDeviceRef, 0, len(deviceInfos))
	seen := make(map[string]struct{}, len(deviceInfos))
	for _, deviceInfo := range deviceInfos {
		if deviceInfo == nil || deviceInfo.UUID == "" {
			continue
		}
		name := canonicalizeNvidiaDeviceUUID(deviceInfo.UUID)
		if _, exists := seen[name]; exists {
			continue
		}
		seen[name] = struct{}{}
		visibleDevices = append(visibleDevices, visibleDeviceRef{
			index: deviceInfo.Index,
			uuid:  name,
		})
	}
	slices.SortFunc(visibleDevices, func(a, b visibleDeviceRef) int {
		switch {
		case a.index < b.index:
			return -1
		case a.index > b.index:
			return 1
		case a.uuid < b.uuid:
			return -1
		case a.uuid > b.uuid:
			return 1
		default:
			return 0
		}
	})
	return lo.Map(visibleDevices, func(device visibleDeviceRef, _ int) string {
		return device.uuid
	})
}

// indexBasedVisibleDevicesEnvName returns the *_VISIBLE_DEVICES env var name
// for vendors whose container runtime hook honors a comma-separated *device
// index* list. NVIDIA is handled separately because its hook canonicalizes
// on UUIDs, not indices. Returns ("", false) for vendors with no such
// convention; the caller skips env injection in that case.
func indexBasedVisibleDevicesEnvName(deviceInfos []*api.DeviceInfo) (string, bool) {
	if len(deviceInfos) == 0 {
		return "", false
	}
	vendor := strings.TrimSpace(deviceInfos[0].Vendor)
	switch {
	case strings.EqualFold(vendor, constants.AcceleratorVendorMThreads):
		return constants.MthreadsVisibleDevicesEnv, true
	case strings.EqualFold(vendor, constants.AcceleratorVendorHuaweiAscendNPU):
		return constants.AscendVisibleDevicesEnv, true
	}
	return "", false
}

// buildPinnedDeviceIndices returns the deduped, sorted device indices the
// worker was allocated, formatted as decimal strings ready for joining into
// a *_VISIBLE_DEVICES env value.
func buildPinnedDeviceIndices(deviceInfos []*api.DeviceInfo) []string {
	indices := make([]int32, 0, len(deviceInfos))
	seen := make(map[int32]struct{}, len(deviceInfos))
	for _, deviceInfo := range deviceInfos {
		if deviceInfo == nil {
			continue
		}
		if _, exists := seen[deviceInfo.Index]; exists {
			continue
		}
		seen[deviceInfo.Index] = struct{}{}
		indices = append(indices, deviceInfo.Index)
	}
	slices.Sort(indices)
	return lo.Map(indices, func(i int32, _ int) string {
		return fmt.Sprintf("%d", i)
	})
}

// canonicalizeNvidiaDeviceUUID restores the NVML-canonical prefix casing
// ("GPU-"/"MIG-") without touching the hex payload. If the input has no
// recognized prefix (e.g. a bare index or "all"), it is returned unchanged.
func canonicalizeNvidiaDeviceUUID(s string) string {
	if len(s) >= 4 {
		switch strings.ToUpper(s[:4]) {
		case "GPU-":
			return "GPU-" + s[4:]
		case "MIG-":
			return "MIG-" + s[4:]
		}
	}
	return s
}
