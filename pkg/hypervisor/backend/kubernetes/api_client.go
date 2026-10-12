package kubernetes

import (
	"context"
	"fmt"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	authv1 "k8s.io/api/authentication/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

var (
	scheme = runtime.NewScheme()
)

func init() {
	utilruntime.Must(tfv1.AddToScheme(scheme))
	utilruntime.Must(authv1.AddToScheme(scheme))
}

// APIClient provides CRUD operations for GPU resources
type APIClient struct {
	client client.Client
	ctx    context.Context
}

// NewAPIClient creates a new API client instance with an existing client
func NewAPIClient(ctx context.Context, k8sClient client.Client) *APIClient {
	return &APIClient{
		client: k8sClient,
		ctx:    ctx,
	}
}

// NewAPIClientFromConfig creates a new API client instance from a rest.Config
func NewAPIClientFromConfig(ctx context.Context, restConfig *rest.Config) (*APIClient, error) {
	k8sClient, err := client.New(restConfig, client.Options{
		Scheme: scheme,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Kubernetes client: %w", err)
	}

	return &APIClient{
		client: k8sClient,
		ctx:    ctx,
	}, nil
}

// GPUInfo contains information needed to create or update a GPU
type GPUInfo struct {
	UUID          string
	DeviceName    string
	VRAMBytes     uint64
	TFlops        resource.Quantity
	Index         int32
	NUMANodeID    int32
	NodeName      string
	Vendor        string
	IsolationMode tfv1.IsolationModeType
}

// CreateOrUpdateGPU creates or updates a GPU resource with metadata and status
func (a *APIClient) CreateOrUpdateGPU(
	gpuNodeName string, gpuID string,
	mutateFn func(gpuNode *tfv1.GPUNode, gpu *tfv1.GPU) error,
) error {
	// Fetch the GPUNode info
	gpuNode := &tfv1.GPUNode{}
	if err := a.client.Get(a.ctx, client.ObjectKey{Name: gpuNodeName}, gpuNode); err != nil {
		return fmt.Errorf("failed to get GPUNode %s: %w", gpuNodeName, err)
	}

	// Create or update GPU metadata
	err := retry.OnError(wait.Backoff{
		Steps:    3,
		Duration: time.Second,
		Factor:   1.0,
		Jitter:   0.1,
	}, func(err error) bool {
		return true // Retry on all errors
	}, func() error {
		gpu := &tfv1.GPU{
			ObjectMeta: metav1.ObjectMeta{
				Name: gpuID,
			},
		}
		_, err := controllerutil.CreateOrPatch(a.ctx, a.client, gpu, func() error {
			// Metadata only. The status is written in one step below, against
			// the object as it is at that moment: capacity and available have
			// to change together, and a status computed from this earlier read
			// could undo what the allocator published in between.
			observedStatus := gpu.Status.DeepCopy()
			if err := mutateFn(gpuNode, gpu); err != nil {
				return err
			}
			gpu.Status = *observedStatus
			return nil
		})
		if err != nil {
			return err
		}
		// Status is a subresource; update it explicitly after CreateOrPatch.
		return a.patchGPUStatus(gpuID, func(gpu *tfv1.GPU) error {
			return mutateFn(gpuNode, gpu)
		})
	})
	return err
}

// GetGPU retrieves a GPU resource by UUID
func (a *APIClient) GetGPU(uuid string) (*tfv1.GPU, error) {
	gpu := &tfv1.GPU{}
	if err := a.client.Get(a.ctx, client.ObjectKey{Name: uuid}, gpu); err != nil {
		return nil, fmt.Errorf("failed to get GPU %s: %w", uuid, err)
	}
	return gpu, nil
}

// UpdateGPUStatus writes the node agent's fields of a GPU status. The fields
// the operator's allocator owns (available, runningApps, allocatedPartitions
// and the dynamic isolation lock) are taken from the resource as it is on the
// API server, not from the caller's copy, which may be older.
func (a *APIClient) UpdateGPUStatus(gpu *tfv1.GPU) error {
	return a.patchGPUStatus(gpu.Name, func(live *tfv1.GPU) error {
		status := gpu.Status.DeepCopy()
		status.Available = live.Status.Available
		status.RunningApps = live.Status.RunningApps
		status.AllocatedPartitions = live.Status.AllocatedPartitions
		status.ActiveIsolationMode = live.Status.ActiveIsolationMode
		status.DynamicIsolationConflict = live.Status.DynamicIsolationConflict
		live.Status = *status
		return nil
	})
}

// patchGPUStatus reads the GPU, lets mutate set the status the node agent
// wants on that fresh copy, and writes the difference. The write carries the
// resource version it was computed from, so a status the allocator publishes
// in between is not overwritten: the write conflicts and is recomputed.
func (a *APIClient) patchGPUStatus(name string, mutate func(gpu *tfv1.GPU) error) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		live := &tfv1.GPU{}
		if err := a.client.Get(a.ctx, client.ObjectKey{Name: name}, live); err != nil {
			return err
		}
		desired := live.DeepCopy()
		if err := mutate(desired); err != nil {
			return err
		}
		settleGPUAvailable(&live.Status, &desired.Status)
		if equality.Semantic.DeepEqual(live.Status, desired.Status) {
			return nil
		}

		patch := client.MergeFromWithOptions(live.DeepCopy(), client.MergeFromWithOptimisticLock{})
		live.Status = desired.Status
		return a.client.Status().Patch(a.ctx, live, patch)
	})
}

// settleGPUAvailable sets desired.Available for the capacity in desired, given
// the status currently on the API server.
//
// status.available belongs to the operator's allocator: it knows the
// allocations and republishes the figure whenever it changes. The node agent
// writes it in two cases only, and with the allocator's rule (capacity minus
// what is allocated, never below zero), so that the two never disagree on a
// GPU for longer than one sync:
//
//   - nothing has been published yet, or mutate reset it: available is the
//     capacity;
//   - it publishes a different capacity (agent upgrade, corrected hardware
//     table): what the resource records as allocated, old capacity minus old
//     available, stays allocated against the new capacity. Leaving the old
//     figure would show available above capacity after a decrease and hide the
//     added capacity after an increase until the allocator's next sync, or for
//     as long as an operator that does not republish on a capacity change is
//     running.
//
// With an unchanged capacity this only brings a figure outside
// [0, capacity] back into range.
func settleGPUAvailable(live, desired *tfv1.GPUStatus) {
	if desired.Capacity == nil {
		return
	}
	if desired.Available == nil {
		desired.Available = desired.Capacity.DeepCopy()
		return
	}

	available := desired.Available.DeepCopy()
	carriedOver := live.Available != nil &&
		live.Available.Tflops.Equal(available.Tflops) &&
		live.Available.Vram.Equal(available.Vram)
	if carriedOver && live.Capacity != nil {
		available.Tflops = availableForCapacity(live.Capacity.Tflops, live.Available.Tflops, desired.Capacity.Tflops)
		available.Vram = availableForCapacity(live.Capacity.Vram, live.Available.Vram, desired.Capacity.Vram)
	} else {
		// Set by mutate (first publication or reset): nothing recorded as
		// allocated, only keep it within the capacity.
		available.Tflops = availableForCapacity(desired.Capacity.Tflops, available.Tflops, desired.Capacity.Tflops)
		available.Vram = availableForCapacity(desired.Capacity.Vram, available.Vram, desired.Capacity.Vram)
	}
	desired.Available = available
}

// availableForCapacity returns capacity - allocated, floored at zero, where
// allocated is what the previous capacity and available record
// (previousCapacity - previousAvailable, itself floored at zero).
func availableForCapacity(previousCapacity, previousAvailable, capacity resource.Quantity) resource.Quantity {
	allocated := previousCapacity.DeepCopy()
	allocated.Sub(previousAvailable)
	if allocated.Sign() <= 0 {
		return capacity.DeepCopy()
	}
	available := capacity.DeepCopy()
	available.Sub(allocated)
	if available.Sign() < 0 {
		return resource.Quantity{}
	}
	return available
}

// UpdateGPUNodeStatus updates the status of a GPUNode resource
func (a *APIClient) UpdateGPUNodeStatus(nodeName string, nodeInfo *api.NodeInfo) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		current := &tfv1.GPUNode{
			ObjectMeta: metav1.ObjectMeta{
				Name: nodeName,
			},
		}
		if err := a.client.Get(a.ctx, client.ObjectKeyFromObject(current), current); err != nil {
			return err
		}
		original := current.DeepCopy()
		patch := client.MergeFrom(original)

		current.Status.TotalTFlops = resource.MustParse(fmt.Sprintf("%f", nodeInfo.TotalTFlops))
		current.Status.TotalVRAM = resource.MustParse(fmt.Sprintf("%d", nodeInfo.TotalVRAMBytes))
		current.Status.TotalGPUs = int32(len(nodeInfo.DeviceIDs))
		current.Status.ManagedGPUs = current.Status.TotalGPUs
		current.Status.ManagedGPUDeviceIDs = nodeInfo.DeviceIDs
		current.Status.NodeInfo = tfv1.GPUNodeInfo{
			RAMSize:      *resource.NewQuantity(nodeInfo.RAMSizeBytes, resource.DecimalSI),
			DataDiskSize: *resource.NewQuantity(nodeInfo.DataDiskBytes, resource.DecimalSI),
		}
		if current.Status.Phase == "" {
			current.Status.Phase = tfv1.TensorFusionGPUNodePhasePending
		}

		if equality.Semantic.DeepEqual(original, current) {
			return nil
		}
		return a.client.Status().Patch(a.ctx, current, patch)
	})
}

// DeleteGPU deletes a GPU resource
func (a *APIClient) DeleteGPU(uuid string) error {
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: uuid,
		},
	}
	if err := a.client.Delete(a.ctx, gpu); err != nil {
		return fmt.Errorf("failed to delete GPU %s: %w", uuid, err)
	}
	return nil
}

// CleanupStaleGPUs cleans up GPU CRs owned by this node whose physical device
// was not enumerated in the current discovery run (e.g. physically removed, or
// removed while the hypervisor was down): idle GPUs are deleted directly, busy
// GPUs are only marked missing so transient NVML/driver hiccups do not wipe
// allocation state, and get deleted by a later run once idle.
func (a *APIClient) CleanupStaleGPUs(gpuNodeName string, aliveDeviceIDs []string) error {
	gpuList := &tfv1.GPUList{}
	if err := a.client.List(a.ctx, gpuList, utils.GPUOwnerSelector(gpuNodeName)); err != nil {
		return fmt.Errorf("list GPUs owned by node %s: %w", gpuNodeName, err)
	}

	alive := make(map[string]struct{}, len(aliveDeviceIDs))
	for _, id := range aliveDeviceIDs {
		alive[strings.ToLower(id)] = struct{}{}
	}

	for i := range gpuList.Items {
		gpu := &gpuList.Items[i]
		if _, ok := alive[strings.ToLower(gpu.Name)]; ok {
			continue
		}
		if err := a.deleteOrMarkMissingGPU(gpu); err != nil {
			return err
		}
	}
	return nil
}

// DeleteOrMarkMissingGPU handles a device that is no longer enumerated: the GPU
// CR is deleted when idle, or marked missing (phase Unknown plus missing-since
// annotation) when workloads still reference it.
func (a *APIClient) DeleteOrMarkMissingGPU(uuid string) error {
	gpu := &tfv1.GPU{}
	if err := a.client.Get(a.ctx, client.ObjectKey{Name: uuid}, gpu); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("failed to get GPU %s: %w", uuid, err)
	}
	return a.deleteOrMarkMissingGPU(gpu)
}

func (a *APIClient) deleteOrMarkMissingGPU(gpu *tfv1.GPU) error {
	// Cards are always drained before being physically pulled, so an idle
	// missing GPU has nothing to lose (Available == Capacity): delete it
	// directly. Even a false positive is harmless, the next discovery run
	// recreates it identically.
	if len(gpu.Status.RunningApps) == 0 && len(gpu.Status.AllocatedPartitions) == 0 {
		klog.Infof("GPU device not enumerated and has no running apps, deleting directly: %s", gpu.Name)
		if err := a.client.Delete(a.ctx, gpu); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("delete missing idle GPU %s: %w", gpu.Name, err)
		}
		return nil
	}

	// A missing GPU still referenced by running apps: deleting now would reset
	// allocation accounting on recreate (transient NVML/driver hiccups), so
	// only mark it. Once its workloads are gone, the next discovery run
	// deletes it via the idle path above.
	// Keep the first missing-since timestamp for observability.
	if _, marked := gpu.Annotations[constants.GPUMissingSinceAnnotationKey]; !marked {
		patch := client.MergeFrom(gpu.DeepCopy())
		if gpu.Annotations == nil {
			gpu.Annotations = map[string]string{}
		}
		gpu.Annotations[constants.GPUMissingSinceAnnotationKey] = time.Now().Format(time.RFC3339)
		if err := a.client.Patch(a.ctx, gpu, patch); err != nil {
			return fmt.Errorf("mark GPU %s missing: %w", gpu.Name, err)
		}
	}

	if gpu.Status.Phase != tfv1.TensorFusionGPUPhaseUnknown {
		patch := client.MergeFrom(gpu.DeepCopy())
		gpu.Status.Phase = tfv1.TensorFusionGPUPhaseUnknown
		if err := a.client.Status().Patch(a.ctx, gpu, patch); err != nil {
			return fmt.Errorf("update phase of missing GPU %s: %w", gpu.Name, err)
		}
	}

	klog.Infof("GPU device not enumerated in this discovery run but still has running apps, "+
		"marked as missing, it will be deleted once idle in a future discovery run unless it recovers: "+
		"uuid=%s missingSince=%s", gpu.Name, gpu.Annotations[constants.GPUMissingSinceAnnotationKey])
	return nil
}
