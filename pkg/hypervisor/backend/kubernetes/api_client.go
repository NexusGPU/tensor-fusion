package kubernetes

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
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
}

// APIClient provides CRUD operations for GPU resources
type APIClient struct {
	client client.Client
	ctx    context.Context

	metadataOnce   sync.Once
	metadataErr    error
	metadata       []tfv1.HardwareModelInfo
	metadataLoaded bool
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
		var desiredStatus tfv1.GPUStatus
		_, err := controllerutil.CreateOrPatch(a.ctx, a.client, gpu, func() error {
			if err := mutateFn(gpuNode, gpu); err != nil {
				return err
			}
			// Capture desired status before CreateOrPatch overwrites it with server response.
			desiredStatus = gpu.Status
			return nil
		})
		if err != nil {
			return err
		}
		gpu.Status = desiredStatus
		// Status is a subresource; update it explicitly after CreateOrPatch.
		if err := a.UpdateGPUStatus(gpu); err != nil {
			return err
		}
		return nil
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

// UpdateGPUStatus updates the status of a GPU resource using merge patch
func (a *APIClient) UpdateGPUStatus(gpu *tfv1.GPU) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		current := &tfv1.GPU{}
		if err := a.client.Get(a.ctx, client.ObjectKeyFromObject(gpu), current); err != nil {
			return err
		}

		patch := client.MergeFrom(current.DeepCopy())
		current.Status = gpu.Status
		return a.client.Status().Patch(a.ctx, current, patch)
	})
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

// ResolveDeviceFp16TFlops tries to find FP16 TFLOPS from provider hardware metadata env (case-insensitive).
func (a *APIClient) ResolveDeviceFp16TFlops(vendor, model string) (resource.Quantity, bool, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return resource.Quantity{}, false, nil
	}

	metadata, err := a.getHardwareMetadataFromEnv()
	if err != nil {
		return resource.Quantity{}, false, err
	}
	if qty, found := matchFp16TFlops(model, vendor, metadata); found {
		return qty, true, nil
	}
	return resource.Quantity{}, false, nil
}

func (a *APIClient) getHardwareMetadataFromEnv() ([]tfv1.HardwareModelInfo, error) {
	a.metadataOnce.Do(func() {
		raw := strings.TrimSpace(os.Getenv(constants.TFProviderHardwareMetadataEnv))
		if raw == "" {
			a.metadataErr = fmt.Errorf("%s is not set", constants.TFProviderHardwareMetadataEnv)
			return
		}
		if err := json.Unmarshal([]byte(raw), &a.metadata); err != nil {
			a.metadataErr = err
			return
		}
		a.metadataLoaded = true
	})
	if a.metadataErr != nil {
		return nil, a.metadataErr
	}
	if !a.metadataLoaded {
		return nil, fmt.Errorf("%s is not set", constants.TFProviderHardwareMetadataEnv)
	}
	return a.metadata, nil
}

func matchFp16TFlops(model, vendor string, metadata []tfv1.HardwareModelInfo) (resource.Quantity, bool) {
	// Devices report different name shapes depending on the runtime path:
	//   * NVML returns "NVIDIA A100-SXM4-80GB" (vendor + full marketing name).
	//   * Some accelerators only fill the short alias (e.g. "A100_SXM_80G").
	// ProviderConfig.hardwareMetadata stores both `model` (short alias) and
	// `fullModelName` (the NVML-style string). Try both so the fp16TFlops
	// fallback fires regardless of which side reports which — important on
	// MIG parent devices where NVML/CUDA can't report SM count and TFlops
	// would otherwise be computed as 0.
	modelsToMatch := []string{model}
	if vendor != "" && len(model) >= len(vendor) {
		if strings.EqualFold(model[:len(vendor)], vendor) {
			trimmed := strings.TrimSpace(model[len(vendor):])
			if trimmed != "" {
				modelsToMatch = append(modelsToMatch, trimmed)
			}
		}
	}
	for _, hw := range metadata {
		for _, candidate := range modelsToMatch {
			if strings.EqualFold(hw.Model, candidate) || strings.EqualFold(hw.FullModelName, candidate) {
				if hw.Fp16TFlops.IsZero() {
					return resource.Quantity{}, false
				}
				return hw.Fp16TFlops.DeepCopy(), true
			}
		}
	}
	return resource.Quantity{}, false
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
	if err := a.client.List(a.ctx, gpuList,
		client.MatchingLabels{constants.LabelKeyOwner: gpuNodeName}); err != nil {
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
