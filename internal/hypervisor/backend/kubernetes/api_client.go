package kubernetes

import (
	"context"
	"fmt"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/retry"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/apiutil"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

const (
	// bytesPerMiB is the number of bytes in a MiB
	bytesPerMiB = 1024 * 1024
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
func (a *APIClient) CreateOrUpdateGPU(gpuNode *tfv1.GPUNode, info GPUInfo) (*tfv1.GPU, error) {
	if len(gpuNode.OwnerReferences) == 0 {
		return nil, fmt.Errorf("GPUNode %s has no owner references", gpuNode.Name)
	}

	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: info.UUID,
		},
	}

	// Create or update GPU metadata
	if err := retry.OnError(wait.Backoff{
		Steps:    10,
		Duration: time.Second,
		Factor:   1.0,
		Jitter:   0.1,
	}, func(err error) bool {
		return true // Retry on all errors
	}, func() error {
		_, err := controllerutil.CreateOrUpdate(a.ctx, a.client, gpu, func() error {
			gpu.Labels = map[string]string{
				constants.LabelKeyOwner: gpuNode.Name,
				constants.GpuPoolKey:    gpuNode.OwnerReferences[0].Name,
			}
			gpu.Annotations = map[string]string{
				constants.LastSyncTimeAnnotationKey: time.Now().Format(time.RFC3339),
			}

			if !metav1.IsControlledBy(gpu, gpuNode) {
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
			return nil
		})
		return err
	}); err != nil {
		return nil, fmt.Errorf("failed to create or update GPU %s: %w", info.UUID, err)
	}

	// Update GPU status with retry on conflict
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		if err := a.client.Get(a.ctx, client.ObjectKey{Name: info.UUID}, gpu); err != nil {
			return err
		}

		patch := client.MergeFrom(gpu.DeepCopy())
		a.setGPUStatus(gpu, info)
		return a.client.Status().Patch(a.ctx, gpu, patch)
	}); err != nil {
		return nil, fmt.Errorf("failed to update GPU %s status: %w", info.UUID, err)
	}

	return gpu, nil
}

// setGPUStatus sets the GPU status fields from GPUInfo
func (a *APIClient) setGPUStatus(gpu *tfv1.GPU, info GPUInfo) {
	gpu.Status.Capacity = &tfv1.Resource{
		Vram:   resource.MustParse(fmt.Sprintf("%dMi", info.VRAMBytes/bytesPerMiB)),
		Tflops: info.TFlops,
	}
	gpu.Status.UUID = info.UUID
	gpu.Status.GPUModel = info.DeviceName
	gpu.Status.Index = ptr.To(info.Index)
	gpu.Status.Vendor = info.Vendor
	gpu.Status.IsolationMode = info.IsolationMode
	gpu.Status.NUMANode = ptr.To(info.NUMANodeID)
	gpu.Status.NodeSelector = map[string]string{
		constants.KubernetesHostNameLabel: info.NodeName,
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
}

// GetGPU retrieves a GPU resource by UUID
func (a *APIClient) GetGPU(uuid string) (*tfv1.GPU, error) {
	gpu := &tfv1.GPU{}
	if err := a.client.Get(a.ctx, client.ObjectKey{Name: uuid}, gpu); err != nil {
		return nil, fmt.Errorf("failed to get GPU %s: %w", uuid, err)
	}
	return gpu, nil
}

// ListGPUs lists all GPU resources
func (a *APIClient) ListGPUs() (*tfv1.GPUList, error) {
	gpuList := &tfv1.GPUList{}
	if err := a.client.List(a.ctx, gpuList); err != nil {
		return nil, fmt.Errorf("failed to list GPUs: %w", err)
	}
	return gpuList, nil
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

// patchGPUStatus patches a specific GPU status field using a function
func (a *APIClient) patchGPUStatus(uuid string, updateFn func(*tfv1.GPU)) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		gpu, err := a.GetGPU(uuid)
		if err != nil {
			return err
		}

		patch := client.MergeFrom(gpu.DeepCopy())
		updateFn(gpu)
		return a.client.Status().Patch(a.ctx, gpu, patch)
	})
}

// UpdateGPUAvailableResources updates the available resources of a GPU
func (a *APIClient) UpdateGPUAvailableResources(uuid string, available *tfv1.Resource) error {
	return a.patchGPUStatus(uuid, func(gpu *tfv1.GPU) {
		gpu.Status.Available = available
	})
}

// UpdateGPUPhase updates the phase of a GPU
func (a *APIClient) UpdateGPUPhase(uuid string, phase tfv1.TensorFusionGPUPhase) error {
	return a.patchGPUStatus(uuid, func(gpu *tfv1.GPU) {
		gpu.Status.Phase = phase
	})
}

// GetGPUNode retrieves a GPUNode resource by name
func (a *APIClient) GetGPUNode(name string) (*tfv1.GPUNode, error) {
	gpuNode := &tfv1.GPUNode{}
	if err := a.client.Get(a.ctx, client.ObjectKey{Name: name}, gpuNode); err != nil {
		return nil, fmt.Errorf("failed to get GPUNode %s: %w", name, err)
	}
	return gpuNode, nil
}

// UpdateGPUNodeStatus updates the status of a GPUNode resource
func (a *APIClient) UpdateGPUNodeStatus(
	gpuNode *tfv1.GPUNode,
	totalTFlops, totalVRAM resource.Quantity,
	totalGPUs int32,
	deviceIDs []string,
) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		current := &tfv1.GPUNode{}
		if err := a.client.Get(a.ctx, client.ObjectKeyFromObject(gpuNode), current); err != nil {
			return err
		}

		patch := client.MergeFrom(current.DeepCopy())
		a.updateGPUNodeStatus(&current.Status, totalTFlops, totalVRAM, totalGPUs, deviceIDs)
		return a.client.Status().Patch(a.ctx, current, patch)
	})
}

// updateGPUNodeStatus updates GPUNode status fields
func (a *APIClient) updateGPUNodeStatus(
	status *tfv1.GPUNodeStatus,
	totalTFlops, totalVRAM resource.Quantity,
	totalGPUs int32,
	deviceIDs []string,
) {
	status.TotalTFlops = totalTFlops
	status.TotalVRAM = totalVRAM
	status.TotalGPUs = totalGPUs
	status.ManagedGPUs = totalGPUs
	status.ManagedGPUDeviceIDs = deviceIDs

	if status.Phase == "" {
		status.Phase = tfv1.TensorFusionGPUNodePhasePending
	}
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
