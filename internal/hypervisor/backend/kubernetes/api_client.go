package kubernetes

import (
	"context"
	"fmt"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

var (
	scheme = runtime.NewScheme()
)

func init() {
	utilruntime.Must(tfv1.AddToScheme(scheme))
}

// sanitizeGPUName converts a GPU UUID to a Kubernetes RFC 1123 compliant name.
// RFC 1123 requires: lowercase alphanumeric characters, '-' or '.', and must
// start and end with an alphanumeric character.
// For PCI-based identifiers (e.g., AMD GPUs), the nodeName is prepended to ensure
// global uniqueness since PCI addresses are only unique within a node.
func sanitizeGPUName(nodeName, uuid string) string {
	// Convert to lowercase
	name := strings.ToLower(uuid)
	// Replace colons with hyphens (e.g., AMD-GPU-0000:05:00.0 -> amd-gpu-0000-05-00-0)
	name = strings.ReplaceAll(name, ":", "-")
	// Replace dots that aren't at the end of segments with hyphens
	// This handles the PCI function separator (e.g., 00.0 -> 00-0)
	name = strings.ReplaceAll(name, ".", "-")

	// Prepend node name to ensure global uniqueness for PCI addresses
	// Node names are already RFC 1123 compliant
	return nodeName + "-" + name
}

// APIClient provides CRUD operations for GPU resources
type APIClient struct {
	client   client.Client
	ctx      context.Context
	nodeName string
}

// NewAPIClient creates a new API client instance with an existing client
func NewAPIClient(ctx context.Context, k8sClient client.Client, nodeName string) *APIClient {
	return &APIClient{
		client:   k8sClient,
		ctx:      ctx,
		nodeName: nodeName,
	}
}

// NewAPIClientFromConfig creates a new API client instance from a rest.Config
func NewAPIClientFromConfig(ctx context.Context, restConfig *rest.Config, nodeName string) (*APIClient, error) {
	k8sClient, err := client.New(restConfig, client.Options{
		Scheme: scheme,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Kubernetes client: %w", err)
	}

	return &APIClient{
		client:   k8sClient,
		ctx:      ctx,
		nodeName: nodeName,
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

	// Sanitize the GPU ID for use as a Kubernetes resource name
	sanitizedName := sanitizeGPUName(a.nodeName, gpuID)

	// Create or update GPU metadata and capture the status we want to set
	var desiredStatus tfv1.GPUStatus
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
				Name: sanitizedName,
			},
		}
		_, err := controllerutil.CreateOrPatch(a.ctx, a.client, gpu, func() error {
			if err := mutateFn(gpuNode, gpu); err != nil {
				return err
			}
			// Capture the status before CreateOrPatch potentially clears it
			// Use DeepCopy on the whole GPU object to properly copy all fields including pointers
			gpuCopy := gpu.DeepCopy()
			desiredStatus = gpuCopy.Status
			return nil
		})
		return err
	})
	if err != nil {
		return err
	}

	// Update status separately with the captured status (status is a subresource in Kubernetes)
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: sanitizedName,
		},
	}
	gpu.Status = desiredStatus
	return a.UpdateGPUStatus(gpu)
}

// GetGPU retrieves a GPU resource by UUID
func (a *APIClient) GetGPU(uuid string) (*tfv1.GPU, error) {
	sanitizedName := sanitizeGPUName(a.nodeName, uuid)
	gpu := &tfv1.GPU{}
	if err := a.client.Get(a.ctx, client.ObjectKey{Name: sanitizedName}, gpu); err != nil {
		return nil, fmt.Errorf("failed to get GPU %s: %w", sanitizedName, err)
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

// DeleteGPU deletes a GPU resource
func (a *APIClient) DeleteGPU(uuid string) error {
	sanitizedName := sanitizeGPUName(a.nodeName, uuid)
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: sanitizedName,
		},
	}
	if err := a.client.Delete(a.ctx, gpu); err != nil {
		return fmt.Errorf("failed to delete GPU %s: %w", sanitizedName, err)
	}
	return nil
}
