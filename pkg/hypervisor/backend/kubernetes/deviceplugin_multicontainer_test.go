package kubernetes

import (
	"context"
	"errors"
	"fmt"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

type stubDeviceController struct {
	framework.DeviceController
	devices map[string]*api.DeviceInfo
}

func (s *stubDeviceController) GetDevice(uuid string) (*api.DeviceInfo, bool) {
	d, ok := s.devices[uuid]
	return d, ok
}

func (s *stubDeviceController) GetVendorMountLibs() ([]*api.Mount, error) { return nil, nil }

// multiContainerFixture is a Pod on 4 GPUs whose containers "avatar" and
// "image" ask for 1 and 3 of them, with the index of the incident that
// motivated this test (index_4 with 4 dummy devices, Pod index 36).
type multiContainerFixture struct {
	dp *DevicePlugin
	// kubeletAllocated is what the fake kubelet reports as already allocated.
	kubeletAllocated map[string][]string
	kubeletErr       error
}

func newMultiContainerFixture(containerGPUs string) *multiContainerFixture {
	const resourceNameIndex, quantity = 4, 4
	resourceName := fmt.Sprintf("%s%s%x", constants.PodIndexAnnotation, constants.PodIndexDelimiter, resourceNameIndex)
	gpus := []string{"gpu-a", "gpu-b", "gpu-c", "gpu-d"}

	devices := make(map[string]*api.DeviceInfo, len(gpus))
	for i, id := range gpus {
		node := fmt.Sprintf("/dev/nvidia%d", i)
		devices[id] = &api.DeviceInfo{
			UUID:       "GPU-" + id[len("gpu-"):],
			Vendor:     constants.AcceleratorVendorNvidia,
			Index:      int32(i),
			DeviceNode: map[string]string{node: node},
		}
	}
	allocator := worker.NewAllocationController(&stubDeviceController{devices: devices})

	claim := corev1.ResourceRequirements{Limits: corev1.ResourceList{
		corev1.ResourceName(resourceName): resource.MustParse(fmt.Sprint(quantity)),
	}}
	annotations := map[string]string{}
	if containerGPUs != "" {
		annotations[constants.ContainerGPUsAnnotation] = containerGPUs
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "imagegenerate", Namespace: "default", UID: "pod-uid", Annotations: annotations},
		Spec: corev1.PodSpec{Containers: []corev1.Container{
			{Name: "image", Resources: claim},
			{Name: "no-gpu"},
			{Name: "avatar", Resources: claim},
		}},
	}

	f := &multiContainerFixture{kubeletAllocated: map[string][]string{}}
	f.dp = &DevicePlugin{
		ctx:               context.Background(),
		resourceNameIndex: resourceNameIndex,
		resourceName:      resourceName,
		kubeletClient: &PodCacheManager{
			ctx:       context.Background(),
			cachedPod: map[string]*corev1.Pod{string(pod.UID): pod},
			indexToWorkerInfo: map[int]*api.WorkerInfo{
				resourceNameIndex*constants.IndexModLength + quantity: {
					WorkerUID:        string(pod.UID),
					WorkerName:       pod.Name,
					Namespace:        pod.Namespace,
					Status:           api.WorkerStatusDeviceAllocating,
					IsolationMode:    tfv1.IsolationModeShared,
					AllocatedDevices: gpus,
					Annotations:      annotations,
				},
			},
			indexSubscribers: map[int]map[*workerInfoSubscriber]struct{}{},
		},
		allocateWorker: func(_ context.Context, w *api.WorkerInfo) (*api.WorkerAllocation, error) {
			return allocator.AllocateWorkerDevices(w)
		},
		allocatedContainerDevices: func(_ context.Context, namespace, name string) (map[string][]string, error) {
			if namespace != pod.Namespace || name != pod.Name {
				return nil, fmt.Errorf("unexpected pod %s/%s", namespace, name)
			}
			return f.kubeletAllocated, f.kubeletErr
		},
	}
	return f
}

// allocate plays kubelet allocating dummy devices ids to one container.
func (f *multiContainerFixture) allocate(ids ...string) (*pluginapi.ContainerAllocateResponse, error) {
	resp, err := f.dp.Allocate(context.Background(), &pluginapi.AllocateRequest{
		ContainerRequests: []*pluginapi.ContainerAllocateRequest{{DevicesIds: ids}},
	})
	if err != nil {
		return nil, err
	}
	return resp.ContainerResponses[0], nil
}

func deviceNodes(resp *pluginapi.ContainerAllocateResponse) []string {
	nodes := make([]string, 0, len(resp.Devices))
	for _, d := range resp.Devices {
		nodes = append(nodes, d.HostPath)
	}
	return nodes
}

const avatarAndImageGPUs = `{"avatar":["gpu-b"],"image":["gpu-a","gpu-c","gpu-d"]}`

func TestDevicePluginAllocate_ContainerSeesOnlyItsOwnGPUs(t *testing.T) {
	f := newMultiContainerFixture(avatarAndImageGPUs)

	// kubelet allocates in spec order: "image" first.
	image, err := f.allocate("4-1", "4-2", "4-3", "4-4")
	require.NoError(t, err)
	assert.Equal(t, "GPU-a,GPU-c,GPU-d", image.Envs[constants.NvidiaVisibleAllDeviceEnv])
	assert.ElementsMatch(t, []string{"/dev/nvidia0", "/dev/nvidia2", "/dev/nvidia3"}, deviceNodes(image))

	f.kubeletAllocated["image"] = []string{"4-1", "4-2", "4-3", "4-4"}
	avatar, err := f.allocate("4-5", "4-6", "4-7", "4-8")
	require.NoError(t, err)
	assert.Equal(t, "GPU-b", avatar.Envs[constants.NvidiaVisibleAllDeviceEnv])
	assert.Equal(t, []string{"/dev/nvidia1"}, deviceNodes(avatar))
}

// kubelet retries only the container whose allocation failed; the ones before
// it keep their devices.
func TestDevicePluginAllocate_RetryResolvesTheSameContainer(t *testing.T) {
	f := newMultiContainerFixture(avatarAndImageGPUs)
	f.kubeletAllocated["image"] = []string{"4-1", "4-2", "4-3", "4-4"}

	for range 2 {
		avatar, err := f.allocate("4-5", "4-6", "4-7", "4-8")
		require.NoError(t, err)
		assert.Equal(t, "GPU-b", avatar.Envs[constants.NvidiaVisibleAllDeviceEnv])
	}

	// A repeated call for devices kubelet already recorded is for their owner.
	image, err := f.allocate("4-4", "4-3", "4-2", "4-1")
	require.NoError(t, err)
	assert.Equal(t, "GPU-a,GPU-c,GPU-d", image.Envs[constants.NvidiaVisibleAllDeviceEnv])
}

func TestDevicePluginAllocate_FailsRatherThanGuessTheContainer(t *testing.T) {
	f := newMultiContainerFixture(avatarAndImageGPUs)
	f.kubeletErr = errors.New("kubelet unavailable")
	_, err := f.allocate("4-1", "4-2", "4-3", "4-4")
	require.ErrorContains(t, err, "kubelet unavailable")

	f = newMultiContainerFixture(avatarAndImageGPUs)
	f.kubeletAllocated["image"] = []string{"4-1", "4-2", "4-3", "4-4"}
	f.kubeletAllocated["avatar"] = []string{"4-5", "4-6", "4-7", "4-8"}
	_, err = f.allocate("4-9", "4-10", "4-11", "4-12")
	require.ErrorContains(t, err, "already allocated")

	f = newMultiContainerFixture(`{"avatar":["gpu-x"],"image":["gpu-a","gpu-c","gpu-d"]}`)
	f.kubeletAllocated["image"] = []string{"4-1", "4-2", "4-3", "4-4"}
	_, err = f.allocate("4-5", "4-6", "4-7", "4-8")
	require.ErrorContains(t, err, "not all among")
}

// Without a per-container split, every container sees the Pod's GPUs and
// kubelet is not asked anything.
func TestDevicePluginAllocate_WithoutContainerGPUsExposesPodGPUs(t *testing.T) {
	f := newMultiContainerFixture("")
	f.kubeletErr = errors.New("must not be called")

	resp, err := f.allocate("4-1", "4-2", "4-3", "4-4")
	require.NoError(t, err)
	assert.Equal(t, "GPU-a,GPU-b,GPU-c,GPU-d", resp.Envs[constants.NvidiaVisibleAllDeviceEnv])
	assert.Len(t, resp.Devices, 4)
}

// A container that claims the index but is not in the split, such as the
// sidecar worker, keeps the Pod's GPUs.
func TestDevicePluginAllocate_ContainerOutsideTheSplitSeesPodGPUs(t *testing.T) {
	f := newMultiContainerFixture(`{"avatar":["gpu-b"]}`)

	resp, err := f.allocate("4-1", "4-2", "4-3", "4-4") // "image"
	require.NoError(t, err)
	assert.Equal(t, "GPU-a,GPU-b,GPU-c,GPU-d", resp.Envs[constants.NvidiaVisibleAllDeviceEnv])
}
