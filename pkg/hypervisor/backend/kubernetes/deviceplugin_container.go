package kubernetes

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	podresv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
)

const kubeletPodResourcesTimeout = 5 * time.Second

// containerExposure returns the environment variables and device nodes for the
// container that this Allocate call is for.
//
// A worker's allocation covers the whole Pod. When several containers of a Pod
// ask for GPUs of their own, the scheduler records which GPUs belong to which
// container in the container-gpus annotation, and each container must see
// only its share. Without that annotation every container sees the Pod's
// devices, as before.
//
// kubelet does not say which container an Allocate call is for, and all
// containers of a Pod claim the same index. The container is therefore the
// first one, in the order kubelet allocates (init containers, then
// containers), that kubelet has not recorded devices for yet. Any doubt is an
// error, which kubelet retries: a guess could hand a container the GPUs of
// another one.
func (dp *DevicePlugin) containerExposure(
	ctx context.Context, allocation *api.WorkerAllocation, requestedIDs []string,
) (map[string]string, []*api.DeviceSpec, error) {
	info := allocation.WorkerInfo
	if info == nil || info.Annotations[constants.ContainerGPUsAnnotation] == "" {
		return allocation.Envs, allocation.Devices, nil
	}
	var containerGPUs map[string][]string
	if err := json.Unmarshal([]byte(info.Annotations[constants.ContainerGPUsAnnotation]), &containerGPUs); err != nil {
		return nil, nil, fmt.Errorf("invalid %s annotation: %w", constants.ContainerGPUsAnnotation, err)
	}

	pod := dp.kubeletClient.GetPodByUID(info.WorkerUID)
	if pod == nil {
		return nil, nil, fmt.Errorf("pod of worker %s is not in the cache", info.WorkerUID)
	}
	containerName, err := dp.allocatingContainer(ctx, pod, requestedIDs)
	if err != nil {
		return nil, nil, err
	}
	gpuIDs, ok := containerGPUs[containerName]
	if !ok {
		// Not a container the scheduler split GPUs for, e.g. the sidecar worker.
		return allocation.Envs, allocation.Devices, nil
	}

	if len(allocation.DeviceInfos) != len(info.AllocatedDevices) {
		return nil, nil, fmt.Errorf("worker %s has %d devices for %d allocated GPUs",
			info.WorkerUID, len(allocation.DeviceInfos), len(info.AllocatedDevices))
	}
	own := make([]*api.DeviceInfo, 0, len(gpuIDs))
	for i, name := range info.AllocatedDevices {
		if slices.ContainsFunc(gpuIDs, func(id string) bool { return strings.EqualFold(id, name) }) {
			own = append(own, allocation.DeviceInfos[i])
		}
	}
	if len(own) != len(gpuIDs) {
		return nil, nil, fmt.Errorf("container %s of worker %s is assigned GPUs %v, which are not all among the Pod's %v",
			containerName, info.WorkerUID, gpuIDs, info.AllocatedDevices)
	}

	isPartitioned := info.IsolationMode == tfv1.IsolationModePartitioned && info.PartitionTemplateID != ""
	podEnvs, _ := worker.DeviceExposure(allocation.DeviceInfos, isPartitioned)
	ownEnvs, ownDevices := worker.DeviceExposure(own, isPartitioned)
	envs := maps.Clone(allocation.Envs)
	for name := range podEnvs {
		delete(envs, name)
	}
	maps.Copy(envs, ownEnvs)

	klog.Infof("Container %s of pod %s/%s gets GPUs %v out of the Pod's %v",
		containerName, pod.Namespace, pod.Name, gpuIDs, info.AllocatedDevices)
	return envs, ownDevices, nil
}

// allocatingContainer names the container of pod that kubelet is allocating
// requestedIDs of this plugin's resource for.
func (dp *DevicePlugin) allocatingContainer(
	ctx context.Context, pod *corev1.Pod, requestedIDs []string,
) (string, error) {
	var claimants []string
	for _, containers := range [][]corev1.Container{pod.Spec.InitContainers, pod.Spec.Containers} {
		for _, container := range containers {
			if _, claims := container.Resources.Limits[corev1.ResourceName(dp.resourceName)]; claims {
				claimants = append(claimants, container.Name)
			}
		}
	}
	switch len(claimants) {
	case 0:
		return "", fmt.Errorf("no container of pod %s/%s claims %s", pod.Namespace, pod.Name, dp.resourceName)
	case 1:
		return claimants[0], nil
	}

	list := dp.allocatedContainerDevices
	if list == nil {
		list = dp.listAllocatedContainerDevices
	}
	allocated, err := list(ctx, pod.Namespace, pod.Name)
	if err != nil {
		return "", fmt.Errorf("list devices kubelet allocated to pod %s/%s: %w", pod.Namespace, pod.Name, err)
	}
	requested := slices.Sorted(slices.Values(requestedIDs))
	for name, ids := range allocated {
		if slices.Equal(slices.Sorted(slices.Values(ids)), requested) {
			return name, nil
		}
	}
	for _, name := range claimants {
		if len(allocated[name]) == 0 {
			return name, nil
		}
	}
	return "", fmt.Errorf("kubelet already allocated %s to every container of pod %s/%s",
		dp.resourceName, pod.Namespace, pod.Name)
}

// listAllocatedContainerDevices asks kubelet which devices of this plugin's
// resource it has allocated to each container of a Pod.
func (dp *DevicePlugin) listAllocatedContainerDevices(
	ctx context.Context, namespace, name string,
) (map[string][]string, error) {
	conn, err := grpc.NewClient("unix://"+upstreamKubeletPodResourcesSocket,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	defer func() { _ = conn.Close() }()

	ctx, cancel := context.WithTimeout(ctx, kubeletPodResourcesTimeout)
	defer cancel()
	resp, err := podresv1.NewPodResourcesListerClient(conn).List(ctx, &podresv1.ListPodResourcesRequest{})
	if err != nil {
		return nil, err
	}
	allocated := map[string][]string{}
	for _, podResources := range resp.GetPodResources() {
		if podResources.GetNamespace() != namespace || podResources.GetName() != name {
			continue
		}
		for _, container := range podResources.GetContainers() {
			for _, devices := range container.GetDevices() {
				if devices.GetResourceName() == dp.resourceName {
					allocated[container.GetName()] = append(allocated[container.GetName()], devices.GetDeviceIds()...)
				}
			}
		}
	}
	return allocated, nil
}
