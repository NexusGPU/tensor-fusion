package kubernetes

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
	corev1 "k8s.io/api/core/v1"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/checkpoint"
)

func TestWorkerRecoveryUsesVerifiedCheckpointUID(t *testing.T) {
	for _, tc := range []struct {
		name, resource                                            string
		wrongUID, corrupt, noDevices, noResponse, invalidResponse bool
		want                                                      bool
	}{
		{name: "legacy", resource: constants.PodIndexAnnotation, want: true},
		{name: "v2", resource: constants.PodIndexAnnotation + "_0", want: true},
		{name: "different UID with reused name and index", resource: constants.PodIndexAnnotation, wrongUID: true},
		{name: "other device plugin", resource: "nvidia.com/gpu"},
		{name: "invalid checksum", resource: constants.PodIndexAnnotation, corrupt: true},
		{name: "no devices", resource: constants.PodIndexAnnotation, noDevices: true},
		{name: "no successful response", resource: constants.PodIndexAnnotation, noResponse: true},
		{name: "invalid protobuf response", resource: constants.PodIndexAnnotation, invalidResponse: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := newWorkerSyncTestBackend(t)
			pod := createTestPodWithIndex(1)
			pod.Status.Phase = corev1.PodPending
			b.podCacher.onPodAdd(pod)
			response, err := proto.Marshal(&pluginapi.ContainerAllocateResponse{
				Envs: map[string]string{"TF_WORKER_NAME": pod.Name},
			})
			require.NoError(t, err)
			entry := checkpoint.PodDevicesEntry{
				PodUID: string(pod.UID), ContainerName: "worker", ResourceName: tc.resource,
				DeviceIDs: checkpoint.DevicesPerNUMA{-1: {"0"}}, AllocResp: response,
			}
			if tc.wrongUID {
				entry.PodUID = "previous-pod-uid"
			}
			if tc.noDevices {
				entry.DeviceIDs = nil
			}
			if tc.noResponse {
				entry.AllocResp = nil
			}
			if tc.invalidResponse {
				entry.AllocResp = []byte{1}
			}
			state := checkpoint.New([]checkpoint.PodDevicesEntry{entry}, nil)
			data, err := state.MarshalCheckpoint()
			require.NoError(t, err)
			if tc.corrupt {
				// Keep the matching UID intact so only checksum verification
				// can reject this otherwise valid recovery record.
				var document map[string]json.RawMessage
				require.NoError(t, json.Unmarshal(data, &document))
				document["Checksum"] = json.RawMessage("0")
				data, err = json.Marshal(document)
				require.NoError(t, err)
			}
			b.checkpointPath = filepath.Join(t.TempDir(), "kubelet_internal_checkpoint")
			require.NoError(t, os.WriteFile(b.checkpointPath, data, 0600))
			var observed *api.WorkerInfo
			b.reconcileWorkers(framework.WorkerChangeHandler{OnAdd: func(info *api.WorkerInfo) { observed = info }})
			require.NotNil(t, observed)
			require.Equal(t, api.WorkerStatusDeviceAllocating, observed.Status, "recovery does not invent a Running state")
			require.Equal(t, tc.want, observed.AllocationConfirmed)
		})
	}
}

func TestTerminalPodReleasesWorkerWithMalformedAnnotations(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	pod := createTestPodWithIndex(1)
	pod.Status.Phase = corev1.PodRunning
	b.podCacher.onPodAdd(pod)
	handler := framework.WorkerChangeHandler{
		OnAdd: func(info *api.WorkerInfo) {
			_, err := b.allocationController.AllocateWorkerDevices(info)
			require.NoError(t, err)
		},
		OnUpdate: func(_, info *api.WorkerInfo) {
			if info.Status == api.WorkerStatusTerminated {
				require.NoError(t, b.allocationController.DeallocateWorker(info.WorkerUID))
			}
		},
	}
	b.reconcileWorkers(handler)
	failed := pod.DeepCopy()
	failed.Status.Phase = corev1.PodFailed
	failed.Annotations[constants.GpuCountAnnotation] = "not-a-count"
	_, _, err := b.podCacher.extractWorkerInfo(failed)
	require.Error(t, err)
	b.podCacher.onPodUpdate(pod, failed)
	b.reconcileWorkers(handler)
	_, exists := b.allocationController.GetWorkerAllocation(string(pod.UID))
	require.False(t, exists, "terminal state must not depend on parsing resource annotations")
}
