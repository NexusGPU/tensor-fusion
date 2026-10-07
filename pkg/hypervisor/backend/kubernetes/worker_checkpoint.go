package kubernetes

import (
	"os"
	"strings"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"google.golang.org/protobuf/proto"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/checkpoint"
)

type checkpointedWorker struct {
	partitions sets.Set[string]
}

// checkpointedWorkers reads successful device-plugin allocations by Pod UID.
// Pod names and the synthetic index count can be reused and are not identities.
func (b *KubeletBackend) checkpointedWorkers() map[string]*checkpointedWorker {
	if b.checkpointPath == "" {
		return nil
	}
	data, err := os.ReadFile(b.checkpointPath)
	if err != nil {
		klog.V(4).Infof("Worker allocation checkpoint unavailable: %v", err)
		return nil
	}
	state := checkpoint.New(nil, nil)
	if err := state.UnmarshalCheckpoint(data); err != nil {
		klog.Errorf("Failed to read worker allocation checkpoint: %v", err)
		return nil
	}
	if err := state.VerifyChecksum(); err != nil {
		klog.Errorf("Invalid worker allocation checkpoint: %v", err)
		return nil
	}
	entries, _ := state.GetData()
	confirmed := make(map[string]*checkpointedWorker)
	for _, entry := range entries {
		if entry.ResourceName != constants.PodIndexAnnotation &&
			!strings.HasPrefix(entry.ResourceName, constants.PodIndexAnnotation+constants.PodIndexDelimiter) {
			continue
		}
		if entry.PodUID == "" || len(entry.AllocResp) == 0 || entry.DeviceIDs.Devices().Len() == 0 {
			continue
		}
		response := &pluginapi.ContainerAllocateResponse{}
		if err := proto.Unmarshal(entry.AllocResp, response); err != nil {
			klog.Errorf("Invalid device allocation response for Pod UID %s: %v", entry.PodUID, err)
			continue
		}
		worker := confirmed[entry.PodUID]
		if worker == nil {
			worker = &checkpointedWorker{partitions: sets.New[string]()}
			confirmed[entry.PodUID] = worker
		}
		// Device-plugin annotations belong to the runtime container, not the
		// API Pod. Kubelet persists them in this allocation response.
		for _, pair := range strings.Split(response.Annotations[constants.PartitionUUIDsAnnotation], ",") {
			if pair = strings.TrimSpace(pair); pair != "" {
				worker.partitions.Insert(pair)
			}
		}
	}
	return confirmed
}
