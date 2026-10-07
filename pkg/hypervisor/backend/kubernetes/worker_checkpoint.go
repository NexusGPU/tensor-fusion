package kubernetes

import (
	"os"
	"strings"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"google.golang.org/protobuf/proto"
	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/checkpoint"
)

// checkpointedWorkers reads successful device-plugin allocations by Pod UID.
// Pod names and the synthetic index count can be reused and are not identities.
func (b *KubeletBackend) checkpointedWorkers() map[string]bool {
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
	confirmed := make(map[string]bool)
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
		confirmed[entry.PodUID] = true
	}
	return confirmed
}
