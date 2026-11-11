package expander

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator/filter"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// A node whose GPUs native Pods hold, or that are not Running, is still a
// valid template: the node expansion would create is new and idle. The
// copies must not inherit the template GPUs' usedBy or phase, or PhaseFilter
// rejects every copy and expansion fails with "no potential GPU nodes can fit".
func TestPrepareNewNodesForScheduleAttemptResetsTemplateGPUState(t *testing.T) {
	utils.SetProgressiveMigration(true)
	defer utils.SetProgressiveMigration(false)

	capacity := &tfv1.Resource{Tflops: resource.MustParse("125"), Vram: resource.MustParse("24Gi")}
	template := map[string]*tfv1.GPU{
		"native": {
			ObjectMeta: metav1.ObjectMeta{Name: "gpu-native"},
			Status: tfv1.GPUStatus{
				Phase:       tfv1.TensorFusionGPUPhaseRunning,
				UsedBy:      tfv1.UsedByNvidiaDevicePlugin,
				Capacity:    capacity,
				Available:   &tfv1.Resource{},
				RunningApps: []*tfv1.RunningAppDetail{{Name: "app", Namespace: "ns"}},
			},
		},
		"pending": {
			ObjectMeta: metav1.ObjectMeta{Name: "gpu-pending"},
			Status: tfv1.GPUStatus{
				Phase:     tfv1.TensorFusionGPUPhasePending,
				UsedBy:    tfv1.UsedByTensorFusion,
				Capacity:  capacity,
				Available: &tfv1.Resource{},
			},
		},
	}
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "template", Labels: map[string]string{}}}

	newNode, gpus := (&NodeExpander{}).prepareNewNodesForScheduleAttempt(node, template)

	if len(gpus) != 2 {
		t.Fatalf("got %d GPUs, want 2", len(gpus))
	}
	for _, gpu := range gpus {
		if gpu.Status.UsedBy != tfv1.UsedByTensorFusion || gpu.Status.Phase != tfv1.TensorFusionGPUPhaseRunning {
			t.Errorf("GPU %s: usedBy=%q phase=%q, want tensor-fusion/Running", gpu.Name, gpu.Status.UsedBy, gpu.Status.Phase)
		}
		if gpu.Status.Available.Vram.Cmp(capacity.Vram) != 0 || len(gpu.Status.RunningApps) != 0 {
			t.Errorf("GPU %s is not idle: available=%v runningApps=%d", gpu.Name, gpu.Status.Available, len(gpu.Status.RunningApps))
		}
	}
	passed, err := filter.NewPhaseFilter(tfv1.TensorFusionGPUPhaseRunning).Filter(context.Background(), tfv1.NameNamespace{}, gpus)
	if err != nil || len(passed) != 2 {
		t.Fatalf("PhaseFilter kept %d of 2 copies (err=%v)", len(passed), err)
	}

	// The template GPUs are left untouched.
	if template["native"].Status.UsedBy != tfv1.UsedByNvidiaDevicePlugin || template["pending"].Status.Phase != tfv1.TensorFusionGPUPhasePending {
		t.Error("template GPUs were modified")
	}
	if newNode.Name == node.Name {
		t.Error("the new node reuses the template node's name")
	}
}
