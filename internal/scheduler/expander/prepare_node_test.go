package expander

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator/filter"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// A node whose GPUs native Pods hold, that are not Running, or that carry
// partitions is still a valid template: the node expansion would create is
// new and idle. The copies must not inherit the template GPUs' usedBy, phase
// or partitions, or PhaseFilter / SharedWholeGPUFilter rejects every copy and
// expansion fails with "no potential GPU nodes can fit".
func TestPrepareNewNodesForScheduleAttemptResetsTemplateGPUState(t *testing.T) {
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
		"partitioned": {
			ObjectMeta: metav1.ObjectMeta{Name: "gpu-partitioned"},
			Status: tfv1.GPUStatus{
				Phase:               tfv1.TensorFusionGPUPhaseRunning,
				UsedBy:              tfv1.UsedByTensorFusion,
				Capacity:            capacity,
				Available:           &tfv1.Resource{},
				AllocatedPartitions: map[string]tfv1.AllocatedPartition{"pod-uid": {TemplateID: "1g.6gb", PodUID: "pod-uid"}},
			},
		},
	}
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "template", Labels: map[string]string{}}}
	expander := &NodeExpander{allocator: &gpuallocator.GpuAllocator{}}

	newNode, gpus := expander.prepareNewNodesForScheduleAttempt(node, template)

	if len(gpus) != len(template) {
		t.Fatalf("got %d GPUs, want %d", len(gpus), len(template))
	}
	for _, gpu := range gpus {
		if gpu.Status.UsedBy != tfv1.UsedByTensorFusion || gpu.Status.Phase != tfv1.TensorFusionGPUPhaseRunning {
			t.Errorf("GPU %s: usedBy=%q phase=%q, want tensor-fusion/Running", gpu.Name, gpu.Status.UsedBy, gpu.Status.Phase)
		}
		if len(gpu.Status.RunningApps) != 0 || len(gpu.Status.AllocatedPartitions) != 0 {
			t.Errorf("GPU %s is not idle: runningApps=%d allocatedPartitions=%d",
				gpu.Name, len(gpu.Status.RunningApps), len(gpu.Status.AllocatedPartitions))
		}
	}
	ctx := context.Background()
	passed, err := filter.NewPhaseFilter(tfv1.TensorFusionGPUPhaseRunning).Filter(ctx, tfv1.NameNamespace{}, gpus)
	if err != nil || len(passed) != len(template) {
		t.Fatalf("PhaseFilter kept %d of %d copies (err=%v)", len(passed), len(template), err)
	}
	passed, err = filter.NewSharedWholeGPUFilter().Filter(ctx, tfv1.NameNamespace{}, gpus)
	if err != nil || len(passed) != len(template) {
		t.Fatalf("SharedWholeGPUFilter kept %d of %d copies (err=%v)", len(passed), len(template), err)
	}

	// The template GPUs are left untouched.
	if template["native"].Status.UsedBy != tfv1.UsedByNvidiaDevicePlugin ||
		template["pending"].Status.Phase != tfv1.TensorFusionGPUPhasePending ||
		len(template["partitioned"].Status.AllocatedPartitions) != 1 {
		t.Error("template GPUs were modified")
	}
	if newNode.Name == node.Name {
		t.Error("the new node reuses the template node's name")
	}
}
