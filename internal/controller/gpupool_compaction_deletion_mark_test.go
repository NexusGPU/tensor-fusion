package controller

import (
	"context"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

// An AutoSelect node marked should-delete while empty must lose the mark once
// it hosts a TF worker again; otherwise defrag keeps excluding a busy node.
func TestCheckNodeCompaction_ClearsDeletionMarkWhenNodeRegainsWorker(t *testing.T) {
	ctx := context.Background()
	pool := newDefragTestPool()
	pool.Spec.NodeManagerConfig.ProvisioningMode = tfv1.ProvisioningModeAutoSelect
	gpuNode := newDefragGPUNode(testDefragNodeA)
	gpuNode.CreationTimestamp = metav1.NewTime(time.Now().Add(-time.Hour))
	gpuNode.Status.TotalTFlops = resource.MustParse("400")
	gpuNode.Status.TotalVRAM = resource.MustParse("40Gi")
	gpuNames := []string{"gpu-0", "gpu-1", "gpu-2", "gpu-3"}
	objects := make([]ctrlclient.Object, 0, 3+len(gpuNames))
	objects = append(objects, pool, gpuNode, &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: testDefragNodeA}})
	for _, name := range gpuNames {
		objects = append(objects, gpuWithUsageOnNode(testDefragNodeA+"-"+name, testDefragNodeA, "100", "10Gi"))
	}

	r, c := newDefragControllerTestReconciler(t, objects...)
	primeAllocatorWorkerStore(t, r)

	hasMark := func() bool {
		t.Helper()
		node := &corev1.Node{}
		if err := c.Get(ctx, ctrlclient.ObjectKey{Name: testDefragNodeA}, node); err != nil {
			t.Fatalf("get node: %v", err)
		}
		return node.Labels[constants.NodeDeletionMark] == constants.TrueStringValue
	}

	if err := r.checkNodeCompaction(ctx, pool); err != nil {
		t.Fatalf("first compaction: %v", err)
	}
	if !hasMark() {
		t.Fatalf("empty node should be marked should-delete")
	}

	// The scheduler ignores should-delete, so a worker can land on the node.
	if err := c.Create(ctx, newDefragWorkerPod("worker-a", testDefragNodeA, time.Now().Add(-time.Hour))); err != nil {
		t.Fatalf("create worker: %v", err)
	}
	primeAllocatorWorkerStore(t, r)

	if err := r.checkNodeCompaction(ctx, pool); err != nil {
		t.Fatalf("second compaction: %v", err)
	}
	if hasMark() {
		t.Fatalf("node hosting a TF worker still carries should-delete")
	}
}
