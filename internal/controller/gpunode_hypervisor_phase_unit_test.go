package controller

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestGPUNodeReconcileHypervisorDeletionMarksGPUsPending(t *testing.T) {
	for _, tc := range []struct {
		name        string
		nodePhase   tfv1.TensorFusionGPUNodePhase
		terminating bool
	}{
		{"rollout deletes old hypervisor", tfv1.TensorFusionGPUNodePhasePending, false},
		{"rollout waits for terminating hypervisor", tfv1.TensorFusionGPUNodePhasePending, true},
		{"running node has terminating hypervisor", tfv1.TensorFusionGPUNodePhaseRunning, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newHypervisorPhaseFixture(t, tc.nodePhase, tc.terminating)
			ctx := context.Background()
			for range 3 {
				result, err := f.reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.node)})
				require.NoError(t, err)
				require.Equal(t, constants.PendingRequeueDuration, result.RequeueAfter)
				f.assertPendingWithoutDeallocation(t)
				pod := &corev1.Pod{}
				require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(f.pod), pod))
				require.False(t, pod.DeletionTimestamp.IsZero(), "must wait for the existing hypervisor to finish deleting")
			}
		})
	}
}

func TestHypervisorStatusTerminatingPodCannotRemainReady(t *testing.T) {
	f := newHypervisorPhaseFixture(t, tfv1.TensorFusionGPUNodePhaseRunning, true)
	err := f.reconciler.checkStatusAndUpdateVirtualCapacity(context.Background(), f.pod.Name, f.node.DeepCopy(), f.pool, f.coreNode)
	require.NoError(t, err)
	f.assertPendingWithoutDeallocation(t)
}

func TestGPUNodeReconcileHealthyHypervisorRestoresGPUPhases(t *testing.T) {
	f := newHypervisorPhaseFixture(t, tfv1.TensorFusionGPUNodePhasePending, true)
	ctx := context.Background()
	req := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.node)}
	_, err := f.reconciler.Reconcile(ctx, req)
	require.NoError(t, err)
	f.assertPendingWithoutDeallocation(t)

	// Simulate kubelet completing deletion, then a replacement becoming Ready.
	pod := &corev1.Pod{}
	require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(f.pod), pod))
	pod.Finalizers = nil
	require.NoError(t, f.reconciler.Update(ctx, pod))
	replacement := f.pod.DeepCopy()
	replacement.ResourceVersion = ""
	replacement.DeletionTimestamp = nil
	replacement.Finalizers = nil
	replacement.Labels[constants.LabelKeyPodTemplateHash] = utils.HypervisorPodTemplateHash(
		f.pool, f.node.Labels[constants.AcceleratorLabelVendor], tfv1.IsolationModeSoft)
	require.NoError(t, f.reconciler.Create(ctx, replacement))

	_, err = f.reconciler.Reconcile(ctx, req)
	require.NoError(t, err)
	updatedNode := &tfv1.GPUNode{}
	require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(f.node), updatedNode))
	require.Equal(t, tfv1.TensorFusionGPUNodePhaseRunning, updatedNode.Status.Phase)
	require.True(t, updatedNode.Status.AvailableTFlops.Equal(f.gpu.Status.Available.Tflops))
	require.True(t, updatedNode.Status.AvailableVRAM.Equal(f.gpu.Status.Available.Vram))
	for _, original := range []*tfv1.GPU{f.gpu, f.missingGPU} {
		updated := &tfv1.GPU{}
		require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(original), updated))
		require.Equal(t, original.Status.Phase, updated.Status.Phase)
		require.True(t, equality.Semantic.DeepEqual(original.Status, updated.Status), "recovery must preserve allocations and keep missing GPUs Unknown")
	}
}

func TestSyncStatusToGPUDevicesReturnsUpdatedPhases(t *testing.T) {
	f := newHypervisorPhaseFixture(t, tfv1.TensorFusionGPUNodePhasePending, true)
	gpus, err := f.reconciler.syncStatusToGPUDevices(context.Background(), f.node, tfv1.TensorFusionGPUPhasePending)
	require.NoError(t, err)
	require.Len(t, gpus, 2)
	for _, gpu := range gpus {
		if gpu.Name == f.missingGPU.Name {
			require.Equal(t, tfv1.TensorFusionGPUPhaseUnknown, gpu.Status.Phase)
		} else {
			require.Equal(t, tfv1.TensorFusionGPUPhasePending, gpu.Status.Phase, "metrics must receive the updated phase")
		}
	}
}

type hypervisorPhaseFixture struct {
	reconciler *GPUNodeReconciler
	node       *tfv1.GPUNode
	coreNode   *corev1.Node
	pool       *tfv1.GPUPool
	pod        *corev1.Pod
	gpu        *tfv1.GPU
	missingGPU *tfv1.GPU
}

func newHypervisorPhaseFixture(t *testing.T, nodePhase tfv1.TensorFusionGPUNodePhase, terminating bool) *hypervisorPhaseFixture {
	t.Helper()
	const vendor = "HypervisorPhaseTestVendor"
	f := &hypervisorPhaseFixture{}
	f.pool = &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "hypervisor-phase-pool"}, Spec: *config.MockGPUPoolSpec.DeepCopy()}
	f.pool.Spec.CapacityConfig = nil
	f.pool.Spec.NodeManagerConfig = &tfv1.NodeManagerConfig{DefaultVendor: vendor}
	allocatedPod := &tfv1.PodGPUInfo{
		Name: "worker", Namespace: "workloads", UID: "worker-uid",
		Requests: tfv1.Resource{Tflops: resource.MustParse("25"), Vram: resource.MustParse("8Gi")},
	}
	f.node = &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "hypervisor-phase-node", Finalizers: []string{constants.Finalizer},
			Labels: map[string]string{
				constants.GPUNodePoolIdentifierLabelPrefix + f.pool.Name: "true",
				constants.AcceleratorLabelVendor:                         vendor,
			},
		},
		Status: tfv1.GPUNodeStatus{
			Phase: nodePhase, TotalGPUs: 2, ManagedGPUs: 1, TotalGPUPods: 1,
			TotalTFlops: resource.MustParse("100"), TotalVRAM: resource.MustParse("24Gi"),
			AvailableTFlops: resource.MustParse("75"), AvailableVRAM: resource.MustParse("16Gi"),
			VirtualTFlops: resource.MustParse("100"), VirtualVRAM: resource.MustParse("24Gi"),
			AllocatedPods: map[string][]*tfv1.PodGPUInfo{"hypervisor-phase-gpu": {allocatedPod}},
			NodeInfo:      tfv1.GPUNodeInfo{RAMSize: resource.MustParse("64Gi"), DataDiskSize: resource.MustParse("100Gi")},
		},
	}
	f.node.Status.VirtualAvailableTFlops = &f.node.Status.AvailableTFlops
	f.node.Status.VirtualAvailableVRAM = &f.node.Status.AvailableVRAM
	f.coreNode = &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: f.node.Name},
		Status:     corev1.NodeStatus{Conditions: []corev1.NodeCondition{{Type: corev1.NodeReady, Status: corev1.ConditionTrue}}},
	}
	f.pod = &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: utils.BuildHypervisorPodName(f.node.Name), Namespace: utils.CurrentNamespace(),
			Labels:     map[string]string{constants.LabelKeyPodTemplateHash: "old-template"},
			Finalizers: []string{"test.tensor-fusion.ai/hold-deletion"},
		},
		Spec: corev1.PodSpec{NodeName: f.node.Name, Containers: []corev1.Container{{
			Name: constants.TFContainerNameHypervisor,
			Env: []corev1.EnvVar{
				{Name: constants.TFIsolationModeEnv, Value: string(tfv1.IsolationModeSoft)},
				{Name: constants.IsolationModePolicyEnv, Value: string(tfv1.IsolationModePolicyStatic)},
			},
		}}},
		// Kubernetes can retain Running and Ready while deletion is in progress.
		Status: corev1.PodStatus{Phase: corev1.PodRunning, Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}},
	}
	if terminating {
		now := metav1.Now()
		f.pod.DeletionTimestamp = &now
	}
	f.gpu = &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{Name: "hypervisor-phase-gpu", Labels: map[string]string{constants.LabelKeyOwner: f.node.Name}},
		Status: tfv1.GPUStatus{
			Phase: tfv1.TensorFusionGPUPhaseRunning, GPUModel: "test-model",
			Capacity:        &tfv1.Resource{Tflops: resource.MustParse("100"), Vram: resource.MustParse("24Gi")},
			Available:       &tfv1.Resource{Tflops: resource.MustParse("75"), Vram: resource.MustParse("16Gi")},
			RunningApps:     []*tfv1.RunningAppDetail{{Name: "workload", Namespace: "workloads", Count: 1, Pods: []*tfv1.PodGPUInfo{allocatedPod}}},
			IsolationPolicy: tfv1.IsolationModePolicyDynamic, ActiveIsolationMode: tfv1.IsolationModeHard,
		},
	}
	f.missingGPU = f.gpu.DeepCopy()
	f.missingGPU.Name = "hypervisor-phase-missing-gpu"
	f.missingGPU.Status.Phase = tfv1.TensorFusionGPUPhaseUnknown
	f.missingGPU.Annotations = map[string]string{constants.GPUMissingSinceAnnotationKey: "2026-07-13T00:00:00Z"}
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPUNode{}, &tfv1.GPU{}, &corev1.Pod{}).
		WithObjects(f.pool, f.node, f.coreNode, f.pod, f.gpu, f.missingGPU).Build()
	f.reconciler = &GPUNodeReconciler{Client: kubeClient, Scheme: scheme, Allocator: &gpuallocator.GpuAllocator{}}
	return f
}

func (f *hypervisorPhaseFixture) assertPendingWithoutDeallocation(t *testing.T) {
	t.Helper()
	ctx := context.Background()
	updatedNode := &tfv1.GPUNode{}
	require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(f.node), updatedNode))
	expectedNodeStatus := f.node.Status.DeepCopy()
	expectedNodeStatus.Phase = tfv1.TensorFusionGPUNodePhasePending
	require.Equal(t, expectedNodeStatus.Phase, updatedNode.Status.Phase)
	require.True(t, equality.Semantic.DeepEqual(*expectedNodeStatus, updatedNode.Status), "only GPUNode phase may change while hypervisor is unavailable")
	for _, original := range []*tfv1.GPU{f.gpu, f.missingGPU} {
		updated := &tfv1.GPU{}
		require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(original), updated))
		expected := original.Status.DeepCopy()
		if original == f.gpu {
			expected.Phase = tfv1.TensorFusionGPUPhasePending
		}
		require.Equal(t, expected.Phase, updated.Status.Phase)
		require.True(t, equality.Semantic.DeepEqual(*expected, updated.Status), "phase synchronization must preserve existing GPU allocations")
	}
}
