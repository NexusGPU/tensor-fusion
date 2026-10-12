package controller

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// The pool's pod template names the hypervisor container, so the tests use a
// name that differs from constants.TFContainerNameHypervisor on purpose.
const (
	sidecarTestHypervisorName = "tensor-fusion-hypervisor"
	sidecarTestSidecarName    = constants.TFContainerVector
)

// podLevelHypervisorNotReady is the expression the controller used before the
// hypervisor container was looked at on its own. Pods without a sidecar, and
// pods the kubelet has not reported yet, must keep giving this result.
func podLevelHypervisorNotReady(pod *corev1.Pod) bool {
	return !pod.DeletionTimestamp.IsZero() || pod.Status.Phase != corev1.PodRunning ||
		!utils.IsPodConditionTrue(pod.Status.Conditions, corev1.PodReady)
}

func runningStatus(name string, ready bool, restarts int32) corev1.ContainerStatus {
	return corev1.ContainerStatus{
		Name: name, Ready: ready, RestartCount: restarts,
		State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}},
	}
}

func waitingStatus(name, reason string, restarts int32) corev1.ContainerStatus {
	return corev1.ContainerStatus{
		Name: name, RestartCount: restarts,
		State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: reason}},
	}
}

func hypervisorTestPod(withSidecar bool, phase corev1.PodPhase, podReady bool, statuses ...corev1.ContainerStatus) *corev1.Pod {
	pod := &corev1.Pod{
		Spec:   corev1.PodSpec{Containers: []corev1.Container{{Name: sidecarTestHypervisorName}}},
		Status: corev1.PodStatus{Phase: phase, ContainerStatuses: statuses},
	}
	if withSidecar {
		pod.Spec.Containers = append(pod.Spec.Containers, corev1.Container{Name: sidecarTestSidecarName})
	}
	if phase == corev1.PodRunning {
		readyStatus := corev1.ConditionFalse
		if podReady {
			readyStatus = corev1.ConditionTrue
		}
		pod.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: readyStatus}}
	}
	return pod
}

func TestIsHypervisorReadyIgnoresSidecar(t *testing.T) {
	for _, tc := range []struct {
		name string
		pod  *corev1.Pod
		want bool
	}{
		{
			name: "sidecar not ready, hypervisor ready",
			pod: hypervisorTestPod(true, corev1.PodRunning, false,
				runningStatus(sidecarTestHypervisorName, true, 0),
				runningStatus(sidecarTestSidecarName, false, 0)),
			want: true,
		},
		{
			name: "sidecar crash-looping, hypervisor ready",
			pod: hypervisorTestPod(true, corev1.PodRunning, false,
				runningStatus(sidecarTestHypervisorName, true, 0),
				waitingStatus(sidecarTestSidecarName, "CrashLoopBackOff", 40)),
			want: true,
		},
		{
			// The pod stays Pending until every container has been started once.
			name: "sidecar image cannot be pulled, hypervisor ready",
			pod: hypervisorTestPod(true, corev1.PodPending, false,
				runningStatus(sidecarTestHypervisorName, true, 0),
				waitingStatus(sidecarTestSidecarName, "ImagePullBackOff", 0)),
			want: true,
		},
		{
			name: "sidecar and hypervisor ready",
			pod: hypervisorTestPod(true, corev1.PodRunning, true,
				runningStatus(sidecarTestHypervisorName, true, 0),
				runningStatus(sidecarTestSidecarName, true, 0)),
			want: true,
		},
		{
			name: "sidecar ready, hypervisor failing its readiness probe",
			pod: hypervisorTestPod(true, corev1.PodRunning, false,
				runningStatus(sidecarTestHypervisorName, false, 0),
				runningStatus(sidecarTestSidecarName, true, 0)),
			want: false,
		},
		{
			name: "sidecar ready, hypervisor crash-looping",
			pod: hypervisorTestPod(true, corev1.PodRunning, false,
				waitingStatus(sidecarTestHypervisorName, "CrashLoopBackOff", 3),
				runningStatus(sidecarTestSidecarName, true, 0)),
			want: false,
		},
		{
			name: "hypervisor not started yet while the sidecar runs",
			pod: hypervisorTestPod(true, corev1.PodPending, false,
				waitingStatus(sidecarTestHypervisorName, "ContainerCreating", 0),
				runningStatus(sidecarTestSidecarName, true, 0)),
			want: false,
		},
		{
			name: "failed pod with a stale ready hypervisor status",
			pod: hypervisorTestPod(true, corev1.PodFailed, false,
				runningStatus(sidecarTestHypervisorName, true, 0),
				runningStatus(sidecarTestSidecarName, true, 0)),
			want: false,
		},
		{
			name: "only the sidecar has been reported, pod not ready",
			pod: hypervisorTestPod(true, corev1.PodRunning, false,
				runningStatus(sidecarTestSidecarName, true, 0)),
			want: false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, isHypervisorReady(tc.pod))
		})
	}

	t.Run("terminating pod is never ready", func(t *testing.T) {
		pod := hypervisorTestPod(true, corev1.PodRunning, true,
			runningStatus(sidecarTestHypervisorName, true, 0),
			runningStatus(sidecarTestSidecarName, true, 0))
		now := metav1.Now()
		pod.DeletionTimestamp = &now
		assert.False(t, isHypervisorReady(pod))
	})

	t.Run("restartable init container is a sidecar too", func(t *testing.T) {
		pod := hypervisorTestPod(false, corev1.PodRunning, false,
			runningStatus(sidecarTestHypervisorName, true, 0))
		pod.Spec.InitContainers = []corev1.Container{
			{Name: "init-shm"},
			{Name: "log-shipper", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways)},
		}
		pod.Status.InitContainerStatuses = []corev1.ContainerStatus{
			waitingStatus("log-shipper", "CrashLoopBackOff", 9),
		}
		assert.True(t, isHypervisorReady(pod))
	})
}

func TestIsHypervisorReadyUnchangedWithoutSidecarOrStatus(t *testing.T) {
	for _, tc := range []struct {
		name string
		pod  *corev1.Pod
		want bool
	}{
		// No sidecar: the pod as a whole decides, whatever the statuses say.
		{"no sidecar, running and ready",
			hypervisorTestPod(false, corev1.PodRunning, true, runningStatus(sidecarTestHypervisorName, true, 0)), true},
		{"no sidecar, running and ready, statuses not reported",
			hypervisorTestPod(false, corev1.PodRunning, true), true},
		{"no sidecar, hypervisor not ready",
			hypervisorTestPod(false, corev1.PodRunning, false, runningStatus(sidecarTestHypervisorName, false, 0)), false},
		{"no sidecar, pod marked not ready although the container status is ready",
			hypervisorTestPod(false, corev1.PodRunning, false, runningStatus(sidecarTestHypervisorName, true, 0)), false},
		{"no sidecar, pending with a running container status",
			hypervisorTestPod(false, corev1.PodPending, false, runningStatus(sidecarTestHypervisorName, true, 0)), false},
		{"no sidecar, just scheduled",
			hypervisorTestPod(false, corev1.PodPending, false), false},
		{"no sidecar, failed",
			hypervisorTestPod(false, corev1.PodFailed, false), false},
		{"no containers in spec, running and ready",
			&corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodRunning,
				Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}}}, true},
		{"no containers in spec, empty status",
			&corev1.Pod{}, false},
		// Sidecar declared but the kubelet has not reported any container yet.
		{"sidecar, just scheduled, statuses missing",
			hypervisorTestPod(true, corev1.PodPending, false), false},
		{"sidecar, statuses empty, running and ready",
			hypervisorTestPod(true, corev1.PodRunning, true), true},
		{"sidecar, statuses empty, running and not ready",
			hypervisorTestPod(true, corev1.PodRunning, false), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := isHypervisorReady(tc.pod)
			assert.Equal(t, tc.want, got)
			assert.Equal(t, !podLevelHypervisorNotReady(tc.pod), got, "must match the pod-level check used before")
		})
	}
}

func TestIsHypervisorStarted(t *testing.T) {
	for _, tc := range []struct {
		name string
		pod  *corev1.Pod
		want bool
	}{
		{"running pod", hypervisorTestPod(false, corev1.PodRunning, false), true},
		{"pending pod without sidecar", hypervisorTestPod(false, corev1.PodPending, false), false},
		{"pending pod without sidecar keeps the pod-level answer",
			hypervisorTestPod(false, corev1.PodPending, false, runningStatus(sidecarTestHypervisorName, true, 0)), false},
		{"pending only because the sidecar has not started",
			hypervisorTestPod(true, corev1.PodPending, false,
				runningStatus(sidecarTestHypervisorName, true, 0),
				waitingStatus(sidecarTestSidecarName, "ImagePullBackOff", 0)), true},
		{"pending, hypervisor not started",
			hypervisorTestPod(true, corev1.PodPending, false,
				waitingStatus(sidecarTestHypervisorName, "ContainerCreating", 0),
				runningStatus(sidecarTestSidecarName, true, 0)), false},
		{"pending with sidecar, statuses missing", hypervisorTestPod(true, corev1.PodPending, false), false},
		{"failed pod", hypervisorTestPod(true, corev1.PodFailed, false,
			runningStatus(sidecarTestHypervisorName, true, 0)), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, isHypervisorStarted(tc.pod))
		})
	}
}

func TestHypervisorPodCrashExceededIgnoresSidecar(t *testing.T) {
	defaultCfg := config.GetGlobalConfig()
	t.Cleanup(func() { config.SetGlobalConfig(defaultCfg) })
	config.SetGlobalConfig(&config.GlobalConfig{})
	over := config.DefaultHypervisorMaxCrashCount + 1

	t.Run("rising sidecar restart count is not counted", func(t *testing.T) {
		for _, restarts := range []int32{over, over + 10, over + 1000} {
			pod := hypervisorTestPod(true, corev1.PodRunning, false,
				runningStatus(sidecarTestHypervisorName, true, 0),
				waitingStatus(sidecarTestSidecarName, "CrashLoopBackOff", restarts))
			_, _, _, exceeded := hypervisorPodCrashExceeded(pod)
			assert.False(t, exceeded, "sidecar restarts=%d", restarts)
		}
	})

	t.Run("hypervisor at the threshold is not pushed over by the sidecar", func(t *testing.T) {
		pod := hypervisorTestPod(true, corev1.PodRunning, false,
			runningStatus(sidecarTestHypervisorName, true, config.DefaultHypervisorMaxCrashCount),
			waitingStatus(sidecarTestSidecarName, "CrashLoopBackOff", over))
		_, _, _, exceeded := hypervisorPodCrashExceeded(pod)
		assert.False(t, exceeded)
	})

	t.Run("hypervisor restarts are still counted next to a sidecar", func(t *testing.T) {
		pod := hypervisorTestPod(true, corev1.PodRunning, false,
			waitingStatus(sidecarTestHypervisorName, "CrashLoopBackOff", over),
			runningStatus(sidecarTestSidecarName, true, over+50))
		name, count, threshold, exceeded := hypervisorPodCrashExceeded(pod)
		assert.True(t, exceeded)
		assert.Equal(t, sidecarTestHypervisorName, name)
		assert.Equal(t, over, count)
		assert.Equal(t, config.DefaultHypervisorMaxCrashCount, threshold)
	})

	t.Run("one-shot init container restarts are still counted next to a sidecar", func(t *testing.T) {
		pod := hypervisorTestPod(true, corev1.PodPending, false)
		pod.Spec.InitContainers = []corev1.Container{{Name: "init-shm"}}
		pod.Status.InitContainerStatuses = []corev1.ContainerStatus{waitingStatus("init-shm", "CrashLoopBackOff", over)}
		name, _, _, exceeded := hypervisorPodCrashExceeded(pod)
		assert.True(t, exceeded)
		assert.Equal(t, "init-shm", name)
	})

	t.Run("restartable init container restarts are not counted", func(t *testing.T) {
		pod := hypervisorTestPod(false, corev1.PodRunning, false, runningStatus(sidecarTestHypervisorName, true, 0))
		pod.Spec.InitContainers = []corev1.Container{
			{Name: "log-shipper", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways)},
		}
		pod.Status.InitContainerStatuses = []corev1.ContainerStatus{waitingStatus("log-shipper", "CrashLoopBackOff", over)}
		_, _, _, exceeded := hypervisorPodCrashExceeded(pod)
		assert.False(t, exceeded)
	})

	t.Run("pod without sidecar is unchanged", func(t *testing.T) {
		pod := hypervisorTestPod(false, corev1.PodRunning, false,
			waitingStatus(sidecarTestHypervisorName, "CrashLoopBackOff", over))
		name, _, _, exceeded := hypervisorPodCrashExceeded(pod)
		assert.True(t, exceeded)
		assert.Equal(t, sidecarTestHypervisorName, name)

		pod = hypervisorTestPod(false, corev1.PodRunning, true,
			runningStatus(sidecarTestHypervisorName, true, config.DefaultHypervisorMaxCrashCount))
		_, _, _, exceeded = hypervisorPodCrashExceeded(pod)
		assert.False(t, exceeded)
	})

	t.Run("statuses missing", func(t *testing.T) {
		_, _, threshold, exceeded := hypervisorPodCrashExceeded(hypervisorTestPod(true, corev1.PodPending, false))
		assert.False(t, exceeded)
		assert.Equal(t, config.DefaultHypervisorMaxCrashCount, threshold)
	})
}

// newHypervisorSidecarFixture stores a hypervisor pod with a sidecar and the
// given kubelet-reported status, with the template hash the pool asks for, so
// the reconciler has no other reason to replace the pod.
func newHypervisorSidecarFixture(
	t *testing.T, nodePhase tfv1.TensorFusionGPUNodePhase, status corev1.PodStatus,
) *hypervisorPhaseFixture {
	t.Helper()
	defaultCfg := config.GetGlobalConfig()
	t.Cleanup(func() { config.SetGlobalConfig(defaultCfg) })
	config.SetGlobalConfig(&config.GlobalConfig{})

	f := newHypervisorPhaseFixture(t, nodePhase, false)
	f.reconciler.Recorder = events.NewFakeRecorder(16)
	ctx := context.Background()

	pod := &corev1.Pod{}
	require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(f.pod), pod))
	pod.Labels[constants.LabelKeyPodTemplateHash] = utils.HypervisorPodTemplateHash(
		f.pool, f.node.Labels[constants.AcceleratorLabelVendor], tfv1.IsolationModeSoft)
	pod.Spec.Containers[0].Name = sidecarTestHypervisorName
	pod.Spec.Containers = append(pod.Spec.Containers, corev1.Container{Name: sidecarTestSidecarName})
	require.NoError(t, f.reconciler.Update(ctx, pod))
	pod.Status = status
	require.NoError(t, f.reconciler.Status().Update(ctx, pod))
	return f
}

func (f *hypervisorPhaseFixture) assertRunningAndPodKept(t *testing.T) {
	t.Helper()
	ctx := context.Background()
	node := &tfv1.GPUNode{}
	require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(f.node), node))
	require.Equal(t, tfv1.TensorFusionGPUNodePhaseRunning, node.Status.Phase)
	gpu := &tfv1.GPU{}
	require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(f.gpu), gpu))
	require.Equal(t, tfv1.TensorFusionGPUPhaseRunning, gpu.Status.Phase)
	pod := &corev1.Pod{}
	require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(f.pod), pod))
	require.True(t, pod.DeletionTimestamp.IsZero(), "hypervisor pod must not be replaced because of its sidecar")
}

func TestGPUNodeReconcileSidecarFailureKeepsGPUsRunning(t *testing.T) {
	over := config.DefaultHypervisorMaxCrashCount + 1
	for _, tc := range []struct {
		name   string
		status corev1.PodStatus
	}{
		{"sidecar not ready", corev1.PodStatus{
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}},
			ContainerStatuses: []corev1.ContainerStatus{
				runningStatus(sidecarTestHypervisorName, true, 0),
				runningStatus(sidecarTestSidecarName, false, 0),
			},
		}},
		{"sidecar crash-looping beyond the crash threshold", corev1.PodStatus{
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}},
			ContainerStatuses: []corev1.ContainerStatus{
				runningStatus(sidecarTestHypervisorName, true, 0),
				waitingStatus(sidecarTestSidecarName, "CrashLoopBackOff", over+20),
			},
		}},
		{"sidecar image cannot be pulled", corev1.PodStatus{
			Phase: corev1.PodPending,
			ContainerStatuses: []corev1.ContainerStatus{
				runningStatus(sidecarTestHypervisorName, true, 0),
				waitingStatus(sidecarTestSidecarName, "ImagePullBackOff", 0),
			},
		}},
	} {
		for _, nodePhase := range []tfv1.TensorFusionGPUNodePhase{
			tfv1.TensorFusionGPUNodePhaseRunning, // a serving node stays in service
			tfv1.TensorFusionGPUNodePhasePending, // a starting node goes into service
		} {
			t.Run(tc.name+"/node "+string(nodePhase), func(t *testing.T) {
				f := newHypervisorSidecarFixture(t, nodePhase, tc.status)
				req := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.node)}
				for range 2 {
					_, err := f.reconciler.Reconcile(context.Background(), req)
					require.NoError(t, err)
					f.assertRunningAndPodKept(t)
				}
			})
		}
	}
}

func TestGPUNodeReconcileHypervisorNotReadyNextToSidecarMarksGPUsPending(t *testing.T) {
	f := newHypervisorSidecarFixture(t, tfv1.TensorFusionGPUNodePhaseRunning, corev1.PodStatus{
		Phase:      corev1.PodRunning,
		Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}},
		ContainerStatuses: []corev1.ContainerStatus{
			runningStatus(sidecarTestHypervisorName, false, 0),
			runningStatus(sidecarTestSidecarName, true, 0),
		},
	})
	_, err := f.reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.node)})
	require.NoError(t, err)
	f.assertPendingWithoutDeallocation(t)
}

func TestGPUNodeReconcileHypervisorCrashLoopNextToSidecarRecreatesPod(t *testing.T) {
	f := newHypervisorSidecarFixture(t, tfv1.TensorFusionGPUNodePhaseRunning, corev1.PodStatus{
		Phase:      corev1.PodRunning,
		Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}},
		ContainerStatuses: []corev1.ContainerStatus{
			waitingStatus(sidecarTestHypervisorName, "CrashLoopBackOff", config.DefaultHypervisorMaxCrashCount+1),
			runningStatus(sidecarTestSidecarName, true, 0),
		},
	})
	ctx := context.Background()
	result, err := f.reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.node)})
	require.NoError(t, err)
	require.Equal(t, constants.PendingRequeueDuration, result.RequeueAfter)
	f.assertPendingWithoutDeallocation(t)
	pod := &corev1.Pod{}
	require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(f.pod), pod))
	require.False(t, pod.DeletionTimestamp.IsZero(), "a crash-looping hypervisor container must still get its pod recreated")
}

func TestRemoveStuckHypervisorPodOnNotReadyNodeKeepsStartedHypervisor(t *testing.T) {
	for _, tc := range []struct {
		name        string
		hypervisor  corev1.ContainerStatus
		wantDeleted bool
	}{
		{"hypervisor running while the sidecar image cannot be pulled",
			runningStatus(sidecarTestHypervisorName, true, 0), false},
		{"hypervisor never started",
			waitingStatus(sidecarTestHypervisorName, "ContainerCreating", 0), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newHypervisorSidecarFixture(t, tfv1.TensorFusionGPUNodePhaseRunning, corev1.PodStatus{
				Phase: corev1.PodPending,
				ContainerStatuses: []corev1.ContainerStatus{
					tc.hypervisor,
					waitingStatus(sidecarTestSidecarName, "ImagePullBackOff", 0),
				},
			})
			ctx := context.Background()
			pod := &corev1.Pod{}
			require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(f.pod), pod))
			require.NoError(t, f.reconciler.removeStuckHypervisorPodOnNotReadyNode(ctx, f.node, pod, true))
			require.NoError(t, f.reconciler.Get(ctx, client.ObjectKeyFromObject(f.pod), pod))
			require.Equal(t, tc.wantDeleted, !pod.DeletionTimestamp.IsZero())
		})
	}
}
