package gpuresources

import (
	"context"
	"strconv"
	"strings"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/gang"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/indexallocator"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	framework "k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	tffwk "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	clientfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// TestPreFilterDoesNotPoolSlotRejectGangAcrossIsolationModes is the regression
// guard for the removed `checkGangPoolFeasibility` shortcut.
//
// Setup is end-to-end: a real gang.Manager is wired in (not nil like the
// existing Ginkgo suite uses), and the manager's podLister sees enough active
// gang peers to satisfy RequiredMembers — so gangManager.PreFilter returns nil.
// PreFilter therefore reaches the post-gang code path where the deleted G2
// check used to live. We then assert PreFilter does NOT reject with a message
// mentioning "pool" or "slot", proving no pool-physical-GPU shortcut remains.
//
// We test soft (default) and hard (the last mode the shortcut was gated to).
// Partitioned mode would need template registry setup unrelated to this guard.
func TestPreFilterDoesNotPoolSlotRejectGangAcrossIsolationModes(t *testing.T) {
	for _, mode := range []tfv1.IsolationModeType{tfv1.IsolationModeSoft, tfv1.IsolationModeHard} {
		t.Run(string(mode), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)

			const (
				ns           = "ns1"
				workloadName = "wl1"
				poolName     = "pool-a"
				required     = 5 // exceeds the 1 GPU in the pool
			)
			groupKey := ns + "/" + workloadName

			// One physical GPU in the pool — far fewer than RequiredMembers.
			// The deleted shortcut would have rejected on this; the correct
			// behavior is to defer capacity judgment to Filter/allocator.
			gpu := &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gpu-1",
					Labels: map[string]string{
						constants.GpuPoolKey:    poolName,
						constants.LabelKeyOwner: "node-a",
					},
				},
				Status: tfv1.GPUStatus{
					Phase:        tfv1.TensorFusionGPUPhaseRunning,
					NodeSelector: map[string]string{constants.KubernetesHostNameLabel: "node-a"},
					UsedBy:       tfv1.UsedByTensorFusion,
					Capacity:     &tfv1.Resource{Tflops: resource.MustParse("1000"), Vram: resource.MustParse("80Gi")},
					Available:    &tfv1.Resource{Tflops: resource.MustParse("1000"), Vram: resource.MustParse("80Gi")},
				},
			}

			scheme := runtime.NewScheme()
			require.NoError(t, tfv1.AddToScheme(scheme))
			require.NoError(t, clientgoscheme.AddToScheme(scheme))
			k8sClient := clientfake.NewClientBuilder().WithScheme(scheme).
				WithRuntimeObjects(
					&tfv1.TensorFusionWorkload{ObjectMeta: metav1.ObjectMeta{Name: workloadName, Namespace: ns}},
				).
				WithStatusSubresource(&tfv1.GPU{}, &tfv1.GPUResourceQuota{}, &tfv1.TensorFusionWorkload{}).
				Build()
			require.NoError(t, k8sClient.Create(ctx, gpu))

			allocator := gpuallocator.NewGpuAllocator(ctx, nil, k8sClient, time.Second)
			require.NoError(t, allocator.InitGPUAndQuotaStore())
			allocator.ReconcileAllocationState()
			allocator.SetAllocatorReady()

			indexAlloc, err := indexallocator.NewIndexAllocator(ctx, k8sClient)
			require.NoError(t, err)
			indexAlloc.IsLeader = true
			indexAlloc.SetReady()

			// Static podLister seeded with enough peer pods so gangManager's
			// active-peer count meets the RequiredMembers threshold. Real
			// informer machinery is unnecessary — the cache.Indexer interface
			// is what the lister uses internally.
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{
				cache.NamespaceIndex: cache.MetaNamespaceIndexFunc,
			})
			for i := 0; i < required; i++ {
				peer := makeGangPeerPod(ns, workloadName, groupKey, mode, required, i)
				require.NoError(t, indexer.Add(peer))
			}
			podLister := corev1listers.NewPodLister(indexer)
			gangMgr := gang.NewManager(podLister, &events.FakeRecorder{}, Name)

			fwkInstance := newMinimalFramework(t, ctx)
			pluginFactory := NewWithDeps(allocator, indexAlloc, gangMgr, k8sClient)
			pluginConfig := &runtime.Unknown{Raw: []byte(`{"maxWorkerPerNode": 8}`)}
			plug, err := pluginFactory(ctx, pluginConfig, fwkInstance)
			require.NoError(t, err)
			gpuFit := plug.(*GPUFit)

			// Test subject: the first peer (already in the cache).
			subject := makeGangPeerPod(ns, workloadName, groupKey, mode, required, 0)
			state := framework.NewCycleState()
			_, status := gpuFit.PreFilter(ctx, state, subject, nil)

			// Either Success, or a non-Success whose message does NOT cite a
			// pool-slot impossibility. (Filter / capacity errors are acceptable;
			// a "pool 'pool-a' only has N GPUs" message is what we're guarding.)
			if !status.IsSuccess() {
				msg := strings.ToLower(status.Message())
				require.NotContains(t, msg, "exclusive gpus", "PreFilter must not reject on physical-GPU-count grounds for mode %s", mode)
				require.NotContains(t, msg, "gang requires", "PreFilter must not reject on physical-GPU-count grounds for mode %s", mode)
				require.NotContains(t, msg, "slot", "PreFilter must not reject on physical-GPU-count grounds for mode %s", mode)
			}
		})
	}
}

func makeGangPeerPod(ns, workloadName, groupKey string, mode tfv1.IsolationModeType, required, idx int) *corev1.Pod {
	name := workloadName + "-peer-" + strconv.Itoa(idx)
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			UID:       types.UID(name + "-uid"),
			Labels: map[string]string{
				constants.WorkloadKey:    workloadName,
				constants.LabelComponent: constants.ComponentWorker,
			},
			Annotations: map[string]string{
				constants.GpuPoolKey:                    "pool-a",
				constants.GpuCountAnnotation:            "1",
				constants.TFLOPSRequestAnnotation:       "100",
				constants.VRAMRequestAnnotation:         "10Gi",
				constants.IsolationModeAnnotation:       string(mode),
				constants.GangEnabledAnnotation:         constants.TrueStringValue,
				constants.GangRequiredMembersAnnotation: strconv.Itoa(required),
				constants.GangDesiredMembersAnnotation:  strconv.Itoa(required),
				constants.GangMinMembersAnnotation:      strconv.Itoa(required),
				constants.GangGroupKeyAnnotation:        groupKey,
			},
		},
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "c", Image: "img"}}},
	}
}

func newMinimalFramework(t *testing.T, ctx context.Context) framework.Framework {
	t.Helper()
	fwkInstance, err := tffwk.NewFramework(
		ctx,
		[]tffwk.RegisterPluginFunc{
			tffwk.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			tffwk.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		},
		"",
		frameworkruntime.WithSnapshotSharedLister(internalcache.NewEmptySnapshot()),
		frameworkruntime.WithEventRecorder(&events.FakeRecorder{}),
	)
	require.NoError(t, err)
	return fwkInstance
}
