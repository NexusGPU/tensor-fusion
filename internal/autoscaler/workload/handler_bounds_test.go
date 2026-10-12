package workload

import (
	"strings"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func testGPU(name, tflops, vram string) *tfv1.GPU {
	gpu := &tfv1.GPU{ObjectMeta: metav1.ObjectMeta{Name: name}}
	if tflops != "" {
		gpu.Status.Capacity = &tfv1.Resource{Tflops: resource.MustParse(tflops), Vram: resource.MustParse(vram)}
	}
	return gpu
}

func testWorker(name string, annotations map[string]string) *corev1.Pod {
	return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name, UID: types.UID(name + "-uid"), Annotations: annotations}}
}

func TestDeviceCapacityOfTheGPUsWorkersHold(t *testing.T) {
	gpuStore := map[types.NamespacedName]*tfv1.GPU{}
	for _, gpu := range []*tfv1.GPU{
		testGPU("rtx3090-0", "71.2", "24Gi"),
		testGPU("rtx3090-1", "71.2", "24Gi"),
		testGPU("t4-0", "40", "16Gi"),
		testGPU("a100-0", "312", "80Gi"),
		testGPU("no-capacity", "", ""),
	} {
		gpuStore[types.NamespacedName{Name: gpu.Name}] = gpu
	}
	holding := func(gpuNames ...string) *tfv1.AllocRequest { return &tfv1.AllocRequest{GPUNames: gpuNames} }

	t.Run("one GPU", func(t *testing.T) {
		capacity := deviceCapacityOf([]*corev1.Pod{testWorker("w0", nil)}, gpuStore,
			map[string]*tfv1.AllocRequest{"w0-uid": holding("rtx3090-0")})
		require.NotNil(t, capacity)
		require.Equal(t, "71200m", capacity.PerGPU.Tflops.String())
		require.Equal(t, "24Gi", capacity.PerGPU.Vram.String())
		require.Equal(t, 1, capacity.GPUsPerWorker)
	})

	t.Run("a worker on two GPUs", func(t *testing.T) {
		// The worker's request and limit apply to each GPU it holds: each may give
		// 71.2 TFLOPS and 24Gi, the worker as a whole twice that.
		capacity := deviceCapacityOf([]*corev1.Pod{testWorker("w0", nil)}, gpuStore,
			map[string]*tfv1.AllocRequest{"w0-uid": holding("rtx3090-0", "rtx3090-1")})
		require.NotNil(t, capacity)
		require.Equal(t, "71200m", capacity.PerGPU.Tflops.String())
		require.Equal(t, "24Gi", capacity.PerGPU.Vram.String())
		require.Equal(t, 2, capacity.GPUsPerWorker)
	})

	t.Run("GPUs of different sizes", func(t *testing.T) {
		// One recommendation is applied to every worker and to each GPU of a
		// worker, so the smallest GPU held is what all of them can be given.
		capacity := deviceCapacityOf(
			[]*corev1.Pod{testWorker("w0", nil), testWorker("w1", nil)}, gpuStore,
			map[string]*tfv1.AllocRequest{"w0-uid": holding("a100-0", "rtx3090-0"), "w1-uid": holding("t4-0")})
		require.NotNil(t, capacity)
		require.Equal(t, "40", capacity.PerGPU.Tflops.String())
		require.Equal(t, "16Gi", capacity.PerGPU.Vram.String())
		require.Equal(t, 2, capacity.GPUsPerWorker)
	})

	t.Run("only GPUs that are held and scaled count", func(t *testing.T) {
		dedicated := testWorker("w1", map[string]string{constants.DedicatedGPUAnnotation: constants.TrueStringValue})
		capacity := deviceCapacityOf(
			[]*corev1.Pod{testWorker("w0", nil), dedicated, testWorker("w2", nil)}, gpuStore,
			map[string]*tfv1.AllocRequest{
				"w0-uid": holding("a100-0", "no-capacity", "gone"),
				// A worker on a dedicated GPU receives no recommendation.
				"w1-uid": holding("t4-0"),
				// w2 has no allocation yet.
			})
		require.NotNil(t, capacity)
		require.Equal(t, "312", capacity.PerGPU.Tflops.String())
		require.Equal(t, 1, capacity.GPUsPerWorker)
	})

	t.Run("unknown when no capacity is reported", func(t *testing.T) {
		require.Nil(t, deviceCapacityOf([]*corev1.Pod{testWorker("w0", nil)}, gpuStore,
			map[string]*tfv1.AllocRequest{"w0-uid": holding("no-capacity")}))
		require.Nil(t, deviceCapacityOf([]*corev1.Pod{testWorker("w0", nil)}, gpuStore, nil))
	})
}

func testResources(tflopsRequest, tflopsLimit, vramRequest string) *tfv1.Resources {
	return &tfv1.Resources{
		Requests: tfv1.Resource{Tflops: resource.MustParse(tflopsRequest), Vram: resource.MustParse(vramRequest)},
		Limits:   tfv1.Resource{Tflops: resource.MustParse(tflopsLimit), Vram: resource.MustParse("8Gi")},
	}
}

func TestScalingEventTellsTheBounds(t *testing.T) {
	current := testResources("10", "20", "4Gi")

	reason, message := scalingEvent(current, testResources("15", "30", "4Gi"),
		"bounds: Compute limit: recommended 82, applied 30 (step limit: at most +50% per change)")
	require.Equal(t, "ResourceScaledUp", reason)
	require.Equal(t, "Resources scaled up: Compute 10->15, VRAM 4Gi->4Gi; limits: Compute 20->30, VRAM 8Gi->8Gi; "+
		"bounds: Compute limit: recommended 82, applied 30 (step limit: at most +50% per change)", message)

	reason, message = scalingEvent(current, testResources("8", "20", "4Gi"), "")
	require.Equal(t, "ResourceScaledDown", reason)
	require.Equal(t, "Resources scaled down: Compute 10->8, VRAM 4Gi->4Gi", message)

	_, message = scalingEvent(current, testResources("15", "30", "4Gi"), strings.Repeat("bound ", 400))
	require.Len(t, message, maxEventNoteLength)
	require.True(t, strings.HasSuffix(message, "..."))
}

func TestWorkerResourcesRange(t *testing.T) {
	annotations := func(tflopsRequest, tflopsLimit, vramRequest, vramLimit string) map[string]string {
		return map[string]string{
			constants.TFLOPSRequestAnnotation: tflopsRequest,
			constants.TFLOPSLimitAnnotation:   tflopsLimit,
			constants.VRAMRequestAnnotation:   vramRequest,
			constants.VRAMLimitAnnotation:     vramLimit,
		}
	}
	state := NewWorkloadState()
	_, _, known := state.WorkerResourcesRange()
	require.False(t, known)

	dedicated := annotations("99", "99", "99Gi", "99Gi")
	dedicated[constants.DedicatedGPUAnnotation] = constants.TrueStringValue
	state.CurrentActiveWorkers = map[string]*corev1.Pod{
		"w0": testWorker("w0", annotations("15", "30", "4Gi", "8Gi")),
		"w1": testWorker("w1", annotations("10", "45", "6Gi", "8Gi")),
		"w2": testWorker("w2", dedicated),
		"w3": testWorker("w3", map[string]string{"unrelated": "annotation"}),
	}
	lowest, highest, known := state.WorkerResourcesRange()
	require.True(t, known)
	require.True(t, lowest.Equal(testResources("10", "30", "4Gi")), "lowest %v", lowest)
	require.True(t, highest.Equal(testResources("15", "45", "6Gi")), "highest %v", highest)
}

func TestLastResourceChangeOnlyMovesForward(t *testing.T) {
	state := NewWorkloadState()
	require.True(t, state.LastResourceChange().IsZero())
	now := time.Now()
	state.MarkResourcesChanged(now)
	state.MarkResourcesChanged(now.Add(-time.Minute))
	require.True(t, state.LastResourceChange().Equal(now))
}
