package kubernetes

import (
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/stretchr/testify/require"
)

// The compute burst window is an annotation of the worker Pod. The backend
// hands the Pod's annotations to the worker handlers as they are, and reports a
// change of this one on a running Pod through OnUpdate, like a change of a
// limit: that is how it takes effect without a restart.
func TestWorkerSyncReportsBurstWindowAnnotation(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	pod := createTestPodWithIndex(1)
	pod.Annotations[constants.ComputeBurstWindowAnnotation] = "2s"
	added := make(chan *api.WorkerInfo, 1)
	updates := make(chan *api.WorkerInfo, 4)
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnAdd:    func(info *api.WorkerInfo) { added <- info },
		OnUpdate: func(_, updated *api.WorkerInfo) { updates <- updated },
	}))
	b.podCacher.onPodAdd(pod)
	select {
	case info := <-added:
		require.Equal(t, "2s", info.Annotations[constants.ComputeBurstWindowAnnotation])
	case <-time.After(time.Second):
		t.Fatal("Add was not delivered")
	}

	current := pod
	for _, raw := range []string{"500ms", ""} {
		patched := current.DeepCopy()
		if raw == "" {
			delete(patched.Annotations, constants.ComputeBurstWindowAnnotation)
		} else {
			patched.Annotations[constants.ComputeBurstWindowAnnotation] = raw
		}
		b.podCacher.onPodUpdate(current, patched)
		select {
		case info := <-updates:
			require.Equal(t, raw, info.Annotations[constants.ComputeBurstWindowAnnotation])
			require.Equal(t, string(pod.UID), info.WorkerUID)
		case <-time.After(time.Second):
			t.Fatalf("the change of the burst window to %q was not reported to the worker handler", raw)
		}
		current = patched
	}
}
