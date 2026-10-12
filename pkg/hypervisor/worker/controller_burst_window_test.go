package worker

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
)

// The burst window a worker asks for with the annotation is part of what the
// quota controller reads about it in every control interval, next to its limit.
// A change on the running worker is there with the next read: the mapping and
// the allocation stay as they are. The annotation can be changed after
// admission, so a value outside the bounds is brought to the nearest bound and
// one that is not a duration counts as not set; either is reported once.
func TestBurstWindowAnnotationReachesTheQuotaController(t *testing.T) {
	w, allocations := newLimitTestController(t, testA100Backend(), testA100())
	handler := w.workerChangeHandler()
	worker := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndPercent("8Gi", "50"), "gpu-0")
	handler.OnAdd(worker)

	burstWindow := func() *float64 {
		t.Helper()
		snapshot := w.buildWorkerInfoSnapshots()[worker.WorkerUID]
		require.NotNil(t, snapshot)
		require.Len(t, snapshot.Devices, 1)
		require.Equal(t, uint32(50), snapshot.Devices[0].UpLimit)
		return snapshot.Devices[0].BurstWindow
	}
	require.Nil(t, burstWindow(), "without the annotation the pool's burst window applies")
	mapping := w.getShmHandle(worker.WorkerUID)
	require.NotNil(t, mapping)
	allocation, _ := allocations.GetWorkerAllocation(worker.WorkerUID)

	seconds := func(v float64) *float64 { return &v }
	current := worker
	for _, tc := range []struct {
		raw      string
		want     *float64
		reported bool
	}{
		{raw: "2s", want: seconds(2)},
		{raw: "500ms", want: seconds(0.5)},
		{raw: "0", want: seconds(0)},
		{raw: "10s", want: seconds(10)},
		{raw: "1h", want: seconds(10), reported: true},
		{raw: "-1s", want: seconds(0), reported: true},
		{raw: "fast", reported: true},
		{raw: "0.5", reported: true},
		{raw: "1s", want: seconds(1)},
		{raw: ""},
	} {
		updated := *current
		updated.Annotations = map[string]string{}
		if tc.raw != "" {
			updated.Annotations[constants.ComputeBurstWindowAnnotation] = tc.raw
		}
		handler.OnUpdate(current, &updated)
		current = &updated

		// Read twice: the second read finds the value already reported.
		got := burstWindow()
		require.Equal(t, tc.want, burstWindow(), "annotation %q", tc.raw)
		require.Equal(t, tc.want, got, "annotation %q", tc.raw)
		reported, isReported := w.burstWindowReported.Load(worker.WorkerUID)
		require.Equal(t, tc.reported, isReported, "annotation %q", tc.raw)
		if tc.reported {
			require.Equal(t, tc.raw, reported)
		}
	}
	require.Same(t, mapping, w.getShmHandle(worker.WorkerUID), "the mapping was replaced by a change of the burst window")
	after, _ := allocations.GetWorkerAllocation(worker.WorkerUID)
	require.Equal(t, allocation.DeviceInfos, after.DeviceInfos)

	// Nothing is kept for a worker that is gone.
	bad := *current
	bad.Annotations = map[string]string{constants.ComputeBurstWindowAnnotation: "never"}
	handler.OnUpdate(current, &bad)
	require.Nil(t, burstWindow())
	_, isReported := w.burstWindowReported.Load(worker.WorkerUID)
	require.True(t, isReported)
	handler.OnRemove(&bad)
	_, isReported = w.burstWindowReported.Load(worker.WorkerUID)
	require.False(t, isReported)
}

// A worker that is not soft-isolated has no token bucket; the annotation is
// read for nobody.
func TestBurstWindowAnnotationIsOnlyReadForSoftWorkers(t *testing.T) {
	w, _ := newLimitTestController(t, testA100Backend(), testA100())
	worker := newLimitTestWorker(tfv1.IsolationModeHard, vramAndPercent("8Gi", "50"), "gpu-0")
	worker.Annotations = map[string]string{constants.ComputeBurstWindowAnnotation: "2s"}
	w.workerChangeHandler().OnAdd(worker)
	require.Empty(t, w.buildWorkerInfoSnapshots())
}
