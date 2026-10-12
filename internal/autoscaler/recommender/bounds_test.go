package recommender

import (
	"context"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func quantity(value string) resource.Quantity { return resource.MustParse(value) }

func quantityRef(value string) *resource.Quantity {
	q := resource.MustParse(value)
	return &q
}

func TestQuantityBounds(t *testing.T) {
	const stepUp, stepDown = "step limit: at most +50% per change", "step limit: at most -25% per change"
	compute := func(current string) quantityBounds {
		return quantityBounds{name: "Compute limit", current: quantity(current), minStep: minComputeStep}
	}
	withFloor := func(bounds quantityBounds, floor string) quantityBounds {
		bounds.usageFloor, bounds.usageFloorSource = quantity(floor), "the median of the observed usage"
		return bounds
	}
	withHeld := func(bounds quantityBounds, lowest, highest string) quantityBounds {
		bounds.lowestHeld, bounds.highestHeld = quantityRef(lowest), quantityRef(highest)
		return bounds
	}

	for _, test := range []struct {
		name        string
		bounds      quantityBounds
		recommended string
		applied     string
		bound       string
	}{
		{"an increase within the step is applied as it is", compute("20"), "28", "28", ""},
		{"an increase of exactly the step is applied as it is", compute("20"), "30", "30", ""},
		{"a larger increase stops at the step", compute("20"), "82", "30", stepUp},
		{"the step is rounded down to whole units", compute("45"), "82", "67", stepUp},
		{"a small value still moves by a whole unit", compute("1"), "40", "2", stepUp},
		{"no change needs no bound", compute("20"), "20", "20", ""},
		{"without a current value there is no step", compute("0"), "82", "82", ""},

		{"a decrease within the step is applied as it is", compute("40"), "31", "31", ""},
		{"a larger decrease stops at the step", compute("40"), "3", "30", stepDown},
		{"the step down is rounded up to whole units", compute("30"), "3", "23", stepDown},
		{"a small value still moves down by a whole unit", compute("2"), "0", "1", stepDown},

		{"a decrease stops at the usage floor", withFloor(compute("40"), "36"), "3", "36",
			"usage floor: the median of the observed usage is 36"},
		{"the step applies when the floor is further away", withFloor(compute("40"), "12"), "3", "30", stepDown},
		{"a floor above the current value does not raise it", withFloor(compute("40"), "55"), "3", "40",
			"usage floor: the median of the observed usage is 55"},
		{"the floor does not limit an increase", withFloor(compute("20"), "70"), "25", "25", ""},
		{"a decrease to above the floor is applied as it is", withFloor(compute("40"), "30"), "35", "35", ""},

		{"no further increase while a worker has not received the previous one",
			withHeld(compute("30"), "20", "30"), "82", "30", boundNotAppliedYet},
		{"no further decrease while a worker has not received the previous one",
			withHeld(compute("30"), "30", "40"), "3", "30", boundNotAppliedYet},
		{"a worker behind does not hold back a change in the other direction",
			withHeld(compute("30"), "20", "30"), "25", "25", ""},
		{"workers at the current value do not hold anything back",
			withHeld(compute("30"), "30", "30"), "82", "45", stepUp},
	} {
		t.Run(test.name, func(t *testing.T) {
			applied, bound := test.bounds.apply(quantity(test.recommended))
			require.Equal(t, test.applied, applied.String())
			require.Equal(t, test.bound, bound)
		})
	}

	t.Run("memory moves in the same relative steps", func(t *testing.T) {
		bounds := quantityBounds{name: "VRAM limit", current: quantity("8Gi"), minStep: minVRAMStep}
		applied, bound := bounds.apply(quantity("40Gi"))
		require.Equal(t, "12Gi", applied.String())
		require.Equal(t, stepUp, bound)
		applied, bound = bounds.apply(quantity("1Gi"))
		require.Equal(t, "6Gi", applied.String())
		require.Equal(t, stepDown, bound)
	})
}

func TestCapAtDeviceCapacity(t *testing.T) {
	recommended := tfv1.Resources{
		Requests: tfv1.Resource{Tflops: quantity("75"), Vram: quantity("10Gi")},
		Limits:   tfv1.Resource{Tflops: quantity("82"), Vram: quantity("48Gi")},
	}

	t.Run("one GPU", func(t *testing.T) {
		capped, notes := capAtDeviceCapacity(recommended, &workload.DeviceCapacity{
			PerGPU: tfv1.Resource{Tflops: quantity("71.2"), Vram: quantity("24Gi")}, GPUsPerWorker: 1,
		})
		// Request and limit alike, in whole TFLOPS and not above 71.2.
		require.Equal(t, "71", capped.Requests.Tflops.String())
		require.Equal(t, "71", capped.Limits.Tflops.String())
		require.Equal(t, "10Gi", capped.Requests.Vram.String())
		require.Equal(t, "24Gi", capped.Limits.Vram.String())
		require.Equal(t, []string{
			"Compute request: recommended 75, applied 71 (device capacity: 71200m per GPU held)",
			"Compute limit: recommended 82, applied 71 (device capacity: 71200m per GPU held)",
			"VRAM limit: recommended 48Gi, applied 24Gi (device capacity: 24Gi per GPU held)",
		}, notes)
	})

	t.Run("several GPUs", func(t *testing.T) {
		// A worker on two GPUs gets its request and limit on each of them, so the
		// values are capped at what one GPU has (the smaller one, were they
		// different) and the worker as a whole at the sum.
		capped, notes := capAtDeviceCapacity(recommended, &workload.DeviceCapacity{
			PerGPU: tfv1.Resource{Tflops: quantity("71.2"), Vram: quantity("24Gi")}, GPUsPerWorker: 2,
		})
		require.Equal(t, "71", capped.Limits.Tflops.String())
		require.LessOrEqual(t, 2*capped.Limits.Tflops.AsApproximateFloat64(), 2*71.2)
		require.Len(t, notes, 3)
		require.Contains(t, notes[1], "device capacity: 71200m per GPU, each of the 2 GPUs held gets this amount")
	})

	t.Run("values within the capacity are left alone", func(t *testing.T) {
		within := tfv1.Resources{
			Requests: tfv1.Resource{Tflops: quantity("30"), Vram: quantity("10Gi")},
			Limits:   tfv1.Resource{Tflops: quantity("71"), Vram: quantity("24Gi")},
		}
		capped, notes := capAtDeviceCapacity(within, &workload.DeviceCapacity{
			PerGPU: tfv1.Resource{Tflops: quantity("71.2"), Vram: quantity("24Gi")}, GPUsPerWorker: 1,
		})
		require.True(t, capped.Equal(&within))
		require.Empty(t, notes)
	})

	t.Run("an unknown capacity caps nothing", func(t *testing.T) {
		capped, notes := capAtDeviceCapacity(recommended, nil)
		require.True(t, capped.Equal(&recommended))
		require.Empty(t, notes)
		// Neither does a GPU that reports no TFLOPS capacity.
		capped, notes = capAtDeviceCapacity(recommended, &workload.DeviceCapacity{
			PerGPU: tfv1.Resource{Vram: quantity("24Gi")}, GPUsPerWorker: 1,
		})
		require.Equal(t, "82", capped.Limits.Tflops.String())
		require.Len(t, notes, 1)
	})
}

// newBoundsTestWorkload returns a workload past its initial delay whose
// workers hold the given resources.
func newBoundsTestWorkload(current tfv1.Resources) *workload.State {
	state := workload.NewWorkloadState()
	state.CreationTimestamp = metav1.NewTime(time.Now().Add(-3 * time.Hour))
	state.Spec.AutoScalingConfig.AutoSetResources = &tfv1.AutoSetResources{Enable: true}
	state.Spec.Resources = current
	return state
}

func resourceUpdateCondition(t *testing.T, state *workload.State) *metav1.Condition {
	t.Helper()
	_, _, status := state.StatusSnapshot()
	condition := meta.FindStatusCondition(status.Conditions, constants.ConditionStatusTypeResourceUpdate)
	require.NotNil(t, condition)
	return condition
}

func TestDecreaseIsBoundedByStepAndUsageFloor(t *testing.T) {
	current := tfv1.Resources{
		Requests: tfv1.Resource{Tflops: quantity("40"), Vram: quantity("20Gi")},
		Limits:   tfv1.Resource{Tflops: quantity("80"), Vram: quantity("40Gi")},
	}
	// The percentiles put the request at 8 TFLOPS / 5Gi and the memory limit at 10Gi.
	estimations := EstimatedResources{
		LowerBoundTflops: quantity("8"), TargetTflops: quantity("10"), UpperBoundTflops: quantity("20"),
		LowerBoundVram: quantity("5Gi"), TargetVram: quantity("8Gi"), UpperBoundVram: quantity("10Gi"),
	}

	t.Run("by the step", func(t *testing.T) {
		state := newBoundsTestWorkload(current)
		recommender := &PercentileRecommender{&fakeResourcesEstimator{&estimations}, nil}
		got, err := recommender.Recommend(context.Background(), state)
		require.NoError(t, err)
		require.NotNil(t, got)
		// A quarter down: 40 -> 30, 20Gi -> 15Gi, 40Gi -> 30Gi.
		require.Equal(t, "30", got.Resources.Requests.Tflops.String())
		require.Equal(t, "15Gi", got.Resources.Requests.Vram.String())
		require.Equal(t, "30Gi", got.Resources.Limits.Vram.String())
		// The compute limit is not lowered below the workload's own.
		require.Equal(t, "80", got.Resources.Limits.Tflops.String())

		message := resourceUpdateCondition(t, state).Message
		require.Contains(t, message, "Compute scaled: request 40 -> 30, limit 80 -> 80")
		require.Contains(t, message, "Compute request: recommended 8, applied 30 (step limit: at most -25% per change)")
		require.Contains(t, message, "VRAM limit: recommended 10Gi, applied 30Gi (step limit: at most -25% per change)")
		require.Equal(t, state.ChangeBounds(), message[len(message)-len(state.ChangeBounds()):],
			"the bounds are kept for the scaling event")
	})

	t.Run("by the usage floor", func(t *testing.T) {
		withFloors := estimations
		// The worker's median compute usage is 36 TFLOPS and it was seen holding 38Gi.
		withFloors.RequestFloor = tfv1.Resource{Tflops: quantity("36"), Vram: quantity("4Gi")}
		withFloors.LimitFloor = tfv1.Resource{Tflops: quantity("60"), Vram: quantity("38Gi")}
		state := newBoundsTestWorkload(current)
		recommender := &PercentileRecommender{&fakeResourcesEstimator{&withFloors}, nil}
		got, err := recommender.Recommend(context.Background(), state)
		require.NoError(t, err)
		require.NotNil(t, got)
		require.Equal(t, "36", got.Resources.Requests.Tflops.String(), "not below the observed usage")
		require.Equal(t, "38Gi", got.Resources.Limits.Vram.String(), "not below the observed usage")
		// The floor of the memory request (4Gi) is further away than the step.
		require.Equal(t, "15Gi", got.Resources.Requests.Vram.String())

		message := resourceUpdateCondition(t, state).Message
		require.Contains(t, message, "Compute request: recommended 8, applied 36 (usage floor: the median of the observed usage is 36)")
		require.Contains(t, message, "VRAM limit: recommended 10Gi, applied 38Gi (usage floor: the highest observed usage is 38Gi)")

		// Applied, the next change does not go below the floor either.
		state.SetRecommendation(&got.Resources)
		got, err = recommender.Recommend(context.Background(), state)
		require.NoError(t, err)
		require.NotNil(t, got)
		require.Equal(t, "36", got.Resources.Requests.Tflops.String())
		require.Equal(t, "38Gi", got.Resources.Limits.Vram.String())
	})
}

func TestMinimumIntervalBetweenChanges(t *testing.T) {
	current := tfv1.Resources{
		Requests: tfv1.Resource{Tflops: quantity("10"), Vram: quantity("4Gi")},
		Limits:   tfv1.Resource{Tflops: quantity("20"), Vram: quantity("8Gi")},
	}
	estimations := EstimatedResources{
		LowerBoundTflops: quantity("60"), TargetTflops: quantity("70"), UpperBoundTflops: quantity("82"),
		LowerBoundVram: quantity("4Gi"), TargetVram: quantity("6Gi"), UpperBoundVram: quantity("8Gi"),
	}
	state := newBoundsTestWorkload(current)
	recommender := &PercentileRecommender{&fakeResourcesEstimator{&estimations}, nil}

	changedAt := time.Now()
	clock := changedAt
	timeNow = func() time.Time { return clock }
	t.Cleanup(func() { timeNow = time.Now })

	// Before any change was made there is nothing to wait for.
	got, err := recommender.Recommend(context.Background(), state)
	require.NoError(t, err)
	require.False(t, got.HasApplied)
	require.Equal(t, "30", got.Resources.Limits.Tflops.String())

	// The workers' resources were just changed.
	state.SetRecommendation(&got.Resources)
	state.MarkResourcesChanged(changedAt)
	for _, sinceChange := range []time.Duration{30 * time.Second, minChangeInterval - time.Second} {
		clock = changedAt.Add(sinceChange)
		got, err = recommender.Recommend(context.Background(), state)
		require.NoError(t, err)
		require.True(t, got.HasApplied, "a change %s after the previous one", sinceChange)
		require.True(t, got.Resources.IsZero())
		condition := resourceUpdateCondition(t, state)
		require.Equal(t, "MinChangeInterval", condition.Reason)
		require.Contains(t, condition.Message, "minimum interval 5m0s")
		// The same recommendation over the recommenders leaves the workload as it is.
		recommendation, err := GetRecommendation(context.Background(), state, []Interface{recommender})
		require.NoError(t, err)
		require.Nil(t, recommendation)
	}

	clock = changedAt.Add(minChangeInterval)
	got, err = recommender.Recommend(context.Background(), state)
	require.NoError(t, err)
	require.False(t, got.HasApplied)
	require.Equal(t, "45", got.Resources.Limits.Tflops.String(), "the next step")
	require.Equal(t, "Updated", resourceUpdateCondition(t, state).Reason)
}

func TestNoFurtherStepWhileAWorkerIsBehind(t *testing.T) {
	current := tfv1.Resources{
		Requests: tfv1.Resource{Tflops: quantity("10"), Vram: quantity("4Gi")},
		Limits:   tfv1.Resource{Tflops: quantity("20"), Vram: quantity("8Gi")},
	}
	estimations := EstimatedResources{
		LowerBoundTflops: quantity("60"), TargetTflops: quantity("70"), UpperBoundTflops: quantity("82"),
		LowerBoundVram: quantity("4Gi"), TargetVram: quantity("6Gi"), UpperBoundVram: quantity("8Gi"),
	}
	worker := func(name, request, limit string) *corev1.Pod {
		return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name, Annotations: map[string]string{
			constants.TFLOPSRequestAnnotation: request,
			constants.TFLOPSLimitAnnotation:   limit,
			constants.VRAMRequestAnnotation:   "4Gi",
			constants.VRAMLimitAnnotation:     "8Gi",
		}}}
	}
	state := newBoundsTestWorkload(current)
	// The first step (15 / 30) reached one worker; changing the other failed.
	state.SetRecommendation(&tfv1.Resources{
		Requests: tfv1.Resource{Tflops: quantity("15"), Vram: quantity("4Gi")},
		Limits:   tfv1.Resource{Tflops: quantity("30"), Vram: quantity("8Gi")},
	})
	state.CurrentActiveWorkers = map[string]*corev1.Pod{
		"worker-0": worker("worker-0", "15", "30"),
		"worker-1": worker("worker-1", "10", "20"),
	}
	recommender := &PercentileRecommender{&fakeResourcesEstimator{&estimations}, nil}

	got, err := recommender.Recommend(context.Background(), state)
	require.NoError(t, err)
	require.NotNil(t, got)
	// A second step would take worker-1 from 20 to 45 in one change.
	require.True(t, got.HasApplied)
	require.Equal(t, "30", got.Resources.Limits.Tflops.String())

	// Once the worker has caught up, the next step is made.
	state.UpdateWorkerAnnotations("worker-1", worker("worker-1", "15", "30").Annotations)
	got, err = recommender.Recommend(context.Background(), state)
	require.NoError(t, err)
	require.False(t, got.HasApplied)
	require.Equal(t, "22", got.Resources.Requests.Tflops.String())
	require.Equal(t, "45", got.Resources.Limits.Tflops.String())

	// Workers that differ from the workload's own resources, with no
	// recommendation made yet, are not waited for: nothing would bring them there.
	fresh := newBoundsTestWorkload(tfv1.Resources{
		Requests: tfv1.Resource{Tflops: quantity("20"), Vram: quantity("4Gi")},
		Limits:   tfv1.Resource{Tflops: quantity("40"), Vram: quantity("8Gi")},
	})
	fresh.CurrentActiveWorkers = map[string]*corev1.Pod{"worker-0": worker("worker-0", "10", "20")}
	got, err = recommender.Recommend(context.Background(), fresh)
	require.NoError(t, err)
	require.False(t, got.HasApplied)
	require.Equal(t, "60", got.Resources.Limits.Tflops.String())
}
