package recommender

import (
	"fmt"
	"math"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"k8s.io/apimachinery/pkg/api/resource"
)

// Bounds on the changes the usage based (percentile) recommender makes to
// running workers. The autoscaling spec has no fields for them (it bounds the
// result against the original resources only, see Min/Max*ResourcesRatio, and
// sets how often the usage is evaluated), so they are compiled in.
//
// Cron scaling rules and the external scaler name the resources to apply
// themselves and are not stepped; like every recommendation they are capped at
// the capacity of the GPUs held (see recommendationProcessor.Apply).
const (
	// maxStepUpFraction and maxStepDownFraction limit how far one change moves a
	// request or limit away from its current value: up by at most half of it,
	// down by at most a quarter. A larger difference is covered in several
	// changes, each of them made on a fresh look at the usage. Lowering is the
	// slower direction because taking resources away from a running worker is
	// the riskier mistake.
	maxStepUpFraction   = 0.5
	maxStepDownFraction = 0.25

	// minChangeInterval is the time left between two changes of a workload's
	// running workers, whatever the evaluation interval is: a worker needs time
	// to show its usage under the new resources before they are changed again.
	minChangeInterval = 5 * time.Minute

	// A request is not lowered below the median of the observed usage, and a
	// limit not below the highest observed usage (no margin added to either),
	// whatever percentiles and ratios the workload configures.
	usageFloorRequestPercentile = 0.5
	usageFloorLimitPercentile   = 1.0
)

var (
	// The least a change may move a value, so that the rounding of a relative
	// step can not pin a small value where it is.
	minComputeStep = resource.MustParse("1")
	minVRAMStep    = resource.MustParse("256Mi")
)

// How the bounded quantities are called in messages.
const (
	computeRequest = scaleResourceCompute + " request"
	computeLimit   = scaleResourceCompute + " limit"
	vramRequest    = scaleResourceVram + " request"
	vramLimit      = scaleResourceVram + " limit"

	usageMedian  = "the median of the observed usage"
	usageHighest = "the highest observed usage"
)

// timeNow is the clock the minimum interval is measured with.
var timeNow = time.Now

const boundNotAppliedYet = "the previous change is not applied to all workers yet"

// quantityBounds bounds the change of one request or limit.
type quantityBounds struct {
	// name is how the quantity is called in messages, e.g. "Compute limit".
	name string
	// current is the value in effect: the latest recommendation, or the
	// workload's own resources when there is none yet.
	current resource.Quantity
	// lowestHeld and highestHeld are the lowest and the highest value running
	// workers hold; nil when that is not known.
	lowestHeld, highestHeld *resource.Quantity
	// usageFloor is the value a decrease does not go below; zero for none.
	usageFloor resource.Quantity
	// usageFloorSource says what the floor is, for messages.
	usageFloorSource string
	minStep          resource.Quantity
}

// apply returns the value to change to for the given recommended value, and
// the bound that made it differ from the recommended one ("" when none did).
func (b quantityBounds) apply(recommended resource.Quantity) (resource.Quantity, string) {
	// Without a current value there is nothing a step could be relative to.
	if b.current.IsZero() || recommended.Cmp(b.current) == 0 {
		return recommended, ""
	}
	current := b.current.AsApproximateFloat64()
	minStep := b.minStep.AsApproximateFloat64()

	if recommended.Cmp(b.current) > 0 {
		// While a worker is still below the current value, the previous increase
		// has not reached it; a further one would take it more than one step.
		if b.lowestHeld != nil && b.lowestHeld.Cmp(b.current) < 0 {
			return b.current.DeepCopy(), boundNotAppliedYet
		}
		highest := math.Max(math.Floor(current*(1+maxStepUpFraction)), current+minStep)
		if recommended.AsApproximateFloat64() > highest {
			return *resource.NewQuantity(int64(highest), recommended.Format),
				fmt.Sprintf("step limit: at most +%.0f%% per change", maxStepUpFraction*100)
		}
		return recommended, ""
	}

	if b.highestHeld != nil && b.highestHeld.Cmp(b.current) > 0 {
		return b.current.DeepCopy(), boundNotAppliedYet
	}
	value, bound := recommended, ""
	lowest := math.Max(math.Min(math.Ceil(current*(1-maxStepDownFraction)), current-minStep), 0)
	if value.AsApproximateFloat64() < lowest {
		value = *resource.NewQuantity(int64(lowest), recommended.Format)
		bound = fmt.Sprintf("step limit: at most -%.0f%% per change", maxStepDownFraction*100)
	}
	if !b.usageFloor.IsZero() && value.Cmp(b.usageFloor) < 0 {
		// The floor stops a decrease; it does not turn it into an increase.
		value = b.usageFloor.DeepCopy()
		if value.Cmp(b.current) > 0 {
			value = b.current.DeepCopy()
		}
		value.Format = recommended.Format
		bound = fmt.Sprintf("usage floor: %s is %s", b.usageFloorSource, b.usageFloor.String())
	}
	return value, bound
}

// boundNote words one bound that took effect: what was recommended, what is
// applied instead, and why.
func boundNote(name string, recommended, applied resource.Quantity, bound string) string {
	return fmt.Sprintf("%s: recommended %s, applied %s (%s)", name, recommended.String(), applied.String(), bound)
}

// boundChange applies the step limit and the usage floor to a recommendation
// of the usage based recommender, relative to the current resources. It
// returns the bounded recommendation and a note for every bound that took
// effect.
func boundChange(state *workload.State, current *tfv1.Resources, recommended tfv1.Resources, estimations *EstimatedResources) (tfv1.Resources, []string) {
	lowestHeld, highestHeld, heldKnown := state.WorkerResourcesRange()
	// A worker that is behind the current values is only waited for when those
	// are a recommendation: that one is applied again until every worker has
	// it. Workers that differ from the workload's own resources are not brought
	// to them by anything, and waiting for that would end all scaling.
	heldKnown = heldKnown && state.LatestRecommendation() != nil
	held := func(quantity *resource.Quantity) *resource.Quantity {
		if !heldKnown {
			return nil
		}
		return quantity
	}

	bounded := *recommended.DeepCopy()
	notes := []string{}
	for _, quantity := range []struct {
		value  *resource.Quantity
		bounds quantityBounds
	}{
		{&bounded.Requests.Tflops, quantityBounds{
			name: computeRequest, current: current.Requests.Tflops,
			lowestHeld: held(&lowestHeld.Requests.Tflops), highestHeld: held(&highestHeld.Requests.Tflops),
			usageFloor: estimations.RequestFloor.Tflops, usageFloorSource: usageMedian,
			minStep: minComputeStep,
		}},
		{&bounded.Limits.Tflops, quantityBounds{
			name: computeLimit, current: current.Limits.Tflops,
			lowestHeld: held(&lowestHeld.Limits.Tflops), highestHeld: held(&highestHeld.Limits.Tflops),
			usageFloor: estimations.LimitFloor.Tflops, usageFloorSource: usageHighest,
			minStep: minComputeStep,
		}},
		{&bounded.Requests.Vram, quantityBounds{
			name: vramRequest, current: current.Requests.Vram,
			lowestHeld: held(&lowestHeld.Requests.Vram), highestHeld: held(&highestHeld.Requests.Vram),
			usageFloor: estimations.RequestFloor.Vram, usageFloorSource: usageMedian,
			minStep: minVRAMStep,
		}},
		{&bounded.Limits.Vram, quantityBounds{
			name: vramLimit, current: current.Limits.Vram,
			lowestHeld: held(&lowestHeld.Limits.Vram), highestHeld: held(&highestHeld.Limits.Vram),
			usageFloor: estimations.LimitFloor.Vram, usageFloorSource: usageHighest,
			minStep: minVRAMStep,
		}},
	} {
		recommendedValue := quantity.value.DeepCopy()
		applied, bound := quantity.bounds.apply(recommendedValue)
		if bound == "" {
			continue
		}
		*quantity.value = applied
		notes = append(notes, boundNote(quantity.bounds.name, recommendedValue, applied, bound))
	}
	return bounded, notes
}

// capAtDeviceCapacity lowers the requests and limits of a recommendation to
// what the GPUs held by the workers have, and returns a note for every value
// it lowered. TFLOPS are capped at whole TFLOPS, as recommendations are made.
func capAtDeviceCapacity(recommended tfv1.Resources, capacity *workload.DeviceCapacity) (tfv1.Resources, []string) {
	if capacity == nil {
		return recommended, nil
	}
	tflopsCap := capacity.PerGPU.Tflops.DeepCopy()
	if whole := math.Floor(tflopsCap.AsApproximateFloat64()); whole >= 1 {
		tflopsCap = *resource.NewQuantity(int64(whole), resource.DecimalSI)
	}
	held := "GPU held"
	if capacity.GPUsPerWorker > 1 {
		held = fmt.Sprintf("GPU, each of the %d GPUs held gets this amount", capacity.GPUsPerWorker)
	}

	capped := *recommended.DeepCopy()
	notes := []string{}
	for _, quantity := range []struct {
		name     string
		value    *resource.Quantity
		limit    resource.Quantity
		capacity resource.Quantity
	}{
		{computeRequest, &capped.Requests.Tflops, tflopsCap, capacity.PerGPU.Tflops},
		{computeLimit, &capped.Limits.Tflops, tflopsCap, capacity.PerGPU.Tflops},
		{vramRequest, &capped.Requests.Vram, capacity.PerGPU.Vram, capacity.PerGPU.Vram},
		{vramLimit, &capped.Limits.Vram, capacity.PerGPU.Vram, capacity.PerGPU.Vram},
	} {
		// A capacity of zero is one that is not reported, not a GPU without any.
		if quantity.capacity.IsZero() || quantity.value.Cmp(quantity.capacity) <= 0 {
			continue
		}
		recommendedValue := quantity.value.DeepCopy()
		*quantity.value = quantity.limit.DeepCopy()
		notes = append(notes, boundNote(quantity.name, recommendedValue, *quantity.value,
			fmt.Sprintf("device capacity: %s per %s", quantity.capacity.String(), held)))
	}
	return capped, notes
}

// describeBounds joins the notes of the bounds that took effect into the text
// shown in the workload's status condition and in the scaling event.
func describeBounds(notes []string) string {
	if len(notes) == 0 {
		return ""
	}
	return "bounds: " + strings.Join(notes, "; ")
}
