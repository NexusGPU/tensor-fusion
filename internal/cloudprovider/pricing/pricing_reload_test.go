package pricing

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Instance types of the embedded data used by these tests.
const (
	awsL4Instance    = "g6.xlarge" // 1x L4
	awsA10GInstance  = "g5.xlarge" // 1x A10G
	azureV100Spot    = "NC24s v3"  // 4x V100, no reserved price
	l4OnDemandPrice  = 0.8048
	l4ReservedPrice  = 0.5239
	v100OnDemand     = 12.24
	v100Spot         = 2.0735
	testRegion       = "us-east-2"
	longTestTimeout  = 30 * time.Second
	shortTestTimeout = 300 * time.Millisecond
)

func testGpuInfo(model string, fp16TFlops string) config.GpuInfo {
	return config.GpuInfo{
		Model:         model,
		Vendor:        "NVIDIA",
		Fp16TFlops:    resource.MustParse(fp16TFlops),
		FullModelName: "NVIDIA " + model,
	}
}

// newIsolatedProvider returns a provider whose tables are separate from the
// process-wide ones, so a test controls exactly what was loaded and when.
func newIsolatedProvider(firstLoadTimeout time.Duration) (*StaticPricingProvider, *pricingStore) {
	store := newPricingStore(firstLoadTimeout)
	return &StaticPricingProvider{store: store}, store
}

// (a) A cluster that starts with no ProviderConfig has no GPU info. Adding one
// later must make prices available through the exported entry point that
// start-up and the ProviderConfig controller call, without a restart.
func TestReload_PricesBecomeAvailableAfterGpuInfoIsAdded(t *testing.T) {
	ctx := t.Context()
	t.Cleanup(func() { SetTflopsMapAndInitGPUPricingInfo(context.Background(), config.MockGpuInfo()) })
	provider := NewStaticPricingProvider()

	SetTflopsMapAndInitGPUPricingInfo(ctx, &[]config.GpuInfo{})

	_, err := provider.LookupPricing(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion)
	require.ErrorIs(t, err, ErrInstanceTypeNotFound)
	price, found := provider.GetPricing(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion)
	assert.False(t, found)
	assert.Zero(t, price)
	instanceTypes, found := provider.GetRegionalGPUNodeInstanceTypes(testRegion)
	assert.False(t, found)
	assert.Empty(t, instanceTypes)

	SetTflopsMapAndInitGPUPricingInfo(ctx, &[]config.GpuInfo{testGpuInfo("L4", "121")})

	price, err = provider.LookupPricing(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion)
	require.NoError(t, err)
	assert.InDelta(t, l4OnDemandPrice, price, 1e-9)
	price, found = provider.GetPricing(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion)
	assert.True(t, found)
	assert.InDelta(t, l4OnDemandPrice, price, 1e-9)

	instanceTypes, found = provider.GetRegionalGPUNodeInstanceTypes(testRegion)
	require.True(t, found)
	for _, instanceType := range instanceTypes {
		assert.Equal(t, "L4", instanceType.GPUModel)
		assert.InDelta(t, 121.0, instanceType.FP16TFlopsPerGPU, 1e-9)
	}

	// Removing the GPU info again (the ProviderConfig was deleted) removes the
	// instance types that depended on it.
	SetTflopsMapAndInitGPUPricingInfo(ctx, &[]config.GpuInfo{})
	_, err = provider.LookupPricing(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion)
	require.ErrorIs(t, err, ErrInstanceTypeNotFound)
}

// (b) Correcting the TFlops of a model must change everything derived from it.
func TestReload_TflopsUpdateChangesDerivedValues(t *testing.T) {
	ctx := t.Context()
	provider, store := newIsolatedProvider(longTestTimeout)

	store.load(ctx, []config.GpuInfo{testGpuInfo("L4", "121")})

	info, err := provider.LookupGPUNodeInstanceTypeInfo(awsL4Instance, testRegion)
	require.NoError(t, err)
	assert.InDelta(t, 121.0, info.FP16TFlopsPerGPU, 1e-9)
	tflops, vram, err := provider.LookupGPUCapacityByModel("L4")
	require.NoError(t, err)
	assert.InDelta(t, 121.0, tflops.AsApproximateFloat64(), 1e-9)
	assert.Equal(t, int64(24)*constants.GiBToBytes, vram.Value())

	store.load(ctx, []config.GpuInfo{testGpuInfo("L4", "242")})

	info, found := provider.GetGPUNodeInstanceTypeInfoByInstance(awsL4Instance, testRegion)
	require.True(t, found)
	assert.InDelta(t, 242.0, info.FP16TFlopsPerGPU, 1e-9)
	tflops, _, found = provider.GetGPUCapacityByModel("NVIDIA L4")
	require.True(t, found)
	assert.InDelta(t, 242.0, tflops.AsApproximateFloat64(), 1e-9)
	instanceTypes, found := provider.GetRegionalGPUNodeInstanceTypes(testRegion)
	require.True(t, found)
	for _, instanceType := range instanceTypes {
		assert.InDelta(t, 242.0, instanceType.FP16TFlopsPerGPU, 1e-9, instanceType.InstanceType)
	}
}

// (c) Readers running while the tables are rebuilt must always see one
// complete generation. The two generations differ in both the TFlops value and
// the set of instance types, so a mixed or partly filled table is detectable.
// Run with -race.
func TestReload_ConcurrentReadersSeeCompleteTables(t *testing.T) {
	ctx := t.Context()
	provider, store := newIsolatedProvider(longTestTimeout)

	small := []config.GpuInfo{testGpuInfo("L4", "100")}
	large := []config.GpuInfo{testGpuInfo("L4", "200"), testGpuInfo("A10G", "200"), testGpuInfo("T4", "200")}

	store.load(ctx, small)
	smallTypes, err := provider.LookupRegionalGPUNodeInstanceTypes(testRegion)
	require.NoError(t, err)
	store.load(ctx, large)
	largeTypes, err := provider.LookupRegionalGPUNodeInstanceTypes(testRegion)
	require.NoError(t, err)
	require.NotEmpty(t, smallTypes)
	require.Greater(t, len(largeTypes), len(smallTypes))

	stop := make(chan struct{})
	var readers sync.WaitGroup
	var reads atomic.Int64
	failures := make(chan string, 16)
	report := func(msg string) {
		select {
		case failures <- msg:
		default:
		}
	}

	for range 8 {
		readers.Go(func() {
			for {
				select {
				case <-stop:
					return
				default:
				}

				reads.Add(1)
				instanceTypes, found := provider.GetRegionalGPUNodeInstanceTypes(testRegion)
				if !found {
					report("instance types disappeared during a reload")
					continue
				}
				tflops := instanceTypes[0].FP16TFlopsPerGPU
				want := len(smallTypes)
				if tflops == 200 {
					want = len(largeTypes)
				}
				if len(instanceTypes) != want {
					report("instance type list does not match its generation")
				}
				for _, instanceType := range instanceTypes {
					if instanceType.FP16TFlopsPerGPU != tflops {
						report("instance types of two generations in one result")
						break
					}
				}

				if price, found := provider.GetPricing(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion); !found || price <= 0 {
					report("price of an instance type present in both generations was not found")
				}
				if info, found := provider.GetGPUNodeInstanceTypeInfoByInstance(awsL4Instance, testRegion); !found ||
					(info.FP16TFlopsPerGPU != 100 && info.FP16TFlopsPerGPU != 200) {
					report("instance info of an instance type present in both generations was wrong")
				}
				if _, _, found := provider.GetGPUCapacityByModel("L4"); !found {
					report("capacity of a model present in both generations was not found")
				}
				// Only in the large generation; must be absent or complete.
				_, _ = provider.GetPricing(awsA10GInstance, tfv1.CapacityTypeSpot, testRegion)
				_, _, _ = provider.GetGPUCapacityByModel("A10G")
			}
		})
	}

	var writers sync.WaitGroup
	for w := range 2 {
		writers.Go(func() {
			for i := range 300 {
				if (i+w)%2 == 0 {
					store.load(ctx, small)
				} else {
					store.load(ctx, large)
				}
			}
		})
	}
	writers.Wait()
	close(stop)
	readers.Wait()
	close(failures)
	require.Positive(t, reads.Load(), "readers must have run while the tables were reloaded")

	for msg := range failures {
		t.Error(msg)
	}
}

// (d) A reader that arrives before anything was loaded waits for a bounded
// time and then gets an error, never a zero price or a panic. The wait is
// spent once: later readers fail immediately.
func TestReaderBeforeFirstLoad_TimesOutWithError(t *testing.T) {
	provider, store := newIsolatedProvider(shortTestTimeout)

	start := time.Now()
	price, err := provider.LookupPricing(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion)
	waited := time.Since(start)
	require.ErrorIs(t, err, ErrPricingNotLoaded)
	assert.Zero(t, price)
	assert.GreaterOrEqual(t, waited, shortTestTimeout-20*time.Millisecond, "the reader must wait for the first load")
	assert.Less(t, waited, longTestTimeout, "the wait must be bounded")

	start = time.Now()
	_, err = provider.LookupRegionalGPUNodeInstanceTypes(testRegion)
	require.ErrorIs(t, err, ErrPricingNotLoaded)
	_, _, err = provider.LookupGPUCapacityByModel("L4")
	require.ErrorIs(t, err, ErrPricingNotLoaded)
	_, err = provider.LookupGPUNodeInstanceTypeInfo(awsL4Instance, testRegion)
	require.ErrorIs(t, err, ErrPricingNotLoaded)
	_, err = provider.HourlyPriceForCapacityType(awsL4Instance, tfv1.CapacityTypeSpot, testRegion)
	require.ErrorIs(t, err, ErrPricingNotLoaded)

	// The signatures without an error report "not found".
	price, found := provider.GetPricing(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion)
	assert.False(t, found)
	assert.Zero(t, price)
	instanceTypes, found := provider.GetRegionalGPUNodeInstanceTypes(testRegion)
	assert.False(t, found)
	assert.Empty(t, instanceTypes)
	_, _, found = provider.GetGPUCapacityByModel("L4")
	assert.False(t, found)
	_, found = provider.GetGPUNodeInstanceTypeInfoByInstance(awsL4Instance, testRegion)
	assert.False(t, found)
	assert.Less(t, time.Since(start), shortTestTimeout, "readers after the deadline must not wait again")

	// A late first load still makes the data available.
	store.load(t.Context(), []config.GpuInfo{testGpuInfo("L4", "121")})
	price, err = provider.LookupPricing(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion)
	require.NoError(t, err)
	assert.InDelta(t, l4OnDemandPrice, price, 1e-9)
}

// (d) A reader that is already waiting is released by the first load.
func TestReaderBeforeFirstLoad_IsReleasedByTheLoad(t *testing.T) {
	provider, store := newIsolatedProvider(longTestTimeout)

	type result struct {
		price float64
		err   error
	}
	results := make(chan result, 4)
	for range 4 {
		go func() {
			price, err := provider.LookupPricing(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion)
			results <- result{price, err}
		}()
	}

	time.Sleep(50 * time.Millisecond)
	store.load(t.Context(), []config.GpuInfo{testGpuInfo("L4", "121")})

	for range 4 {
		select {
		case res := <-results:
			require.NoError(t, res.err)
			assert.InDelta(t, l4OnDemandPrice, res.price, 1e-9)
		case <-time.After(10 * time.Second):
			t.Fatal("reader was not released by the first load")
		}
	}
}

// (e) A GPU model that is not (or no longer) described by any ProviderConfig
// is reported as unknown; nothing dereferences its missing entry.
func TestMissingGPUModel_IsUnknownAndDoesNotPanic(t *testing.T) {
	ctx := t.Context()
	provider, store := newIsolatedProvider(longTestTimeout)

	store.load(ctx, []config.GpuInfo{testGpuInfo("L4", "121"), testGpuInfo("A10G", "125")})
	_, err := provider.LookupGPUNodeInstanceTypeInfo(awsA10GInstance, testRegion)
	require.NoError(t, err)

	// The A10G metadata goes away while its instance types are still in the
	// embedded instance data.
	store.load(ctx, []config.GpuInfo{testGpuInfo("L4", "121")})

	_, err = provider.LookupGPUNodeInstanceTypeInfo(awsA10GInstance, testRegion)
	require.ErrorIs(t, err, ErrInstanceTypeNotFound)
	info, found := provider.GetGPUNodeInstanceTypeInfoByInstance(awsA10GInstance, testRegion)
	assert.False(t, found)
	assert.Equal(t, types.GPUNodeInstanceInfo{}, info)
	_, found = provider.GetPricing(awsA10GInstance, tfv1.CapacityTypeOnDemand, testRegion)
	assert.False(t, found)
	_, _, err = provider.LookupGPUCapacityByModel("A10G")
	require.ErrorIs(t, err, ErrGPUModelUnknown)
	_, _, found = provider.GetGPUCapacityByModel("A10G")
	assert.False(t, found)

	instanceTypes, found := provider.GetRegionalGPUNodeInstanceTypes(testRegion)
	require.True(t, found)
	for _, instanceType := range instanceTypes {
		assert.Equal(t, "L4", instanceType.GPUModel)
	}
	assert.Contains(t, store.current.Load().skippedModels, "A10G")

	// Entries without GPU info are treated the same way.
	snap := buildSnapshot([]config.GpuInfo{testGpuInfo("L4", "121")})
	snap.gpuInfo["nil-entry"] = nil
	snap.gpuInfo["empty-entry"] = &CompleteGPUInfo{}
	store.current.Store(snap)
	for _, model := range []string{"nil-entry", "empty-entry", "never-configured", ""} {
		assert.NotPanics(t, func() {
			_, _, err := provider.LookupGPUCapacityByModel(model)
			require.ErrorIs(t, err, ErrGPUModelUnknown)
		}, model)
	}
	rows := []csvInstance{{info: types.GPUNodeInstanceInfo{InstanceType: "x.nil", GPUModel: "nil-entry"}}}
	skipped := map[string]struct{}{}
	assert.NotPanics(t, func() {
		assert.Empty(t, joinInstanceData(rows, snap.gpuInfo, skipped))
	})
	assert.Contains(t, skipped, "nil-entry")
}

// (e) The log lines about unknown models are written once, not on every
// lookup and not on every re-sync that reloads unchanged GPU info.
func TestMissingGPUModel_IsLoggedOnce(t *testing.T) {
	provider, store := newIsolatedProvider(longTestTimeout)

	var mu sync.Mutex
	var skippedLines []string
	logger := funcr.New(func(_, args string) {
		if strings.Contains(args, "instance types skipped") {
			mu.Lock()
			skippedLines = append(skippedLines, args)
			mu.Unlock()
		}
	}, funcr.Options{})
	ctx := log.IntoContext(t.Context(), logger)
	skippedLineCount := func() int {
		mu.Lock()
		defer mu.Unlock()
		return len(skippedLines)
	}

	gpuInfos := []config.GpuInfo{testGpuInfo("L4", "121")}
	for range 5 {
		store.load(ctx, gpuInfos)
		for range 20 {
			_, _, found := provider.GetGPUCapacityByModel("A10G")
			assert.False(t, found)
		}
	}
	assert.Equal(t, 1, skippedLineCount(), "unchanged reloads must not repeat the skipped models line")
	require.Len(t, skippedLines, 1)
	assert.Contains(t, skippedLines[0], "A10G")
	assert.Equal(t, int32(1), store.current.Load().warned.count.Load(), "repeated lookups of one unknown model log once")

	// A real change is reported again.
	store.load(ctx, []config.GpuInfo{testGpuInfo("L4", "121"), testGpuInfo("T4", "65")})
	assert.Equal(t, 2, skippedLineCount())
	assert.Equal(t, int32(0), store.current.Load().warned.count.Load())

	// Lookups with many different names cannot grow the record without bound.
	warned := &warnSet{}
	logged := 0
	for i := range maxWarnedKeys * 4 {
		if warned.first(strings.Repeat("m", i+1)) {
			logged++
		}
	}
	assert.Equal(t, maxWarnedKeys, logged)
}

// The rate of a node depends on how its capacity was bought.
func TestHourlyPriceForCapacityType(t *testing.T) {
	provider, store := newIsolatedProvider(longTestTimeout)
	store.load(t.Context(), []config.GpuInfo{testGpuInfo("L4", "121"), testGpuInfo("V100", "125")})

	// AWS: on-demand and reserved come from the instance data, spot is the
	// assumed discount on the on-demand price.
	price, err := provider.HourlyPriceForCapacityType(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion)
	require.NoError(t, err)
	assert.InDelta(t, l4OnDemandPrice, price, 1e-9)
	price, err = provider.HourlyPriceForCapacityType(awsL4Instance, tfv1.CapacityTypeReserved, testRegion)
	require.NoError(t, err)
	assert.InDelta(t, l4ReservedPrice, price, 1e-9)
	price, err = provider.HourlyPriceForCapacityType(awsL4Instance, tfv1.CapacityTypeSpot, testRegion)
	require.NoError(t, err)
	assert.InDelta(t, l4OnDemandPrice*constants.SpotInstanceAssumedDiscountRatio, price, 1e-9)
	assert.Less(t, price, l4OnDemandPrice, "a spot node must not be rated at the on-demand price")

	// Azure: the instance data has real spot prices.
	price, err = provider.HourlyPriceForCapacityType(azureV100Spot, tfv1.CapacityTypeSpot, "eastus")
	require.NoError(t, err)
	assert.InDelta(t, v100Spot, price, 1e-9)
	price, err = provider.HourlyPriceForCapacityType(azureV100Spot, tfv1.CapacityTypeOnDemand, "eastus")
	require.NoError(t, err)
	assert.InDelta(t, v100OnDemand, price, 1e-9)

	// A capacity type without a price in the data is an error, not $0.
	price, err = provider.HourlyPriceForCapacityType(azureV100Spot, tfv1.CapacityTypeReserved, "eastus")
	require.ErrorIs(t, err, ErrPriceUnavailable)
	assert.Zero(t, price)

	_, err = provider.HourlyPriceForCapacityType(awsL4Instance, tfv1.CapacityTypeEnum("burstable"), testRegion)
	require.ErrorIs(t, err, ErrUnsupportedCapacityType)
	_, err = provider.HourlyPriceForCapacityType("no-such-instance", tfv1.CapacityTypeSpot, testRegion)
	require.ErrorIs(t, err, ErrInstanceTypeNotFound)

	// The price used to compare instance types for provisioning is unchanged.
	price, err = provider.LookupPricing(awsL4Instance, tfv1.CapacityTypeSpot, testRegion)
	require.NoError(t, err)
	assert.InDelta(t, l4OnDemandPrice, price, 1e-9)
	_, err = provider.LookupPricing(awsL4Instance, tfv1.CapacityTypeEnum(""), testRegion)
	require.ErrorIs(t, err, ErrUnsupportedCapacityType)
}

func TestParseCapacityType(t *testing.T) {
	cases := map[string]tfv1.CapacityTypeEnum{
		"on-demand": tfv1.CapacityTypeOnDemand,
		"OnDemand":  tfv1.CapacityTypeOnDemand,
		"spot":      tfv1.CapacityTypeSpot,
		"Spot":      tfv1.CapacityTypeSpot,
		" SPOT ":    tfv1.CapacityTypeSpot,
		"reserved":  tfv1.CapacityTypeReserved,
		"Reserved":  tfv1.CapacityTypeReserved,
	}
	for value, want := range cases {
		got, ok := ParseCapacityType(value)
		assert.True(t, ok, value)
		assert.Equal(t, want, got, value)
	}
	for _, value := range []string{"", "unknown", "savings-plan"} {
		_, ok := ParseCapacityType(value)
		assert.False(t, ok, value)
	}
}

// A nil provider pointer is used by callers that never construct one.
func TestNilProviderReadsProcessWideTables(t *testing.T) {
	SetTflopsMapAndInitGPUPricingInfo(t.Context(), config.MockGpuInfo())
	var provider *StaticPricingProvider
	price, found := provider.GetPricing(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion)
	assert.True(t, found)
	assert.InDelta(t, l4OnDemandPrice, price, 1e-9)
}
