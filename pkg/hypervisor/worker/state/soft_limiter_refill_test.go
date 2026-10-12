package worker

import (
	"encoding/hex"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/stretchr/testify/require"
)

func newRefillTestBucket(tokens, capacity, lastUpdate float64) *SharedDeviceInfoV2 {
	info := NewSharedDeviceInfoV2(1024, 50, 1<<30)
	info.SetERLCurrentTokens(tokens)
	info.SetERLTokenCapacity(capacity)
	info.SetERLLastTokenUpdate(lastUpdate)
	return info
}

func TestRefillERLTokensAddsWhatWasEarnedSinceTheLastRefill(t *testing.T) {
	const start = 1_800_000_000.0
	info := newRefillTestBucket(0, 2000, start)

	// 4000 tokens/s for 10 ms.
	require.InDelta(t, 40, info.RefillERLTokens(start+0.010, 4000, 2000, 5), 1e-3)
	require.InDelta(t, 40, info.GetERLCurrentTokens(), 1e-3)
	require.Equal(t, start+0.010, info.GetERLLastTokenUpdate())

	// The same instant earns nothing, and neither does an earlier one.
	require.Zero(t, info.RefillERLTokens(start+0.010, 4000, 2000, 5))
	require.Zero(t, info.RefillERLTokens(start+0.005, 4000, 2000, 5))
	require.Equal(t, start+0.010, info.GetERLLastTokenUpdate(), "the timestamp never moves backwards")

	// A stalled ticker is caught up in one step: 300 ms of tokens at once.
	require.InDelta(t, 1200, info.RefillERLTokens(start+0.310, 4000, 2000, 5), 1e-3)
	require.InDelta(t, 1240, info.GetERLCurrentTokens(), 1e-3)

	// Never beyond the capacity, however long the pause.
	require.InDelta(t, 760, info.RefillERLTokens(start+3, 4000, 2000, 5), 1e-3)
	require.Equal(t, 2000.0, info.GetERLCurrentTokens())
	require.Zero(t, info.RefillERLTokens(start+4, 4000, 2000, 5))
	require.Equal(t, 2000.0, info.GetERLCurrentTokens())
	require.Equal(t, start+4.0, info.GetERLLastTokenUpdate(), "a full bucket still advances the timestamp")

	// A balance above a capacity that shrank is brought down to it.
	require.Equal(t, -1500.0, info.RefillERLTokens(start+4.5, 4000, 500, 5))
	require.Equal(t, 500.0, info.GetERLCurrentTokens())
}

func TestRefillERLTokensCapsACatchUp(t *testing.T) {
	const start = 1_800_000_000.0
	// In debt, so the capacity does not hide the cap on the elapsed time.
	info := newRefillTestBucket(-100_000, 2000, start)
	require.InDelta(t, 20_000, info.RefillERLTokens(start+3600, 4000, 2000, 5), 1e-3)
	require.InDelta(t, -80_000, info.GetERLCurrentTokens(), 1e-3)
	require.Equal(t, start+3600.0, info.GetERLLastTokenUpdate())
}

func TestRefillERLTokensStartsFromAnUnusableTimestamp(t *testing.T) {
	const now = 1_800_000_000.0
	for name, last := range map[string]float64{
		"never refilled": 0, "negative": -5, "NaN": math.NaN(), "infinite": math.Inf(1),
		"further ahead than any catch-up": now + 600,
	} {
		t.Run(name, func(t *testing.T) {
			info := newRefillTestBucket(10, 2000, last)
			require.Zero(t, info.RefillERLTokens(now, 4000, 2000, 5), "nothing is known to have been earned")
			require.Equal(t, 10.0, info.GetERLCurrentTokens())
			require.Equal(t, now, info.GetERLLastTokenUpdate())
			require.InDelta(t, 40, info.RefillERLTokens(now+0.010, 4000, 2000, 5), 1e-3)
		})
	}

	// A timestamp ahead of this caller's clock by less than a catch-up belongs
	// to a writer that read its clock later. It is never moved backwards: the
	// time up to it is credited already.
	for _, ahead := range []float64{0.002, 0.3, 1.5, 4.9} {
		info := newRefillTestBucket(10, 2000, now+ahead)
		require.Zero(t, info.RefillERLTokens(now, 4000, 2000, 5))
		require.Equal(t, now+ahead, info.GetERLLastTokenUpdate(), "ahead by %v s", ahead)
	}
}

// A pass that reads its clock and is then held up must not undo what faster
// passes have credited in the meantime, however long it was held up.
func TestRefillWithAStaleClockCreditsNothingTwice(t *testing.T) {
	const start, rate = 1_800_000_000.0, 1000.0
	for _, stall := range []float64{0.3, 1.5, 3.0} {
		info := newRefillTestBucket(0, 1e9, start)
		// A slow pass read its clock at start+0.1; meanwhile the refill loop
		// carried on every 10 ms up to start+0.1+stall.
		credited := 0.0
		for now := start + 0.010; now <= start+0.1+stall+1e-9; now += 0.010 {
			credited += info.RefillERLTokens(now, rate, 1e9, 5)
		}
		upTo := info.GetERLLastTokenUpdate()
		// The slow pass arrives with its stale reading.
		require.Zero(t, info.RefillERLTokens(start+0.1, rate, 1e9, 5), "stall of %v s", stall)
		require.Equal(t, upTo, info.GetERLLastTokenUpdate(), "the timestamp moved backwards after a stall of %v s", stall)
		// And the refill loop carries on.
		credited += info.RefillERLTokens(upTo+0.010, rate, 1e9, 5)
		require.InEpsilon(t, rate*(upTo+0.010-start), credited, 1e-6, "stall of %v s", stall)
		require.InEpsilon(t, credited, info.GetERLCurrentTokens(), 1e-9)
	}
}

// The balance is a word the workload's processes write as well. A
// compare-and-swap on it that never succeeds must not hold up the caller.
func TestTokenUpdatesGiveUpOnAContendedBalance(t *testing.T) {
	info := newRefillTestBucket(100, 1e9, 1)
	stop := make(chan struct{})
	var writer sync.WaitGroup
	writer.Go(func() {
		for value := 100.0; ; value++ {
			select {
			case <-stop:
				return
			default:
				info.SetERLCurrentTokens(value)
			}
		}
	})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range 20000 {
			info.AddERLTokens(1, 1e9)
			info.DrainERLTokens(1, 0)
		}
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Error("token updates did not return while the balance was being rewritten")
	}
	close(stop)
	writer.Wait()
}

func TestRefillERLTokensHandlesDebtAndInvalidBalances(t *testing.T) {
	const start = 1_800_000_000.0
	// Debt is repaid, not forgiven.
	info := newRefillTestBucket(-700, 1000, start)
	require.InDelta(t, 250, info.RefillERLTokens(start+0.5, 500, 1000, 5), 1e-6)
	require.InDelta(t, -450, info.GetERLCurrentTokens(), 1e-6)

	for name, invalid := range map[string]float64{"NaN": math.NaN(), "+Inf": math.Inf(1), "-Inf": math.Inf(-1)} {
		info := newRefillTestBucket(invalid, 1000, start)
		require.InDelta(t, 250, info.RefillERLTokens(start+0.5, 500, 1000, 5), 1e-6, name)
		require.InDelta(t, 250, info.GetERLCurrentTokens(), 1e-6, name)
	}

	// A rate that is not a positive number adds nothing but still claims the span.
	for name, rate := range map[string]float64{"zero": 0, "negative": -5, "NaN": math.NaN(), "infinite": math.Inf(1)} {
		info := newRefillTestBucket(10, 1000, start)
		require.Zero(t, info.RefillERLTokens(start+0.5, rate, 1000, 5), name)
		require.Equal(t, 10.0, info.GetERLCurrentTokens(), name)
		require.Equal(t, start+0.5, info.GetERLLastTokenUpdate(), name)
	}
}

func TestDrainERLTokens(t *testing.T) {
	info := newRefillTestBucket(300, 1000, 1)
	require.Equal(t, 100.0, info.DrainERLTokens(100, 50))
	require.Equal(t, 200.0, info.GetERLCurrentTokens())
	require.Equal(t, 150.0, info.DrainERLTokens(500, 50), "never below the floor")
	require.Equal(t, 50.0, info.GetERLCurrentTokens())
	require.Zero(t, info.DrainERLTokens(500, 50))
	require.Zero(t, info.DrainERLTokens(-5, 0))
	require.Zero(t, info.DrainERLTokens(math.NaN(), 0))
	info.SetERLCurrentTokens(math.NaN())
	require.Zero(t, info.DrainERLTokens(5, 0))
}

// Several writers can refill one bucket: the hypervisor's two loops, and
// limiter processes while the hypervisor's heartbeat is stale. Whatever their
// interleaving, a span of time is never credited twice. A writer that claimed a
// span and then cannot get its tokens in after a bounded number of attempts
// gives up (it must not spin on a balance others keep changing), so under
// heavy contention a sliver of a span can go uncredited: never more than
// earned, and all but a negligible part of it.
func TestConcurrentRefillCreditsEverySpanOnce(t *testing.T) {
	const (
		start    = 1_800_000_000.0
		rate     = 5000.0
		writers  = 4
		attempts = 20_000
	)
	// Large enough that the capacity never limits what is added.
	info := newRefillTestBucket(0, 1e12, start)

	// One shared clock in microseconds; every attempt reads it afresh, so
	// writers race on overlapping and identical instants.
	var clockMicros atomic.Int64
	var credited [writers]float64
	var wins [writers]int
	var group sync.WaitGroup
	for writer := range writers {
		group.Go(func() {
			for attempt := range attempts {
				micros := clockMicros.Load()
				if attempt%3 == writer%3 {
					micros = clockMicros.Add(int64(50 + 37*writer))
				}
				added := info.RefillERLTokens(start+float64(micros)/1e6, rate, 1e12, 5)
				if added != 0 {
					credited[writer] += added
					wins[writer]++
				}
			}
		})
	}
	group.Wait()

	total := 0.0
	for writer := range writers {
		total += credited[writer]
		require.NotZero(t, wins[writer], "writer %d never won a refill; the test did not exercise a race", writer)
	}
	last := info.GetERLLastTokenUpdate()
	earned := rate * (last - start)
	require.LessOrEqual(t, last, start+float64(clockMicros.Load())/1e6)
	require.LessOrEqual(t, total, earned*(1+1e-9), "writers credited more than rate x elapsed")
	require.GreaterOrEqual(t, total, earned*(1-1e-3), "writers left more than a sliver uncredited")
	require.InEpsilon(t, total, info.GetERLCurrentTokens(), 1e-9, "the balance is what the writers say they added")
	require.LessOrEqual(t, info.GetERLCurrentTokens(), rate*float64(clockMicros.Load())/1e6*(1+1e-9))
}

// The byte image of one device's limiter state, as the limiter library and the
// hypervisor both have to read and write it (the device record of the shared
// layout, 64 bytes, little endian): up_limit u32 at 0, mem_limit u64 at 8,
// total_cuda_cores u32 at 16, pod_memory_used u64 at 24, then four IEEE-754
// doubles stored as their bit patterns: erl_token_refill_rate at 32,
// erl_token_capacity at 40, erl_current_tokens at 48 and erl_last_token_update
// at 56, the last one in seconds since the Unix epoch on the wall clock.
//
// The image holds: up_limit 57, mem_limit 8 GiB, total_cuda_cores 1343488,
// pod_memory_used 3 GiB, rate 2500.5, capacity 1250.25, tokens -3.5 and
// last update 1800000000.123456. The same bytes belong in a test on the
// limiter's side; whichever side changes its encoding breaks its own test.
const goldenDeviceInfoV2 = "3900000000000000" + "0000000002000000" + "0080140000000000" + "000000c000000000" +
	"000000000089a340" + "0000000000899340" + "0000000000000cc0" + "b4e6078074d2da41"

func TestDeviceInfoEncodingMatchesTheLimiterLayout(t *testing.T) {
	golden, err := hex.DecodeString(goldenDeviceInfoV2)
	require.NoError(t, err)
	require.Len(t, golden, 64)
	require.Equal(t, uintptr(64), unsafe.Sizeof(SharedDeviceInfoV2{}))

	// Written through the accessors the hypervisor uses.
	info := NewSharedDeviceInfoV2(1343488, 57, 8<<30)
	atomic.StoreUint64(&info.PodMemoryUsed, 3<<30)
	info.SetERLTokenRefillRate(2500.5)
	info.SetERLTokenCapacity(1250.25)
	info.SetERLCurrentTokens(-3.5)
	info.SetERLLastTokenUpdate(1800000000.123456)
	written := unsafe.Slice((*byte)(unsafe.Pointer(info)), 64)
	require.Equal(t, goldenDeviceInfoV2, hex.EncodeToString(written))

	// Read from an image a limiter wrote.
	var read SharedDeviceInfoV2
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&read)), 64), golden)
	require.Equal(t, uint32(57), read.UpLimit)
	require.Equal(t, uint64(8<<30), read.MemLimit)
	require.Equal(t, uint32(1343488), read.TotalCudaCores)
	require.Equal(t, uint64(3<<30), read.PodMemoryUsed)
	require.Equal(t, 2500.5, read.GetERLTokenRefillRate())
	require.Equal(t, 1250.25, read.GetERLTokenCapacity())
	require.Equal(t, -3.5, read.GetERLCurrentTokens())
	require.Equal(t, 1800000000.123456, read.GetERLLastTokenUpdate())

	// The refill timestamp is what clock_gettime(CLOCK_REALTIME) yields as
	// seconds + nanoseconds / 1e9, the heartbeat the same clock in whole seconds.
	at := time.Unix(1800000000, 123456000)
	require.InDelta(t, 1800000000.123456, float64(at.UnixMicro())/1e6, 1e-6)
	require.Equal(t, uint64(1800000000), uint64(at.Unix()))
	require.Equal(t, uint64(4745336555194279604), math.Float64bits(1800000000.123456))
}

// A claim reports the timestamp it found, so that a writer can tell whether
// somebody else refilled since its own last claim.
func TestClaimERLRefillReportsWhatItFound(t *testing.T) {
	const start = 1_800_000_000.0
	info := newRefillTestBucket(0, 1000, start)

	elapsed, found, claimed := info.ClaimERLRefill(start+0.010, 5)
	require.True(t, claimed)
	require.InDelta(t, 0.010, elapsed, 1e-6)
	require.Equal(t, math.Float64bits(start), found)
	own := math.Float64bits(start + 0.010)

	// Another writer (a limiter that found nobody refilling) claims a span.
	_, _, claimed = info.ClaimERLRefill(start+0.150, 5)
	require.True(t, claimed)
	require.Equal(t, 0.0, info.AddERLTokens(0, 1000))

	elapsed, found, claimed = info.ClaimERLRefill(start+0.160, 5)
	require.True(t, claimed)
	require.NotEqual(t, own, found, "the timestamp is not the one this writer stored")
	require.InDelta(t, 0.010, elapsed, 1e-6, "only the time since the other writer's claim is left")

	// Nothing to claim at the same instant, and the found timestamp says so.
	elapsed, found, claimed = info.ClaimERLRefill(start+0.160, 5)
	require.False(t, claimed)
	require.Zero(t, elapsed)
	require.Equal(t, math.Float64bits(start+0.160), found)
}

// After the wall clock was set back, the timestamp of the last refill lies
// ahead of it. A claim leaves it alone, so that a writer with a slightly older
// clock reading never credits a span twice; a writer that has seen it unchanged
// and ahead for several of its own passes restarts the clock from it, and that
// credits nothing.
func TestRestartERLRefillClock(t *testing.T) {
	bucket := newRefillTestBucket(40, 1000, 1000)

	// The clock reads three seconds less than the timestamp.
	elapsed, found, claimed := bucket.ClaimERLRefill(997, 5)
	if claimed || elapsed != 0 || math.Float64frombits(found) != 1000 {
		t.Fatalf("claim with the clock behind the timestamp = (%v, %v, %v), want it left alone",
			elapsed, math.Float64frombits(found), claimed)
	}
	if !bucket.RestartERLRefillClock(found, 997) {
		t.Fatal("the clock was not restarted from the timestamp that was found")
	}
	if got := bucket.GetERLLastTokenUpdate(); got != 997 {
		t.Fatalf("timestamp = %v, want 997", got)
	}
	if got := bucket.GetERLCurrentTokens(); got != 40 {
		t.Fatalf("restarting the clock changed the balance to %v", got)
	}
	// From there on refills credit the time that passes, once.
	if added := bucket.RefillERLTokens(997.5, 100, 1000, 5); added != 50 {
		t.Fatalf("refill after the restart added %v tokens, want 50", added)
	}

	// Somebody refilled in the meantime: the timestamp is theirs, nothing changes.
	if bucket.RestartERLRefillClock(found, 990) {
		t.Fatal("the clock was restarted although the timestamp had been advanced by a refill")
	}
	if got := bucket.GetERLLastTokenUpdate(); got != 997.5 {
		t.Fatalf("timestamp = %v, want 997.5", got)
	}
	for _, now := range []float64{math.NaN(), math.Inf(1), 0, -1} {
		if bucket.RestartERLRefillClock(math.Float64bits(997.5), now) {
			t.Fatalf("the clock was restarted to %v", now)
		}
	}
}
