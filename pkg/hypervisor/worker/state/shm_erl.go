//go:build darwin || linux || freebsd || netbsd

package worker

import (
	"math"
	"sync/atomic"
)

// ERL token bucket operations for SharedDeviceInfoV2

// GetERLTokenCapacity returns the token capacity
func (d *SharedDeviceInfoV2) GetERLTokenCapacity() float64 {
	return math.Float64frombits(atomic.LoadUint64(&d.ERLTokenCapacity))
}

// SetERLTokenCapacity sets the token capacity
func (d *SharedDeviceInfoV2) SetERLTokenCapacity(capacity float64) {
	atomic.StoreUint64(&d.ERLTokenCapacity, math.Float64bits(capacity))
}

// GetERLTokenRefillRate returns the refill rate
func (d *SharedDeviceInfoV2) GetERLTokenRefillRate() float64 {
	return math.Float64frombits(atomic.LoadUint64(&d.ERLTokenRefillRate))
}

// SetERLTokenRefillRate sets the refill rate
func (d *SharedDeviceInfoV2) SetERLTokenRefillRate(rate float64) {
	atomic.StoreUint64(&d.ERLTokenRefillRate, math.Float64bits(rate))
}

// GetERLCurrentTokens returns the current tokens
func (d *SharedDeviceInfoV2) GetERLCurrentTokens() float64 {
	return math.Float64frombits(atomic.LoadUint64(&d.ERLCurrentTokens))
}

// SetERLCurrentTokens sets the current tokens
func (d *SharedDeviceInfoV2) SetERLCurrentTokens(tokens float64) {
	atomic.StoreUint64(&d.ERLCurrentTokens, math.Float64bits(tokens))
}

// GetERLLastTokenUpdate returns the last token update timestamp
func (d *SharedDeviceInfoV2) GetERLLastTokenUpdate() float64 {
	return math.Float64frombits(atomic.LoadUint64(&d.ERLLastTokenUpdate))
}

// SetERLLastTokenUpdate sets the last token update timestamp
func (d *SharedDeviceInfoV2) SetERLLastTokenUpdate(timestamp float64) {
	atomic.StoreUint64(&d.ERLLastTokenUpdate, math.Float64bits(timestamp))
}

// LoadERLTokenState loads the token state atomically
func (d *SharedDeviceInfoV2) LoadERLTokenState() (float64, float64) {
	return d.GetERLCurrentTokens(), d.GetERLLastTokenUpdate()
}

// StoreERLTokenState stores the token state atomically
func (d *SharedDeviceInfoV2) StoreERLTokenState(tokens, timestamp float64) {
	d.SetERLCurrentTokens(tokens)
	d.SetERLLastTokenUpdate(timestamp)
}

// LoadERLQuota loads the quota configuration
func (d *SharedDeviceInfoV2) LoadERLQuota() (float64, float64) {
	return d.GetERLTokenCapacity(), d.GetERLTokenRefillRate()
}

// FetchSubERLTokens atomically subtracts tokens and returns the value before subtraction
func (d *SharedDeviceInfoV2) FetchSubERLTokens(cost float64) float64 {
	for {
		currentBits := atomic.LoadUint64(&d.ERLCurrentTokens)
		current := math.Float64frombits(currentBits)

		if current < cost {
			return current
		}

		newValue := math.Max(0.0, current-cost)
		newBits := math.Float64bits(newValue)

		if atomic.CompareAndSwapUint64(&d.ERLCurrentTokens, currentBits, newBits) {
			return current
		}
	}
}

// FetchAddERLTokens atomically adds tokens (capped at capacity) and returns the value before addition.
// A negative balance is debt left by the limiter: the addition repays it rather
// than resetting the bucket to empty. The addition itself never creates debt.
func (d *SharedDeviceInfoV2) FetchAddERLTokens(amount float64) float64 {
	capacity := d.GetERLTokenCapacity()

	for {
		currentBits := atomic.LoadUint64(&d.ERLCurrentTokens)
		current := math.Float64frombits(currentBits)

		newValue := math.Max(math.Min(0.0, current), math.Min(capacity, current+amount))
		newBits := math.Float64bits(newValue)

		if atomic.CompareAndSwapUint64(&d.ERLCurrentTokens, currentBits, newBits) {
			return current
		}
	}
}

// erlCASAttempts bounds the retries of a compare-and-swap on a word that the
// workload's processes write as well. When that many attempts in a row have
// failed, the hypervisor leaves the bucket alone for this pass: the loop that
// serves every worker on the node does not wait on one bucket.
const erlCASAttempts = 16

// RefillERLTokens adds the tokens the bucket earned since it was last refilled
// and returns how many it added (negative when the balance was above the
// capacity and was brought down to it; zero when another writer got there
// first). It is safe to call from several writers at any cadence.
//
// The protocol is shared by every writer that refills a bucket: the hypervisor
// in normal operation, and limiter processes while the hypervisor's heartbeat
// is stale. It only uses atomic operations on existing fields, so the layout is
// unchanged:
//
//  1. Load erl_last_token_update (seconds, f64 bits) and take the time elapsed
//     since then on the same clock. Nothing has been earned if it is not
//     positive.
//  2. Compare-and-swap the timestamp forward to now. A writer that loses the
//     swap adds nothing: the winner accounts for that span.
//  3. The winner alone adds rate x elapsed to erl_current_tokens, never taking
//     the balance above capacity.
//
// The span is therefore credited exactly once however many writers there are,
// and handing refilling back and forth neither loses nor doubles tokens.
//
// elapsed is capped at maxElapsed, so a long pause is caught up in one bounded
// step. A bucket that was never refilled only gets its timestamp set, and so
// does one whose timestamp lies further in the future than maxElapsed; a
// timestamp ahead by less is left alone until the clock has passed it. A balance
// below zero is debt and is repaid; a balance that is not a number is restarted
// from an empty bucket, because nothing can be added to it and a limiter cannot
// charge it.
//
// The clock is the wall clock: seconds since the Unix epoch as an IEEE-754
// double, stored as its bit pattern in the u64 field, the same value a limiter
// computes from clock_gettime(CLOCK_REALTIME). The heartbeat is on the same
// clock, in whole seconds as a plain u64.
func (d *SharedDeviceInfoV2) RefillERLTokens(nowSecs, rate, capacity, maxElapsed float64) float64 {
	elapsed, _, claimed := d.ClaimERLRefill(nowSecs, maxElapsed)
	if !claimed {
		return 0
	}
	return d.AddERLTokens(rate*elapsed, capacity)
}

// ClaimERLRefill is steps 1 and 2 of the refill protocol: it claims the time
// since the bucket was last refilled by advancing erl_last_token_update to
// nowSecs. It returns the span claimed, in seconds and capped at maxElapsed
// (zero when the claim only set a timestamp that was unusable), the timestamp
// it found as stored (f64 bits), and whether the claim succeeded. A writer that
// remembers the timestamp it stored last can tell from the one it finds whether
// somebody else refilled in between. Only a successful claim may be followed by
// AddERLTokens.
func (d *SharedDeviceInfoV2) ClaimERLRefill(nowSecs, maxElapsed float64) (elapsed float64, found uint64, claimed bool) {
	found = atomic.LoadUint64(&d.ERLLastTokenUpdate)
	last := math.Float64frombits(found)
	elapsed = nowSecs - last
	switch {
	case math.IsNaN(elapsed) || math.IsInf(elapsed, 0) || last <= 0 || elapsed < -maxElapsed:
		// Never refilled, or not a time anybody refilling this bucket could have
		// stored: further ahead than the longest span that is ever caught up.
		// The timestamp is set, and nothing is credited for getting there.
		return 0, found, atomic.CompareAndSwapUint64(&d.ERLLastTokenUpdate, found, math.Float64bits(nowSecs))
	case elapsed <= 0:
		// Up to date, or stamped by a writer whose clock reading is newer than
		// this caller's (a pass that was held up after reading its clock). The
		// timestamp is never moved backwards: the span up to it is credited
		// already, and moving it would credit that span again.
		return 0, found, false
	}
	if !atomic.CompareAndSwapUint64(&d.ERLLastTokenUpdate, found, math.Float64bits(nowSecs)) {
		return 0, found, false
	}
	return math.Min(elapsed, maxElapsed), found, true
}

// RestartERLRefillClock sets erl_last_token_update to nowSecs if it still holds
// found, the timestamp a claim returned, and reports whether it did. It credits
// nothing and is not part of the refill arithmetic (the test vectors do not
// change): it is for a writer that has established, over several of its own
// passes, that the timestamp it found lies ahead of the clock and that nobody
// is advancing it, as after the wall clock was set back. ClaimERLRefill alone
// leaves such a timestamp in place until the clock has passed it, which stops
// refilling for the length of the step (up to maxElapsed). The swap fails, and
// nothing changes, when anybody stored a timestamp in the meantime.
func (d *SharedDeviceInfoV2) RestartERLRefillClock(found uint64, nowSecs float64) bool {
	if math.IsNaN(nowSecs) || math.IsInf(nowSecs, 0) || nowSecs <= 0 {
		return false
	}
	return atomic.CompareAndSwapUint64(&d.ERLLastTokenUpdate, found, math.Float64bits(nowSecs))
}

// AddERLTokens is step 3 of the refill protocol: it adds amount to the balance
// without taking it above capacity, and returns the change (negative when the
// balance was above the capacity and was brought down to it). A balance below
// zero is debt and is repaid; one that is not a number is restarted from an
// empty bucket. An amount that is not a positive number adds nothing.
func (d *SharedDeviceInfoV2) AddERLTokens(amount, capacity float64) float64 {
	if math.IsNaN(amount) || math.IsInf(amount, 0) || amount <= 0 || math.IsNaN(capacity) {
		return 0
	}
	for range erlCASAttempts {
		currentBits := atomic.LoadUint64(&d.ERLCurrentTokens)
		current := math.Float64frombits(currentBits)
		if math.IsNaN(current) || math.IsInf(current, 0) {
			current = 0
		}
		refilled := math.Min(capacity, current+amount)
		if atomic.CompareAndSwapUint64(&d.ERLCurrentTokens, currentBits, math.Float64bits(refilled)) {
			return refilled - current
		}
	}
	return 0
}

// DrainERLTokens atomically removes up to amount tokens without taking the
// balance below floor, and returns how many it removed.
func (d *SharedDeviceInfoV2) DrainERLTokens(amount, floor float64) float64 {
	if math.IsNaN(amount) || amount <= 0 {
		return 0
	}
	for range erlCASAttempts {
		currentBits := atomic.LoadUint64(&d.ERLCurrentTokens)
		current := math.Float64frombits(currentBits)
		// A balance that is not a number is left to the next refill.
		if math.IsNaN(current) || current <= floor {
			return 0
		}
		drained := math.Max(floor, current-amount)
		if atomic.CompareAndSwapUint64(&d.ERLCurrentTokens, currentBits, math.Float64bits(drained)) {
			return current - drained
		}
	}
	return 0
}
