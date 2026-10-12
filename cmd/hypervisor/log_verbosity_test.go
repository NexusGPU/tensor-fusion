package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/klog/v2"
)

// verbosityTestClock stands in for the wall clock: timers fire when the test
// moves the time past them.
type verbosityTestClock struct {
	mu     sync.Mutex
	now    time.Duration
	timers []*verbosityTestTimer
}

type verbosityTestTimer struct {
	clock   *verbosityTestClock
	at      time.Duration
	f       func()
	stopped bool
	fired   bool
}

func (t *verbosityTestTimer) Stop() bool {
	t.clock.mu.Lock()
	defer t.clock.mu.Unlock()
	pending := !t.stopped && !t.fired
	t.stopped = true
	return pending
}

func (c *verbosityTestClock) afterFunc(d time.Duration, f func()) logVerbosityTimer {
	c.mu.Lock()
	defer c.mu.Unlock()
	timer := &verbosityTestTimer{clock: c, at: c.now + d, f: f}
	c.timers = append(c.timers, timer)
	return timer
}

func (c *verbosityTestClock) advance(d time.Duration) {
	c.mu.Lock()
	c.now += d
	var due []*verbosityTestTimer
	for _, timer := range c.timers {
		if !timer.stopped && !timer.fired && timer.at <= c.now {
			timer.fired = true
			due = append(due, timer)
		}
	}
	c.mu.Unlock()
	for _, timer := range due {
		timer.f()
	}
}

func (c *verbosityTestClock) pending() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	pending := 0
	for _, timer := range c.timers {
		if !timer.stopped && !timer.fired {
			pending++
		}
	}
	return pending
}

// verbosityTestLevel is the verbosity the raiser sets, with every value it was
// set to and every line the raiser logged.
type verbosityTestLevel struct {
	mu      sync.Mutex
	level   int
	sets    []int
	lines   []string
	changed chan int
}

func (l *verbosityTestLevel) set(level int) error {
	l.mu.Lock()
	l.level = level
	l.sets = append(l.sets, level)
	l.mu.Unlock()
	if l.changed != nil {
		l.changed <- level
	}
	return nil
}

func (l *verbosityTestLevel) logf(format string, args ...any) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.lines = append(l.lines, fmt.Sprintf(format, args...))
}

func (l *verbosityTestLevel) get() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.level
}

func (l *verbosityTestLevel) logged() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.lines...)
}

func newTestVerbosityRaiser(configured int) (*logVerbosityRaiser, *verbosityTestLevel, *verbosityTestClock) {
	level := &verbosityTestLevel{level: configured}
	clock := &verbosityTestClock{}
	return &logVerbosityRaiser{
		configured: configured,
		setLevel:   level.set,
		afterFunc:  clock.afterFunc,
		logf:       level.logf,
	}, level, clock
}

func wantVerbosity(t *testing.T, level *verbosityTestLevel, want int) {
	t.Helper()
	if got := level.get(); got != want {
		t.Fatalf("log verbosity = %d, want %d", got, want)
	}
}

func TestLogVerbosityRaiseLastsTenMinutesThenReverts(t *testing.T) {
	raiser, level, clock := newTestVerbosityRaiser(2)

	raiser.Raise()
	wantVerbosity(t, level, 4)
	if lines := level.logged(); len(lines) != 1 || !strings.Contains(lines[0], "raised from 2 to 4 for 10m0s") {
		t.Fatalf("log lines after the raise = %q, want one line about the raise", lines)
	}

	clock.advance(10*time.Minute - time.Nanosecond)
	wantVerbosity(t, level, 4)

	clock.advance(time.Nanosecond)
	wantVerbosity(t, level, 2)
	if lines := level.logged(); len(lines) != 2 || !strings.Contains(lines[1], "back to 2") {
		t.Fatalf("log lines after the revert = %q, want one more line about the revert", lines)
	}

	// Nothing is left to fire, and the next raise works like the first.
	clock.advance(time.Hour)
	wantVerbosity(t, level, 2)
	raiser.Raise()
	wantVerbosity(t, level, 4)
	clock.advance(10 * time.Minute)
	wantVerbosity(t, level, 2)
}

func TestLogVerbosityRevertRestoresTheConfiguredLevel(t *testing.T) {
	for _, configured := range []int{0, 1, 3} {
		t.Run(fmt.Sprintf("configured %d", configured), func(t *testing.T) {
			raiser, level, clock := newTestVerbosityRaiser(configured)

			raiser.Raise()
			clock.advance(10 * time.Minute)

			wantVerbosity(t, level, configured)
			if len(level.sets) != 2 || level.sets[0] != 4 || level.sets[1] != configured {
				t.Fatalf("verbosity was set to %v, want [4 %d]", level.sets, configured)
			}
		})
	}
}

func TestLogVerbositySecondRaiseStartsTheTenMinutesAgain(t *testing.T) {
	raiser, level, clock := newTestVerbosityRaiser(0)

	raiser.Raise()
	clock.advance(7 * time.Minute)
	raiser.Raise()
	if pending := clock.pending(); pending != 1 {
		t.Fatalf("%d reverts pending after the second raise, want 1", pending)
	}

	// Ten minutes after the first raise: still raised.
	clock.advance(3 * time.Minute)
	wantVerbosity(t, level, 4)
	clock.advance(7*time.Minute - time.Nanosecond)
	wantVerbosity(t, level, 4)

	// Ten minutes after the second.
	clock.advance(time.Nanosecond)
	wantVerbosity(t, level, 0)
	if lines := level.logged(); len(lines) != 3 || !strings.Contains(lines[2], "back to 0") {
		t.Fatalf("log lines = %q, want two raises and one revert", lines)
	}
}

// A timer that fires at the moment it is replaced can no longer be stopped. Its
// revert must not cut the new raise short.
func TestLogVerbosityRevertOfAReplacedRaiseDoesNothing(t *testing.T) {
	raiser, level, clock := newTestVerbosityRaiser(0)

	raiser.Raise()
	first := clock.timers[0]
	raiser.Raise()
	first.f()

	wantVerbosity(t, level, 4)
	clock.advance(10 * time.Minute)
	wantVerbosity(t, level, 0)
}

func TestLogVerbosityRaiseDoesNothingWhenConfiguredHighEnough(t *testing.T) {
	for _, configured := range []int{4, 6} {
		t.Run(fmt.Sprintf("configured %d", configured), func(t *testing.T) {
			raiser, level, clock := newTestVerbosityRaiser(configured)

			raiser.Raise()

			wantVerbosity(t, level, configured)
			if len(level.sets) != 0 {
				t.Fatalf("verbosity was set to %v, want no change", level.sets)
			}
			if pending := clock.pending(); pending != 0 {
				t.Fatalf("%d reverts pending, want none", pending)
			}
			if lines := level.logged(); len(lines) != 1 || !strings.Contains(lines[0], "nothing to raise") {
				t.Fatalf("log lines = %q, want one line saying that nothing changes", lines)
			}
		})
	}
}

func TestLogVerbosityRunRaisesPerSignalAndStopsWithContext(t *testing.T) {
	raiser, level, clock := newTestVerbosityRaiser(1)
	level.changed = make(chan int, 4)
	ctx, cancel := context.WithCancel(context.Background())
	signals := make(chan os.Signal)
	done := make(chan struct{})
	go func() {
		defer close(done)
		raiser.run(ctx, signals)
	}()

	signals <- os.Interrupt
	if got := <-level.changed; got != 4 {
		t.Fatalf("verbosity set to %d on the signal, want 4", got)
	}

	cancel()
	<-done
	// The raise that was in effect ended with the handler.
	wantVerbosity(t, level, 1)
	if pending := clock.pending(); pending != 0 {
		t.Fatalf("%d reverts pending after the handler stopped, want none", pending)
	}
	clock.advance(time.Hour)
	wantVerbosity(t, level, 1)
}

// Signals and the revert timer come from different goroutines.
func TestLogVerbosityConcurrentRaisesAndReverts(t *testing.T) {
	raiser, level, clock := newTestVerbosityRaiser(0)

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for range 50 {
				raiser.Raise()
			}
		}()
		go func() {
			defer wg.Done()
			for range 50 {
				clock.advance(4 * time.Minute)
			}
		}()
	}
	wg.Wait()

	clock.advance(10 * time.Minute)
	wantVerbosity(t, level, 0)
	if pending := clock.pending(); pending != 0 {
		t.Fatalf("%d reverts pending, want none", pending)
	}
}

// The raiser of the binary changes what klog actually logs, through the -v flag.
func TestKlogVerbosityRaiserChangesTheVerbosityFlag(t *testing.T) {
	flags := flag.NewFlagSet("klog", flag.ContinueOnError)
	klog.InitFlags(flags)
	verbosity := flags.Lookup("v")
	before := verbosity.Value.String()
	t.Cleanup(func() { _ = verbosity.Value.Set(before) })
	if err := flags.Parse([]string{"-v=2"}); err != nil {
		t.Fatal(err)
	}

	raiser, err := newKlogVerbosityRaiser(flags)
	if err != nil {
		t.Fatal(err)
	}
	clock := &verbosityTestClock{}
	raiser.afterFunc = clock.afterFunc
	if raiser.configured != 2 {
		t.Fatalf("configured verbosity = %d, want 2", raiser.configured)
	}

	raiser.Raise()
	if !klog.V(4).Enabled() || klog.V(5).Enabled() {
		t.Fatalf("after the raise: -v=%s, want level 4 logged and level 5 not", verbosity.Value)
	}

	clock.advance(10 * time.Minute)
	if got := verbosity.Value.String(); got != "2" {
		t.Fatalf("after the revert: -v=%s, want 2", got)
	}
	if !klog.V(2).Enabled() || klog.V(3).Enabled() {
		t.Fatal("after the revert: want level 2 logged and level 3 not")
	}
}

func TestKlogVerbosityRaiserNeedsTheLogFlags(t *testing.T) {
	if _, err := newKlogVerbosityRaiser(flag.NewFlagSet("empty", flag.ContinueOnError)); err == nil {
		t.Fatal("want an error for a flag set without the log flags")
	}
}
