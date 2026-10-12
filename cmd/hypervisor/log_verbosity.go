package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

const (
	// raisedLogVerbosity is the level the detail of the control loop is logged
	// at, one line per worker and tick.
	raisedLogVerbosity = 4
	// raisedLogVerbosityDuration bounds a raise: nobody has to remember to
	// lower the verbosity again, and the log volume stays bounded.
	raisedLogVerbosityDuration = 10 * time.Minute
)

// logVerbosityTimer is the part of *time.Timer the raiser uses.
type logVerbosityTimer interface {
	Stop() bool
}

// logVerbosityRaiser raises the log verbosity of the running process for a
// limited time and puts the configured verbosity back afterwards. It lets the
// per-tick detail be read on a node that misbehaves without restarting the
// hypervisor, which would end the situation that is being looked at.
type logVerbosityRaiser struct {
	// configured is the verbosity the process was started with.
	configured int
	setLevel   func(level int) error
	afterFunc  func(d time.Duration, f func()) logVerbosityTimer
	logf       func(format string, args ...any)

	mu sync.Mutex
	// timer is the pending revert, nil while the verbosity is the configured one.
	timer logVerbosityTimer
	// raises counts the raises. A timer that fires while it is being replaced
	// cannot be stopped any more; its revert finds another count and does nothing.
	raises uint64
}

// newKlogVerbosityRaiser returns a raiser for the klog verbosity that is
// registered as the "v" flag of flags, taking the current value of the flag as
// the configured verbosity. Call it after the flags were parsed.
func newKlogVerbosityRaiser(flags *flag.FlagSet) (*logVerbosityRaiser, error) {
	verbosity := flags.Lookup("v")
	if verbosity == nil {
		return nil, fmt.Errorf("the log flags are not registered")
	}
	configured, err := strconv.Atoi(verbosity.Value.String())
	if err != nil {
		return nil, fmt.Errorf("read the configured log verbosity: %w", err)
	}
	return &logVerbosityRaiser{
		configured: configured,
		setLevel:   func(level int) error { return verbosity.Value.Set(strconv.Itoa(level)) },
		afterFunc:  func(d time.Duration, f func()) logVerbosityTimer { return time.AfterFunc(d, f) },
		logf:       klog.Infof,
	}, nil
}

// Raise sets the verbosity to raisedLogVerbosity for raisedLogVerbosityDuration
// from now. While it is raised, that starts the duration again. It does nothing
// when the configured verbosity is not lower.
func (r *logVerbosityRaiser) Raise() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.configured >= raisedLogVerbosity {
		r.logf("Log verbosity is %d as configured, nothing to raise (a raise sets it to %d)",
			r.configured, raisedLogVerbosity)
		return
	}
	if err := r.setLevel(raisedLogVerbosity); err != nil {
		klog.Errorf("Failed to raise the log verbosity to %d: %v", raisedLogVerbosity, err)
		return
	}
	if r.timer != nil {
		r.timer.Stop()
	}
	r.raises++
	raise := r.raises
	r.timer = r.afterFunc(raisedLogVerbosityDuration, func() { r.revert(raise) })
	r.logf("Log verbosity raised from %d to %d for %s", r.configured, raisedLogVerbosity, raisedLogVerbosityDuration)
}

// revert ends the raise with the given count, unless a later one replaced it.
func (r *logVerbosityRaiser) revert(raise uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.timer == nil || raise != r.raises {
		return
	}
	r.restoreLocked()
}

// Stop ends a raise now.
func (r *logVerbosityRaiser) Stop() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.timer == nil {
		return
	}
	r.timer.Stop()
	r.restoreLocked()
}

func (r *logVerbosityRaiser) restoreLocked() {
	r.timer = nil
	if err := r.setLevel(r.configured); err != nil {
		klog.Errorf("Failed to set the log verbosity back to %d: %v", r.configured, err)
		return
	}
	r.logf("Log verbosity is back to %d", r.configured)
}

// run raises the verbosity for every signal received until ctx is done, and
// ends a raise that is still in effect then.
func (r *logVerbosityRaiser) run(ctx context.Context, signals <-chan os.Signal) {
	defer r.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-signals:
			r.Raise()
		}
	}
}

// raiseLogVerbosityOnSignal makes the signals of logVerbositySignals raise the
// klog verbosity (see logVerbosityRaiser.Raise) until ctx is done. A signal
// needs no listening port and can only be sent from the node or from inside
// the container.
//
// The registration is not undone when ctx is done: the default action of the
// signal would be back, which is to end the process in the middle of its
// shutdown.
func raiseLogVerbosityOnSignal(ctx context.Context) {
	if len(logVerbositySignals) == 0 {
		return
	}
	raiser, err := newKlogVerbosityRaiser(flag.CommandLine)
	if err != nil {
		klog.Warningf("The log verbosity cannot be raised by signal: %v", err)
		return
	}
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, logVerbositySignals...)
	go raiser.run(ctx, signals)
}
