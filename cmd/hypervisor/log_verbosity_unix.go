//go:build unix

package main

import (
	"os"
	"syscall"
)

// logVerbositySignals are the signals that raise the log verbosity.
var logVerbositySignals = []os.Signal{syscall.SIGUSR1}
