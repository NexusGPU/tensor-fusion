//go:build !unix

package main

import "os"

// logVerbositySignals is empty where SIGUSR1 does not exist: the log verbosity
// is then only set by the -v flag.
var logVerbositySignals []os.Signal
