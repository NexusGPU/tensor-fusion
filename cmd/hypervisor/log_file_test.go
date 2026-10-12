package main

import (
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/klog/v2"
)

// The log file goes into the volume the metrics file is in, under the name of
// the Pod, and only when there is such a volume.
func TestLogFilePath(t *testing.T) {
	for _, test := range []struct {
		name, explicit, metricsPath, podName, want string
	}{
		{name: "next to the metrics file the operator configures", metricsPath: "/logs/metrics.log",
			podName: "hypervisor-node-a", want: "/logs/pods/hypervisor-node-a/hypervisor.log"},
		{name: "default metrics path: no shared volume, no file", metricsPath: "metrics.log", podName: "hypervisor-node-a"},
		{name: "unknown pod name", metricsPath: "/logs/metrics.log", want: "/logs/pods/hypervisor/hypervisor.log"},
		{name: "a pod name that is not a file name", metricsPath: "/logs/metrics.log", podName: "../elsewhere",
			want: "/logs/pods/hypervisor/hypervisor.log"},
		{name: "explicit path", explicit: "/var/log/tf/hypervisor.log", metricsPath: "metrics.log",
			want: "/var/log/tf/hypervisor.log"},
		{name: "turned off", explicit: "none", metricsPath: "/logs/metrics.log", podName: "hypervisor-node-a"},
	} {
		if got := logFilePath(test.explicit, test.metricsPath, test.podName); got != test.want {
			t.Errorf("%s: logFilePath(%q, %q, %q) = %q, want %q",
				test.name, test.explicit, test.metricsPath, test.podName, got, test.want)
		}
	}
}

// Every message reaches the file exactly once, whatever its severity, the
// verbosity applies to the file as it does to standard error, and the file is
// bounded.
func TestLogIsMirroredToABoundedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pods", "hypervisor-node-a", "hypervisor.log")
	flags := flag.NewFlagSet("klog", flag.ContinueOnError)
	klog.InitFlags(flags)
	writer := mirrorLogToFile(path)
	if writer == nil {
		t.Fatal("no log file although no log flag asks for another one")
	}
	t.Cleanup(func() {
		klog.Flush()
		for name, value := range map[string]string{
			"logtostderr": "true", "alsologtostderr": "false", "one_output": "false", "v": "0",
		} {
			_ = flags.Set(name, value)
		}
		_ = writer.Close()
	})
	if writer.MaxSize != logFileMaxSizeMB || writer.MaxBackups != logFileMaxBackups ||
		writer.MaxAge != logFileMaxAgeDays || writer.MaxSize <= 0 || writer.MaxBackups <= 0 {
		t.Errorf("log file is not bounded as documented: %+v", writer)
	}

	klog.Info("mirrored info line")
	klog.Warning("mirrored warning line")
	klog.Error("mirrored error line")
	klog.V(4).Info("mirrored debug line before")
	if err := flags.Set("v", "4"); err != nil {
		t.Fatal(err)
	}
	klog.V(4).Info("mirrored debug line after")
	klog.Flush()

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for message, want := range map[string]int{
		"mirrored info line": 1, "mirrored warning line": 1, "mirrored error line": 1,
		"mirrored debug line before": 0, "mirrored debug line after": 1,
	} {
		if got := strings.Count(string(raw), message); got != want {
			t.Errorf("%q is in the log file %d times, want %d:\n%s", message, got, want, raw)
		}
	}

	// Log flags that already name a file are left alone.
	if err := flags.Set("log_file", filepath.Join(t.TempDir(), "own.log")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = flags.Set("log_file", "") })
	if other := mirrorLogToFile(filepath.Join(t.TempDir(), "unused.log")); other != nil {
		t.Error("a second log file was set up although -log_file is given")
	}
}
