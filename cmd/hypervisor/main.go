package main

import (
	"flag"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/device"
	"k8s.io/klog/v2"
)

func main() {
	var (
		acceleratorLibPath = flag.String("accelerator-lib",
			"../provider/build/libaccelerator_stub.so", "Path to accelerator library")
		discoveryInterval = flag.Duration("discovery-interval",
			30*time.Second, "Device discovery interval")
		isolationMode = flag.String("isolation-mode", "shared",
			"Isolation mode: shared, soft, hard, partitioned")
	)
	flag.Parse()

	klog.InitFlags(nil)
	defer klog.Flush()

	// Create device manager
	mgr, err := device.NewManager(*acceleratorLibPath, *discoveryInterval)
	if err != nil {
		klog.Fatalf("Failed to create device manager: %v", err)
	}

	// Start device manager
	if err := mgr.Start(); err != nil {
		klog.Fatalf("Failed to start device manager: %v", err)
	}
	defer mgr.Stop()
	klog.Info("Device manager started")

	devices := mgr.GetDevices()
	if len(devices) == 0 {
		klog.Fatalf("No devices found")
	}

	// Parse isolation mode
	var mode device.IsolationMode
	switch *isolationMode {
	case "shared":
		mode = device.IsolationModeShared
	case "soft":
		mode = device.IsolationModeSoft
	case "hard":
		mode = device.IsolationModeHard
	case "partitioned":
		mode = device.IsolationModePartitioned
	default:
		klog.Fatalf("Invalid isolation mode: %s", *isolationMode)
	}

	klog.Infof("Registered devices: %s with %d devices, isolation mode: %s", devices[0].Vendor, len(devices), mode)

	// Wait for interrupt signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	klog.Info("Hypervisor running")
	<-sigCh
	klog.Info("Shutting down...")
}
