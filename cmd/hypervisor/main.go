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

	// Discover devices
	devices := mgr.GetDevices()
	klog.Infof("Discovered %d devices", len(devices))

	if len(devices) == 0 {
		klog.Warning("No devices discovered, waiting...")
		time.Sleep(2 * time.Second)
		devices = mgr.GetDevices()
		if len(devices) == 0 {
			klog.Fatalf("No devices available")
		}
	}

	// Register default pool
	deviceUUIDs := make([]string, 0, len(devices))
	for _, d := range devices {
		deviceUUIDs = append(deviceUUIDs, d.UUID)
		klog.Infof("Device: UUID=%s, Vendor=%s, Model=%s, Memory=%d GB",
			d.UUID, d.Vendor, d.Model, d.TotalMemory/(1024*1024*1024))
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

	pool := &device.DevicePool{
		Vendor:         devices[0].Vendor,
		IsolationMode:  mode,
		DeviceUUIDs:    deviceUUIDs,
		AcceleratorLib: *acceleratorLibPath,
	}

	if err := mgr.RegisterPool(pool); err != nil {
		klog.Fatalf("Failed to register pool: %v", err)
	}
	klog.Infof("Registered devices: %s with %d devices, isolation mode: %s", devices[0].Vendor, len(deviceUUIDs), mode)

	// TODO: 2. If k8s mode, listen Pods from kubelet socket and build a map
	// TODO: 3. Extensible Device Plugin, to read config yaml of pool and
	// TODO: 4. Report GPU CR to API server, if DRA enabled, report ResourceSlice
	// TODO: 5. Build shm handle or ivshmem device for soft isolation mode for
	//        limiter and hard isolation mode, manage shm lifecycle
	// TODO: 6. Expose HTTP APIs for watch worker pod status, or create workers process,
	//  manage workers lifecycle in VM mode

	// Wait for interrupt signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	klog.Info("Hypervisor running, press Ctrl+C to stop")
	<-sigCh
	klog.Info("Shutting down...")
}
