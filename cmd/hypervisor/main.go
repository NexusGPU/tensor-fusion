package main

import (
	"context"
	"flag"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/cmd/hypervisor/shm_init"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/backend/kubernetes"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/backend/single_node"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/device"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/server"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/worker"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"
)

var (
	acceleratorLibPath = flag.String("accelerator-lib",
		"./provider/build/libaccelerator_stub.so", "Path to accelerator library")
	isolationMode = flag.String("isolation-mode", "shared",
		"Isolation mode: shared, soft, hard, partitioned")
	backendType       = flag.String("backend-type", "kubernetes", "Backend type: kubernetes, simple")
	discoveryInterval = flag.Duration("discovery-interval",
		12*time.Hour, "Device discovery interval")
	metricsPath = flag.String("metrics-output-path", "metrics.log", "Path to metrics output file")

	httpPort = flag.Int("port", int(constants.HypervisorDefaultPortNumber), "HTTP port for hypervisor API")
)

const (
	TFHardwareVendorEnv     = "TF_HARDWARE_VENDOR"
	TFAcceleratorLibPathEnv = "TF_ACCELERATOR_LIB_PATH"
)

const (
	MountShmSubcommand = "mount-shm"
)

func main() {
	// Check for subcommands (used inside init container for initializing shared memory of limiter of soft isolation)
	if len(os.Args) > 1 && os.Args[1] == MountShmSubcommand {
		shm_init.RunMountShm()
		return
	}

	flag.Parse()
	klog.InitFlags(nil)
	defer klog.Flush()

	ctx, cancel := context.WithCancel(context.Background())

	utils.NormalizeKubeConfigEnv()

	// Determine accelerator library path from env var or flag
	libPath := *acceleratorLibPath
	if envLibPath := os.Getenv(TFAcceleratorLibPathEnv); envLibPath != "" {
		libPath = envLibPath
		klog.Infof("Using accelerator library path from env: %s", libPath)
	}
	if vendor := os.Getenv(TFHardwareVendorEnv); vendor != "" {
		klog.Infof("Hardware vendor from env: %s", vendor)
	}

	// Create and start device controller
	deviceController, err := device.NewController(ctx, libPath, *discoveryInterval)
	if err != nil {
		klog.Fatalf("Failed to create device controller: %v", err)
	}
	if err := deviceController.Start(); err != nil {
		klog.Fatalf("Failed to start device manager: %v", err)
	}
	klog.Info("Device manager started")

	// Parse isolation mode
	var mode tfv1.IsolationModeType
	switch *isolationMode {
	case string(tfv1.IsolationModeShared):
		mode = tfv1.IsolationModeShared
	case string(tfv1.IsolationModeSoft):
		mode = tfv1.IsolationModeSoft
	case string(tfv1.IsolationModeHard):
		mode = tfv1.IsolationModeHard
	case string(tfv1.IsolationModePartitioned):
		mode = tfv1.IsolationModePartitioned
	}

	// initialize data backend
	var backend framework.Backend
	switch *backendType {
	case "kubernetes":
		// Get Kubernetes rest config
		var restConfig *rest.Config
		kubeconfig := os.Getenv("KUBECONFIG")
		if kubeconfig != "" {
			restConfig, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
		} else {
			restConfig, err = rest.InClusterConfig()
		}
		if err != nil {
			klog.Fatalf("Failed to get Kubernetes config: %v", err)
		}
		backend, err = kubernetes.NewKubeletBackend(ctx, deviceController, restConfig)
		if err != nil {
			klog.Fatalf("Failed to create Kubernetes backend: %v", err)
		}
	case "simple":
		backend = single_node.NewSingleNodeBackend(ctx, deviceController)
	default:
		klog.Fatalf("Invalid backend type: %s", *backendType)
	}

	// initialize worker controller
	workerController := worker.NewWorkerController(deviceController, mode, backend)
	err = workerController.Start()
	if err != nil {
		klog.Fatalf("Failed to start worker controller: %v", err)
	}
	defer func() {
		_ = workerController.Stop()
	}()
	klog.Info("Worker controller started")

	// initialize metrics recorder
	metricsRecorder := metrics.NewHypervisorMetricsRecorder(ctx, *metricsPath, deviceController, workerController)
	metricsRecorder.Start()
	klog.Info("Metrics recorder started")

	// initialize and start HTTP server
	httpPortNum := *httpPort
	if httpPortEnv := os.Getenv(constants.HypervisorPortEnv); httpPortEnv != "" {
		httpPortNum, err = strconv.Atoi(httpPortEnv)
		if err != nil {
			klog.Fatalf("Failed to convert HTTP port from env: %v", err)
		}
	}
	httpServer := server.NewServer(ctx, deviceController, workerController, metricsRecorder, backend, httpPortNum)
	go func() {
		if err := httpServer.Start(); err != nil && err != http.ErrServerClosed {
			klog.Fatalf("Failed to start HTTP server: %v", err)
		}
	}()
	klog.Info("HTTP server started")

	// Wait for interrupt signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	klog.Info("Hypervisor running")
	<-sigCh
	klog.Info("Stopping hypervisor...")

	// Shutdown HTTP server
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	if err := httpServer.Stop(shutdownCtx); err != nil {
		klog.Errorf("Error shutting down HTTP server: %v", err)
	}

	cancel()
	klog.Info("Hypervisor stopped")
}
