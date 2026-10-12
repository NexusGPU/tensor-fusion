package main

import (
	"context"
	"flag"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/cmd/hypervisor/shm_init"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/internal/version"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/backend/kubernetes"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/backend/single_node"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/device"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/metrics"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/server"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker"
	"gopkg.in/natefinch/lumberjack.v2"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

var (
	acceleratorVendor  = flag.String("vendor", "NVIDIA", "Accelerator vendor: NVIDIA, AMD, Intel, etc.")
	acceleratorLibPath = flag.String("accelerator-lib",
		"./provider/build/libaccelerator_example.so", "Path to accelerator library")
	isolationMode = flag.String("isolation-mode", "soft",
		"Isolation mode: shared, soft, hard, partitioned")
	isolationPolicy = flag.String("isolation-policy", "static",
		"Isolation policy: static or dynamic")
	backendType       = flag.String("backend-type", "kubernetes", "Backend type: kubernetes, simple")
	discoveryInterval = flag.Duration("discovery-interval",
		12*time.Hour, "Device discovery interval")
	metricsPath     = flag.String("metrics-output-path", "metrics.log", "Path to metrics output file")
	metricsInterval = flag.Duration("metrics-interval", 60*time.Second, "Metrics interval")
	logPath         = flag.String("log-output-path", "",
		"File the log is also written to, for the log shipping sidecar. Default: pods/<pod name>/hypervisor.log "+
			"next to an absolute -metrics-output-path; \"none\" turns it off")

	httpPort = flag.Int("port", int(constants.HypervisorDefaultPortNumber), "HTTP port for hypervisor API")
)

func main() {
	// Check for subcommands (used inside init container for initializing shared memory of limiter of soft isolation)
	if len(os.Args) > 1 && os.Args[1] == constants.MountShmSubcommand {
		shm_init.RunMountShm()
		return
	}

	// Keep the v1 image command compatible. v1 invokes the binary as
	// `hypervisor daemon`; v2 has a single daemon process and does not need a
	// subcommand. Strip the legacy token before parsing flags so flags following
	// it are still honored when a wrapper does forward them.
	*isolationMode = envOrDefault(constants.TFIsolationModeEnv, *isolationMode)
	*isolationPolicy = envOrDefault(constants.IsolationModePolicyEnv, *isolationPolicy)
	// Registered before parsing, so that log flags such as -v are accepted.
	klog.InitFlags(nil)
	if err := flag.CommandLine.Parse(legacyCompatibleFlagArgs(os.Args[1:])); err != nil {
		klog.Fatalf("failed to parse hypervisor flags: %v", err)
	}
	policy := tfv1.IsolationModePolicyStatic
	switch strings.ToLower(strings.TrimSpace(*isolationPolicy)) {
	case "static":
		*isolationPolicy = "static"
	case "dynamic":
		*isolationPolicy = "dynamic"
		policy = tfv1.IsolationModePolicyDynamic
	default:
		klog.Fatalf("invalid isolation policy: %s", *isolationPolicy)
	}
	defer klog.Flush()
	if path := logFilePath(*logPath, *metricsPath, os.Getenv(constants.PodNameEnv)); path != "" {
		if writer := mirrorLogToFile(path); writer != nil {
			defer func() { _ = writer.Close() }()
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	// Before anything that takes time: without the handler SIGUSR1 ends a Go
	// process, and the signal may be sent as soon as the Pod is there.
	raiseLogVerbosityOnSignal(ctx)
	klog.Info("tensor fusion hypervisor starting. ", version.VersionInfo())

	utils.NormalizeKubeConfigEnv()

	// Determine accelerator library path from env var or flag
	libPath := *acceleratorLibPath
	if envLibPath := os.Getenv(constants.TFAcceleratorLibPathEnv); envLibPath != "" {
		libPath = envLibPath
		klog.Infof("Using accelerator library path from env: %s", libPath)
	}
	if vendor := os.Getenv(constants.TFHardwareVendorEnv); vendor != "" {
		acceleratorVendor = ptr.To(vendor)
		klog.Infof("Hardware vendor from env: %s", vendor)
	}
	configureNvidiaDriverLibraries(*acceleratorVendor, constants.NvidiaDriverReadyPath)

	// Create and start device controller
	deviceController, err := device.NewController(ctx, libPath, *acceleratorVendor, *discoveryInterval, *isolationMode)
	if err != nil {
		klog.Fatalf("Failed to create device controller: %v", err)
	}
	deviceController.SetIsolationPolicy(policy)
	if *backendType == "kubernetes" {
		// TFLOPS capacities are vendor-published FP16 figures from the
		// ProviderConfig hardware table, which the operator passes in the
		// environment (it recreates this Pod when the ProviderConfig changes).
		// Installed before discovery so that devices never appear with another
		// capacity first. Without a usable table every device is reported as
		// falling back to the provider's estimate.
		metadata, err := device.LoadHardwareMetadataFromEnv()
		if err != nil {
			klog.Warningf("ProviderConfig hardware table is unavailable, TFLOPS capacities fall back to "+
				"provider estimates: %v", err)
		}
		deviceController.SetHardwareMetadata(metadata)
	}
	if err := deviceController.Start(); err != nil {
		klog.Fatalf("Failed to start device manager: %v", err)
	}
	klog.Info("Device manager started")

	mode := tfv1.IsolationModeType(*isolationMode)

	// Create allocation controller first - it's a shared dependency
	allocationController := worker.NewAllocationController(deviceController)
	allocationController.SetIsolationPolicy(policy)
	deviceController.SetAllocationController(allocationController)
	klog.Info("Allocation controller created")

	// initialize data backend and worker controller
	var backend framework.Backend
	var workerController framework.WorkerController

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

		backend, err = kubernetes.NewKubeletBackend(ctx, deviceController, allocationController, restConfig)
		if err != nil {
			klog.Fatalf("Failed to create Kubernetes backend: %v", err)
		}
		workerController = worker.NewWorkerControllerWithPolicy(deviceController, allocationController, mode, policy, backend)
	case "simple":
		backend = single_node.NewSingleNodeBackend(ctx, deviceController, allocationController)
		workerController = worker.NewWorkerControllerWithPolicy(deviceController, allocationController, mode, policy, backend)
	default:
		klog.Fatalf("Invalid backend type: %s", *backendType)
	}
	deviceController.RegisterDeviceUpdateHandler(backend.GetDeviceChangeHandler())
	klog.Infof("Device change handler registered from backend: %s", *backendType)

	err = workerController.Start()
	if err != nil {
		klog.Fatalf("Failed to start worker controller: %v", err)
	}
	defer func() {
		_ = workerController.Stop()
	}()

	klog.Info("Worker controller started")

	// initialize metrics recorder
	metricsRecorder := metrics.NewHypervisorMetricsRecorder(
		ctx, *metricsPath, deviceController, workerController, allocationController, *metricsInterval,
	)
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
	httpServer := server.NewServer(
		ctx, deviceController, workerController, allocationController, metricsRecorder, backend, httpPortNum,
	)
	// The server owns the counters of the limiter events endpoint; the recorder,
	// which had to exist first, exports them from now on.
	metricsRecorder.SetLimiterEventSource(httpServer.EachLimiterEventCount)
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

// legacyCompatibleFlagArgs removes the v1 daemon subcommand. The v2 binary
// has a single daemon process, but accepting this token keeps existing TFC pod
// templates usable when only the Hypervisor image is changed.
func legacyCompatibleFlagArgs(args []string) []string {
	if len(args) > 0 && args[0] == "daemon" {
		return args[1:]
	}
	return args
}

const (
	// logFileDisabled is the -log-output-path value that turns the log file off.
	logFileDisabled = "none"
	// The log file is bounded: at most logFileMaxBackups+1 files of
	// logFileMaxSizeMB each, none of them older than logFileMaxAgeDays. A
	// verbosity somebody forgot to lower again cannot fill the volume.
	logFileMaxSizeMB  = 50
	logFileMaxBackups = 2
	logFileMaxAgeDays = 7
)

// logFilePath returns the file the log is also written to, or "" for none.
//
// The log of this container is not readable by the log shipping sidecar of the
// Pod, which mounts only the volume it shares with the hypervisor. The operator
// points -metrics-output-path into that volume, so the log goes there too, into
// a directory named after the Pod: that is how the sidecar knows whose log it
// ships. A relative metrics path (the default) means there is no such volume.
func logFilePath(explicit, metricsPath, podName string) string {
	switch {
	case explicit == logFileDisabled:
		return ""
	case explicit != "":
		return explicit
	case !filepath.IsAbs(metricsPath):
		return ""
	}
	if podName == "" || podName != filepath.Base(podName) || podName == "." || podName == ".." {
		podName = "hypervisor"
	}
	return filepath.Join(filepath.Dir(metricsPath), "pods", podName, "hypervisor.log")
}

// mirrorLogToFile makes klog write every message to a size-bounded, rotating
// file in addition to standard error, which stays as it was. It returns the
// file, or nil when the log flags already send the log to files of their own.
// A file that cannot be written costs the copy, never the hypervisor: klog
// ignores the error and standard error still has the message.
func mirrorLogToFile(path string) *lumberjack.Logger {
	flags := flag.NewFlagSet("klog", flag.ContinueOnError)
	klog.InitFlags(flags)
	for _, name := range []string{"log_file", "log_dir"} {
		if flags.Lookup(name).Value.String() != "" {
			return nil
		}
	}
	// klog writes to files only with logtostderr off. alsologtostderr keeps
	// standard error complete, and one_output keeps a warning or an error from
	// being written once more for every lower severity.
	for name, value := range map[string]string{"logtostderr": "false", "alsologtostderr": "true", "one_output": "true"} {
		_ = flags.Set(name, value)
	}
	writer := &lumberjack.Logger{
		Filename:   path,
		MaxSize:    logFileMaxSizeMB,
		MaxBackups: logFileMaxBackups,
		MaxAge:     logFileMaxAgeDays,
	}
	klog.SetOutput(writer)
	return writer
}

func envOrDefault(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}
