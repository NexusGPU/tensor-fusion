package metrics

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/internal/version"
	"github.com/posthog/posthog-go"
	"golang.org/x/sys/unix"
	"gopkg.in/natefinch/lumberjack.v2"
	"k8s.io/klog/v2"
)

type HypervisorMetricsRecorder struct {
	ctx                  context.Context
	outputPath           string
	nodeName             string
	gpuPool              string
	deviceController     framework.DeviceController
	workerController     framework.WorkerController
	allocationController framework.WorkerAllocationController
	extraLabelsMap       map[string]string // podLabelKey -> tagName mapping from env config
}

const (
	defaultNodeName = "unknown"
	defaultGPUPool  = "unknown"
)

var (
	startTime            = time.Now()
	telemetryClient      posthog.Client
	telemetryClientMu    sync.Once
	telemetryLockMu      sync.Mutex
	telemetryMinInterval = 24 * time.Hour
)

func NewHypervisorMetricsRecorder(
	ctx context.Context, outputPath string,
	deviceController framework.DeviceController,
	workerController framework.WorkerController,
	allocationController framework.WorkerAllocationController,
) *HypervisorMetricsRecorder {
	nodeName := os.Getenv(constants.HypervisorGPUNodeNameEnv)
	if nodeName == "" {
		nodeName = defaultNodeName
	}
	gpuPool := os.Getenv(constants.HypervisorPoolNameEnv)
	if gpuPool == "" {
		gpuPool = defaultGPUPool
	}

	// Parse extra labels config once at initialization
	extraLabelsMap := make(map[string]string)
	extraLabelsConfig := os.Getenv(constants.HypervisorMetricsExtraLabelsEnv)
	if extraLabelsConfig != "" {
		if err := json.Unmarshal([]byte(extraLabelsConfig), &extraLabelsMap); err != nil {
			// Log error but continue without extra labels
			extraLabelsMap = make(map[string]string)
		}
	}

	return &HypervisorMetricsRecorder{
		ctx:                  ctx,
		outputPath:           outputPath,
		nodeName:             nodeName,
		gpuPool:              gpuPool,
		deviceController:     deviceController,
		workerController:     workerController,
		allocationController: allocationController,
		extraLabelsMap:       extraLabelsMap,
	}
}

func (h *HypervisorMetricsRecorder) Start() {
	writer := &lumberjack.Logger{
		Filename:   h.outputPath,
		MaxSize:    100,
		MaxBackups: 10,
		MaxAge:     14,
	}

	// Record device and worker metrics
	deviceMetricsTicker := time.NewTicker(10 * time.Second)
	go func() {
		for {
			select {
			case <-h.ctx.Done():
				return
			case <-deviceMetricsTicker.C:
				h.RecordDeviceMetrics(writer)
				h.RecordWorkerMetrics(writer)
			}
		}
	}()
}

func (h *HypervisorMetricsRecorder) RecordDeviceMetrics(writer io.Writer) {
	gpuMetrics, err := h.deviceController.GetDeviceMetrics()
	if err != nil {
		return
	}

	// Output GPU metrics directly
	now := time.Now()
	enc := metrics.NewEncoder(os.Getenv(constants.HypervisorMetricsFormatEnv))

	for gpuUUID, metrics := range gpuMetrics {
		enc.StartLine("tf_gpu_usage")
		enc.AddTag("uuid", gpuUUID)
		enc.AddTag("node", h.nodeName)
		enc.AddTag("pool", h.gpuPool)

		enc.AddField("rx", metrics.Rx)
		enc.AddField("tx", metrics.Tx)
		enc.AddField("temperature", metrics.Temperature)
		enc.AddField("memory_bytes", int64(metrics.MemoryBytes))
		enc.AddField("memory_percentage", metrics.MemoryPercentage)
		enc.AddField("compute_percentage", metrics.ComputePercentage)
		enc.AddField("compute_tflops", metrics.ComputeTflops)
		enc.AddField("power_usage", float64(metrics.PowerUsage))
		if metrics.ExtraMetrics != nil {
			for key, value := range metrics.ExtraMetrics {
				enc.AddField(key, value)
			}
		}
		enc.EndLine(now)
	}

	if err := enc.Err(); err == nil {
		_, _ = writer.Write(enc.Bytes())
	}
}

func (h *HypervisorMetricsRecorder) RecordWorkerMetrics(writer io.Writer) {
	workerMetrics, err := h.workerController.GetWorkerMetrics()
	if err != nil {
		return
	}

	workers, err := h.workerController.ListWorkers()
	if err != nil {
		return
	}

	// Get worker allocations for metadata
	workerAllocations := make(map[string]*api.WorkerAllocation)
	for _, worker := range workers {
		allocation, found := h.allocationController.GetWorkerAllocation(worker.WorkerUID)
		if found && allocation != nil {
			workerAllocations[worker.WorkerUID] = allocation
		}
	}

	// Output worker metrics directly
	now := time.Now()
	enc := metrics.NewEncoder(os.Getenv(constants.HypervisorMetricsFormatEnv))

	for deviceUUID, workerMap := range workerMetrics {
		for workerUID, processMap := range workerMap {
			allocation, ok := workerAllocations[workerUID]
			if !ok {
				continue
			}

			var memoryBytes uint64
			var computePercentage float64
			var computeTflops float64
			var memoryPercentage float64

			// Sum up metrics from all processes for this worker
			for _, metrics := range processMap {
				memoryBytes += metrics.MemoryBytes
				computePercentage += metrics.ComputePercentage
				computeTflops += metrics.ComputeTflops

				// Calculate memory percentage
				vramLimit := float64(0)
				if allocation.WorkerInfo != nil {
					vramLimit = float64(allocation.WorkerInfo.Limits.Vram.Value())
				}
				if vramLimit > 0 {
					memoryPercentage += float64(metrics.MemoryBytes) / vramLimit * 100.0
				}
			}

			enc.StartLine("tf_worker_usage")
			enc.AddTag("uuid", deviceUUID)
			enc.AddTag("node", h.nodeName)
			enc.AddTag("pool", h.gpuPool)
			if allocation.WorkerInfo != nil {
				enc.AddTag("pod_name", allocation.WorkerInfo.WorkerName)
				enc.AddTag("namespace", allocation.WorkerInfo.Namespace)
			}

			workloadName := "unknown"
			// Try to get workload name from worker ID or pod name
			if allocation.WorkerInfo != nil && allocation.WorkerInfo.WorkerUID != "" {
				workloadName = allocation.WorkerInfo.WorkerUID
			}
			enc.AddTag("workload", workloadName)
			enc.AddTag("worker", workerUID)

			// Add extra labels if configured
			h.addExtraLabels(enc, allocation)

			enc.AddField("memory_bytes", int64(memoryBytes))
			enc.AddField("compute_percentage", computePercentage)
			enc.AddField("compute_tflops", computeTflops)
			enc.AddField("memory_percentage", memoryPercentage)

			enc.EndLine(now)
		}
	}

	if err := enc.Err(); err == nil {
		_, _ = writer.Write(enc.Bytes())
	}
}

// addExtraLabels adds dynamic tags based on HypervisorMetricsExtraLabelsEnv configuration
// The config is a JSON map where keys are tag names and values are pod label keys to extract
// Labels are read directly from allocation.Labels which is populated by the backend
func (h *HypervisorMetricsRecorder) addExtraLabels(enc metrics.Encoder, allocation *api.WorkerAllocation) {
	if len(h.extraLabelsMap) == 0 {
		return
	}

	if allocation.WorkerInfo == nil || len(allocation.WorkerInfo.Annotations) == 0 {
		return
	}

	// Add tags based on the mapping
	for podLabelKey, tagName := range h.extraLabelsMap {
		if labelValue, exists := allocation.WorkerInfo.Annotations[podLabelKey]; exists && labelValue != "" {
			enc.AddTag(tagName, labelValue)
		}
	}
}

// TelemetryConfig contains optional telemetry parameters
type TelemetryConfig struct {
	WorkersCount     int
	IsolationMode    string
	SampleGPUModel   string
	DeviceController framework.DeviceController
}

// getPostHogClient initializes and returns the PostHog client (singleton)
func getPostHogClient() posthog.Client {
	telemetryClientMu.Do(func() {
		endpoint := os.Getenv(constants.TelemetryEndpointEnvVar)
		if endpoint == "" {
			endpoint = constants.DefaultTelemetryEndpoint
		}

		pubKey := os.Getenv(constants.TelemetryPublicKeyEnvVar)
		if pubKey == "" {
			pubKey = constants.DefaultTelemetryPublicKey
		}

		client, err := posthog.NewWithConfig(pubKey, posthog.Config{
			Endpoint: endpoint,
		})
		if err != nil {
			klog.V(4).Infof("Failed to initialize PostHog client: %v", err)
			return
		}
		telemetryClient = client
	})
	return telemetryClient
}

// fileLock and fileUnlock use flock for file locking on Unix-like systems
func fileLock(fd uintptr) error {
	return unix.Flock(int(fd), unix.LOCK_EX|unix.LOCK_NB)
}

func fileUnlock(fd uintptr) error {
	return unix.Flock(int(fd), unix.LOCK_UN)
}

func ShouldSendTelemetry() bool {
	if os.Getenv("DISABLE_TENSOR_FUSION_TELEMETRY") != "" {
		return false
	}
	if utils.IsTestMode {
		return false
	}

	telemetryLockMu.Lock()
	defer telemetryLockMu.Unlock()

	// Try to open or create the lock file
	telemetryLockFile := filepath.Join(os.TempDir(), "tensor-fusion-telemetry.lock")
	file, err := os.OpenFile(telemetryLockFile, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		klog.V(4).Infof("Failed to open telemetry lock file: %v", err)
		return false
	}
	defer func() {
		if err := file.Close(); err != nil {
			klog.V(4).Infof("Failed to close telemetry lock file: %v", err)
		}
	}()

	// Try to acquire an exclusive lock (non-blocking)
	err = fileLock(file.Fd())
	if err != nil {
		klog.V(4).Infof("Failed to acquire telemetry lock: %v", err)
		// Lock is already held by another process
		return false
	}
	defer func() {
		if err := fileUnlock(file.Fd()); err != nil {
			klog.V(4).Infof("Failed to release telemetry lock: %v", err)
		}
	}()

	// Read and parse the timestamp from the file
	var lastSentTime time.Time
	if data, err := io.ReadAll(file); err == nil {
		if timestamp, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64); err == nil {
			lastSentTime = time.Unix(timestamp, 0)
		}
	}
	if !lastSentTime.IsZero() && time.Since(lastSentTime) < telemetryMinInterval {
		return false
	}

	// Write current timestamp to the file
	now := time.Now()
	timestampStr := strconv.FormatInt(now.Unix(), 10)
	if _, err := file.Seek(0, 0); err != nil {
		klog.V(4).Infof("Failed to seek telemetry lock file: %v", err)
		return false
	}
	if err := file.Truncate(0); err != nil {
		klog.V(4).Infof("Failed to truncate telemetry lock file: %v", err)
		return false
	}
	if _, err := file.WriteString(timestampStr); err != nil {
		klog.V(4).Infof("Failed to write telemetry lock file: %v", err)
		return false
	}
	if err := file.Sync(); err != nil {
		klog.V(4).Infof("Failed to sync telemetry lock file: %v", err)
		return false
	}
	return true
}

// SendAnonymousTelemetry sends Anonymous telemetry data without ANY sensitive data
func SendAnonymousTelemetry(nodeInfo *api.NodeInfo, hardwareVendor string, sampleGPUModel string, workersCount int, isolationMode api.IsolationMode) {
	// Get PostHog client
	client := getPostHogClient()
	if client == nil {
		klog.V(4).Infof("PostHog client not available, skipping telemetry")
		return
	}

	// Prepare event properties
	properties := posthog.NewProperties().
		Set("ramSizeBytes", nodeInfo.RAMSizeBytes).
		Set("totalTFlops", nodeInfo.TotalTFlops).
		Set("totalVRAMBytes", nodeInfo.TotalVRAMBytes).
		Set("totalDevices", len(nodeInfo.DeviceIDs)).
		Set("brand", constants.Domain).
		Set("version", version.BuildVersion).
		Set("uptime", time.Since(startTime).String()).
		Set("workersCount", workersCount).
		Set("isolationMode", string(isolationMode)).
		Set("vendor", hardwareVendor).
		Set("sampleGPUModel", sampleGPUModel)

	// Send event to PostHog
	err := client.Enqueue(posthog.Capture{
		Event:      "hypervisor_telemetry",
		Properties: properties,
	})
	if err != nil {
		klog.V(4).Infof("Failed to send telemetry: %v", err)
		return
	}
}
