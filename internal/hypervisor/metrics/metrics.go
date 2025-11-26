package metrics

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"gopkg.in/natefinch/lumberjack.v2"
)

type HypervisorMetricsRecorder struct {
	ctx              context.Context
	outputPath       string
	nodeName         string
	gpuPool          string
	deviceController framework.DeviceController
	workerController framework.WorkerController
	gpuCapacityMap   map[string]float64 // GPU UUID -> MaxTflops
	extraLabelsMap   map[string]string  // podLabelKey -> tagName mapping from env config
}

const (
	defaultNodeName = "unknown"
	defaultGPUPool  = "unknown"
)

func NewHypervisorMetricsRecorder(
	ctx context.Context, outputPath string,
	deviceController framework.DeviceController,
	workerController framework.WorkerController,
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
		ctx:              ctx,
		outputPath:       outputPath,
		nodeName:         nodeName,
		gpuPool:          gpuPool,
		deviceController: deviceController,
		workerController: workerController,
		gpuCapacityMap:   make(map[string]float64),
		extraLabelsMap:   extraLabelsMap,
	}
}

func (h *HypervisorMetricsRecorder) Start() {
	writer := &lumberjack.Logger{
		Filename:   h.outputPath,
		MaxSize:    100,
		MaxBackups: 10,
		MaxAge:     14,
	}

	// Initialize GPU capacity map from devices
	h.initGPUCapacityMap()

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

func (h *HypervisorMetricsRecorder) initGPUCapacityMap() {
	devices, err := h.deviceController.ListDevices()
	if err != nil {
		return
	}
	for _, device := range devices {
		h.gpuCapacityMap[device.UUID] = device.MaxTflops
	}
}

func (h *HypervisorMetricsRecorder) RecordDeviceMetrics(writer io.Writer) {
	gpuMetrics, err := h.deviceController.GetGPUMetrics()
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
		// Add vendor-specific metrics from ExtraMetrics map
		if metrics.ExtraMetrics != nil {
			for key, value := range metrics.ExtraMetrics {
				enc.AddField(key, value)
			}
		}
		enc.AddField("temperature", metrics.Temperature)
		enc.AddField("graphics_clock_mhz", metrics.GraphicsClockMHz)
		enc.AddField("sm_clock_mhz", metrics.SMClockMHz)
		enc.AddField("memory_clock_mhz", metrics.MemoryClockMHz)
		enc.AddField("video_clock_mhz", metrics.VideoClockMHz)
		enc.AddField("memory_bytes", int64(metrics.MemoryBytes))
		enc.AddField("memory_percentage", metrics.MemoryPercentage)
		enc.AddField("compute_percentage", metrics.ComputePercentage)
		enc.AddField("compute_tflops", metrics.ComputeTflops)
		enc.AddField("power_usage", float64(metrics.PowerUsage))

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

	workerUIDs, err := h.workerController.ListWorkers()
	if err != nil {
		return
	}

	// Get worker allocations for metadata
	workerAllocations := make(map[string]*api.WorkerAllocation)
	for _, workerUID := range workerUIDs {
		allocation, err := h.workerController.GetWorkerAllocation(workerUID)
		if err == nil && allocation != nil {
			workerAllocations[workerUID] = allocation
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
					vramLimit = float64(allocation.WorkerInfo.MemoryLimitBytes)
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
				enc.AddTag("pod_name", allocation.WorkerInfo.PodName)
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
