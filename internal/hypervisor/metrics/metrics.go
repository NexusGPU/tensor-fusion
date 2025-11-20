package metrics

import (
	"context"
	"io"
	"os"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/config"
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

	return &HypervisorMetricsRecorder{
		ctx:              ctx,
		outputPath:       outputPath,
		nodeName:         nodeName,
		gpuPool:          gpuPool,
		deviceController: deviceController,
		workerController: workerController,
		gpuCapacityMap:   make(map[string]float64),
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
	devices, err := h.deviceController.ListDevices(h.ctx)
	if err != nil {
		return
	}
	for _, device := range devices {
		h.gpuCapacityMap[device.UUID] = device.MaxTflops
	}
}

func (h *HypervisorMetricsRecorder) RecordDeviceMetrics(writer io.Writer) {
	gpuMetrics, err := h.deviceController.GetGPUMetrics(h.ctx)
	if err != nil {
		return
	}

	// Output GPU metrics directly
	now := time.Now()
	enc := metrics.NewEncoder(config.GetGlobalConfig().MetricsFormat)

	for gpuUUID, metrics := range gpuMetrics {
		enc.StartLine("tf_gpu_usage")
		enc.AddTag("uuid", gpuUUID)
		enc.AddTag("node", h.nodeName)
		enc.AddTag("pool", h.gpuPool)

		enc.AddField("rx", metrics.Rx)
		enc.AddField("tx", metrics.Tx)
		enc.AddField("nvlink_rx", float64(metrics.NvlinkRxBandwidth))
		enc.AddField("nvlink_tx", float64(metrics.NvlinkTxBandwidth))
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
	workerMetrics, err := h.workerController.GetWorkerMetrics(h.ctx)
	if err != nil {
		return
	}

	workerUIDs, err := h.workerController.ListWorkers(h.ctx)
	if err != nil {
		return
	}

	// Get worker allocations for metadata
	workerAllocations := make(map[string]*api.DeviceAllocation)
	for _, workerUID := range workerUIDs {
		allocation, err := h.workerController.GetWorkerAllocation(h.ctx, workerUID)
		if err == nil && allocation != nil {
			workerAllocations[workerUID] = allocation
		}
	}

	// Get extra labels config
	extraLabelsConfig := config.GetGlobalConfig().MetricsExtraPodLabels
	_ = len(extraLabelsConfig) > 0 // hasDynamicMetricsLabels - reserved for future use

	// Output worker metrics directly
	now := time.Now()
	// TODO: use config from flag parser, not global config
	enc := metrics.NewEncoder("influx")

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
				vramLimit := float64(allocation.MemoryLimit)
				if vramLimit > 0 {
					memoryPercentage += float64(metrics.MemoryBytes) / vramLimit * 100.0
				}
			}

			enc.StartLine("tf_worker_usage")
			enc.AddTag("uuid", deviceUUID)
			enc.AddTag("node", h.nodeName)
			enc.AddTag("pool", h.gpuPool)
			enc.AddTag("pod_name", allocation.PodName)
			enc.AddTag("namespace", allocation.Namespace)

			workloadName := "unknown"
			// Try to get workload name from worker ID or pod name
			if allocation.WorkerID != "" {
				workloadName = allocation.WorkerID
			}
			enc.AddTag("workload", workloadName)
			enc.AddTag("worker", workerUID)

			// Add extra labels if configured
			// Note: In Rust code, labels come from pod_state.info.labels
			// Here we would need to get pod labels from allocation or another source
			// For now, we'll skip extra labels as we don't have access to pod labels
			_ = extraLabelsConfig // Reserved for future use

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
