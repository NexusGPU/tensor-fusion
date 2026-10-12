package metrics

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/internal/version"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/posthog/posthog-go"
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
	metricsInterval      time.Duration

	// lastUsage is the accumulated compute usage read at the previous record,
	// the reference for this interval's average. Only the recording goroutine
	// touches it.
	lastUsage map[string]map[string]api.WorkerComputeUsage
	// writeFailing is set while writes of the metrics file fail, so that a
	// failure is reported when it starts and when it ends, not on every record.
	writeFailing atomic.Bool

	// limiterEvents supplies the counters of the limiter events endpoint; nil
	// until SetLimiterEventSource is called.
	limiterEvents atomic.Pointer[LimiterEventSource]
	// limiterEventsWritten holds the count last written per tf_limiter_events
	// row. Only the recording goroutine touches it. It is replaced on every
	// record by the rows the source supplied, so it never holds more than the
	// source does (see RecordLimiterEventMetrics).
	limiterEventsWritten map[limiterEventRow]uint64
}

// LimiterEventSource supplies the counters of the limiter events the workloads
// of this node reported: it calls yield once per Pod and kind with the
// occurrences counted so far.
type LimiterEventSource func(yield func(namespace, pod, kind string, count uint64))

// limiterEventRow identifies one tf_limiter_events row.
type limiterEventRow struct {
	namespace, pod, kind string
}

// limiterEventsMeasurement is the table the limiter event counters are written
// to; see RecordLimiterEventMetrics.
const limiterEventsMeasurement = "tf_limiter_events"

const (
	defaultNodeName = "unknown"
	defaultGPUPool  = "unknown"
)

// The type a field is written with is part of the table: GreptimeDB creates a
// column with the type of the first value it receives and from then on rejects
// every row that carries another type for it (HTTP 400, "column ... expect
// type ..."), and the hypervisor never sees that answer because the sidecar
// posts the file. tf_gpu_usage and tf_worker_usage were first written by the
// earlier hypervisor, so the fields below keep the types it used:
//
//   - memory_bytes: unsigned integer in both tables (a uint64 value);
//   - power_usage, nvlink_rx, nvlink_tx of tf_gpu_usage: signed integer;
//   - every other field: float.
//
// TestHypervisorLineProtocolFieldTypes pins the type of every field. Changing
// one of them makes existing installations lose the rows of that table.

// gpuUsageFixedFields are the tf_gpu_usage fields RecordDeviceMetrics writes
// itself. A provider's extra metric of the same name is not written: a line
// with the field twice, or with the provider's type, would be rejected.
var gpuUsageFixedFields = map[string]struct{}{
	"rx": {}, "tx": {}, "temperature": {}, "memory_bytes": {}, "memory_percentage": {},
	"compute_percentage": {}, "compute_tflops": {}, "power_usage": {},
}

// gpuUsageIntegerExtraFields are the provider extra metrics whose tf_gpu_usage
// column is a signed integer in existing tables.
var gpuUsageIntegerExtraFields = map[string]struct{}{
	"nvlink_rx": {}, "nvlink_tx": {},
}

// gpuUsageExtraField returns the value a provider's extra metric is written to
// tf_gpu_usage with, and false when it must not be written. Extra metrics are
// floats, so a metric keeps one type whatever its values are.
func gpuUsageExtraField(key string, value float64) (any, bool) {
	if _, fixed := gpuUsageFixedFields[key]; fixed || key == "" {
		return nil, false
	}
	if math.IsNaN(value) || math.IsInf(value, 0) {
		// Not a line-protocol value: one such field would cost the whole line.
		return nil, false
	}
	if _, integer := gpuUsageIntegerExtraFields[key]; integer {
		return int64(math.Round(value)), true
	}
	return value, true
}

var (
	startTime            = time.Now()
	telemetryClient      posthog.Client
	telemetryClientMu    sync.Once
	telemetryLockMu      sync.Mutex
	telemetryLastSent    time.Time
	telemetryMinInterval = 24 * time.Hour
)

func NewHypervisorMetricsRecorder(
	ctx context.Context, outputPath string,
	deviceController framework.DeviceController,
	workerController framework.WorkerController,
	allocationController framework.WorkerAllocationController,
	metricsInterval time.Duration,
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
		metricsInterval:      metricsInterval,
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
	deviceMetricsTicker := time.NewTicker(h.metricsInterval)
	go func() {
		for {
			select {
			case <-h.ctx.Done():
				return
			case <-deviceMetricsTicker.C:
				h.RecordDeviceMetrics(writer)
				h.RecordWorkerMetrics(writer)
				h.RecordLimiterEventMetrics(writer)
			}
		}
	}()
}

// write appends one record to the metrics file. The file is a bounded,
// rotating one nobody has to read, and a record that cannot be written (a full
// or read-only volume) is dropped: metrics never hold up the hypervisor. It
// reports whether the record was written.
func (h *HypervisorMetricsRecorder) write(writer io.Writer, data []byte) bool {
	if _, err := writer.Write(data); err != nil {
		if !h.writeFailing.Swap(true) {
			klog.Warningf("Failed to write metrics, dropping records until it works again: %v", err)
		}
		return false
	}
	if h.writeFailing.Swap(false) {
		klog.Infof("Metrics are written again")
	}
	return true
}

func (h *HypervisorMetricsRecorder) RecordDeviceMetrics(writer io.Writer) {
	// A device whose telemetry could not be read is absent from the result, so
	// no row is written for it instead of a row of zeros.
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
		// Units: memory_bytes in bytes; memory_percentage is memory in use as a
		// share of the device's memory; compute_percentage is the device's
		// utilization (0-100); compute_tflops is that share of the device's TFLOPS
		// capacity and is left out for a device whose capacity is unknown.
		// Memory or utilization the provider could not read in this sample is
		// left out as well: a missing field is "unknown", zero is a measurement.
		if !metrics.MemoryUnavailable {
			enc.AddField("memory_bytes", metrics.MemoryBytes)
			enc.AddField("memory_percentage", metrics.MemoryPercentage)
		}
		if !metrics.ComputeUnavailable {
			enc.AddField("compute_percentage", metrics.ComputePercentage)
			if device, exists := h.deviceController.GetDevice(gpuUUID); exists && device.EffectiveTflopsCapacity() > 0 {
				enc.AddField("compute_tflops", metrics.ComputeTflops)
			}
		}
		enc.AddField("power_usage", metrics.PowerUsage)
		for key, value := range metrics.ExtraMetrics {
			if field, ok := gpuUsageExtraField(key, value); ok {
				enc.AddField(key, field)
			}
		}
		enc.EndLine(now)
	}

	if err := enc.Err(); err == nil {
		h.write(writer, enc.Bytes())
	}
}

// RecordWorkerMetrics writes one tf_worker_usage row per worker and device the
// worker has GPU processes on.
//
//   - memory_bytes: device memory held by the worker's processes, in bytes.
//   - memory_percentage: memory_bytes as a share of the worker's VRAM limit, or
//     of the device's memory when the worker has no limit.
//   - compute_percentage: the worker's own utilization of the device (0-100),
//     averaged over the time since the previous record (see worker/usage.go for
//     the signal behind it). For the first record of a worker, and with a worker
//     controller that does not integrate usage, it is the sum of what the
//     provider reports for the worker's processes at this instant.
//   - compute_tflops: TFLOPS in use = compute_percentage / 100 x the device's
//     TFLOPS capacity (api.DeviceInfo.TflopsCapacity, FP16 TFLOPS when the
//     model is listed in the ProviderConfig).
//
// compute_percentage and compute_tflops are left out when the utilization was
// not measured, and compute_tflops also when the device's capacity is unknown.
// A missing field is "unknown"; zero is a measured idle worker.
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

	var usage map[string]map[string]api.WorkerComputeUsage
	if provider, ok := h.workerController.(framework.WorkerComputeUsageProvider); ok {
		usage = provider.GetWorkerComputeUsage()
	}
	previousUsage := h.lastUsage
	h.lastUsage = usage

	// Output worker metrics directly
	now := time.Now()
	enc := metrics.NewEncoder(os.Getenv(constants.HypervisorMetricsFormatEnv))

	for deviceUUID, workerMap := range workerMetrics {
		device, _ := h.deviceController.GetDevice(deviceUUID)
		for workerUID, processMap := range workerMap {
			allocation, ok := workerAllocations[workerUID]
			if !ok {
				continue
			}

			var memoryBytes uint64
			// The utilization of this instant is a sum over the worker's processes
			// and is only known when the provider supplied it for every one of them.
			computePercentage, computeAvailable := 0.0, true
			for _, metrics := range processMap {
				memoryBytes += metrics.MemoryBytes
				if metrics.ComputeUnavailable {
					computeAvailable = false
				} else {
					computePercentage += metrics.ComputePercentage
				}
			}
			// The average over the interval, when it was measured, is the better figure.
			if current, tracked := usage[deviceUUID][workerUID]; tracked {
				if before, known := previousUsage[deviceUUID][workerUID]; known {
					computePercentage, computeAvailable = current.MeanPercentSince(before)
				}
			}

			vramLimit := float64(0)
			if allocation.WorkerInfo != nil {
				vramLimit = float64(allocation.WorkerInfo.Limits.Vram.Value())
			}
			if vramLimit <= 0 && device != nil {
				vramLimit = float64(device.TotalMemoryBytes)
			}
			memoryPercentage := float64(0)
			if vramLimit > 0 {
				memoryPercentage = float64(memoryBytes) / vramLimit * 100.0
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
			if allocation.WorkerInfo != nil && allocation.WorkerInfo.WorkloadName != "" {
				workloadName = allocation.WorkerInfo.WorkloadName
			}

			enc.AddTag("workload", workloadName)
			enc.AddTag("worker", workerUID)

			// Add extra labels if configured
			h.addExtraLabels(enc, allocation)

			enc.AddField("memory_bytes", memoryBytes)
			if computeAvailable {
				enc.AddField("compute_percentage", computePercentage)
				if computeTflops, known := api.TflopsInUse(computePercentage, device); known {
					enc.AddField("compute_tflops", computeTflops)
				}
			}
			enc.AddField("memory_percentage", memoryPercentage)

			enc.EndLine(now)
		}
	}

	if err := enc.Err(); err == nil {
		h.write(writer, enc.Bytes())
	}
}

// SetLimiterEventSource makes the recorder export the counters of the limiter
// events endpoint. The HTTP server that owns them is created after the
// recorder, so the source is set once it exists; until then, and with a nil
// source, no tf_limiter_events rows are written. Safe to call while the
// recorder runs.
func (h *HypervisorMetricsRecorder) SetLimiterEventSource(source LimiterEventSource) {
	if source == nil {
		h.limiterEvents.Store(nil)
		return
	}
	h.limiterEvents.Store(&source)
}

// RecordLimiterEventMetrics writes the counters of the limiter events the
// workloads of this node reported (POST /api/v1/limiter/events) as
// tf_limiter_events rows, one per namespace, Pod and kind.
//
//   - tags: node, pool, namespace, pod and kind. kind is one of the fixed kinds
//     of the endpoint (a kind it does not know is counted as "other"); nothing
//     else a workload sent is in a tag.
//   - count (integer): the occurrences of the kind the Pod has reported since
//     this hypervisor process started counting them for it. It is a resettable
//     counter, kept in memory only: it starts again from zero when the
//     hypervisor restarts, and also when the endpoint has forgotten a Pod that
//     did not report for a while and the Pod reports again. A value lower than
//     the previous one of the row means the counter restarted, and that value
//     is the increase since.
//
// Only rows whose count differs from the one written last are written, so a
// Pod that reports nothing new costs nothing; the first record after the source
// is set writes every counter. What was written last is remembered for the
// rows of the current record only, and those come from the endpoint's own
// bounded state (at most handlers.LimiterEventsMaxPods Pods, 1024, with one
// counter per kind, each forgotten after handlers.LimiterEventsPodTTL without
// a report). Pods of the same name and namespace (one replaced by another
// while the first is still remembered) share a row, with their counts added.
// Rows that could not be written are written with the next record.
func (h *HypervisorMetricsRecorder) RecordLimiterEventMetrics(writer io.Writer) {
	source := h.limiterEvents.Load()
	if source == nil {
		return
	}
	counts := make(map[limiterEventRow]uint64, len(h.limiterEventsWritten))
	(*source)(func(namespace, pod, kind string, count uint64) {
		counts[limiterEventRow{namespace: namespace, pod: pod, kind: kind}] += count
	})

	now := time.Now()
	enc := metrics.NewEncoder(os.Getenv(constants.HypervisorMetricsFormatEnv))
	changed := false
	for row, count := range counts {
		if written, known := h.limiterEventsWritten[row]; known && written == count {
			continue
		}
		changed = true
		enc.StartLine(limiterEventsMeasurement)
		enc.AddTag("node", h.nodeName)
		enc.AddTag("pool", h.gpuPool)
		enc.AddTag("namespace", row.namespace)
		enc.AddTag("pod", row.pod)
		enc.AddTag("kind", row.kind)
		// A signed integer since the table exists (see the note on field types).
		enc.AddField("count", int64(count))
		enc.EndLine(now)
	}
	if changed && (enc.Err() != nil || !h.write(writer, enc.Bytes())) {
		// Keep what was written before, so that these rows are tried again.
		return
	}
	h.limiterEventsWritten = counts
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

func ShouldSendTelemetry() bool {
	if os.Getenv("DISABLE_TENSOR_FUSION_TELEMETRY") != "" {
		return false
	}
	if utils.IsTestMode {
		return false
	}

	telemetryLockMu.Lock()
	defer telemetryLockMu.Unlock()

	// Check if enough time has passed since last telemetry send
	if !telemetryLastSent.IsZero() && time.Since(telemetryLastSent) < telemetryMinInterval {
		return false
	}

	// Update last sent time
	telemetryLastSent = time.Now()
	return true
}

// SendAnonymousTelemetry sends Anonymous telemetry data without ANY sensitive data
func SendAnonymousTelemetry(
	nodeInfo *api.NodeInfo,
	hardwareVendor string,
	sampleGPUModel string,
	workersCount int,
	isolationMode api.IsolationMode,
) {
	// Get PostHog client
	client := getPostHogClient()
	if client == nil {
		klog.V(4).Infof("PostHog client not available, skipping telemetry")
		return
	}

	// Prepare event properties
	properties := posthog.NewProperties().
		Set("tflops", nodeInfo.TotalTFlops).
		Set("vram", nodeInfo.TotalVRAMBytes).
		Set("hostRam", nodeInfo.RAMSizeBytes).
		Set("devicesCnt", len(nodeInfo.DeviceIDs)).
		Set("brand", constants.Domain).
		Set("version", version.BuildVersion).
		Set("uptime", time.Since(startTime).String()).
		Set("workersCnt", workersCount).
		Set("mode", string(isolationMode)).
		Set("vendor", hardwareVendor).
		Set("gpuModel", sampleGPUModel).
		Set("product", getProductName()).
		Set("os", runtime.GOOS).
		Set("arch", runtime.GOARCH)

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

func getProductName() string {
	productName := os.Getenv(constants.TFProductNameEnv)
	if productName != "" {
		if _, ok := constants.ProductNameMap[productName]; ok {
			return constants.ProductNameMap[productName]
		}
	}
	return constants.ProductNameUnknown
}
