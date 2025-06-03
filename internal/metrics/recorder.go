package metrics

import (
	"io"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	metricsProto "github.com/influxdata/line-protocol/v2/lineprotocol"
	"gopkg.in/natefinch/lumberjack.v2"
	corev1 "k8s.io/api/core/v1"
	ctrl "sigs.k8s.io/controller-runtime"
)

// Worker level metrics, include worker resources/costs status
// map updated in one reconcile loop in single goroutine, thus no RW lock needed
var workerMetricsLock sync.RWMutex
var workerMetricsMap = map[string]*WorkerMetrics{}

// Node level metrics, include node allocation/costs status
var nodeMetricsLock sync.RWMutex
var NodeMetricsMap = map[string]*NodeMetrics{}

var log = ctrl.Log.WithName("metrics-recorder")

type MetricsRecorder struct {
	MetricsOutputPath string

	// Raw billing result for node and workers
	HourlyUnitPriceMap map[string]float64

	// Worker level unit price map, key is pool name, second level key is QoS level
	WorkerUnitPriceMap map[string]map[string]RawBillingPricing
}

func RemoveWorkerMetrics(workerName string, deletionTime time.Time) {
	workerMetricsLock.Lock()
	// to get more accurate metrics, should record the deletion timestamp to calculate duration for the last metrics
	workerMetricsMap[workerName].DeletionTimestamp = deletionTime
	workerMetricsLock.Unlock()
}

func RemoveNodeMetrics(nodeName string) {
	nodeMetricsLock.Lock()
	// Node lifecycle is much longer than worker, so just delete the metrics, 1 minute metrics interval is enough
	delete(NodeMetricsMap, nodeName)
	nodeMetricsLock.Unlock()
}

func SetWorkerMetricsByWorkload(pod *corev1.Pod, workload *tfv1.TensorFusionWorkload, now time.Time) {
	workerMetricsLock.Lock()
	defer workerMetricsLock.Unlock()

	// Initialize metrics
	if _, ok := workerMetricsMap[pod.Name]; !ok {
		workerMetricsMap[pod.Name] = &WorkerMetrics{
			WorkerName:     pod.Name,
			WorkloadName:   workload.Name,
			PoolName:       workload.Spec.PoolName,
			Namespace:      pod.Namespace,
			QoS:            string(workload.Spec.Qos),
			RawCost:        0,
			LastRecordTime: now,
		}
	}

	// Update metrics fields that are mutable
	metricsItem := workerMetricsMap[pod.Name]
	metricsItem.TflopsRequest = workload.Spec.Resources.Requests.Tflops.AsApproximateFloat64()
	metricsItem.TflopsLimit = workload.Spec.Resources.Limits.Tflops.AsApproximateFloat64()
	metricsItem.VramBytesRequest = workload.Spec.Resources.Requests.Vram.AsApproximateFloat64()
	metricsItem.VramBytesLimit = workload.Spec.Resources.Limits.Vram.AsApproximateFloat64()
	if workload.Spec.GPUCount <= 0 {
		// handle invalid data if exists
		metricsItem.GPUCount = 1
	} else {
		metricsItem.GPUCount = int(workload.Spec.GPUCount)
	}
	metricsItem.WorkloadName = workload.Name

}

func SetNodeMetrics(node *tfv1.GPUNode, poolObj *tfv1.GPUPool, gpuModels []string) {
	nodeMetricsLock.Lock()
	defer nodeMetricsLock.Unlock()

	if _, ok := NodeMetricsMap[node.Name]; !ok {
		NodeMetricsMap[node.Name] = &NodeMetrics{
			NodeName:       node.Name,
			RawCost:        0,
			LastRecordTime: time.Now(),
		}
	}
	// Fields that possibly change after initialization
	metricsItem := NodeMetricsMap[node.Name]
	metricsItem.PoolName = poolObj.Name
	metricsItem.GPUModels = gpuModels

	totalTflops := node.Status.TotalTFlops.AsApproximateFloat64()
	totalVram := node.Status.TotalVRAM.AsApproximateFloat64()

	metricsItem.AllocatedTflops = totalTflops - node.Status.AvailableTFlops.AsApproximateFloat64()
	if totalTflops <= 0 {
		metricsItem.AllocatedTflopsPercent = 0
	} else {
		metricsItem.AllocatedTflopsPercent = metricsItem.AllocatedTflops / totalTflops * 100
	}

	metricsItem.AllocatedVramBytes = totalVram - node.Status.AvailableVRAM.AsApproximateFloat64()
	if totalVram <= 0 {
		metricsItem.AllocatedVramPercent = 0
	} else {
		metricsItem.AllocatedVramPercent = metricsItem.AllocatedVramBytes / totalVram * 100
	}

	totalVirtualTflops := node.Status.VirtualTFlops.AsApproximateFloat64()
	totalVirtualVram := node.Status.VirtualVRAM.AsApproximateFloat64()
	if totalVirtualTflops <= 0 {
		metricsItem.AllocatedTflopsPercentToVirtualCap = 0
	} else {
		metricsItem.AllocatedTflopsPercentToVirtualCap = metricsItem.AllocatedTflops / totalVirtualTflops * 100
	}
	if totalVirtualVram <= 0 {
		metricsItem.AllocatedVramPercentToVirtualCap = 0
	} else {
		metricsItem.AllocatedVramPercentToVirtualCap = metricsItem.AllocatedVramBytes / totalVirtualVram * 100
	}
}

// Start metrics recorder
// The leader container will fill the metrics map, so followers don't have metrics point
// thus metrics recorder only printed in one controller instance
// One minute interval could cause some metrics ignored or billing not accurate, known issue
func (mr *MetricsRecorder) Start() {

	ticker := time.NewTicker(time.Minute)

	writer := &lumberjack.Logger{
		Filename:   mr.MetricsOutputPath,
		MaxSize:    100,
		MaxBackups: 10,
		MaxAge:     28,
	}

	// Record metrics
	go func() {
		for {
			<-ticker.C
			mr.RecordMetrics(writer)
		}
	}()

	// Clean up worker metrics that have been deleted
	go func() {
		for {
			time.Sleep(5 * time.Minute)
			workerMetricsLock.Lock()
			for _, metrics := range workerMetricsMap {
				if !metrics.DeletionTimestamp.IsZero() {
					delete(workerMetricsMap, metrics.WorkerName)
				}
			}
			workerMetricsLock.Unlock()
		}
	}()
}

func (mr *MetricsRecorder) RecordMetrics(writer io.Writer) {
	if len(workerMetricsMap) <= 0 && len(NodeMetricsMap) <= 0 {
		return
	}

	now := time.Now()

	var enc metricsProto.Encoder
	enc.SetPrecision(metricsProto.Millisecond)

	workerMetricsLock.RLock()

	activeWorkerCnt := 0
	for _, metrics := range workerMetricsMap {

		if !metrics.DeletionTimestamp.IsZero() {
			metrics.RawCost = mr.getWorkerRawCost(metrics, metrics.DeletionTimestamp.Sub(metrics.LastRecordTime))
		} else {
			metrics.RawCost = mr.getWorkerRawCost(metrics, now.Sub(metrics.LastRecordTime))
		}
		metrics.LastRecordTime = now

		// Skip recording metrics if raw cost is negative
		// which means worker already deleted waiting for cleanup
		if metrics.RawCost < 0 {
			continue
		}
		activeWorkerCnt++
		enc.StartLine("tf_worker_metrics")
		enc.AddTag("namespace", metrics.Namespace)
		enc.AddTag("pool_name", metrics.PoolName)
		enc.AddTag("qos", metrics.QoS)
		enc.AddTag("worker_name", metrics.WorkerName)
		enc.AddTag("workload_name", metrics.WorkloadName)

		enc.AddField("gpu_count", metricsProto.MustNewValue(int64(metrics.GPUCount)))
		enc.AddField("tflops_limit", metricsProto.MustNewValue(metrics.TflopsLimit))
		enc.AddField("tflops_request", metricsProto.MustNewValue(metrics.TflopsRequest))
		enc.AddField("raw_cost", metricsProto.MustNewValue(metrics.RawCost))
		enc.AddField("vram_bytes_limit", metricsProto.MustNewValue(metrics.VramBytesLimit))
		enc.AddField("vram_bytes_request", metricsProto.MustNewValue(metrics.VramBytesRequest))

		enc.EndLine(now)
	}
	enc.StartLine("tf_system_metrics")
	enc.AddField("total_workers_cnt", metricsProto.MustNewValue(int64(activeWorkerCnt)))
	workerMetricsLock.RUnlock()

	nodeMetricsLock.RLock()
	for _, metrics := range NodeMetricsMap {
		metrics.RawCost = mr.getNodeRawCost(metrics, now.Sub(metrics.LastRecordTime), mr.HourlyUnitPriceMap)
		metrics.LastRecordTime = now

		enc.StartLine("tf_node_metrics")

		enc.AddTag("node_name", metrics.NodeName)
		enc.AddTag("pool_name", metrics.PoolName)

		enc.AddField("allocated_tflops", metricsProto.MustNewValue(metrics.AllocatedTflops))
		enc.AddField("allocated_tflops_percent", metricsProto.MustNewValue(metrics.AllocatedTflopsPercent))
		enc.AddField("allocated_vram_bytes", metricsProto.MustNewValue(metrics.AllocatedVramBytes))
		enc.AddField("allocated_vram_percent", metricsProto.MustNewValue(metrics.AllocatedVramPercent))
		enc.AddField("gpu_count", metricsProto.MustNewValue(int64(len(metrics.GPUModels))))
		enc.AddField("raw_cost", metricsProto.MustNewValue(metrics.RawCost))
		enc.EndLine(now)
	}
	enc.StartLine("tf_system_metrics")
	enc.AddField("total_nodes_cnt", metricsProto.MustNewValue(int64(len(NodeMetricsMap))))
	enc.EndLine(now)

	nodeMetricsLock.RUnlock()

	if err := enc.Err(); err != nil {
		log.Error(err, "metrics encoding error", "workerCount", activeWorkerCnt, "nodeCount", len(NodeMetricsMap))
	}

	if _, err := writer.Write(enc.Bytes()); err != nil {
		log.Error(err, "metrics writing error", "workerCount", activeWorkerCnt, "nodeCount", len(NodeMetricsMap))
	}
	log.Info("metrics and raw billing recorded:", "workerCount", activeWorkerCnt, "nodeCount", len(NodeMetricsMap))
}

func (mr *MetricsRecorder) getWorkerRawCost(metrics *WorkerMetrics, duration time.Duration) float64 {
	qosPricing, ok := mr.WorkerUnitPriceMap[metrics.PoolName]
	// The qos pricing for this pool not set
	if !ok {
		return 0
	}
	// The price of current qos not defined for this pool
	qosLevel := metrics.QoS
	if qosLevel == "" {
		qosLevel = constants.QoSLevelMedium
	}
	pricing, ok := qosPricing[qosLevel]
	if !ok {
		return 0
	}
	if duration < 0 {
		return -1
	}

	rawCostTflopsLimitOverRequest := (metrics.TflopsLimit - metrics.TflopsRequest) * pricing.TflopsOverRequestPerSecond
	rawCostPerTflops := pricing.TflopsPerSecond * metrics.TflopsRequest

	rawCostVRAMLimitOverRequest := (metrics.VramBytesLimit - metrics.VramBytesRequest) * pricing.VramOverRequestPerSecond / constants.GiBToBytes
	rawCostPerVRAM := pricing.VramPerSecond * metrics.VramBytesRequest / constants.GiBToBytes

	return (rawCostPerTflops + rawCostPerVRAM + rawCostTflopsLimitOverRequest + rawCostVRAMLimitOverRequest) * duration.Seconds() * float64(metrics.GPUCount)
}

// unit price data comes from global config map, and multi-GPU instance should normalized with per GPU pricing, e.g. 8xA100 p4d.24xlarge price should divide by 8
func (mr *MetricsRecorder) getNodeRawCost(metrics *NodeMetrics, duration time.Duration, hourlyUnitPriceMap map[string]float64) float64 {
	cost := 0.0
	for _, gpuModel := range metrics.GPUModels {
		cost += metrics.AllocatedTflops * duration.Hours() * hourlyUnitPriceMap[gpuModel]
	}
	return cost
}
