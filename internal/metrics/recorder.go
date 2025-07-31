package metrics

import (
	"io"
	"strconv"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"gopkg.in/natefinch/lumberjack.v2"
	corev1 "k8s.io/api/core/v1"
	ctrl "sigs.k8s.io/controller-runtime"
)

// Worker level metrics, include worker resources/costs status
// map updated in one reconcile loop in single goroutine, thus no RW lock needed
var workerMetricsLock sync.RWMutex
var workerMetricsMap = map[string]*WorkerResourceMetrics{}

// Node level metrics, include node allocation/costs status
var nodeMetricsLock sync.RWMutex
var nodeMetricsMap = map[string]*NodeResourceMetrics{}

var log = ctrl.Log.WithName("metrics-recorder")

type MetricsRecorder struct {
	MetricsOutputPath string

	// Raw billing result for node and workers
	HourlyUnitPriceMap map[string]float64

	// Worker level unit price map, key is pool name, second level key is QoS level
	WorkerUnitPriceMap map[string]map[string]RawBillingPricing
}

type ActiveNodeAndWorker struct {
	workerCnt int
	nodeCnt   int
}

func RemoveWorkerMetrics(workerName string, deletionTime time.Time) {
	defer workerMetricsLock.Unlock()
	workerMetricsLock.Lock()

	// to get more accurate metrics, should record the deletion timestamp to calculate duration for the last metrics
	if _, ok := workerMetricsMap[workerName]; ok {
		workerMetricsMap[workerName].deletionTimestamp = &deletionTime
	}
}

func RemoveNodeMetrics(nodeName string) {
	defer nodeMetricsLock.Unlock()
	nodeMetricsLock.Lock()
	// Node lifecycle is much longer than worker, so just delete the metrics, 1 minute metrics interval is enough
	delete(nodeMetricsMap, nodeName)
}

func SetWorkerMetricsByWorkload(pod *corev1.Pod) {
	workerMetricsLock.Lock()
	defer workerMetricsLock.Unlock()

	gpuRequestResource, err := utils.GetGPUResource(pod, true)
	if err != nil {
		return
	}
	gpuLimitResource, err := utils.GetGPUResource(pod, false)
	if err != nil {
		return
	}
	count, _ := strconv.ParseUint(pod.Annotations[constants.GpuCountAnnotation], 10, 32)

	// Initialize metrics
	if _, ok := workerMetricsMap[pod.Name]; !ok {
		workerMetricsMap[pod.Name] = &WorkerResourceMetrics{
			WorkerName:     pod.Name,
			WorkloadName:   pod.Labels[constants.WorkloadKey],
			PoolName:       pod.Annotations[constants.GpuPoolKey],
			Namespace:      pod.Namespace,
			QoS:            pod.Annotations[constants.QoSLevelAnnotation],
			podLabels:      pod.Labels,
			RawCost:        0,
			LastRecordTime: time.Now(),
		}
	}

	// Update metrics fields that are mutable
	metricsItem := workerMetricsMap[pod.Name]
	metricsItem.TflopsRequest = gpuRequestResource.Tflops.AsApproximateFloat64()
	metricsItem.TflopsLimit = gpuLimitResource.Tflops.AsApproximateFloat64()
	metricsItem.VramBytesRequest = gpuRequestResource.Vram.AsApproximateFloat64()
	metricsItem.VramBytesLimit = gpuLimitResource.Vram.AsApproximateFloat64()
	if count <= 0 {
		// handle invalid data if exists
		metricsItem.GPUCount = 1
	} else {
		metricsItem.GPUCount = int(count)
	}
	metricsItem.WorkloadName = pod.Labels[constants.WorkloadKey]
}

func SetNodeMetrics(node *tfv1.GPUNode, poolObj *tfv1.GPUPool, gpuModels []string) {
	nodeMetricsLock.Lock()
	defer nodeMetricsLock.Unlock()

	if _, ok := nodeMetricsMap[node.Name]; !ok {
		nodeMetricsMap[node.Name] = &NodeResourceMetrics{
			NodeName:       node.Name,
			RawCost:        0,
			LastRecordTime: time.Now(),
		}
	}
	// Fields that possibly change after initialization
	metricsItem := nodeMetricsMap[node.Name]
	metricsItem.PoolName = poolObj.Name
	metricsItem.SetGPUModelAndCount(gpuModels)

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

func SetSchedulerMetrics(poolName string, isSuccess bool) {
	if _, ok := TensorFusionSystemMetricsMap[poolName]; !ok {
		TensorFusionSystemMetricsMap[poolName] = &TensorFusionSystemMetrics{
			PoolName: poolName,
		}
	}
	if isSuccess {
		TensorFusionSystemMetricsMap[poolName].TotalAllocationSuccessCount++
	} else {
		TensorFusionSystemMetricsMap[poolName].TotalAllocationFailCount++
	}
}

// TODO should record metrics after autoscaling feature added
func SetAutoscalingMetrics(poolName string, isScaleUp bool) {
	if _, ok := TensorFusionSystemMetricsMap[poolName]; !ok {
		TensorFusionSystemMetricsMap[poolName] = &TensorFusionSystemMetrics{
			PoolName: poolName,
		}
	}
	if isScaleUp {
		TensorFusionSystemMetricsMap[poolName].TotalScaleUpCount++
	} else {
		TensorFusionSystemMetricsMap[poolName].TotalScaleDownCount++
	}
}

func getSchedulerMetricsByPool(poolName string) (int64, int64, int64, int64) {
	if item, ok := TensorFusionSystemMetricsMap[poolName]; !ok {
		return 0, 0, 0, 0
	} else {
		return item.TotalAllocationSuccessCount, item.TotalAllocationFailCount, item.TotalScaleUpCount, item.TotalScaleDownCount
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
		MaxAge:     14,
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
				if metrics.deletionTimestamp != nil && !metrics.deletionTimestamp.IsZero() {
					delete(workerMetricsMap, metrics.WorkerName)
				}
			}
			workerMetricsLock.Unlock()
		}
	}()
}

func (mr *MetricsRecorder) RecordMetrics(writer io.Writer) {
	if len(workerMetricsMap) <= 0 && len(nodeMetricsMap) <= 0 {
		return
	}

	now := time.Now()
	enc := NewEncoder(config.GetGlobalConfig().MetricsFormat)
	workerMetricsLock.RLock()

	activeWorkerCnt := 0
	activeWorkerAndNodeByPool := map[string]*ActiveNodeAndWorker{}

	for _, metrics := range workerMetricsMap {

		if metrics.deletionTimestamp != nil && !metrics.deletionTimestamp.IsZero() {
			metrics.RawCost = mr.getWorkerRawCost(metrics, metrics.deletionTimestamp.Sub(metrics.LastRecordTime))
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

		if _, ok := activeWorkerAndNodeByPool[metrics.PoolName]; !ok {
			activeWorkerAndNodeByPool[metrics.PoolName] = &ActiveNodeAndWorker{
				workerCnt: 0,
				nodeCnt:   0,
			}
		}
		activeWorkerAndNodeByPool[metrics.PoolName].workerCnt++

		enc.StartLine("tf_worker_resources")
		enc.AddTag("namespace", metrics.Namespace)
		enc.AddTag("pool", metrics.PoolName)

		if metrics.QoS == "" {
			metrics.QoS = constants.QoSLevelMedium
		}
		enc.AddTag("qos", metrics.QoS)
		enc.AddTag("worker", metrics.WorkerName)
		enc.AddTag("workload", metrics.WorkloadName)

		if config.GetGlobalConfig().MetricsExtraPodLabels != nil {
			for _, label := range config.GetGlobalConfig().MetricsExtraPodLabels {
				enc.AddTag(label, metrics.podLabels[label])
			}
		}

		enc.AddField("gpu_count", int64(metrics.GPUCount))
		enc.AddField("tflops_limit", metrics.TflopsLimit)
		enc.AddField("tflops_request", metrics.TflopsRequest)
		enc.AddField("raw_cost", metrics.RawCost)
		enc.AddField("vram_bytes_limit", metrics.VramBytesLimit)
		enc.AddField("vram_bytes_request", metrics.VramBytesRequest)

		enc.EndLine(now)
	}
	workerMetricsLock.RUnlock()

	nodeMetricsLock.RLock()

	for _, metrics := range nodeMetricsMap {
		metrics.RawCost = mr.getNodeRawCost(metrics, now.Sub(metrics.LastRecordTime), mr.HourlyUnitPriceMap)
		metrics.LastRecordTime = now

		if _, ok := activeWorkerAndNodeByPool[metrics.PoolName]; !ok {
			activeWorkerAndNodeByPool[metrics.PoolName] = &ActiveNodeAndWorker{
				workerCnt: 0,
				nodeCnt:   0,
			}
		}
		activeWorkerAndNodeByPool[metrics.PoolName].nodeCnt++

		enc.StartLine("tf_node_metrics")

		enc.AddTag("node", metrics.NodeName)
		enc.AddTag("pool", metrics.PoolName)

		enc.AddField("allocated_tflops", metrics.AllocatedTflops)
		enc.AddField("allocated_tflops_percent", metrics.AllocatedTflopsPercent)
		enc.AddField("allocated_tflops_percent_virtual", metrics.AllocatedTflopsPercentToVirtualCap)
		enc.AddField("allocated_vram_bytes", metrics.AllocatedVramBytes)
		enc.AddField("allocated_vram_percent", metrics.AllocatedVramPercent)
		enc.AddField("allocated_vram_percent_virtual", metrics.AllocatedVramPercentToVirtualCap)
		enc.AddField("gpu_count", int64(metrics.GPUCount))
		enc.AddField("raw_cost", metrics.RawCost)
		enc.EndLine(now)
	}

	for poolName, activeNodeAndWorker := range activeWorkerAndNodeByPool {
		enc.StartLine("tf_system_metrics")
		successCount, failCount, scaleUpCount, scaleDownCount := getSchedulerMetricsByPool(poolName)
		enc.AddTag("pool", poolName)
		enc.AddField("total_workers_cnt", int64(activeNodeAndWorker.workerCnt))
		enc.AddField("total_nodes_cnt", int64(activeNodeAndWorker.nodeCnt))
		enc.AddField("total_allocation_fail_cnt", failCount)
		enc.AddField("total_allocation_success_cnt", successCount)
		enc.AddField("total_scale_up_cnt", scaleUpCount)
		enc.AddField("total_scale_down_cnt", scaleDownCount)
		enc.EndLine(now)
	}

	nodeMetricsLock.RUnlock()

	if err := enc.Err(); err != nil {
		log.Error(err, "metrics encoding error", "workerCount", activeWorkerCnt, "nodeCount", len(nodeMetricsMap))
	}

	if _, err := writer.Write(enc.Bytes()); err != nil {
		log.Error(err, "metrics writing error", "workerCount", activeWorkerCnt, "nodeCount", len(nodeMetricsMap))
	}
	log.Info("metrics and raw billing recorded:", "workerCount", activeWorkerCnt, "nodeCount", len(nodeMetricsMap))
}

func (mr *MetricsRecorder) getWorkerRawCost(metrics *WorkerResourceMetrics, duration time.Duration) float64 {
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
func (mr *MetricsRecorder) getNodeRawCost(metrics *NodeResourceMetrics, duration time.Duration, hourlyUnitPriceMap map[string]float64) float64 {
	cost := 0.0
	for _, gpuModel := range metrics.gpuModels {
		cost += metrics.AllocatedTflops * duration.Hours() * hourlyUnitPriceMap[gpuModel]
	}
	return cost
}
