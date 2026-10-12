package metrics

import (
	"io"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"gopkg.in/natefinch/lumberjack.v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
)

// Worker level metrics, include worker resources/costs status
// map updated in one reconcile loop in single goroutine, thus no RW lock needed
var workerMetricsLock sync.RWMutex
var workerMetricsMap = make(map[string]*WorkerResourceMetrics, 200)

// Node level metrics, include node allocation/costs status
var nodeMetricsLock sync.RWMutex
var nodeMetricsMap = make(map[string]*NodeResourceMetrics, 100)

// Pool level metrics, include pool allocation/costs status
var poolMetricsLock sync.RWMutex
var poolMetricsMap = make(map[string]*PoolResourceMetrics, 4)

// GPU allocation metrics, key is podName (one record per pod, aggregated from all GPUs)
var gpuAllocationMetricsLock sync.RWMutex
var gpuAllocationMetricsMap = make(map[string]*GPUAllocationMetrics, 200)

// Per-GPU card resource metrics, key is GPU name (UUID)
var gpuResourceMetricsLock sync.RWMutex
var gpuResourceMetricsMap = make(map[string]*GPUResourceMetrics, 100)

var settingLock sync.RWMutex
var tensorFusionSystemMetricsLock sync.RWMutex

var log = ctrl.Log.WithName("metrics-recorder")

type MetricsRecorder struct {
	MetricsOutputPath string

	// ProviderManager is the live source of GPU pricing data. The recorder
	// pulls pricing on every metric cycle via currentPricing(), so changes to
	// ProviderConfig take effect without restarting the operator.
	ProviderManager *provider.Manager

	// ConfigMapPricing is the legacy pricing source loaded from a ConfigMap at
	// startup. ProviderManager pricing takes precedence over it; kept for
	// backward compatibility with deployments that still rely on the
	// ConfigMap.
	ConfigMapPricing map[string]float64

	// Pool level eviction protection price ratio map, key is pool name
	PoolEvictionProtectionPriceRatioMap map[string]string

	// Worker level unit price map, key is pool name, second level key is QoS level
	WorkerUnitPriceMap map[string]map[string]RawBillingPricing

	// NodeHourlyPrice, when set, tells the hourly rate of a whole node bought
	// as the given capacity type. See nodeHourlyPrice.
	NodeHourlyPrice NodeHourlyPriceFunc
}

// NodeHourlyPriceFunc returns the hourly price of a node of an instance type
// bought as a capacity type (on-demand, spot or reserved, as on the
// karpenter.sh/capacity-type label) in a region, and false when it has no price
// for that combination. The region is empty when the node does not say.
//
// source says where the price comes from and is one of the PriceSource values:
// PriceSourceList for a listed price, PriceSourceSpotEstimate for a figure that
// was derived from an assumption. It is written as the price_source tag, so an
// estimate must never be returned as PriceSourceList.
type NodeHourlyPriceFunc func(instanceType, capacityType, region string) (hourlyPrice float64, source string, ok bool)

// Where the hourly rate of a node comes from; these are the values of the
// price_source tag of tf_node_metrics. A row without the tag has no price.
const (
	// PriceSourceList is a price listed in the pricing data for the instance
	// type and capacity type of the node.
	PriceSourceList = "list"
	// PriceSourceSpotEstimate is a listed on-demand price multiplied by an
	// assumed spot factor: an estimate, not what the node is actually charged.
	PriceSourceSpotEstimate = "spot-estimate"
	// PriceSourceConfigured is the sum of the per-GPU unit prices the cluster
	// administrator configured for the GPU models of the node.
	PriceSourceConfigured = "configured"

	// ListPriceCurrency is the currency of the built-in instance price lists,
	// and so of every rate whose source is PriceSourceList or
	// PriceSourceSpotEstimate.
	ListPriceCurrency = "USD"
)

// knownPriceSource keeps the price_source tag to its documented values: a
// source the recorder does not know is left out rather than passed on.
func knownPriceSource(source string) string {
	switch source {
	case PriceSourceList, PriceSourceSpotEstimate, PriceSourceConfigured:
		return source
	}
	return ""
}

// Capacity types as the karpenter.sh/capacity-type node label spells them;
// these are the values of the capacity_type tag of tf_node_metrics.
const (
	CapacityTypeOnDemand = "on-demand"
	CapacityTypeSpot     = "spot"
	CapacityTypeReserved = "reserved"
)

// normalizeCapacityType maps a capacity type label value to one of the three
// tag values, and anything else to "" so that the tag only ever takes them.
func normalizeCapacityType(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case CapacityTypeOnDemand, "ondemand":
		return CapacityTypeOnDemand
	case CapacityTypeSpot, "preemptive", "preemptible":
		return CapacityTypeSpot
	case CapacityTypeReserved:
		return CapacityTypeReserved
	}
	return ""
}

// currentPricing returns the live GPU model → hourly unit price map, merging
// ConfigMap pricing (legacy) with ProviderManager pricing (authoritative,
// updated by providerconfig_controller on every ProviderConfig change).
func (mr *MetricsRecorder) currentPricing() map[string]float64 {
	merged := make(map[string]float64, len(mr.ConfigMapPricing))
	for k, v := range mr.ConfigMapPricing {
		merged[k] = v
	}
	if mr.ProviderManager != nil {
		for k, v := range mr.ProviderManager.GetGPUPricingMap() {
			merged[k] = v
		}
	}
	return merged
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

func SetGPUMetrics(gpuList []tfv1.GPU, nodeName, poolName string) {
	gpuResourceMetricsLock.Lock()
	defer gpuResourceMetricsLock.Unlock()

	// Remove stale entries for this node that are no longer in the GPU list
	currentGPUs := make(map[string]struct{}, len(gpuList))
	for i := range gpuList {
		currentGPUs[gpuList[i].Name] = struct{}{}
	}
	for key, m := range gpuResourceMetricsMap {
		if m.NodeName == nodeName {
			if _, ok := currentGPUs[key]; !ok {
				delete(gpuResourceMetricsMap, key)
			}
		}
	}

	for i := range gpuList {
		gpu := &gpuList[i]
		if gpu.Status.Available == nil || gpu.Status.Capacity == nil {
			continue
		}

		capTflops := gpu.Status.Capacity.Tflops.AsApproximateFloat64()
		availTflops := gpu.Status.Available.Tflops.AsApproximateFloat64()
		allocTflops := capTflops - availTflops

		capVram := gpu.Status.Capacity.Vram.AsApproximateFloat64()
		availVram := gpu.Status.Available.Vram.AsApproximateFloat64()
		allocVram := capVram - availVram

		var allocTflopsPercent, allocVramPercent float64
		if capTflops > 0 {
			allocTflopsPercent = allocTflops / capTflops * 100
		}
		if capVram > 0 {
			allocVramPercent = allocVram / capVram * 100
		}

		gpuResourceMetricsMap[gpu.Name] = &GPUResourceMetrics{
			GPUName:                gpu.Name,
			NodeName:               nodeName,
			PoolName:               poolName,
			GPUModel:               gpu.Status.GPUModel,
			Phase:                  string(gpu.Status.Phase),
			CapacityTflops:         capTflops,
			AvailableTflops:        availTflops,
			AllocatedTflops:        allocTflops,
			AllocatedTflopsPercent: allocTflopsPercent,
			CapacityVramBytes:      capVram,
			AvailableVramBytes:     availVram,
			AllocatedVramBytes:     allocVram,
			AllocatedVramPercent:   allocVramPercent,
			LastRecordTime:         time.Now(),
		}
	}
}

func RemoveGPUMetricsByNode(nodeName string) {
	gpuResourceMetricsLock.Lock()
	defer gpuResourceMetricsLock.Unlock()
	for key, m := range gpuResourceMetricsMap {
		if m.NodeName == nodeName {
			delete(gpuResourceMetricsMap, key)
		}
	}
}

func SetWorkerMetricsByWorkload(
	pod *corev1.Pod, gpuStore map[types.NamespacedName]*tfv1.GPU,
) {
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
			WorkerName:         pod.Name,
			WorkloadName:       pod.Labels[constants.WorkloadKey],
			PoolName:           pod.Annotations[constants.GpuPoolKey],
			Namespace:          pod.Namespace,
			QoS:                pod.Annotations[constants.QoSLevelAnnotation],
			podLabels:          pod.Labels,
			RawCost:            0,
			LastRecordTime:     time.Now(),
			creationTime:       pod.CreationTimestamp.Time,
			evictionProtection: pod.Annotations[constants.EvictionProtectionAnnotation],
		}
	}

	// Update metrics fields that are mutable
	metricsItem := workerMetricsMap[pod.Name]
	metricsItem.TflopsRequest, metricsItem.TflopsLimit = getWorkerTflopsRequestLimit(gpuStore, pod, gpuRequestResource, gpuLimitResource)
	metricsItem.VramBytesRequest = gpuRequestResource.Vram.AsApproximateFloat64()
	metricsItem.VramBytesLimit = getWorkerVramBytesLimit(gpuStore, pod, gpuLimitResource)
	metricsItem.Ready = utils.IsPodConditionTrue(pod.Status.Conditions, corev1.PodReady)
	if count <= 0 || count > uint64(math.MaxInt32) {
		// handle invalid or out-of-bounds data
		metricsItem.GPUCount = 1
	} else {
		metricsItem.GPUCount = int(count)
	}
	metricsItem.WorkloadName = pod.Labels[constants.WorkloadKey]
	if pod.Spec.NodeName != "" {
		metricsItem.NodeName = pod.Spec.NodeName
	}
}

func getWorkerTflopsRequestLimit(
	gpuStore map[types.NamespacedName]*tfv1.GPU,
	pod *corev1.Pod,
	request tfv1.Resource,
	limit tfv1.Resource,
) (float64, float64) {
	reqTflops := request.Tflops.AsApproximateFloat64()
	limitTflops := limit.Tflops.AsApproximateFloat64()

	// if both request and limit already have explicit TFLOPs, return as-is.
	if reqTflops > 0 || limitTflops > 0 {
		return reqTflops, limitTflops
	}

	if request.ComputePercent.IsZero() && limit.ComputePercent.IsZero() {
		return 0, 0
	}

	var capTflops float64
	derive := func(res tfv1.Resource) float64 {
		// Prefer explicit TFLOPs when present.
		if !res.Tflops.IsZero() {
			return res.Tflops.AsApproximateFloat64()
		}
		if res.ComputePercent.IsZero() {
			return 0
		}
		return res.ComputePercent.AsApproximateFloat64() * capTflops / 100
	}
	if gpuDeviceIDs, exists := pod.Annotations[constants.GPUDeviceIDsAnnotation]; exists && gpuDeviceIDs != "" {
		gpuIDList := strings.Split(gpuDeviceIDs, ",")
		sampleGPU := strings.TrimSpace(gpuIDList[0])
		if sampleGPU != "" {
			gpu, exists := gpuStore[types.NamespacedName{Name: sampleGPU}]
			if !exists || gpu == nil {
				log.V(5).Info("GPU not found in store for GPU allocation metrics", "gpuID", sampleGPU, "pod", pod.Name)
				return 0, 0
			}
			if gpu.Status.Capacity == nil {
				log.V(5).Info("GPU capacity not available", "gpuID", sampleGPU, "pod", pod.Name)
				return 0, 0
			}
			capTflops = gpu.Status.Capacity.Tflops.AsApproximateFloat64()
		}
	}
	return derive(request), derive(limit)
}

func getWorkerVramBytesLimit(gpuStore map[types.NamespacedName]*tfv1.GPU, pod *corev1.Pod, limit tfv1.Resource) float64 {
	limitVram := limit.Vram.AsApproximateFloat64()
	if limitVram > 0 {
		return limitVram
	}

	if gpuDeviceIDs, exists := pod.Annotations[constants.GPUDeviceIDsAnnotation]; exists && gpuDeviceIDs != "" {
		gpuIDList := strings.Split(gpuDeviceIDs, ",")
		sampleGPU := strings.TrimSpace(gpuIDList[0])
		if sampleGPU != "" {
			gpu, exists := gpuStore[types.NamespacedName{Name: sampleGPU}]
			if !exists || gpu == nil {
				log.V(5).Info("GPU not found in store for GPU allocation metrics", "gpuID", sampleGPU, "pod", pod.Name)
				return 0
			}
			if gpu.Status.Capacity == nil {
				log.V(5).Info("GPU capacity not available", "gpuID", sampleGPU, "pod", pod.Name)
				return 0
			}
			return gpu.Status.Capacity.Vram.AsApproximateFloat64()
		}
	}
	return 0
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
	metricsItem.Phase = string(node.Status.Phase)
	metricsItem.SetGPUModelAndCount(gpuModels)
	// The node controller copies these labels from the Kubernetes Node.
	labels := node.GetLabels()
	metricsItem.CapacityType = normalizeCapacityType(labels[string(tfv1.NodeRequirementKeyCapacityType)])
	metricsItem.InstanceType = labels[string(tfv1.NodeRequirementKeyInstanceType)]
	metricsItem.region = labels[string(tfv1.NodeRequirementKeyRegion)]

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

	// Limits assigned on the node, against its virtual (oversold) capacity. Known
	// once the allocator has published the node's virtual available capacity.
	metricsItem.limitedKnown = node.Status.VirtualAvailableTFlops != nil && node.Status.VirtualAvailableVRAM != nil
	metricsItem.LimitedTFlops, metricsItem.LimitedVramBytes = 0, 0
	metricsItem.LimitedTFlopsPercentToVirtualCap, metricsItem.LimitedVramPercentToVirtualCap = 0, 0
	if metricsItem.limitedKnown {
		metricsItem.LimitedTFlops = totalVirtualTflops - node.Status.VirtualAvailableTFlops.AsApproximateFloat64()
		metricsItem.LimitedVramBytes = totalVirtualVram - node.Status.VirtualAvailableVRAM.AsApproximateFloat64()
		if totalVirtualTflops > 0 {
			metricsItem.LimitedTFlopsPercentToVirtualCap = metricsItem.LimitedTFlops / totalVirtualTflops * 100
		}
		if totalVirtualVram > 0 {
			metricsItem.LimitedVramPercentToVirtualCap = metricsItem.LimitedVramBytes / totalVirtualVram * 100
		}
	}
}

func InitPoolMetricsWhenNotExists(poolObj *tfv1.GPUPool) {
	poolMetricsLock.RLock()
	_, ok := poolMetricsMap[poolObj.Name]
	poolMetricsLock.RUnlock()
	if !ok {
		SetPoolMetrics(poolObj)
	}
}

func SetPoolMetrics(poolObj *tfv1.GPUPool) {
	if poolObj == nil {
		return
	}
	poolMetricsLock.Lock()
	defer poolMetricsLock.Unlock()
	if _, ok := poolMetricsMap[poolObj.Name]; !ok {
		poolMetricsMap[poolObj.Name] = &PoolResourceMetrics{
			PoolName: poolObj.Name,
		}
	}
	poolMetricsMap[poolObj.Name].Phase = string(poolObj.Status.Phase)

	if poolObj.Status.TotalVRAM.AsApproximateFloat64() > 0 && poolObj.Status.TotalTFlops.AsApproximateFloat64() > 0 {
		poolMetricsMap[poolObj.Name].AllocatedTflops = poolObj.Status.TotalTFlops.AsApproximateFloat64() -
			poolObj.Status.AvailableTFlops.AsApproximateFloat64()
		poolMetricsMap[poolObj.Name].AllocatedTflopsPercent = poolMetricsMap[poolObj.Name].AllocatedTflops /
			poolObj.Status.TotalTFlops.AsApproximateFloat64() * 100

		poolMetricsMap[poolObj.Name].AllocatedVramBytes = poolObj.Status.TotalVRAM.AsApproximateFloat64() -
			poolObj.Status.AvailableVRAM.AsApproximateFloat64()
		poolMetricsMap[poolObj.Name].AllocatedVramPercent = poolMetricsMap[poolObj.Name].AllocatedVramBytes /
			poolObj.Status.TotalVRAM.AsApproximateFloat64() * 100
	}

	if poolObj.Status.VirtualAvailableTFlops != nil && poolObj.Status.VirtualAvailableVRAM != nil {
		virtualVRAM := poolObj.Status.VirtualVRAM.AsApproximateFloat64()
		virtualTFlops := poolObj.Status.VirtualTFlops.AsApproximateFloat64()

		if virtualVRAM > 0 {
			poolMetricsMap[poolObj.Name].AllocatedVramPercentToVirtualCap = poolMetricsMap[poolObj.Name].AllocatedVramBytes / virtualVRAM * 100
		} else {
			poolMetricsMap[poolObj.Name].AllocatedVramPercentToVirtualCap = 0
		}

		if virtualTFlops > 0 {
			poolMetricsMap[poolObj.Name].AllocatedTflopsPercentToVirtualCap = poolMetricsMap[poolObj.Name].AllocatedTflops / virtualTFlops * 100
		} else {
			poolMetricsMap[poolObj.Name].AllocatedTflopsPercentToVirtualCap = 0
		}

		poolMetricsMap[poolObj.Name].AssignedLimitedTFlops = virtualTFlops -
			poolObj.Status.VirtualAvailableTFlops.AsApproximateFloat64()
		poolMetricsMap[poolObj.Name].AssignedLimitedVramBytes = virtualVRAM -
			poolObj.Status.VirtualAvailableVRAM.AsApproximateFloat64()

		if virtualTFlops > 0 {
			poolMetricsMap[poolObj.Name].AssignedLimitedTFlopsPercentToVirtualCap = poolMetricsMap[poolObj.Name].AssignedLimitedTFlops / virtualTFlops * 100
		} else {
			poolMetricsMap[poolObj.Name].AssignedLimitedTFlopsPercentToVirtualCap = 0
		}

		if virtualVRAM > 0 {
			poolMetricsMap[poolObj.Name].AssignedLimitedVramPercentToVirtualCap = poolMetricsMap[poolObj.Name].AssignedLimitedVramBytes / virtualVRAM * 100
		} else {
			poolMetricsMap[poolObj.Name].AssignedLimitedVramPercentToVirtualCap = 0
		}
	}
	poolMetricsMap[poolObj.Name].GPUCount = int(poolObj.Status.TotalGPUs)
}

func RemovePoolMetrics(poolName string) {
	poolMetricsLock.Lock()
	defer poolMetricsLock.Unlock()
	delete(poolMetricsMap, poolName)
}

func RemoveGPUAllocationMetrics(podName string) {
	gpuAllocationMetricsLock.Lock()
	defer gpuAllocationMetricsLock.Unlock()
	delete(gpuAllocationMetricsMap, podName)
}

func HasGPUAllocationMetrics(podName string) bool {
	gpuAllocationMetricsLock.RLock()
	defer gpuAllocationMetricsLock.RUnlock()
	_, exists := gpuAllocationMetricsMap[podName]
	return exists
}

// SetGPUAllocationMetrics sets aggregated GPU allocation metrics for a pod
func SetGPUAllocationMetrics(gpuStore map[types.NamespacedName]*tfv1.GPU, pod *corev1.Pod) {
	gpuAllocationMetricsLock.Lock()
	defer gpuAllocationMetricsLock.Unlock()
	// Get GPU device IDs from pod annotations
	gpuDeviceIDs, exists := pod.Annotations[constants.GPUDeviceIDsAnnotation]
	if !exists || gpuDeviceIDs == "" {
		return
	}

	gpuIDList := strings.Split(gpuDeviceIDs, ",")

	// Collect information from all GPUs
	var allGPUModelIndices []string

	for _, gpuID := range gpuIDList {
		gpuID = strings.TrimSpace(gpuID)
		if gpuID == "" {
			continue
		}
		key := types.NamespacedName{Name: gpuID}
		gpu, exists := gpuStore[key]
		if !exists || gpu == nil {
			log.V(5).Info("GPU not found in store for GPU allocation metrics", "gpuID", gpuID, "pod", pod.Name)
			continue
		}

		// Get GPU model
		var gpuModel string
		if gpu.Status.GPUModel != "" {
			gpuModel = gpu.Status.GPUModel
		}

		// Get GPU index
		var gpuIndex int32
		if gpu.Status.Index != nil {
			gpuIndex = *gpu.Status.Index
		}

		// Build GPU model with index: "Model:Index"
		var gpuModelIndex string
		if gpuModel != "" {
			if gpuIndex >= 0 {
				gpuModelIndex = gpuModel + ":" + strconv.Itoa(int(gpuIndex))
			} else {
				gpuModelIndex = gpuModel
			}
		} else if gpuIndex >= 0 {
			gpuModelIndex = ":" + strconv.Itoa(int(gpuIndex))
		}

		// Collect model:index
		if gpuModelIndex != "" {
			allGPUModelIndices = append(allGPUModelIndices, gpuModelIndex)
		}
	}

	// Create aggregated metric (one record per pod)
	gpuCount := len(allGPUModelIndices)
	if gpuCount == 0 {
		// If no valid GPU model indices, use the count of GPU IDs from annotation
		gpuCount = len(gpuIDList)
	}

	gpuAllocationMetricsMap[pod.Name] = &GPUAllocationMetrics{
		PodName:       pod.Name,
		Namespace:     pod.Namespace,
		PodUUID:       string(pod.UID),
		NodeName:      pod.Spec.NodeName,
		GPUModelIndex: strings.Join(allGPUModelIndices, ","),
		GPUName:       gpuDeviceIDs,
		GPUCount:      gpuCount,
		Timestamp:     time.Now(),
		podLabels:     pod.Labels,
	}
}

func SetSchedulerMetrics(poolName string, isSuccess bool) {
	tensorFusionSystemMetricsLock.Lock()
	defer tensorFusionSystemMetricsLock.Unlock()
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

// SetTopologyMetrics records GPU topology scheduling metrics.
func SetTopologyMetrics(poolName string, satisfied bool, isFallback bool, isDegraded bool) {
	tensorFusionSystemMetricsLock.Lock()
	defer tensorFusionSystemMetricsLock.Unlock()
	if _, ok := TensorFusionSystemMetricsMap[poolName]; !ok {
		TensorFusionSystemMetricsMap[poolName] = &TensorFusionSystemMetrics{
			PoolName: poolName,
		}
	}
	m := TensorFusionSystemMetricsMap[poolName]
	if satisfied {
		m.TotalTopoSatisfiedCount++
	} else {
		m.TotalTopoUnsatisfiedCount++
	}
	if isFallback {
		m.TotalTopoFallbackCount++
	}
	if isDegraded {
		m.TotalTopoSearchDegradedCount++
	}
}

// SetAutoscalingMetrics counts one node scaling decision of a pool: a node
// requested (scale up: a GPUNodeClaim or Karpenter NodeClaim was created) or a
// node selected for removal by compaction (scale down). Like the allocation
// counters, the totals count from the start of this operator process.
//
// They are resettable counters: kept in memory only, never seeded from the
// time-series database, and back at zero after an operator restart or a leader
// change. See TensorFusionSystemMetrics for what that asks of consumers.
func SetAutoscalingMetrics(poolName string, isScaleUp bool) {
	if poolName == "" {
		return
	}
	tensorFusionSystemMetricsLock.Lock()
	defer tensorFusionSystemMetricsLock.Unlock()
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
	tensorFusionSystemMetricsLock.RLock()
	defer tensorFusionSystemMetricsLock.RUnlock()
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
			time.Sleep(1 * time.Minute)
			workerMetricsLock.Lock()
			var keysToDelete []string
			for key, metrics := range workerMetricsMap {
				if metrics.deletionTimestamp != nil && !metrics.deletionTimestamp.IsZero() {
					keysToDelete = append(keysToDelete, key)
				}
			}
			for _, key := range keysToDelete {
				delete(workerMetricsMap, key)
			}
			workerMetricsLock.Unlock()
		}
	}()
}

func (mr *MetricsRecorder) RecordMetrics(writer io.Writer) {
	workerMetricsLock.RLock()
	workerMetricsCount := len(workerMetricsMap)
	workerMetricsLock.RUnlock()
	nodeMetricsLock.RLock()
	nodeMetricsCount := len(nodeMetricsMap)
	nodeMetricsLock.RUnlock()
	gpuResourceMetricsLock.RLock()
	gpuResourceMetricsCount := len(gpuResourceMetricsMap)
	gpuResourceMetricsLock.RUnlock()
	poolMetricsLock.RLock()
	poolNames := make([]string, 0, len(poolMetricsMap))
	for poolName := range poolMetricsMap {
		poolNames = append(poolNames, poolName)
	}
	poolMetricsLock.RUnlock()
	if workerMetricsCount <= 0 && nodeMetricsCount <= 0 && gpuResourceMetricsCount <= 0 && len(poolNames) == 0 {
		return
	}

	now := time.Now()
	enc := NewEncoder(config.GetGlobalConfig().MetricsFormat)
	// The currency of the cost figures is a label the administrator may set; it
	// is left out of the cost rows when it is not.
	currency := config.GetCurrency()
	workerMetricsLock.Lock()

	activeWorkerCnt := 0
	// Every known pool gets a tf_system_metrics row, also one without nodes or
	// workers: a pool that scaled to zero is a point on its node count chart,
	// not a gap in it.
	activeWorkerAndNodeByPool := make(map[string]*ActiveNodeAndWorker, len(poolNames))
	for _, poolName := range poolNames {
		activeWorkerAndNodeByPool[poolName] = &ActiveNodeAndWorker{}
	}

	for _, metrics := range workerMetricsMap {
		interval := now.Sub(metrics.LastRecordTime)
		if metrics.deletionTimestamp != nil && !metrics.deletionTimestamp.IsZero() {
			interval = metrics.deletionTimestamp.Sub(metrics.LastRecordTime)
		}
		metrics.RawCost = mr.getWorkerRawCost(metrics, interval)
		metrics.IntervalSeconds = math.Max(interval.Seconds(), 0)
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
		if metrics.deletionTimestamp == nil || metrics.deletionTimestamp.IsZero() {
			activeWorkerAndNodeByPool[metrics.PoolName].workerCnt++
		}

		enc.StartLine("tf_worker_resources")
		enc.AddTag("namespace", metrics.Namespace)
		addTagIfSet(enc, "node", metrics.NodeName)
		enc.AddTag("pool", metrics.PoolName)

		if metrics.QoS == "" {
			metrics.QoS = constants.QoSLevelMedium
		}
		enc.AddTag("qos", metrics.QoS)
		enc.AddTag("worker", metrics.WorkerName)
		enc.AddTag("workload", metrics.WorkloadName)
		metrics.Currency = currency
		addTagIfSet(enc, "currency", metrics.Currency)

		if config.GetGlobalConfig().MetricsExtraPodLabels != nil {
			for k, v := range config.GetGlobalConfig().MetricsExtraPodLabels {
				enc.AddTag(v, metrics.podLabels[k])
			}
		}

		enc.AddField("gpu_count", int64(metrics.GPUCount))
		enc.AddField("tflops_limit", metrics.TflopsLimit)
		enc.AddField("tflops_request", metrics.TflopsRequest)
		enc.AddField("raw_cost", metrics.RawCost)
		enc.AddField("interval_seconds", metrics.IntervalSeconds)
		enc.AddField("vram_bytes_limit", metrics.VramBytesLimit)
		enc.AddField("vram_bytes_request", metrics.VramBytesRequest)
		enc.AddField("ready", metrics.Ready)

		enc.EndLine(now)
	}
	workerMetricsLock.Unlock()

	nodeMetricsLock.Lock()

	gpuHourlyUnitPrices := mr.currentPricing()
	for _, metrics := range nodeMetricsMap {
		interval := now.Sub(metrics.LastRecordTime)
		metrics.HourlyPrice, metrics.PriceSource = mr.nodeHourlyPrice(metrics, gpuHourlyUnitPrices)
		metrics.Currency = nodeRateCurrency(metrics.PriceSource, currency)
		metrics.RawCost = nodeRawCost(metrics.HourlyPrice, interval)
		metrics.IntervalSeconds = interval.Seconds()
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
		enc.AddTag("phase", metrics.Phase)
		addTagIfSet(enc, "capacity_type", metrics.CapacityType)
		addTagIfSet(enc, "instance_type", metrics.InstanceType)
		addTagIfSet(enc, "price_source", metrics.PriceSource)
		addTagIfSet(enc, "currency", metrics.Currency)

		enc.AddField("allocated_tflops", metrics.AllocatedTflops)
		enc.AddField("allocated_tflops_percent", metrics.AllocatedTflopsPercent)
		enc.AddField("allocated_tflops_percent_virtual", metrics.AllocatedTflopsPercentToVirtualCap)
		enc.AddField("allocated_vram_bytes", metrics.AllocatedVramBytes)
		enc.AddField("allocated_vram_percent", metrics.AllocatedVramPercent)
		enc.AddField("allocated_vram_percent_virtual", metrics.AllocatedVramPercentToVirtualCap)
		if metrics.limitedKnown {
			enc.AddField("limited_tflops", metrics.LimitedTFlops)
			enc.AddField("limited_vram_bytes", metrics.LimitedVramBytes)
			enc.AddField("limited_tflops_percent_virtual", metrics.LimitedTFlopsPercentToVirtualCap)
			enc.AddField("limited_vram_percent_virtual", metrics.LimitedVramPercentToVirtualCap)
		}
		enc.AddField("gpu_count", int64(metrics.GPUCount))
		enc.AddField("raw_cost", metrics.RawCost)
		enc.AddField("hourly_price", metrics.HourlyPrice)
		enc.AddField("interval_seconds", metrics.IntervalSeconds)
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
		// Resettable counters: they count since this process started, so a
		// reader sees them drop to zero after an operator restart or a leader
		// change and has to treat that as a reset (rate/increase style).
		enc.AddField("total_scale_up_cnt", scaleUpCount)
		enc.AddField("total_scale_down_cnt", scaleDownCount)
		tensorFusionSystemMetricsLock.RLock()
		if item, ok := TensorFusionSystemMetricsMap[poolName]; ok {
			enc.AddField("total_topo_satisfied_cnt", item.TotalTopoSatisfiedCount)
			enc.AddField("total_topo_unsatisfied_cnt", item.TotalTopoUnsatisfiedCount)
			enc.AddField("total_topo_fallback_cnt", item.TotalTopoFallbackCount)
			enc.AddField("total_topo_search_degraded_cnt", item.TotalTopoSearchDegradedCount)
		}
		tensorFusionSystemMetricsLock.RUnlock()
		enc.EndLine(now)
	}
	nodeMetricsLock.Unlock()

	// Additional Pool level metrics
	poolMetricsLock.RLock()
	for _, metrics := range poolMetricsMap {
		enc.StartLine("tf_pool_metrics")
		enc.AddTag("pool", metrics.PoolName)
		enc.AddTag("phase", metrics.Phase)
		enc.AddField("allocated_tflops", metrics.AllocatedTflops)
		enc.AddField("allocated_tflops_percent", metrics.AllocatedTflopsPercent)
		enc.AddField("allocated_tflops_percent_virtual", metrics.AllocatedTflopsPercentToVirtualCap)
		enc.AddField("allocated_vram_bytes", metrics.AllocatedVramBytes)
		enc.AddField("allocated_vram_percent", metrics.AllocatedVramPercent)
		enc.AddField("allocated_vram_percent_virtual", metrics.AllocatedVramPercentToVirtualCap)
		enc.AddField("limited_tflops", metrics.AssignedLimitedTFlops)
		enc.AddField("limited_vram_bytes", metrics.AssignedLimitedVramBytes)
		enc.AddField("limited_tflops_percent_virtual", metrics.AssignedLimitedTFlopsPercentToVirtualCap)
		enc.AddField("limited_vram_percent_virtual", metrics.AssignedLimitedVramPercentToVirtualCap)
		enc.AddField("gpu_count", int64(metrics.GPUCount))
		enc.EndLine(now)
	}
	poolMetricsLock.RUnlock()

	// GPU allocation metrics
	gpuAllocationMetricsLock.RLock()
	for _, metrics := range gpuAllocationMetricsMap {
		enc.StartLine("tf_gpu_allocation")
		enc.AddTag("pod", metrics.PodName)
		enc.AddTag("namespace", metrics.Namespace)
		enc.AddTag("pod_uuid", metrics.PodUUID)
		enc.AddTag("node", metrics.NodeName)
		enc.AddTag("gpu_model_index", metrics.GPUModelIndex)
		enc.AddTag("gpu_name", metrics.GPUName)
		enc.AddField("gpu_count", int64(metrics.GPUCount))

		if config.GetGlobalConfig().MetricsExtraPodLabels != nil {
			for k, v := range config.GetGlobalConfig().MetricsExtraPodLabels {
				enc.AddTag(v, metrics.podLabels[k])
			}
		}
		enc.EndLine(now)
	}
	gpuAllocationMetricsLock.RUnlock()

	// Per-GPU card resource metrics
	gpuResourceMetricsLock.RLock()
	for _, metrics := range gpuResourceMetricsMap {
		enc.StartLine("tf_gpu_metrics")
		enc.AddTag("gpu", metrics.GPUName)
		enc.AddTag("node", metrics.NodeName)
		enc.AddTag("pool", metrics.PoolName)
		enc.AddTag("gpu_model", metrics.GPUModel)
		enc.AddTag("phase", metrics.Phase)
		enc.AddField("capacity_tflops", metrics.CapacityTflops)
		enc.AddField("available_tflops", metrics.AvailableTflops)
		enc.AddField("allocated_tflops", metrics.AllocatedTflops)
		enc.AddField("allocated_tflops_percent", metrics.AllocatedTflopsPercent)
		enc.AddField("capacity_vram_bytes", metrics.CapacityVramBytes)
		enc.AddField("available_vram_bytes", metrics.AvailableVramBytes)
		enc.AddField("allocated_vram_bytes", metrics.AllocatedVramBytes)
		enc.AddField("allocated_vram_percent", metrics.AllocatedVramPercent)
		enc.EndLine(now)
	}
	gpuResourceMetricsLock.RUnlock()

	if err := enc.Err(); err != nil {
		log.Error(err, "metrics encoding error", "workerCount", activeWorkerCnt, "nodeCount", nodeMetricsCount)
	}

	if _, err := writer.Write(enc.Bytes()); err != nil {
		log.Error(err, "metrics writing error", "workerCount", activeWorkerCnt, "nodeCount", nodeMetricsCount)
	}
	log.Info("metrics and raw billing recorded:", "workerCount", activeWorkerCnt, "nodeCount", nodeMetricsCount)
}

// Update metrics recorder's raw billing map
func (r *MetricsRecorder) UpdateMetricsRecorder(pool *tfv1.GPUPool, specChanged bool) {
	const dollarSign = "$"
	settingLock.Lock()
	defer settingLock.Unlock()
	if pool.Spec.QosConfig == nil {
		log.Info("QosConfig is nil, skip updating metrics recorder", "pool", pool.Name)
		return
	}

	qosConfig := pool.Spec.QosConfig
	if _, ok := r.WorkerUnitPriceMap[pool.Name]; !ok {
		r.WorkerUnitPriceMap[pool.Name] = make(map[string]RawBillingPricing)
	}

	if r.PoolEvictionProtectionPriceRatioMap == nil {
		r.PoolEvictionProtectionPriceRatioMap = make(map[string]string, 4)
	}
	r.PoolEvictionProtectionPriceRatioMap[pool.Name] = qosConfig.EvictionProtectionPriceRatio

	pricingDetail := r.WorkerUnitPriceMap[pool.Name]
	if !specChanged && len(pricingDetail) == 0 {
		return
	}
	// Pricing potentially changed
	for _, pricing := range qosConfig.Pricing {
		tflopsPerHour, _ := strconv.ParseFloat(strings.TrimPrefix(pricing.Requests.PerFP16TFlopsPerHour, dollarSign), 64)
		vramPerHour, _ := strconv.ParseFloat(strings.TrimPrefix(pricing.Requests.PerGBOfVRAMPerHour, dollarSign), 64)
		limitOverRequestChargingRatio, _ := strconv.ParseFloat(pricing.LimitsOverRequestsChargingRatio, 64)

		pricingDetail[string(pricing.Qos)] = RawBillingPricing{
			TflopsPerSecond: tflopsPerHour / float64(3600),
			VramPerSecond:   vramPerHour / float64(3600),

			TflopsOverRequestPerSecond: tflopsPerHour / float64(3600) * limitOverRequestChargingRatio,
			VramOverRequestPerSecond:   vramPerHour / float64(3600) * limitOverRequestChargingRatio,
		}
	}

	log.V(5).Info("Updated metrics recorder", "pool", pool.Name, "pricing", pricingDetail)
}

func (mr *MetricsRecorder) getWorkerRawCost(metrics *WorkerResourceMetrics, duration time.Duration) float64 {
	settingLock.RLock()
	defer settingLock.RUnlock()
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

	baseCost := (rawCostPerTflops + rawCostPerVRAM + rawCostTflopsLimitOverRequest + rawCostVRAMLimitOverRequest) * duration.Seconds() * float64(metrics.GPUCount)

	// Apply eviction protection price ratio if the pod is under protection and QoS is not critical
	if metrics.evictionProtection != "" && qosLevel != constants.QoSLevelCritical {
		if isUnderProtection := mr.isUnderEvictionProtection(metrics); isUnderProtection {
			protectionPriceRatio := mr.PoolEvictionProtectionPriceRatioMap[metrics.PoolName]
			protectionPriceRatioFloat, _ := strconv.ParseFloat(protectionPriceRatio, 64)
			if protectionPriceRatioFloat < 1 {
				protectionPriceRatioFloat = constants.DefaultEvictionProtectionPriceRatio
			}
			baseCost *= protectionPriceRatioFloat
		}
	}

	return baseCost
}

// isUnderEvictionProtection checks if a worker is under eviction protection
func (mr *MetricsRecorder) isUnderEvictionProtection(metrics *WorkerResourceMetrics) bool {
	if metrics.evictionProtection == "" {
		return false
	}

	// Parse protection duration (1h, 5h, 24h, etc.)
	duration, err := time.ParseDuration(metrics.evictionProtection)
	if err != nil {
		return false
	}

	protectionEndTime := metrics.creationTime.Add(duration)
	return time.Now().Before(protectionEndTime)
}

// nodeHourlyPrice returns the hourly rate a node is charged at and where that
// rate comes from. They are the rate behind raw_cost, the hourly_price field
// and the price_source tag of tf_node_metrics. The source is empty when the
// node has no price.
func (mr *MetricsRecorder) nodeHourlyPrice(metrics *NodeResourceMetrics, hourlyUnitPriceMap map[string]float64) (float64, string) {
	// PRICING HOOK (capacity type). This is the one place the recorder asks what
	// a node costs per hour. With NodeHourlyPrice set, a node whose instance type
	// and capacity type are known is charged the price of that capacity type, so
	// a spot node is not billed at the on-demand rate. Without it, or when it has
	// no price for the node, the rate is the sum of the per-GPU unit prices below.
	//
	// The hook also says whether its figure is a listed price or an estimate
	// (the spot rate of a provider whose pricing data lists no spot prices is
	// one); the row carries that, so an assumed rate is not read as a fact.
	if mr.NodeHourlyPrice != nil && metrics.InstanceType != "" && metrics.CapacityType != "" {
		if price, source, ok := mr.NodeHourlyPrice(metrics.InstanceType, metrics.CapacityType, metrics.region); ok {
			if price <= 0 {
				return price, ""
			}
			return price, knownPriceSource(source)
		}
	}

	// unit price data comes from global config map, and multi-GPU instance should normalized with per GPU pricing, e.g. 8xA100 p4d.24xlarge price should divide by 8
	price := 0.0
	for _, gpuModel := range metrics.gpuModels {
		price += hourlyUnitPriceMap[gpuModel]
	}
	if price <= 0 {
		return price, ""
	}
	return price, PriceSourceConfigured
}

func nodeRawCost(hourlyPrice float64, duration time.Duration) float64 {
	return duration.Seconds() * hourlyPrice / 3600.0
}

// addTagIfSet leaves a tag without a value out of the line: a line with an
// empty tag value is not valid line protocol, and the column reads NULL.
// nodeRateCurrency is the currency of a node's hourly rate: a rate taken from
// the built-in price lists is in the currency of those lists, whatever the
// administrator's own prices are in; any other rate is in the configured one.
func nodeRateCurrency(priceSource, configured string) string {
	if priceSource == PriceSourceList || priceSource == PriceSourceSpotEstimate {
		return ListPriceCurrency
	}
	return configured
}

func addTagIfSet(enc Encoder, key, value string) {
	if value != "" {
		enc.AddTag(key, value)
	}
}
