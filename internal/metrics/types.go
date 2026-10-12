// NOTE: Make sure any new field/tag to existing metrics or new metrics
// should be added to SetupTable function for manual DB migration
package metrics

import (
	"time"
)

// TensorFusionSystemMetrics is one tf_system_metrics row of a pool.
//
// TotalScaleUpCount and TotalScaleDownCount (total_scale_up_cnt and
// total_scale_down_cnt) are resettable counters. They count the node scaling
// decisions made since the operator process that writes them started, and are
// kept in memory only: they start again from zero when the operator restarts
// or leadership moves to another replica. They are not seeded from the
// time-series database. A consumer must not read them as all-time totals; it
// has to handle resets the way rate/increase functions do: a value lower than
// the previous one means the counter restarted, and that value is the
// increase since the restart.
type TensorFusionSystemMetrics struct {
	PoolName string `json:"poolName" gorm:"column:pool;index:,class:INVERTED"`

	TotalWorkerCount             int64 `json:"totalWorkerCount" gorm:"column:total_workers_cnt"`
	TotalNodeCount               int64 `json:"totalNodeCount" gorm:"column:total_nodes_cnt"`
	TotalAllocationFailCount     int64 `json:"totalAllocationFailCount" gorm:"column:total_allocation_fail_cnt"`
	TotalAllocationSuccessCount  int64 `json:"totalAllocationSuccessCount" gorm:"column:total_allocation_success_cnt"`
	TotalScaleUpCount            int64 `json:"totalScaleUpCount" gorm:"column:total_scale_up_cnt"`
	TotalScaleDownCount          int64 `json:"totalScaleDownCount" gorm:"column:total_scale_down_cnt"`
	TotalTopoSatisfiedCount      int64 `json:"totalTopoSatisfiedCount" gorm:"column:total_topo_satisfied_cnt"`
	TotalTopoUnsatisfiedCount    int64 `json:"totalTopoUnsatisfiedCount" gorm:"column:total_topo_unsatisfied_cnt"`
	TotalTopoFallbackCount       int64 `json:"totalTopoFallbackCount" gorm:"column:total_topo_fallback_cnt"`
	TotalTopoSearchDegradedCount int64 `json:"totalTopoSearchDegradedCount" gorm:"column:total_topo_search_degraded_cnt"`

	// NOTE: make sure new fields will be migrated in SetupTable function

	Timestamp time.Time `json:"ts" gorm:"column:greptime_timestamp;index:,class:TIME;precision:ns"`
}

func (wm TensorFusionSystemMetrics) TableName() string {
	return "tf_system_metrics"
}

var TensorFusionSystemMetricsMap = make(map[string]*TensorFusionSystemMetrics)

// Metrics will be stored in a map, key is the worker name, value is the metrics
// By default, metrics will be updated every minute
type WorkerResourceMetrics struct {
	WorkerName   string `json:"workerName" gorm:"column:worker;index:,class:SKIPPING"`
	WorkloadName string `json:"workloadName" gorm:"column:workload;index:,class:INVERTED"`
	PoolName     string `json:"poolName" gorm:"column:pool;index:,class:INVERTED"`
	Namespace    string `json:"namespace" gorm:"column:namespace;index:,class:INVERTED"`
	// NodeName is the node the worker runs on, so that the cost of a node can
	// be apportioned to the workers on it.
	NodeName string `json:"nodeName" gorm:"column:node;index:,class:INVERTED"`
	// Currency labels the currency RawCost is in, as configured cluster-wide
	// (GlobalConfig.Currency). It is empty, and the tag left out, unless an
	// administrator configured it.
	Currency string `json:"currency" gorm:"column:currency;index:,class:INVERTED"`
	QoS      string `json:"qos" gorm:"column:qos"`

	TflopsRequest    float64 `json:"tflopsRequest" gorm:"column:tflops_request"`
	TflopsLimit      float64 `json:"tflopsLimit" gorm:"column:tflops_limit"`
	VramBytesRequest float64 `json:"vramBytesRequest" gorm:"column:vram_bytes_request"`
	VramBytesLimit   float64 `json:"vramBytesLimit" gorm:"column:vram_bytes_limit"`
	GPUCount         int     `json:"gpuCount" gorm:"column:gpu_count"`
	RawCost          float64 `json:"rawCost" gorm:"column:raw_cost"`
	// IntervalSeconds is the duration RawCost of this row covers.
	IntervalSeconds float64 `json:"intervalSeconds" gorm:"column:interval_seconds"`
	Ready           bool    `json:"ready" gorm:"column:ready"`

	// NOTE: make sure new fields will be migrated in SetupTable function

	LastRecordTime time.Time `json:"lastRecordTime" gorm:"column:greptime_timestamp;index:,class:TIME;precision:ns"`

	// For more accurate metrics, should record the deletion timestamp to calculate duration for the last metrics
	deletionTimestamp *time.Time

	// Fields for eviction protection tracking - private, not stored in TSDB
	creationTime       time.Time
	evictionProtection string

	podLabels map[string]string
}

func (wm WorkerResourceMetrics) TableName() string {
	return "tf_worker_resources"
}

type NodeResourceMetrics struct {
	NodeName string `json:"nodeName" gorm:"column:node;index:,class:INVERTED"`
	PoolName string `json:"poolName" gorm:"column:pool;index:,class:INVERTED"`
	Phase    string `json:"phase" gorm:"column:phase;index:,class:INVERTED"`
	// CapacityType is how the node is bought: on-demand, spot or reserved. It is
	// empty when the node does not say.
	CapacityType string `json:"capacityType" gorm:"column:capacity_type;index:,class:INVERTED"`
	InstanceType string `json:"instanceType" gorm:"column:instance_type;index:,class:INVERTED"`
	// PriceSource says where HourlyPrice comes from: PriceSourceList,
	// PriceSourceSpotEstimate or PriceSourceConfigured. It is empty, and the tag
	// left out, when the node has no price at all.
	PriceSource string `json:"priceSource" gorm:"column:price_source;index:,class:INVERTED"`
	// Currency labels the currency RawCost and HourlyPrice are in, as configured
	// cluster-wide (GlobalConfig.Currency). It is empty, and the tag left out,
	// unless an administrator configured it.
	Currency string `json:"currency" gorm:"column:currency;index:,class:INVERTED"`

	AllocatedTflops        float64 `json:"allocatedTflops" gorm:"column:allocated_tflops"`
	AllocatedTflopsPercent float64 `json:"allocatedTflopsPercent" gorm:"column:allocated_tflops_percent"`
	AllocatedVramBytes     float64 `json:"allocatedVramBytes" gorm:"column:allocated_vram_bytes"`
	AllocatedVramPercent   float64 `json:"allocatedVramPercent" gorm:"column:allocated_vram_percent"`

	AllocatedTflopsPercentToVirtualCap float64 `json:"allocatedTflopsPercentToVirtualCap" gorm:"column:allocated_tflops_percent_virtual"`
	AllocatedVramPercentToVirtualCap   float64 `json:"allocatedVramPercentToVirtualCap" gorm:"column:allocated_vram_percent_virtual"`

	LimitedTFlops                    float64 `json:"limitedTFlops" gorm:"column:limited_tflops"`
	LimitedVramBytes                 float64 `json:"limitedVramBytes" gorm:"column:limited_vram_bytes"`
	LimitedTFlopsPercentToVirtualCap float64 `json:"limitedTFlopsPercentToVirtualCap" gorm:"column:limited_tflops_percent_virtual"`
	LimitedVramPercentToVirtualCap   float64 `json:"limitedVramPercentToVirtualCap" gorm:"column:limited_vram_percent_virtual"`

	RawCost float64 `json:"rawCost" gorm:"column:raw_cost"`
	// HourlyPrice is the hourly rate RawCost of this row was charged at, and
	// IntervalSeconds the duration it covers.
	HourlyPrice     float64 `json:"hourlyPrice" gorm:"column:hourly_price"`
	IntervalSeconds float64 `json:"intervalSeconds" gorm:"column:interval_seconds"`
	GPUCount        int     `json:"gpuCount" gorm:"column:gpu_count"`

	// NOTE: make sure new fields will be migrated in SetupTable function

	LastRecordTime time.Time `json:"lastRecordTime" gorm:"column:greptime_timestamp;index:,class:TIME;precision:ns"`

	// additional field for raw cost calculation since each GPU has different price
	// private field automatically ignored in gorm
	gpuModels []string
	// limitedKnown says whether the limited_* figures could be derived; they
	// are only written when they could.
	limitedKnown bool
	// region is where the node runs, for looking up its price.
	region string
}

func (nm NodeResourceMetrics) TableName() string {
	return "tf_node_metrics"
}

func (nm *NodeResourceMetrics) SetGPUModelAndCount(gpuModels []string) {
	if gpuModels == nil {
		return
	}
	nm.gpuModels = gpuModels
	nm.GPUCount = len(gpuModels)
}

type PoolResourceMetrics struct {
	PoolName string `json:"poolName" gorm:"column:pool;index:,class:INVERTED"`

	Phase string `json:"phase" gorm:"column:phase;index:,class:INVERTED"`

	AllocatedTflops                    float64 `json:"allocatedTflops" gorm:"column:allocated_tflops"`
	AllocatedTflopsPercent             float64 `json:"allocatedTflopsPercent" gorm:"column:allocated_tflops_percent"`
	AllocatedTflopsPercentToVirtualCap float64 `json:"allocatedTflopsPercentToVirtualCap" gorm:"column:allocated_tflops_percent_virtual"`
	AllocatedVramBytes                 float64 `json:"allocatedVramBytes" gorm:"column:allocated_vram_bytes"`
	AllocatedVramPercent               float64 `json:"allocatedVramPercent" gorm:"column:allocated_vram_percent"`
	AllocatedVramPercentToVirtualCap   float64 `json:"allocatedVramPercentToVirtualCap" gorm:"column:allocated_vram_percent_virtual"`

	AssignedLimitedTFlops                    float64 `json:"limitedTFlops" gorm:"column:limited_tflops"`
	AssignedLimitedVramBytes                 float64 `json:"limitedVramBytes" gorm:"column:limited_vram_bytes"`
	AssignedLimitedTFlopsPercentToVirtualCap float64 `json:"limitedTFlopsPercentToVirtualCap" gorm:"column:limited_tflops_percent_virtual"`
	AssignedLimitedVramPercentToVirtualCap   float64 `json:"limitedVramPercentToVirtualCap" gorm:"column:limited_vram_percent_virtual"`

	GPUCount int `json:"gpuCount" gorm:"column:gpu_count"`

	// NOTE: make sure new fields will be migrated in SetupTable function

	LastRecordTime time.Time `json:"lastRecordTime" gorm:"column:greptime_timestamp;index:,class:TIME;precision:ns"`
}

func (pm PoolResourceMetrics) TableName() string {
	return "tf_pool_metrics"
}

type RawBillingPricing struct {
	TflopsPerSecond float64
	VramPerSecond   float64

	TflopsOverRequestPerSecond float64
	VramOverRequestPerSecond   float64
}

// Other tables are used to store metrics from hypervisor or system logs

type TFSystemLog struct {
	Component string `json:"component" gorm:"column:component;index:,class:INVERTED"`
	Container string `json:"container" gorm:"column:container;index:,class:INVERTED"`
	Message   string `json:"message" gorm:"column:message;index:,class:FULLTEXT,option:WITH (analyzer = 'English' $comma$ case_sensitive = 'false')"`
	Namespace string `json:"namespace" gorm:"column:namespace;index:,class:INVERTED"`
	Pod       string `json:"pod" gorm:"column:pod;index:,class:SKIPPING"`
	Stream    string `json:"stream" gorm:"column:stream"`
	// message written timestamp
	Timestamp string `json:"timestamp" gorm:"column:timestamp"`

	// NOTE: make sure new fields will be migrated in SetupTable function

	GreptimeTimestamp time.Time `json:"greptime_timestamp" gorm:"column:greptime_timestamp;index:,class:TIME;precision:ms"`
}

func (sl TFSystemLog) TableName() string {
	return "tf_system_log"
}

type HypervisorWorkerUsageMetrics struct {
	WorkloadName string `json:"workloadName" gorm:"column:workload;index:,class:INVERTED"`
	WorkerName   string `json:"workerName" gorm:"column:worker;index:,class:SKIPPING"`
	Namespace    string `json:"namespace" gorm:"column:namespace;index:,class:INVERTED"`
	PoolName     string `json:"poolName" gorm:"column:pool;index:,class:INVERTED"`
	NodeName     string `json:"nodeName" gorm:"column:node;index:,class:INVERTED"`
	UUID         string `json:"uuid" gorm:"column:uuid;index:,class:INVERTED"`

	ComputeTflops  float64 `json:"computeTflops" gorm:"column:compute_tflops"`
	ComputePercent float64 `json:"computePercent" gorm:"column:compute_percentage"`
	VRAMBytes      uint64  `json:"vramBytes" gorm:"column:memory_bytes"`
	VRAMPercent    float64 `json:"vramPercent" gorm:"column:memory_percentage"`

	ComputeThrottledCount int64 `json:"computeThrottledCount" gorm:"column:compute_throttled_cnt"`
	VRAMFreezedCount      int64 `json:"vramFreezedCount" gorm:"column:vram_freezed_cnt"`
	VRAMResumedCount      int64 `json:"vramResumedCount" gorm:"column:vram_resumed_cnt"`

	// NOTE: make sure new fields will be migrated in SetupTable function

	Timestamp time.Time `json:"ts" gorm:"column:greptime_timestamp;index:,class:TIME;precision:ns"`
}

func (wu HypervisorWorkerUsageMetrics) TableName() string {
	return "tf_worker_usage"
}

type HypervisorGPUUsageMetrics struct {
	NodeName string `json:"nodeName" gorm:"column:node;index:,class:INVERTED"`
	PoolName string `json:"poolName" gorm:"column:pool;index:,class:INVERTED"`
	UUID     string `json:"uuid" gorm:"column:uuid;index:,class:INVERTED"`

	ComputePercent float64 `json:"computePercent" gorm:"column:compute_percentage"`
	VRAMPercent    float64 `json:"vramPercent" gorm:"column:memory_percentage"`

	VRAMBytes     uint64  `json:"vramBytes" gorm:"column:memory_bytes"`
	ComputeTflops float64 `json:"computeTflops" gorm:"column:compute_tflops"`

	PcieRxKB    float64 `json:"pcieRx" gorm:"column:rx"`
	PcieTxKB    float64 `json:"pcieTx" gorm:"column:tx"`
	Temperature float64 `json:"temperature" gorm:"column:temperature"`

	GraphicsClockMHz float64 `json:"graphicsClockMHz" gorm:"column:graphics_clock_mhz"`
	SMClockMHz       float64 `json:"smClockMHz" gorm:"column:sm_clock_mhz"`
	MemoryClockMHz   float64 `json:"memoryClockMHz" gorm:"column:memory_clock_mhz"`
	VideoClockMHz    float64 `json:"videoClockMHz" gorm:"column:video_clock_mhz"`
	PowerUsage       float64 `json:"powerUsage" gorm:"column:power_usage"`
	NvlinkRx         float64 `json:"nvlinkRx" gorm:"column:nvlink_rx"`
	NvlinkTx         float64 `json:"nvlinkTx" gorm:"column:nvlink_tx"`
	// NOTE: make sure new fields will be migrated in SetupTable function

	Timestamp time.Time `json:"ts" gorm:"column:greptime_timestamp;index:,class:TIME;precision:ns"`
}

func (nu HypervisorGPUUsageMetrics) TableName() string {
	return "tf_gpu_usage"
}

// tf_limiter_events is the third table the hypervisor writes, next to
// tf_worker_usage and tf_gpu_usage: the counters of the events the in-process
// limiters of the workloads on a node report to its hypervisor (see
// RecordLimiterEventMetrics in pkg/hypervisor/metrics). It is described here
// and has no model or DDL of its own yet: the rows arrive as InfluxDB line
// protocol like the other two, and the table is created by the first of them.
//
//	tags    node       node of the hypervisor
//	        pool       GPU pool of the node
//	        namespace  namespace of the Pod that reported
//	        pod        name of the Pod that reported
//	        kind       grace_entered, grace_expired, unwrapped_entry,
//	                   clamped_charge, stalled_bucket, old_hypervisor or other
//	field   count      integer; occurrences of the kind reported by the Pod
//	time    greptime_timestamp
//
// count is a resettable counter like those of tf_system_metrics above: it is
// kept in the hypervisor's memory, counts since that process started and
// starts again from zero when it restarts, or when a Pod reports again after
// the hypervisor had forgotten it (no report for 15 minutes). A value lower
// than the previous one of the series means the counter restarted. A row is
// written only when its count changed, so a series has gaps while nothing is
// reported; the latest row holds the current count.

// GPUAllocationMetrics records GPU allocation information for a pod (aggregated from all GPUs)
type GPUAllocationMetrics struct {
	PodName       string `json:"podName" gorm:"column:pod;index:,class:SKIPPING"`
	Namespace     string `json:"namespace" gorm:"column:namespace;index:,class:INVERTED"`
	PodUUID       string `json:"podUUID" gorm:"column:pod_uuid;index:,class:INVERTED"`
	NodeName      string `json:"nodeName" gorm:"column:node"`
	GPUModelIndex string `json:"gpuModelIndex" gorm:"column:gpu_model_index"`
	GPUName       string `json:"gpuName" gorm:"column:gpu_name;index:,class:INVERTED"`
	GPUCount      int    `json:"gpuCount" gorm:"column:gpu_count"`

	// NOTE: make sure new fields will be migrated in SetupTable function

	Timestamp time.Time `json:"ts" gorm:"column:greptime_timestamp;index:,class:TIME;precision:ns"`

	// Fields for extra pod labels tracking - private, not stored in TSDB
	podLabels map[string]string
}

func (gam GPUAllocationMetrics) TableName() string {
	return "tf_gpu_allocation"
}

// GPUResourceMetrics records per-GPU card capacity and allocation status
type GPUResourceMetrics struct {
	GPUName  string `json:"gpuName" gorm:"column:gpu;index:,class:INVERTED"`
	NodeName string `json:"nodeName" gorm:"column:node;index:,class:INVERTED"`
	PoolName string `json:"poolName" gorm:"column:pool;index:,class:INVERTED"`
	GPUModel string `json:"gpuModel" gorm:"column:gpu_model;index:,class:INVERTED"`
	Phase    string `json:"phase" gorm:"column:phase;index:,class:INVERTED"`

	CapacityTflops         float64 `json:"capacityTflops" gorm:"column:capacity_tflops"`
	AvailableTflops        float64 `json:"availableTflops" gorm:"column:available_tflops"`
	AllocatedTflops        float64 `json:"allocatedTflops" gorm:"column:allocated_tflops"`
	AllocatedTflopsPercent float64 `json:"allocatedTflopsPercent" gorm:"column:allocated_tflops_percent"`

	CapacityVramBytes    float64 `json:"capacityVramBytes" gorm:"column:capacity_vram_bytes"`
	AvailableVramBytes   float64 `json:"availableVramBytes" gorm:"column:available_vram_bytes"`
	AllocatedVramBytes   float64 `json:"allocatedVramBytes" gorm:"column:allocated_vram_bytes"`
	AllocatedVramPercent float64 `json:"allocatedVramPercent" gorm:"column:allocated_vram_percent"`

	// NOTE: make sure new fields will be migrated in SetupTable function

	LastRecordTime time.Time `json:"lastRecordTime" gorm:"column:greptime_timestamp;index:,class:TIME;precision:ns"`
}

func (gm GPUResourceMetrics) TableName() string {
	return "tf_gpu_metrics"
}

// NOTE: make sure new metrics will be migrated in SetupTable function
