// NOTE: Make sure any new field/tag to existing metrics or new metrics
// should be added to SetupTable function for manual DB migration
package metrics

import (
	"time"
)

type TensorFusionSystemMetrics struct {
	PoolName string `json:"poolName" gorm:"column:pool;index:,class:INVERTED"`

	TotalWorkerCount            int64 `json:"totalWorkerCount" gorm:"column:total_workers_cnt"`
	TotalNodeCount              int64 `json:"totalNodeCount" gorm:"column:total_nodes_cnt"`
	TotalAllocationFailCount    int64 `json:"totalAllocationFailCount" gorm:"column:total_allocation_fail_cnt"`
	TotalAllocationSuccessCount int64 `json:"totalAllocationSuccessCount" gorm:"column:total_allocation_success_cnt"`
	TotalScaleUpCount           int64 `json:"totalScaleUpCount" gorm:"column:total_scale_up_cnt"`
	TotalScaleDownCount         int64 `json:"totalScaleDownCount" gorm:"column:total_scale_down_cnt"`

	// NOTE: make sure new fields will be migrated in SetupTable function

	Timestamp time.Time `json:"ts" gorm:"column:ts;index:,class:TIME"`
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
	QoS          string `json:"qos" gorm:"column:qos"`

	TflopsRequest    float64 `json:"tflopsRequest" gorm:"column:tflops_request"`
	TflopsLimit      float64 `json:"tflopsLimit" gorm:"column:tflops_limit"`
	VramBytesRequest float64 `json:"vramBytesRequest" gorm:"column:vram_bytes_request"`
	VramBytesLimit   float64 `json:"vramBytesLimit" gorm:"column:vram_bytes_limit"`
	GPUCount         int     `json:"gpuCount" gorm:"column:gpu_count"`
	RawCost          float64 `json:"rawCost" gorm:"column:raw_cost"`
	Ready            bool    `json:"ready" gorm:"column:ready"`

	// NOTE: make sure new fields will be migrated in SetupTable function

	LastRecordTime time.Time `json:"lastRecordTime" gorm:"column:ts;index:,class:TIME"`

	// For more accurate metrics, should record the deletion timestamp to calculate duration for the last metrics
	deletionTimestamp *time.Time

	podLabels map[string]string
}

func (wm WorkerResourceMetrics) TableName() string {
	return "tf_worker_resources"
}

type NodeResourceMetrics struct {
	NodeName string `json:"nodeName" gorm:"column:node;index:,class:INVERTED"`
	PoolName string `json:"poolName" gorm:"column:pool;index:,class:INVERTED"`
	Phase    string `json:"phase" gorm:"column:phase;index:,class:INVERTED"`

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

	RawCost  float64 `json:"rawCost" gorm:"column:raw_cost"`
	GPUCount int     `json:"gpuCount" gorm:"column:gpu_count"`

	// NOTE: make sure new fields will be migrated in SetupTable function

	LastRecordTime time.Time `json:"lastRecordTime" gorm:"column:ts;index:,class:TIME"`

	// additional field for raw cost calculation since each GPU has different price
	// private field automatically ignored in gorm
	gpuModels []string
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

	LastRecordTime time.Time `json:"lastRecordTime" gorm:"column:ts;index:,class:TIME"`
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

	Timestamp time.Time `json:"ts" gorm:"column:ts;index:,class:TIME"`
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

	Timestamp time.Time `json:"ts" gorm:"column:ts;index:,class:TIME"`
}

func (nu HypervisorGPUUsageMetrics) TableName() string {
	return "tf_gpu_usage"
}

// NOTE: make sure new metrics will be migrated in SetupTable function
