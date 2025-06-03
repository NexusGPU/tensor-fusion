package metrics

import "time"

// Metrics will be stored in a map, key is the worker name, value is the metrics
// By default, metrics will be updated every minute
type WorkerMetrics struct {
	WorkerName   string `json:"workerName"`
	WorkloadName string `json:"workloadName"`
	PoolName     string `json:"poolName"`
	Namespace    string `json:"namespace"`
	QoS          string `json:"qos"`

	TflopsRequest    float64 `json:"tflopsRequest"`
	TflopsLimit      float64 `json:"tflopsLimit"`
	VramBytesRequest float64 `json:"vramBytesRequest"`
	VramBytesLimit   float64 `json:"vramBytesLimit"`
	GPUCount         int     `json:"gpuCount"`
	RawCost          float64 `json:"rawCost"`

	LastRecordTime time.Time `json:"lastRecordTime"`

	// For more accurate metrics, should record the deletion timestamp to calculate duration for the last metrics
	DeletionTimestamp time.Time `json:"deletionTimestamp"`
}

type NodeMetrics struct {
	NodeName string `json:"nodeName"`
	PoolName string `json:"poolName"`

	AllocatedTflops        float64 `json:"allocatedTflops"`
	AllocatedTflopsPercent float64 `json:"allocatedTflopsPercent"`
	AllocatedVramBytes     float64 `json:"allocatedVramBytes"`
	AllocatedVramPercent   float64 `json:"allocatedVramPercent"`

	AllocatedTflopsPercentToVirtualCap float64 `json:"allocatedTflopsPercentToVirtualCap"`
	AllocatedVramPercentToVirtualCap   float64 `json:"allocatedVramPercentToVirtualCap"`

	RawCost float64 `json:"rawCost"`

	LastRecordTime time.Time `json:"lastRecordTime"`

	// additional field for raw cost calculation since each GPU has different price
	GPUModels []string `json:"gpuModels"`
}

type RawBillingPricing struct {
	TflopsPerSecond float64
	VramPerSecond   float64

	TflopsOverRequestPerSecond float64
	VramOverRequestPerSecond   float64
}
