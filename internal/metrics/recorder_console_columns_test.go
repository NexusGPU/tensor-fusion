package metrics

import (
	"bytes"
	"strconv"
	"strings"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// recordedTags returns the tags of the first line of a measurement whose tags
// contain the given fragment.
func recordedTags(t *testing.T, output, measurement, tagFragment string) map[string]string {
	t.Helper()
	for _, line := range strings.Split(output, "\n") {
		if !strings.HasPrefix(line, measurement+",") || !strings.Contains(line, tagFragment) {
			continue
		}
		head, _, _ := strings.Cut(line, " ")
		tags := map[string]string{}
		for _, tag := range strings.Split(head, ",")[1:] {
			key, value, _ := strings.Cut(tag, "=")
			tags[key] = value
		}
		return tags
	}
	t.Fatalf("no %s line with %q in:\n%s", measurement, tagFragment, output)
	return nil
}

func fieldFloat(t *testing.T, fields map[string]string, key string) float64 {
	t.Helper()
	require.Contains(t, fields, key)
	value, err := strconv.ParseFloat(fields[key], 64)
	require.NoError(t, err)
	return value
}

// isolateRecorderState gives a test empty recorder maps and puts the previous
// ones back afterwards.
func isolateRecorderState(t *testing.T) {
	t.Helper()
	workerMetricsLock.Lock()
	workers := workerMetricsMap
	workerMetricsMap = map[string]*WorkerResourceMetrics{}
	workerMetricsLock.Unlock()
	nodeMetricsLock.Lock()
	nodes := nodeMetricsMap
	nodeMetricsMap = map[string]*NodeResourceMetrics{}
	nodeMetricsLock.Unlock()
	poolMetricsLock.Lock()
	pools := poolMetricsMap
	poolMetricsMap = map[string]*PoolResourceMetrics{}
	poolMetricsLock.Unlock()
	gpuResourceMetricsLock.Lock()
	gpus := gpuResourceMetricsMap
	gpuResourceMetricsMap = map[string]*GPUResourceMetrics{}
	gpuResourceMetricsLock.Unlock()
	gpuAllocationMetricsLock.Lock()
	allocations := gpuAllocationMetricsMap
	gpuAllocationMetricsMap = map[string]*GPUAllocationMetrics{}
	gpuAllocationMetricsLock.Unlock()
	tensorFusionSystemMetricsLock.Lock()
	system := TensorFusionSystemMetricsMap
	TensorFusionSystemMetricsMap = map[string]*TensorFusionSystemMetrics{}
	tensorFusionSystemMetricsLock.Unlock()

	t.Cleanup(func() {
		workerMetricsLock.Lock()
		workerMetricsMap = workers
		workerMetricsLock.Unlock()
		nodeMetricsLock.Lock()
		nodeMetricsMap = nodes
		nodeMetricsLock.Unlock()
		poolMetricsLock.Lock()
		poolMetricsMap = pools
		poolMetricsLock.Unlock()
		gpuResourceMetricsLock.Lock()
		gpuResourceMetricsMap = gpus
		gpuResourceMetricsLock.Unlock()
		gpuAllocationMetricsLock.Lock()
		gpuAllocationMetricsMap = allocations
		gpuAllocationMetricsLock.Unlock()
		tensorFusionSystemMetricsLock.Lock()
		TensorFusionSystemMetricsMap = system
		tensorFusionSystemMetricsLock.Unlock()
	})
}

func backdateNodeRecord(nodeName string, by time.Duration) {
	nodeMetricsLock.Lock()
	defer nodeMetricsLock.Unlock()
	nodeMetricsMap[nodeName].LastRecordTime = time.Now().Add(-by)
}

func newTestRecorder() *MetricsRecorder {
	return &MetricsRecorder{WorkerUnitPriceMap: map[string]map[string]RawBillingPricing{}}
}

// A pool without any node, worker or GPU still gets its rows: a pool scaled to
// zero is a point of its node count chart, and its scaling counters stay
// readable. Before, nothing at all was written while every pool was empty.
func TestEmptyPoolStillGetsSystemAndPoolRows(t *testing.T) {
	isolateRecorderState(t)
	const poolName = "empty-pool"
	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: poolName},
		Status:     tfv1.GPUPoolStatus{Phase: tfv1.TensorFusionPoolPhaseRunning},
	}
	SetPoolMetrics(pool)

	var out bytes.Buffer
	newTestRecorder().RecordMetrics(&out)
	fields := recordedLine(t, out.String(), "tf_system_metrics", "pool="+poolName)
	require.Equal(t, "0i", fields["total_nodes_cnt"])
	require.Equal(t, "0i", fields["total_workers_cnt"])
	require.Equal(t, "0i", fields["total_scale_up_cnt"])
	require.Equal(t, "0i", fields["total_scale_down_cnt"])
	poolFields := recordedLine(t, out.String(), "tf_pool_metrics", "pool="+poolName)
	require.Equal(t, "0i", poolFields["gpu_count"])

	// A pool that is gone stops getting rows.
	RemovePoolMetrics(poolName)
	out.Reset()
	newTestRecorder().RecordMetrics(&out)
	require.Empty(t, out.String())
}

// The scaling counters are integers that only grow, per pool, and are still
// reported after the pool lost its last node.
func TestScalingCountersAreCumulativePerPool(t *testing.T) {
	isolateRecorderState(t)
	poolA := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "pool-a"}, Status: tfv1.GPUPoolStatus{Phase: tfv1.TensorFusionPoolPhaseRunning}}
	poolB := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "pool-b"}, Status: tfv1.GPUPoolStatus{Phase: tfv1.TensorFusionPoolPhaseRunning}}
	SetPoolMetrics(poolA)
	SetPoolMetrics(poolB)
	node := &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{Name: "pool-a-node"}}
	recorder := newTestRecorder()

	SetAutoscalingMetrics("pool-a", true)
	SetAutoscalingMetrics("pool-a", true)
	SetNodeMetrics(node, poolA, nil)
	var out bytes.Buffer
	recorder.RecordMetrics(&out)
	fields := recordedLine(t, out.String(), "tf_system_metrics", "pool=pool-a")
	require.Equal(t, "2i", fields["total_scale_up_cnt"])
	require.Equal(t, "0i", fields["total_scale_down_cnt"])
	require.Equal(t, "1i", fields["total_nodes_cnt"])
	fields = recordedLine(t, out.String(), "tf_system_metrics", "pool=pool-b")
	require.Equal(t, "0i", fields["total_scale_up_cnt"], "counted per pool")
	require.Equal(t, "0i", fields["total_nodes_cnt"])

	// The node is removed by a scale down; the totals keep what was counted.
	SetAutoscalingMetrics("pool-a", false)
	RemoveNodeMetrics(node.Name)
	SetAutoscalingMetrics("pool-b", true)
	out.Reset()
	recorder.RecordMetrics(&out)
	fields = recordedLine(t, out.String(), "tf_system_metrics", "pool=pool-a")
	require.Equal(t, "2i", fields["total_scale_up_cnt"])
	require.Equal(t, "1i", fields["total_scale_down_cnt"])
	require.Equal(t, "0i", fields["total_nodes_cnt"])
	fields = recordedLine(t, out.String(), "tf_system_metrics", "pool=pool-b")
	require.Equal(t, "1i", fields["total_scale_up_cnt"])
	require.Equal(t, "0i", fields["total_scale_down_cnt"])
}

func pricedTestNode(name string, labels map[string]string) *tfv1.GPUNode {
	return &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{Name: name, Labels: labels},
		Status:     tfv1.GPUNodeStatus{Phase: tfv1.TensorFusionGPUNodePhaseRunning},
	}
}

// tf_node_metrics says how a node is bought and what it is, and gives the rate
// and the duration behind raw_cost, so that a reader can check or re-derive
// the cost: raw_cost = hourly_price * interval_seconds / 3600.
func TestNodeRowsCarryCapacityTypeInstanceTypeAndRate(t *testing.T) {
	isolateRecorderState(t)
	pool := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "cost-pool"}}
	node := pricedTestNode("cost-node", map[string]string{
		string(tfv1.NodeRequirementKeyCapacityType): "spot",
		string(tfv1.NodeRequirementKeyInstanceType): "g5.2xlarge",
	})
	recorder := newTestRecorder()
	recorder.ConfigMapPricing = map[string]float64{"A10G": 1.2}

	SetNodeMetrics(node, pool, []string{"A10G", "A10G"})
	backdateNodeRecord(node.Name, 90*time.Second)
	var out bytes.Buffer
	recorder.RecordMetrics(&out)

	tags := recordedTags(t, out.String(), "tf_node_metrics", "node=cost-node")
	require.Equal(t, "spot", tags["capacity_type"])
	require.Equal(t, "g5.2xlarge", tags["instance_type"])
	fields := recordedLine(t, out.String(), "tf_node_metrics", "node=cost-node")
	require.InDelta(t, 2.4, fieldFloat(t, fields, "hourly_price"), 1e-9, "two GPUs at the per-GPU unit price")
	interval := fieldFloat(t, fields, "interval_seconds")
	require.InDelta(t, 90, interval, 5)
	require.InDelta(t, 2.4*interval/3600, fieldFloat(t, fields, "raw_cost"), 1e-9)
}

// A node that does not say how it is bought has no capacity_type tag (the
// column reads NULL); the line never carries a tag without a value.
func TestNodeRowsLeaveUnknownCapacityTypeOut(t *testing.T) {
	isolateRecorderState(t)
	pool := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "cost-pool"}}
	node := pricedTestNode("plain-node", map[string]string{
		string(tfv1.NodeRequirementKeyCapacityType): "something-else",
	})
	SetNodeMetrics(node, pool, []string{"A10G"})
	var out bytes.Buffer
	newTestRecorder().RecordMetrics(&out)

	tags := recordedTags(t, out.String(), "tf_node_metrics", "node=plain-node")
	require.NotContains(t, tags, "capacity_type")
	require.NotContains(t, tags, "instance_type")
	require.NotContains(t, out.String(), "capacity_type=,")
	require.NotContains(t, out.String(), "instance_type=,")
}

func TestNormalizeCapacityType(t *testing.T) {
	for value, want := range map[string]string{
		"on-demand":   CapacityTypeOnDemand,
		"OnDemand":    CapacityTypeOnDemand,
		"spot":        CapacityTypeSpot,
		" Spot ":      CapacityTypeSpot,
		"preemptible": CapacityTypeSpot,
		"reserved":    CapacityTypeReserved,
		"Reserved":    CapacityTypeReserved,
		"":            "",
		"burstable":   "",
	} {
		require.Equal(t, want, normalizeCapacityType(value), "value %q", value)
	}
}

// With a price source for capacity types, a node is charged the rate of the
// way it is bought; hourly_price and raw_cost follow it. The per-GPU unit
// price stays the fallback when the source has no price or the node does not
// say what it is.
func TestNodeRateComesFromCapacityTypePriceWhenAvailable(t *testing.T) {
	isolateRecorderState(t)
	pool := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "cost-pool"}}
	labels := func(capacityType, instanceType string) map[string]string {
		return map[string]string{
			string(tfv1.NodeRequirementKeyCapacityType): capacityType,
			string(tfv1.NodeRequirementKeyInstanceType): instanceType,
			string(tfv1.NodeRequirementKeyRegion):       "us-east-1",
		}
	}
	SetNodeMetrics(pricedTestNode("spot-node", labels("spot", "g5.2xlarge")), pool, []string{"A10G"})
	SetNodeMetrics(pricedTestNode("unpriced-node", labels("spot", "unknown.type")), pool, []string{"A10G"})
	SetNodeMetrics(pricedTestNode("unlabelled-node", nil), pool, []string{"A10G"})
	for _, name := range []string{"spot-node", "unpriced-node", "unlabelled-node"} {
		backdateNodeRecord(name, time.Minute)
	}

	type priceRequest struct{ instanceType, capacityType, region string }
	var requests []priceRequest
	recorder := newTestRecorder()
	recorder.ConfigMapPricing = map[string]float64{"A10G": 1.2}
	recorder.NodeHourlyPrice = func(instanceType, capacityType, region string) (float64, string, bool) {
		requests = append(requests, priceRequest{instanceType, capacityType, region})
		if instanceType == "g5.2xlarge" && capacityType == CapacityTypeSpot {
			return 0.45, PriceSourceList, true
		}
		return 0, "", false
	}

	var out bytes.Buffer
	recorder.RecordMetrics(&out)

	fields := recordedLine(t, out.String(), "tf_node_metrics", "node=spot-node")
	require.InDelta(t, 0.45, fieldFloat(t, fields, "hourly_price"), 1e-9)
	require.InDelta(t, 0.45*fieldFloat(t, fields, "interval_seconds")/3600, fieldFloat(t, fields, "raw_cost"), 1e-9)
	fields = recordedLine(t, out.String(), "tf_node_metrics", "node=unpriced-node")
	require.InDelta(t, 1.2, fieldFloat(t, fields, "hourly_price"), 1e-9, "no price for the capacity type: per-GPU unit price")
	fields = recordedLine(t, out.String(), "tf_node_metrics", "node=unlabelled-node")
	require.InDelta(t, 1.2, fieldFloat(t, fields, "hourly_price"), 1e-9)

	require.ElementsMatch(t, []priceRequest{
		{"g5.2xlarge", CapacityTypeSpot, "us-east-1"},
		{"unknown.type", CapacityTypeSpot, "us-east-1"},
	}, requests, "asked once per node that says what it is")
}

// tf_worker_resources names the node a worker runs on, so the cost of a node
// can be apportioned to its workers, and gives the duration raw_cost covers.
func TestWorkerRowsCarryNodeAndInterval(t *testing.T) {
	isolateRecorderState(t)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "billing-worker",
			Namespace: "default",
			Labels:    map[string]string{constants.WorkloadKey: "billing-workload"},
			Annotations: map[string]string{
				constants.TFLOPSRequestAnnotation: "100",
				constants.TFLOPSLimitAnnotation:   "100",
				constants.VRAMRequestAnnotation:   "10Gi",
				constants.VRAMLimitAnnotation:     "10Gi",
				constants.GpuCountAnnotation:      "1",
				constants.GpuPoolKey:              "billing-pool",
				constants.QoSLevelAnnotation:      constants.QoSLevelMedium,
			},
		},
		Spec: corev1.PodSpec{NodeName: "billing-node"},
	}
	SetWorkerMetricsByWorkload(pod, map[types.NamespacedName]*tfv1.GPU{})
	workerMetricsLock.Lock()
	workerMetricsMap[pod.Name].LastRecordTime = time.Now().Add(-2 * time.Minute)
	workerMetricsLock.Unlock()

	recorder := newTestRecorder()
	recorder.WorkerUnitPriceMap["billing-pool"] = map[string]RawBillingPricing{
		constants.QoSLevelMedium: {TflopsPerSecond: 0.001},
	}
	var out bytes.Buffer
	recorder.RecordMetrics(&out)

	tags := recordedTags(t, out.String(), "tf_worker_resources", "worker=billing-worker")
	require.Equal(t, "billing-node", tags["node"])
	fields := recordedLine(t, out.String(), "tf_worker_resources", "worker=billing-worker")
	interval := fieldFloat(t, fields, "interval_seconds")
	require.InDelta(t, 120, interval, 5)
	require.InDelta(t, 0.001*100*interval, fieldFloat(t, fields, "raw_cost"), 1e-6)

	// The last row of a deleted worker covers the time up to its deletion.
	deletedAt := time.Now().Add(-10 * time.Second)
	workerMetricsLock.Lock()
	workerMetricsMap[pod.Name].LastRecordTime = deletedAt.Add(-30 * time.Second)
	workerMetricsLock.Unlock()
	RemoveWorkerMetrics(pod.Name, deletedAt)
	out.Reset()
	recorder.RecordMetrics(&out)
	fields = recordedLine(t, out.String(), "tf_worker_resources", "worker=billing-worker")
	require.InDelta(t, 30, fieldFloat(t, fields, "interval_seconds"), 1e-6)
	require.InDelta(t, 0.001*100*30, fieldFloat(t, fields, "raw_cost"), 1e-6)
}

// A worker that is not bound to a node yet has no node tag rather than an
// empty one.
func TestWorkerRowsLeaveUnknownNodeOut(t *testing.T) {
	isolateRecorderState(t)
	workerMetricsLock.Lock()
	workerMetricsMap["pending-worker"] = &WorkerResourceMetrics{
		WorkerName:     "pending-worker",
		WorkloadName:   "pending-workload",
		PoolName:       "billing-pool",
		Namespace:      "default",
		LastRecordTime: time.Now(),
	}
	workerMetricsLock.Unlock()

	var out bytes.Buffer
	newTestRecorder().RecordMetrics(&out)
	tags := recordedTags(t, out.String(), "tf_worker_resources", "worker=pending-worker")
	require.NotContains(t, tags, "node")
	require.NotContains(t, out.String(), "node=,")
}
