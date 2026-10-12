package metrics

import (
	"bytes"
	"regexp"
	"strings"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/influxdata/line-protocol/v2/lineprotocol"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// controllerFieldTypes is the line-protocol type of every field the controller
// writes, per measurement. It is the type of the column in the tables that
// exist in installations: GreptimeDB creates a column with the type of the
// first value it receives and rejects every later row that carries another
// one, together with the rest of what was posted with it.
//
// Do not change an entry to make the test pass: a changed type means existing
// installations lose the rows of that table until the table is dropped. A new
// field gets a new entry.
var controllerFieldTypes = map[string]map[string]string{
	"tf_worker_resources": {
		"gpu_count":          "int",
		"tflops_limit":       "float",
		"tflops_request":     "float",
		"raw_cost":           "float",
		"interval_seconds":   "float",
		"vram_bytes_limit":   "float",
		"vram_bytes_request": "float",
		"ready":              "bool",
	},
	"tf_node_metrics": {
		"allocated_tflops":                 "float",
		"allocated_tflops_percent":         "float",
		"allocated_tflops_percent_virtual": "float",
		"allocated_vram_bytes":             "float",
		"allocated_vram_percent":           "float",
		"allocated_vram_percent_virtual":   "float",
		"limited_tflops":                   "float",
		"limited_vram_bytes":               "float",
		"limited_tflops_percent_virtual":   "float",
		"limited_vram_percent_virtual":     "float",
		"gpu_count":                        "int",
		"raw_cost":                         "float",
		"hourly_price":                     "float",
		"interval_seconds":                 "float",
	},
	"tf_system_metrics": {
		"total_workers_cnt":              "int",
		"total_nodes_cnt":                "int",
		"total_allocation_fail_cnt":      "int",
		"total_allocation_success_cnt":   "int",
		"total_scale_up_cnt":             "int",
		"total_scale_down_cnt":           "int",
		"total_topo_satisfied_cnt":       "int",
		"total_topo_unsatisfied_cnt":     "int",
		"total_topo_fallback_cnt":        "int",
		"total_topo_search_degraded_cnt": "int",
	},
	"tf_pool_metrics": {
		"allocated_tflops":                 "float",
		"allocated_tflops_percent":         "float",
		"allocated_tflops_percent_virtual": "float",
		"allocated_vram_bytes":             "float",
		"allocated_vram_percent":           "float",
		"allocated_vram_percent_virtual":   "float",
		"limited_tflops":                   "float",
		"limited_vram_bytes":               "float",
		"limited_tflops_percent_virtual":   "float",
		"limited_vram_percent_virtual":     "float",
		"gpu_count":                        "int",
	},
	"tf_gpu_allocation": {
		"gpu_count": "int",
	},
	"tf_gpu_metrics": {
		"capacity_tflops":          "float",
		"available_tflops":         "float",
		"allocated_tflops":         "float",
		"allocated_tflops_percent": "float",
		"capacity_vram_bytes":      "float",
		"available_vram_bytes":     "float",
		"allocated_vram_bytes":     "float",
		"allocated_vram_percent":   "float",
	},
}

// lineProtocolFieldTypes returns the type of every field of every measurement
// in data, and fails when one field is written with two types.
func lineProtocolFieldTypes(t *testing.T, data []byte) map[string]map[string]string {
	t.Helper()
	fieldTypes := map[string]map[string]string{}
	decoder := lineprotocol.NewDecoderWithBytes(data)
	for decoder.Next() {
		name, err := decoder.Measurement()
		require.NoError(t, err)
		measurement := string(name)
		if fieldTypes[measurement] == nil {
			fieldTypes[measurement] = map[string]string{}
		}
		for {
			key, value, err := decoder.NextField()
			require.NoError(t, err)
			if key == nil {
				break
			}
			kind := value.Kind().String()
			if previous, seen := fieldTypes[measurement][string(key)]; seen {
				require.Equal(t, previous, kind, "%s.%s is written with two types", measurement, key)
			}
			fieldTypes[measurement][string(key)] = kind
		}
	}
	require.NoError(t, decoder.Err())
	return fieldTypes
}

var ddlColumn = regexp.MustCompile("`([a-z0-9_]+)` ([A-Za-z_]+( UNSIGNED)?)")

// ddlFieldTypes returns the line-protocol type of every numeric and boolean
// column of a table in the documented schema (TFVersionMigrationMap).
func ddlFieldTypes(table string) map[string]string {
	columns := map[string]string{}
	for _, version := range TFVersionMigrationMap {
		for _, statement := range version.AlterSQL {
			if !strings.HasPrefix(statement, "CREATE TABLE IF NOT EXISTS "+table+" ") {
				continue
			}
			for _, match := range ddlColumn.FindAllStringSubmatch(statement, -1) {
				switch strings.ToLower(match[2]) {
				case "double":
					columns[match[1]] = "float"
				case "bigint":
					columns[match[1]] = "int"
				case "bigint unsigned":
					columns[match[1]] = "uint"
				case "boolean":
					columns[match[1]] = "bool"
				}
			}
		}
	}
	return columns
}

// Every field of every measurement the controller writes has the type the
// existing tables have. The lines are the recorder's own, so a changed Go type
// in a metrics struct or a changed conversion shows up here.
func TestControllerLineProtocolFieldTypes(t *testing.T) {
	isolateRecorderState(t)
	const poolName = "types-pool"
	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: poolName},
		Status:     tfv1.GPUPoolStatus{Phase: tfv1.TensorFusionPoolPhaseRunning, TotalGPUs: 2},
	}
	SetPoolMetrics(pool)

	// Whole numbers on purpose: a float field must stay a float then.
	node := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{Name: "types-node"},
		Status: tfv1.GPUNodeStatus{
			Phase:                  tfv1.TensorFusionGPUNodePhaseRunning,
			TotalTFlops:            resource.MustParse("312"),
			TotalVRAM:              resource.MustParse("80Gi"),
			AvailableTFlops:        resource.MustParse("212"),
			AvailableVRAM:          resource.MustParse("60Gi"),
			VirtualTFlops:          resource.MustParse("624"),
			VirtualVRAM:            resource.MustParse("120Gi"),
			VirtualAvailableTFlops: quantity("468"),
			VirtualAvailableVRAM:   quantity("90Gi"),
		},
	}
	SetNodeMetrics(node, pool, []string{"A100", "A100"})
	backdateNodeRecord(node.Name, time.Minute)

	index := int32(0)
	gpu := tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{Name: "types-gpu"},
		Status: tfv1.GPUStatus{
			GPUModel:  "A100",
			Index:     &index,
			Phase:     tfv1.TensorFusionGPUPhaseRunning,
			Capacity:  &tfv1.Resource{Tflops: resource.MustParse("312"), Vram: resource.MustParse("80Gi")},
			Available: &tfv1.Resource{Tflops: resource.MustParse("212"), Vram: resource.MustParse("60Gi")},
		},
	}
	SetGPUMetrics([]tfv1.GPU{gpu}, node.Name, poolName)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "types-worker",
			Namespace: "default",
			UID:       "types-worker-uid",
			Labels:    map[string]string{constants.WorkloadKey: "types-workload"},
			Annotations: map[string]string{
				constants.TFLOPSRequestAnnotation: "100",
				constants.TFLOPSLimitAnnotation:   "100",
				constants.VRAMRequestAnnotation:   "10Gi",
				constants.VRAMLimitAnnotation:     "10Gi",
				constants.GpuCountAnnotation:      "1",
				constants.GpuPoolKey:              poolName,
				constants.QoSLevelAnnotation:      constants.QoSLevelMedium,
				constants.GPUDeviceIDsAnnotation:  gpu.Name,
			},
		},
		Spec: corev1.PodSpec{NodeName: node.Name},
	}
	gpuStore := map[types.NamespacedName]*tfv1.GPU{{Name: gpu.Name}: &gpu}
	SetWorkerMetricsByWorkload(pod, gpuStore)
	SetGPUAllocationMetrics(gpuStore, pod)
	workerMetricsLock.Lock()
	workerMetricsMap[pod.Name].LastRecordTime = time.Now().Add(-time.Minute)
	workerMetricsLock.Unlock()

	// The counters of tf_system_metrics, the topology ones included.
	SetSchedulerMetrics(poolName, true)
	SetSchedulerMetrics(poolName, false)
	SetAutoscalingMetrics(poolName, true)
	SetAutoscalingMetrics(poolName, false)
	SetTopologyMetrics(poolName, true, true, true)

	recorder := newTestRecorder()
	recorder.WorkerUnitPriceMap[poolName] = map[string]RawBillingPricing{
		constants.QoSLevelMedium: {TflopsPerSecond: 0.001},
	}
	var out bytes.Buffer
	recorder.RecordMetrics(&out)

	written := lineProtocolFieldTypes(t, out.Bytes())
	require.Equal(t, controllerFieldTypes, written, "lines:\n%s", out.String())

	// The documented schema says the same of every field it lists, and lists
	// every field.
	for table, fields := range controllerFieldTypes {
		documented := ddlFieldTypes(table)
		for field, kind := range fields {
			require.Contains(t, documented, field, "%s.%s is missing from the documented schema", table, field)
			require.Equal(t, kind, documented[field], "documented type of %s.%s", table, field)
		}
	}
}
