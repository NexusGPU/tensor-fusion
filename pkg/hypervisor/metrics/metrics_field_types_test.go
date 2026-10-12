package metrics

import (
	"bytes"
	"math"
	"regexp"
	"strings"
	"testing"

	"github.com/influxdata/line-protocol/v2/lineprotocol"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	controllermetrics "github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
)

// hypervisorFieldTypes is the line-protocol type of every field the hypervisor
// writes, per measurement. It is the type of the column in the tables that
// exist in installations: GreptimeDB rejects every row that carries another
// one. tf_gpu_usage and tf_worker_usage keep the types of the earlier
// hypervisor, which created those tables; tf_limiter_events is written since
// this recorder exists.
//
// Do not change an entry to make the test pass: a changed type means existing
// installations lose the rows of that table until the table is dropped.
var hypervisorFieldTypes = map[string]map[string]string{
	"tf_gpu_usage": {
		"rx":                 "float",
		"tx":                 "float",
		"temperature":        "float",
		"memory_bytes":       "uint",
		"memory_percentage":  "float",
		"compute_percentage": "float",
		"compute_tflops":     "float",
		"power_usage":        "int",
		// Provider extra metrics: integers for the two columns that were
		// integers, floats otherwise.
		"nvlink_rx":          "int",
		"nvlink_tx":          "int",
		"graphics_clock_mhz": "float",
		"sm_clock_mhz":       "float",
		"memory_clock_mhz":   "float",
		"video_clock_mhz":    "float",
		"vendor_metric":      "float",
	},
	"tf_worker_usage": {
		"memory_bytes":       "uint",
		"memory_percentage":  "float",
		"compute_percentage": "float",
		"compute_tflops":     "float",
	},
	"tf_limiter_events": {
		"count": "int",
	},
}

// lineProtocolFieldTypes returns the type of every field of every measurement
// in data, and fails when one field is written with two types.
func lineProtocolFieldTypes(t *testing.T, data []byte) map[string]map[string]string {
	t.Helper()
	types := map[string]map[string]string{}
	decoder := lineprotocol.NewDecoderWithBytes(data)
	for decoder.Next() {
		name, err := decoder.Measurement()
		require.NoError(t, err)
		measurement := string(name)
		if types[measurement] == nil {
			types[measurement] = map[string]string{}
		}
		for {
			key, value, err := decoder.NextField()
			require.NoError(t, err)
			if key == nil {
				break
			}
			kind := value.Kind().String()
			if previous, seen := types[measurement][string(key)]; seen {
				require.Equal(t, previous, kind, "%s.%s is written with two types", measurement, key)
			}
			types[measurement][string(key)] = kind
		}
	}
	require.NoError(t, decoder.Err())
	return types
}

var ddlColumn = regexp.MustCompile("`([a-z0-9_]+)` ([A-Za-z_]+( UNSIGNED)?)")

// ddlFieldType returns the line-protocol type of a column of a table in the
// documented schema (controllermetrics.TFVersionMigrationMap), and false when
// the schema does not list it.
func ddlFieldType(table, column string) (string, bool) {
	for _, version := range controllermetrics.TFVersionMigrationMap {
		for _, statement := range version.AlterSQL {
			if !strings.HasPrefix(statement, "CREATE TABLE IF NOT EXISTS "+table+" ") {
				continue
			}
			for _, match := range ddlColumn.FindAllStringSubmatch(statement, -1) {
				if match[1] != column {
					continue
				}
				switch strings.ToLower(match[2]) {
				case "double":
					return "float", true
				case "bigint":
					return "int", true
				case "bigint unsigned":
					return "uint", true
				case "boolean":
					return "bool", true
				default:
					return strings.ToLower(match[2]), true
				}
			}
		}
	}
	return "", false
}

// Every field of every measurement the hypervisor writes has the type the
// existing tables have. The lines are the recorder's own, so a changed Go type
// in a metrics struct or a changed conversion shows up here.
func TestHypervisorLineProtocolFieldTypes(t *testing.T) {
	worker := &api.WorkerInfo{
		WorkerUID: "uid-0", WorkerName: "worker-0", Namespace: "ns", WorkloadName: "app",
		Limits: tfv1.Resource{Vram: resource.MustParse("8Gi")},
	}
	devices := &usageTestDevices{
		devices: map[string]*api.DeviceInfo{"gpu-0": rtx3090()},
		metrics: map[string]*api.GPUUsageMetrics{"gpu-0": {
			DeviceUUID: "gpu-0", MemoryBytes: 6 << 30, MemoryPercentage: 25, ComputePercentage: 50, ComputeTflops: 35.6,
			// Whole numbers on purpose: a float field must stay a float then.
			Rx: 12, Tx: 34, Temperature: 65, PowerUsage: 180,
			ExtraMetrics: map[string]float64{
				"nvlink_rx": 1024, "nvlink_tx": 2048.4,
				"graphics_clock_mhz": 1500, "sm_clock_mhz": 1500, "memory_clock_mhz": 9500, "video_clock_mhz": 1350,
				"vendor_metric": 3,
				// Not written: a name the recorder writes itself, and values
				// that are not numbers.
				"memory_bytes": 1.5, "power_usage": 0.5, "not_a_number": math.NaN(), "infinite": math.Inf(1),
			},
		}},
	}
	workers := &usageTestWorkers{
		workers: []*api.WorkerInfo{worker},
		metrics: map[string]map[string]map[string]*api.WorkerMetrics{"gpu-0": {
			"uid-0": {"1": {MemoryBytes: 2 << 30, ComputePercentage: 40}},
		}},
	}
	recorder := &HypervisorMetricsRecorder{
		nodeName: "node", gpuPool: "pool", deviceController: devices, workerController: workers,
		allocationController: &usageTestAllocations{workers: []*api.WorkerInfo{worker}},
	}
	recorder.SetLimiterEventSource(func(yield func(namespace, pod, kind string, count uint64)) {
		yield("ns", "worker-0", "stalled_bucket", 3)
	})

	var out bytes.Buffer
	recorder.RecordDeviceMetrics(&out)
	recorder.RecordWorkerMetrics(&out)
	recorder.RecordLimiterEventMetrics(&out)

	written := lineProtocolFieldTypes(t, out.Bytes())
	require.Equal(t, hypervisorFieldTypes, written, "lines:\n%s", out.String())

	// The values of the fields whose type was corrected.
	require.Contains(t, out.String(), "memory_bytes=6442450944u")
	require.Contains(t, out.String(), "memory_bytes=2147483648u")
	require.Contains(t, out.String(), "power_usage=180i")
	require.Contains(t, out.String(), "nvlink_rx=1024i")
	require.Contains(t, out.String(), "nvlink_tx=2048i")

	// The documented schema of the two tables says the same.
	for _, table := range []string{"tf_gpu_usage", "tf_worker_usage"} {
		for field, kind := range hypervisorFieldTypes[table] {
			documented, listed := ddlFieldType(table, field)
			if !listed {
				require.Equal(t, "vendor_metric", field, "%s.%s is missing from the documented schema", table, field)
				continue
			}
			require.Equal(t, kind, documented, "documented type of %s.%s", table, field)
		}
	}
}
