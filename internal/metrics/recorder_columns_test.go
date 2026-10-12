package metrics

import (
	"bytes"
	"strconv"
	"strings"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// recordedLine returns the fields of the first line of a measurement whose
// tags contain the given fragment.
func recordedLine(t *testing.T, output, measurement, tagFragment string) map[string]string {
	t.Helper()
	for _, line := range strings.Split(output, "\n") {
		if !strings.HasPrefix(line, measurement+",") || !strings.Contains(line, tagFragment) {
			continue
		}
		parts := strings.Split(line, " ")
		require.Len(t, parts, 3, "line %q", line)
		fields := map[string]string{}
		for _, field := range strings.Split(parts[1], ",") {
			key, value, _ := strings.Cut(field, "=")
			fields[key] = value
		}
		return fields
	}
	t.Fatalf("no %s line with %q in:\n%s", measurement, tagFragment, output)
	return nil
}

func quantity(value string) *resource.Quantity {
	q := resource.MustParse(value)
	return &q
}

// tf_node_metrics carries the limits assigned on a node against its virtual
// capacity once the allocator has published the node's virtual availability,
// and leaves those fields out until then.
func TestNodeMetricsCarryAssignedLimits(t *testing.T) {
	pool := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "limits-pool"}}
	node := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{Name: "limits-node"},
		Status: tfv1.GPUNodeStatus{
			Phase:           tfv1.TensorFusionGPUNodePhaseRunning,
			TotalTFlops:     resource.MustParse("312"),
			TotalVRAM:       resource.MustParse("80Gi"),
			AvailableTFlops: resource.MustParse("212"),
			AvailableVRAM:   resource.MustParse("60Gi"),
			VirtualTFlops:   resource.MustParse("624"),
			VirtualVRAM:     resource.MustParse("120Gi"),
		},
	}
	t.Cleanup(func() { RemoveNodeMetrics(node.Name) })
	recorder := &MetricsRecorder{WorkerUnitPriceMap: map[string]map[string]RawBillingPricing{}}

	SetNodeMetrics(node, pool, []string{"A100"})
	var out bytes.Buffer
	recorder.RecordMetrics(&out)
	fields := recordedLine(t, out.String(), "tf_node_metrics", "node=limits-node")
	require.Equal(t, "100", fields["allocated_tflops"])
	require.NotContains(t, fields, "limited_tflops", "not derivable yet: left out rather than written as zero")
	require.NotContains(t, fields, "limited_vram_percent_virtual")

	node.Status.VirtualAvailableTFlops = quantity("468")
	node.Status.VirtualAvailableVRAM = quantity("90Gi")
	SetNodeMetrics(node, pool, []string{"A100"})
	out.Reset()
	recorder.RecordMetrics(&out)
	fields = recordedLine(t, out.String(), "tf_node_metrics", "node=limits-node")
	require.Equal(t, "156", fields["limited_tflops"])
	require.Equal(t, "25", fields["limited_tflops_percent_virtual"])
	limitedVram, err := strconv.ParseFloat(fields["limited_vram_bytes"], 64)
	require.NoError(t, err)
	require.Equal(t, float64(30<<30), limitedVram)
	require.Equal(t, "25", fields["limited_vram_percent_virtual"])
}

// The node scaling counters of tf_system_metrics count the decisions made for
// a pool: nodes requested and nodes selected for removal.
func TestSystemMetricsCountNodeScaling(t *testing.T) {
	const poolName = "scaling-pool"
	pool := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: poolName}}
	node := &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{Name: "scaling-node"}}
	t.Cleanup(func() {
		RemoveNodeMetrics(node.Name)
		tensorFusionSystemMetricsLock.Lock()
		delete(TensorFusionSystemMetricsMap, poolName)
		tensorFusionSystemMetricsLock.Unlock()
	})
	SetNodeMetrics(node, pool, nil)
	recorder := &MetricsRecorder{WorkerUnitPriceMap: map[string]map[string]RawBillingPricing{}}

	var out bytes.Buffer
	recorder.RecordMetrics(&out)
	fields := recordedLine(t, out.String(), "tf_system_metrics", "pool="+poolName)
	require.Equal(t, "0i", fields["total_scale_up_cnt"])
	require.Equal(t, "0i", fields["total_scale_down_cnt"])

	for range 3 {
		SetAutoscalingMetrics(poolName, true)
	}
	SetAutoscalingMetrics(poolName, false)
	// A decision that cannot be attributed to a pool is not counted anywhere.
	SetAutoscalingMetrics("", true)
	out.Reset()
	recorder.RecordMetrics(&out)
	fields = recordedLine(t, out.String(), "tf_system_metrics", "pool="+poolName)
	require.Equal(t, "3i", fields["total_scale_up_cnt"])
	require.Equal(t, "1i", fields["total_scale_down_cnt"])
	require.Equal(t, "1i", fields["total_nodes_cnt"])
	tensorFusionSystemMetricsLock.RLock()
	_, unnamed := TensorFusionSystemMetricsMap[""]
	tensorFusionSystemMetricsLock.RUnlock()
	require.False(t, unnamed)
}
