package metrics

import (
	"bytes"
	"strings"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func cloudNodeLabels(capacityType, instanceType string) map[string]string {
	return map[string]string{
		string(tfv1.NodeRequirementKeyCapacityType): capacityType,
		string(tfv1.NodeRequirementKeyInstanceType): instanceType,
	}
}

// useGlobalConfig installs a global config for one test.
func useGlobalConfig(t *testing.T, cfg *config.GlobalConfig) {
	t.Helper()
	previous := config.GetGlobalConfig()
	config.SetGlobalConfig(cfg)
	t.Cleanup(func() { config.SetGlobalConfig(previous) })
}

// Every node row says where its hourly rate comes from, so that a reader can
// tell a listed price from an estimate and from the prices an administrator
// configured. A node without any price has no price_source tag.
func TestNodeRowsCarryThePriceSource(t *testing.T) {
	isolateRecorderState(t)
	pool := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "cost-pool"}}
	nodes := map[string]struct {
		labels    map[string]string
		gpuModels []string
	}{
		"list-node":       {cloudNodeLabels("on-demand", "listed.type"), []string{"A10G"}},
		"estimate-node":   {cloudNodeLabels("spot", "estimated.type"), []string{"A10G"}},
		"configured-node": {cloudNodeLabels("spot", "unknown.type"), []string{"A10G", "A10G"}},
		"on-prem-node":    {nil, []string{"A10G"}},
		"unpriced-node":   {cloudNodeLabels("spot", "unknown.type"), []string{"NoPriceModel"}},
		"free-node":       {cloudNodeLabels("on-demand", "free.type"), []string{"A10G"}},
		"odd-node":        {cloudNodeLabels("on-demand", "odd.type"), []string{"A10G"}},
	}
	for name, node := range nodes {
		SetNodeMetrics(pricedTestNode(name, node.labels), pool, node.gpuModels)
		backdateNodeRecord(name, time.Minute)
	}

	recorder := newTestRecorder()
	recorder.ConfigMapPricing = map[string]float64{"A10G": 1.2}
	recorder.NodeHourlyPrice = func(instanceType, capacityType, region string) (float64, string, bool) {
		switch instanceType {
		case "listed.type":
			return 2.0, PriceSourceList, true
		case "estimated.type":
			return 0.6, PriceSourceSpotEstimate, true
		case "free.type":
			return 0, PriceSourceList, true
		case "odd.type":
			return 3.0, "somewhere-else", true
		}
		return 0, "", false
	}

	var out bytes.Buffer
	recorder.RecordMetrics(&out)

	for _, tc := range []struct {
		node       string
		wantSource string
		wantPrice  float64
		why        string
	}{
		{"list-node", PriceSourceList, 2.0, "a price from the pricing data"},
		{"estimate-node", PriceSourceSpotEstimate, 0.6, "an assumed spot rate is marked as an estimate"},
		{"configured-node", PriceSourceConfigured, 2.4, "no price for the instance type: the configured per-GPU prices"},
		{"on-prem-node", PriceSourceConfigured, 1.2, "a node that is not a cloud instance: the configured per-GPU prices"},
		{"unpriced-node", "", 0, "no price at all: no source"},
		{"free-node", "", 0, "a rate of zero has no source either"},
		{"odd-node", "", 3.0, "a source the recorder does not know is not passed on"},
	} {
		tags := recordedTags(t, out.String(), "tf_node_metrics", "node="+tc.node)
		fields := recordedLine(t, out.String(), "tf_node_metrics", "node="+tc.node)
		require.InDelta(t, tc.wantPrice, fieldFloat(t, fields, "hourly_price"), 1e-9, tc.node)
		if tc.wantSource == "" {
			require.NotContains(t, tags, "price_source", "%s: %s", tc.node, tc.why)
			continue
		}
		require.Equal(t, tc.wantSource, tags["price_source"], "%s: %s", tc.node, tc.why)
	}
	// The tag is left out, never written without a value.
	require.NotContains(t, out.String(), "price_source=,")
	require.NotContains(t, out.String(), "price_source= ")
}

// Without a price hook every priced node is priced from the configured per-GPU
// unit prices, and says so.
func TestNodeRowsWithoutPriceHookAreConfigured(t *testing.T) {
	isolateRecorderState(t)
	pool := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "cost-pool"}}
	SetNodeMetrics(pricedTestNode("spot-node", cloudNodeLabels("spot", "g5.2xlarge")), pool, []string{"A10G"})
	recorder := newTestRecorder()
	recorder.ConfigMapPricing = map[string]float64{"A10G": 1.2}

	var out bytes.Buffer
	recorder.RecordMetrics(&out)
	tags := recordedTags(t, out.String(), "tf_node_metrics", "node=spot-node")
	require.Equal(t, PriceSourceConfigured, tags["price_source"])
}

// The currency of the cost figures is written on the two cost tables, and only
// when an administrator configured one.
func TestCostRowsCarryTheCurrencyOnlyWhenSet(t *testing.T) {
	isolateRecorderState(t)
	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: "cost-pool"},
		Status:     tfv1.GPUPoolStatus{Phase: tfv1.TensorFusionPoolPhaseRunning},
	}
	SetPoolMetrics(pool)
	SetNodeMetrics(pricedTestNode("cost-node", nil), pool, []string{"A10G"})
	workerMetricsLock.Lock()
	workerMetricsMap["cost-worker"] = &WorkerResourceMetrics{
		WorkerName:     "cost-worker",
		WorkloadName:   "cost-workload",
		PoolName:       pool.Name,
		Namespace:      "default",
		NodeName:       "cost-node",
		LastRecordTime: time.Now(),
	}
	workerMetricsLock.Unlock()
	recorder := newTestRecorder()
	recorder.ConfigMapPricing = map[string]float64{"A10G": 1.2}

	record := func() string {
		var out bytes.Buffer
		recorder.RecordMetrics(&out)
		return out.String()
	}

	for _, unset := range []string{"", "  "} {
		useGlobalConfig(t, &config.GlobalConfig{Currency: unset})
		output := record()
		require.NotContains(t, recordedTags(t, output, "tf_node_metrics", "node=cost-node"), "currency")
		require.NotContains(t, recordedTags(t, output, "tf_worker_resources", "worker=cost-worker"), "currency")
		require.NotContains(t, output, "currency=", "currency %q is unspecified", unset)
	}

	useGlobalConfig(t, &config.GlobalConfig{Currency: "USD"})
	output := record()
	require.Equal(t, "USD", recordedTags(t, output, "tf_node_metrics", "node=cost-node")["currency"])
	require.Equal(t, "USD", recordedTags(t, output, "tf_worker_resources", "worker=cost-worker")["currency"])
	// Tables without cost figures do not get the tag.
	for _, line := range strings.Split(output, "\n") {
		if strings.HasPrefix(line, "tf_node_metrics,") || strings.HasPrefix(line, "tf_worker_resources,") || line == "" {
			continue
		}
		require.NotContains(t, line, "currency=", "line %q", line)
	}
	require.NotContains(t, recordedTags(t, output, "tf_system_metrics", "pool=cost-pool"), "currency")
	require.NotContains(t, recordedTags(t, output, "tf_pool_metrics", "pool=cost-pool"), "currency")

	// A change of the setting is picked up by the next cycle.
	useGlobalConfig(t, &config.GlobalConfig{Currency: " CNY "})
	output = record()
	require.Equal(t, "CNY", recordedTags(t, output, "tf_node_metrics", "node=cost-node")["currency"])
	require.Equal(t, "CNY", recordedTags(t, output, "tf_worker_resources", "worker=cost-worker")["currency"])
}

// A rate from the built-in price lists is in the currency of those lists,
// whatever currency the administrator configured for the cluster's own prices;
// only rates built from the configured prices carry the configured currency.
func TestNodeRowsCarryTheCurrencyOfTheirRate(t *testing.T) {
	for _, configured := range []string{"", "CNY"} {
		isolateRecorderState(t)
		useGlobalConfig(t, &config.GlobalConfig{Currency: configured})
		pool := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "cost-pool"}}
		for name, instanceType := range map[string]string{
			"list-node": "listed.type", "estimate-node": "estimated.type", "configured-node": "unknown.type",
		} {
			SetNodeMetrics(pricedTestNode(name, cloudNodeLabels("spot", instanceType)), pool, []string{"A10G"})
			backdateNodeRecord(name, time.Minute)
		}
		recorder := newTestRecorder()
		recorder.ConfigMapPricing = map[string]float64{"A10G": 1.2}
		recorder.NodeHourlyPrice = func(instanceType, capacityType, region string) (float64, string, bool) {
			switch instanceType {
			case "listed.type":
				return 2.0, PriceSourceList, true
			case "estimated.type":
				return 0.6, PriceSourceSpotEstimate, true
			}
			return 0, "", false
		}

		var out bytes.Buffer
		recorder.RecordMetrics(&out)

		require.Equal(t, ListPriceCurrency, recordedTags(t, out.String(), "tf_node_metrics", "node=list-node")["currency"])
		require.Equal(t, ListPriceCurrency, recordedTags(t, out.String(), "tf_node_metrics", "node=estimate-node")["currency"])
		tags := recordedTags(t, out.String(), "tf_node_metrics", "node=configured-node")
		if configured == "" {
			require.NotContains(t, tags, "currency", "no currency configured: the tag is left out")
		} else {
			require.Equal(t, configured, tags["currency"])
		}
	}
}
