package main

import (
	"bytes"
	"strings"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/pricing"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Instance types of the embedded pricing data used by these tests.
const (
	awsL4Instance     = "g6.xlarge" // AWS: the data lists no spot price
	azureV100Instance = "NC24s v3"  // Azure: the data lists a spot price
)

// loadTestPricing loads the pricing tables with the GPU models of the instance
// types above.
func loadTestPricing(t *testing.T) {
	t.Helper()
	gpuInfos := append([]config.GpuInfo{}, *config.MockGpuInfo()...)
	gpuInfos = append(gpuInfos, config.GpuInfo{
		Model:         "V100",
		Vendor:        "NVIDIA",
		Fp16TFlops:    resource.MustParse("125"),
		FullModelName: "NVIDIA V100",
	})
	pricing.SetTflopsMapAndInitGPUPricingInfo(t.Context(), &gpuInfos)
}

// The metrics recorder asks for a node's hourly rate by instance type and the
// capacity type on its karpenter.sh/capacity-type label. A spot node gets the
// spot rate, and a node the pricing tables do not know gets no answer, so that
// the recorder prices it by its GPUs as before.
func TestNodeHourlyPriceFollowsTheCapacityType(t *testing.T) {
	loadTestPricing(t)
	const instanceType = awsL4Instance
	provider := pricing.NewStaticPricingProvider()

	onDemand, _, ok := nodeHourlyPrice(instanceType, "on-demand", "")
	require.True(t, ok)
	want, err := provider.HourlyPriceForCapacityType(instanceType, tfv1.CapacityTypeOnDemand, "")
	require.NoError(t, err)
	require.Equal(t, want, onDemand)

	spot, _, ok := nodeHourlyPrice(instanceType, "spot", "")
	require.True(t, ok)
	require.Less(t, spot, onDemand, "a spot node is not costed at the on-demand rate")

	for _, tc := range []struct{ name, instanceType, capacityType string }{
		{"unknown instance type", "on-prem-server", "on-demand"},
		{"unknown capacity type", instanceType, "burstable"},
		{"no capacity type", instanceType, ""},
	} {
		_, source, ok := nodeHourlyPrice(tc.instanceType, tc.capacityType, "")
		require.False(t, ok, tc.name)
		require.Empty(t, source, tc.name)
	}
}

// The rate comes with its source. A listed price is "list"; the AWS spot rate,
// which the pricing data does not list and which is the on-demand price times
// an assumed factor, is "spot-estimate"; the Azure spot rate, which the data
// lists, is "list".
func TestNodeHourlyPriceSaysWhereTheRateComesFrom(t *testing.T) {
	loadTestPricing(t)

	// The two packages spell the sources the same way.
	require.Equal(t, metrics.PriceSourceList, string(pricing.PriceSourceList))
	require.Equal(t, metrics.PriceSourceSpotEstimate, string(pricing.PriceSourceSpotEstimate))

	onDemand, source, ok := nodeHourlyPrice(awsL4Instance, "on-demand", "")
	require.True(t, ok)
	require.Equal(t, metrics.PriceSourceList, source)

	spot, source, ok := nodeHourlyPrice(awsL4Instance, "spot", "")
	require.True(t, ok)
	require.Equal(t, metrics.PriceSourceSpotEstimate, source, "an assumed AWS spot rate is not presented as a price")
	require.InDelta(t, onDemand*pricing.AWSSpotAssumedPriceRatio, spot, 1e-9)

	_, source, ok = nodeHourlyPrice(awsL4Instance, "reserved", "")
	require.True(t, ok)
	require.Equal(t, metrics.PriceSourceList, source)

	// Asked by the name the pricing data knows the instance type by.
	azureSpot, source, ok := nodeHourlyPrice(azureV100Instance, "spot", "")
	require.True(t, ok)
	require.Equal(t, metrics.PriceSourceList, source, "the Azure data lists the spot price")
	require.InDelta(t, 2.0735, azureSpot, 1e-9)
}

// End to end, through the recorder as the operator wires it: the row of an AWS
// spot node is marked as an estimate, the row of an on-demand node carries a
// listed price, and a node the pricing tables do not know and that has no
// configured GPU price has no price_source at all.
func TestNodeMetricsRowsMarkAnEstimatedSpotRate(t *testing.T) {
	loadTestPricing(t)
	pool := &tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: "price-source-pool"}}
	nodes := map[string][2]string{
		"aws-spot-node":      {"spot", awsL4Instance},
		"aws-on-demand-node": {"on-demand", awsL4Instance},
		"unknown-node":       {"spot", "on-prem-server"},
	}
	for name, labels := range nodes {
		node := &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{
			string(tfv1.NodeRequirementKeyCapacityType): labels[0],
			string(tfv1.NodeRequirementKeyInstanceType): labels[1],
		}}}
		metrics.SetNodeMetrics(node, pool, []string{"unpriced-model"})
		t.Cleanup(func() { metrics.RemoveNodeMetrics(name) })
	}
	recorder := &metrics.MetricsRecorder{
		WorkerUnitPriceMap: map[string]map[string]metrics.RawBillingPricing{},
		NodeHourlyPrice:    nodeHourlyPrice,
	}

	var out bytes.Buffer
	recorder.RecordMetrics(&out)

	sources := map[string]string{}
	for _, line := range strings.Split(out.String(), "\n") {
		if !strings.HasPrefix(line, "tf_node_metrics,") {
			continue
		}
		head, _, _ := strings.Cut(line, " ")
		tags := map[string]string{}
		for _, tag := range strings.Split(head, ",")[1:] {
			key, value, _ := strings.Cut(tag, "=")
			tags[key] = value
		}
		source, tagged := tags["price_source"]
		if !tagged {
			source = "<none>"
		}
		sources[tags["node"]] = source
	}
	require.Equal(t, map[string]string{
		"aws-spot-node":      metrics.PriceSourceSpotEstimate,
		"aws-on-demand-node": metrics.PriceSourceList,
		"unknown-node":       "<none>",
	}, sources, out.String())
}
