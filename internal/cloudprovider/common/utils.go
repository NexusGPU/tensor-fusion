package common

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"math/rand"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Avoid creating too many nodes at once
const MAX_NODES_PER_RECONCILE_LOOP = 100

func GetAccessKeyOrSecretFromPath(filePath string) (string, error) {
	if filePath != "" {
		bytes, err := os.ReadFile(filePath)
		if err != nil {
			return "", err
		}
		return strings.TrimSpace(string(bytes)), nil
	}
	return "", fmt.Errorf("no env var or file path provided")
}

// Pool config contains node requirements, nodeClass indicates some base template info for creating VM nodes, this func should output the list of VM to be created to meet TFlops and VRAM gap
// Simple algorithm, try to find the instance type that best meets the gap
// TODO: implement a more advanced algorithm to combine multiple instance types
func CalculateLeastCostGPUNodes(ctx context.Context, provider types.GPUNodeProvider, cluster *tfv1.TensorFusionCluster, pool *tfv1.GPUPool, nodeClassRef tfv1.GroupKindName, tflopsGap int64, vramGap int64) ([]tfv1.GPUNodeClaimSpec, error) {
	if tflopsGap <= 0 && vramGap <= 0 {
		return []tfv1.GPUNodeClaimSpec{}, nil
	}

	requirements := pool.Spec.NodeManagerConfig.NodeProvisioner.GPURequirements
	region := cluster.Spec.ComputingVendor.Params.DefaultRegion
	zones := []string{}

	// check region override
	for _, req := range requirements {
		if req.Key == tfv1.NodeRequirementKeyRegion {
			region = req.Values[0]
			break
		}
	}

	eligibleInstances := getEligibleInstances(pool, region, provider)
	if len(eligibleInstances) == 0 {
		return nil, fmt.Errorf("no eligible instances types found, can not start creating nodes")
	}

	var bestInstance *types.GPUNodeInstanceInfo
	minCost := math.MaxFloat64
	var bestNumInstances int64

	// Default to spot, it's cheaper
	preferredCapacityType := tfv1.CapacityTypeOnDemand

	for _, req := range requirements {
		if req.Key == tfv1.NodeRequirementKeyCapacityType && req.Operator == corev1.NodeSelectorOpIn {
			// user can specify other capacity types
			hasSpot := false
			for _, capacityType := range req.Values {
				if capacityType == string(tfv1.CapacityTypeSpot) {
					hasSpot = true
				}
			}
			if hasSpot {
				preferredCapacityType = tfv1.CapacityTypeSpot
			}
		} else if req.Key == tfv1.NodeRequirementKeyRegion && req.Operator == corev1.NodeSelectorOpIn {
			// single pool can only leverage one region
			region = req.Values[0]
		} else if req.Key == tfv1.NodeRequirementKeyZone && req.Operator == corev1.NodeSelectorOpIn {
			// single pool can use multiple zones
			// TODO need balance the node number among zones, need maxScrew factor like Kubernetes, using simple random balancing as of now
			zones = req.Values
		}
	}

	for _, instance := range eligibleInstances {

		costPerHour, err := provider.GetInstancePricing(instance.InstanceType, preferredCapacityType, region)
		if err != nil {
			continue
		}

		tflopsPerInstance := instance.FP16TFlopsPerGPU * float64(instance.GPUCount)
		vramPerInstance := instance.VRAMGigabytesPerGPU * instance.GPUCount

		if tflopsPerInstance <= 0 || vramPerInstance <= 0 {
			log.FromContext(ctx).Error(nil, "tflopsPerInstance or vramPerInstance for GPU instance is zero",
				"tflopsPerInstance", tflopsPerInstance, "vramPerInstance", vramPerInstance, "instance", instance.InstanceType)
			continue // Avoid division by zero
		}

		tflopsRequired := math.Ceil(float64(tflopsGap) / tflopsPerInstance)
		vramRequired := math.Ceil(float64(vramGap) / float64(vramPerInstance) / float64(1024*1024*1024)) // convert GiB to bytes
		numInstances := int64(math.Max(tflopsRequired, vramRequired))

		totalCost := costPerHour * float64(numInstances)

		if totalCost < minCost {
			minCost = totalCost
			bestInstance = &instance
			bestNumInstances = numInstances
		}
		// if the instance type is too large to fit the gap, wait next compaction controller reconcile to make half
	}

	if bestInstance == nil {
		return nil, fmt.Errorf("no eligible instances found")
	}

	if bestNumInstances > MAX_NODES_PER_RECONCILE_LOOP {
		log.FromContext(ctx).Info("[Warn] Limiting the number of nodes to 200(default limit value)", "parsed number", bestNumInstances)
		bestNumInstances = MAX_NODES_PER_RECONCILE_LOOP
	}

	zone := ""
	if len(zones) > 0 {
		zone = zones[rand.Intn(len(zones))]
	}

	nodes := make([]tfv1.GPUNodeClaimSpec, 0, bestNumInstances)
	for i := int64(0); i < bestNumInstances; i++ {
		nodes = append(nodes, tfv1.GPUNodeClaimSpec{
			NodeName:     fmt.Sprintf("%s-%s", pool.Name, generateRandomString(8)),
			InstanceType: bestInstance.InstanceType,
			NodeClassRef: nodeClassRef,
			Region:       region,
			Zone:         zone,
			CapacityType: preferredCapacityType,

			TFlopsOffered:    resource.MustParse(fmt.Sprintf("%f", bestInstance.FP16TFlopsPerGPU*float64(bestInstance.GPUCount))),
			VRAMOffered:      resource.MustParse(fmt.Sprintf("%dGi", bestInstance.VRAMGigabytesPerGPU*bestInstance.GPUCount)),
			GPUDeviceOffered: bestInstance.GPUCount,

			ExtraParams: cluster.Spec.ComputingVendor.Params.ExtraParams,
		})
	}

	return nodes, nil
}

func getEligibleInstances(pool *tfv1.GPUPool, region string, provider types.GPUNodeProvider) []types.GPUNodeInstanceInfo {
	eligible := []types.GPUNodeInstanceInfo{}

	for _, instance := range provider.GetGPUNodeInstanceTypeInfo(region) {
		meetsRequirements := true

		for _, req := range pool.Spec.NodeManagerConfig.NodeProvisioner.GPURequirements {

			if req.Key == tfv1.NodeRequirementKeyCapacityType {
				continue
			}

			switch req.Key {

			case tfv1.NodeRequirementKeyInstanceType:
				if req.Operator == corev1.NodeSelectorOpIn {
					if !contains(req.Values, instance.InstanceType) {
						meetsRequirements = false
						break
					}
				}
			case tfv1.NodeRequirementKeyArchitecture:
				if req.Operator == corev1.NodeSelectorOpIn {
					if !contains(req.Values, string(instance.CPUArchitecture)) {
						meetsRequirements = false
						break
					}
				}
			case tfv1.NodeRequirementKeyGPUVendor:
				if req.Operator == corev1.NodeSelectorOpIn {
					if !contains(req.Values, string(instance.GPUVendor)) {
						meetsRequirements = false
						break
					}
				}

			}

		}

		if meetsRequirements {
			eligible = append(eligible, instance)
		}
	}

	return eligible
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func generateRandomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyz"
	source := rand.NewSource(time.Now().UnixNano())
	rand := rand.New(source)

	result := make([]byte, length)
	for i := range result {
		result[i] = charset[rand.Intn(len(charset))]
	}
	return string(result)
}
