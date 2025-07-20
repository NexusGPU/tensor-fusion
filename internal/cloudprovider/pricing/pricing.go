/*
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
 * GPU instance data is from:https://instances.vantage.sh/ ,Thanks a lot!
 */
package pricing

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/csv"
	"regexp"
	"strconv"
	"strings"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	providerAWS   = "aws"
	providerAzure = "azure"
)

// Global data initialized at package load time
var (
	globalAWSGPUInstanceData   map[string]GPUNodeInstanceInfoAndPrice
	globalAzureGPUInstanceData map[string]GPUNodeInstanceInfoAndPrice
	tflopsMap                  map[string]*config.GpuInfo
)

var readyCh = make(chan struct{})
var initOnce sync.Once

// PricingProvider provides pricing information and calculations for instance types
type PricingProvider interface {
	GetPricing(instanceType, capacityType tfv1.CapacityTypeEnum) (float64, bool)
	GetGPUNodeInstanceTypeInfo(region string) ([]string, bool)
}

type GPUNodeInstanceInfoAndPrice struct {
	GPUNodeInstanceInfo types.GPUNodeInstanceInfo
	onDemandPrice       float64
	spotPrice           float64
	reservedPrice       float64
}

// StaticPricingProvider implements PricingProvider using static pricing data
// Data is now stored in global variables and initialized during package init
type StaticPricingProvider struct{}

func NewStaticPricingProvider() *StaticPricingProvider {
	return &StaticPricingProvider{}
}

//go:embed pricing-data/aws-gpu.csv
var awsCSV string

//go:embed pricing-data/azure-gpu.csv
var azureCSV string

func init() {
	tflopsMap = make(map[string]*config.GpuInfo, 100)
}

func SetTflopsMapAndInitGPUPricingInfo(ctx context.Context, gpuInfos *[]config.GpuInfo) {
	if gpuInfos == nil {
		log.FromContext(ctx).Info("gpuInfos is empty, check public-gpu-info config file")
		return
	}
	for _, gpuInfo := range *gpuInfos {
		tflopsMap[gpuInfo.FullModelName] = &gpuInfo
		tflopsMap[gpuInfo.Model] = &gpuInfo
	}

	initOnce.Do(func() {
		globalAWSGPUInstanceData = make(map[string]GPUNodeInstanceInfoAndPrice)
		globalAzureGPUInstanceData = make(map[string]GPUNodeInstanceInfoAndPrice)

		loadCSVInstanceDataFromPath(context.Background(), []byte(awsCSV), providerAWS)
		loadCSVInstanceDataFromPath(context.Background(), []byte(azureCSV), providerAzure)

		close(readyCh)
	})
}

// loadCSVInstanceDataFromPath loads instance data from a single CSV file
func loadCSVInstanceDataFromPath(ctx context.Context, data []byte, provider string) {
	reader := csv.NewReader(bytes.NewReader(data))
	records, err := reader.ReadAll()
	if err != nil {
		log.FromContext(ctx).Error(err, "Error reading CSV file", "provider", provider)
		return
	}

	localAWSGPUInstanceData := make(map[string]GPUNodeInstanceInfoAndPrice)
	localAzureGPUInstanceData := make(map[string]GPUNodeInstanceInfoAndPrice)
	processedCount := 0
	// Parse CSV records (skip header)
	for i, record := range records {
		if i == 0 {
			continue // Skip header
		}

		// Determine required record length based on provider
		if provider == providerAWS && len(record) < 13 {
			continue // AWS needs at least 13 fields
		}
		if provider == providerAzure && len(record) < 11 {
			continue // Azure needs at least 11 fields
		}

		var instanceInfo types.GPUNodeInstanceInfo
		var prices [3]float64 // onDemand, reserved, spot

		switch provider {
		case providerAWS:
			instanceInfo, prices = parseAWSRecord(record)
		case providerAzure:
			instanceInfo, prices = parseAzureRecord(record)
			// Filter out Azure instances with fractional GPU counts (GPUCount = 0)
			if instanceInfo.GPUCount == 0 {
				continue // Skip this record, don't store it in memory
			}
		default:
			continue
		}

		// Store in appropriate map
		gpuInfo, exists := tflopsMap[instanceInfo.GPUModel]
		if !exists {
			// not configured, skip
			continue
		}
		instanceInfo.FP16TFlopsPerGPU = gpuInfo.Fp16TFlops.AsApproximateFloat64()

		instanceInfoAndPrice := GPUNodeInstanceInfoAndPrice{
			GPUNodeInstanceInfo: instanceInfo,
			onDemandPrice:       prices[0],
			reservedPrice:       prices[1],
			spotPrice:           prices[2],
		}

		if provider == providerAWS {
			localAWSGPUInstanceData[instanceInfo.InstanceType] = instanceInfoAndPrice
		} else {
			localAzureGPUInstanceData[instanceInfo.InstanceType] = instanceInfoAndPrice
		}
		processedCount++
	}
	if provider == providerAWS {
		globalAWSGPUInstanceData = localAWSGPUInstanceData
	} else {
		globalAzureGPUInstanceData = localAzureGPUInstanceData
	}

	log.FromContext(ctx).V(6).Info("Loaded GPU instance types", "count", processedCount, "provider", provider)
}

// parseAWSRecord parses a single AWS CSV record
func parseAWSRecord(record []string) (types.GPUNodeInstanceInfo, [3]float64) {
	instanceType := record[1]      // API Name
	memory := record[2]            // Instance Memory
	cpuCountStr := record[3]       // vCPUs
	gpuCountStr := record[4]       // GPUs
	gpuModel := record[5]          // GPU model
	gpuMemory := record[6]         // GPU memory
	cpuArchStr := record[7]        // CPU architecture, x86_64 or arm64
	onDemandPriceStr := record[9]  // On Demand
	reservedPriceStr := record[10] // Linux Reserved cost
	totalMemory := parseMemory(gpuMemory)
	gpuCount := parseGPUCount(gpuCountStr)

	// Default to amd64 (x86_64)
	cpuArch := types.CPUArchitectureAMD64
	if cpuArchStr == "arm64" {
		cpuArch = types.CPUArchitectureARM64
	}

	var perGPUMemory int32
	if gpuCount != 0 {
		perGPUMemory = totalMemory / gpuCount
	}

	info := types.GPUNodeInstanceInfo{
		InstanceType:        instanceType,
		MemoryGiB:           parseMemory(memory),
		VRAMGigabytesPerGPU: perGPUMemory,
		GPUModel:            gpuModel,
		GPUCount:            gpuCount,
		CPUArchitecture:     cpuArch,
		CPUs:                parseCPUCount(cpuCountStr),
		GPUVendor:           types.GPUVendorNvidia,
	}

	prices := [3]float64{
		parsePrice(onDemandPriceStr),
		parsePrice(reservedPriceStr),
		parsePrice(onDemandPriceStr) * constants.SpotInstanceAssumedDiscountRatio, // Spot price not available in current CSV
	}

	return info, prices
}

// parseAzureRecord parses a single Azure CSV record
func parseAzureRecord(record []string) (types.GPUNodeInstanceInfo, [3]float64) {
	instanceType := record[1]     // API Name
	memory := record[2]           // Instance Memory
	gpuSpec := record[3]          // GPUs (e.g., "8X V100 (NVlink)")
	onDemandPriceStr := record[4] // Linux On Demand cost
	reservedPriceStr := record[5] // Linux Reserved cost
	spotPriceStr := record[6]     // Linux Spot cost
	gpuMemory := record[11]       // GPU memory(per gpu)
	gpuMemoryInt := parseMemory(gpuMemory)

	// Parse GPU info from spec
	gpuCount, gpuModel := parseAzureGPUSpec(gpuSpec)

	info := types.GPUNodeInstanceInfo{
		InstanceType:        instanceType,
		MemoryGiB:           parseMemory(memory), // Now Azure has memory info
		VRAMGigabytesPerGPU: gpuMemoryInt,        // Not provided in Azure CSV
		GPUModel:            gpuModel,
		GPUCount:            gpuCount,
	}

	prices := [3]float64{
		parsePrice(onDemandPriceStr),
		parsePrice(reservedPriceStr),
		parsePrice(spotPriceStr),
	}

	return info, prices
}

// parsePrice parses price string like "$21.5000 hourly" or "unavailable"
func parsePrice(priceStr string) float64 {
	if strings.Contains(priceStr, "unavailable") {
		return 0.0
	}

	// Remove $ and " hourly" parts
	priceStr = strings.ReplaceAll(priceStr, "$", "")
	priceStr = strings.ReplaceAll(priceStr, " hourly", "")

	if price, err := strconv.ParseFloat(priceStr, 64); err == nil {
		return price
	}

	return 0.0
}

// parseGPUCount parses GPU count from string like "16" or "8"
func parseGPUCount(countStr string) int32 {
	if count, err := strconv.ParseInt(countStr, 10, 32); err == nil {
		return int32(count)
	}
	return 0
}

func parseCPUCount(countStr string) int32 {
	countStr = strings.ReplaceAll(countStr, "vCPUs", "")
	countStr = strings.TrimSpace(countStr)
	if count, err := strconv.ParseInt(countStr, 10, 32); err == nil {
		return int32(count)
	}
	return 0
}

// parseMemory parses memory string like "512 GiB" or "2048 GiB"
func parseMemory(memoryStr string) int32 {

	memoryStr = strings.ReplaceAll(memoryStr, " GiB", "")
	memoryStr = strings.ReplaceAll(memoryStr, "GB", "")
	memoryStr = strings.TrimSpace(memoryStr)

	if memory, err := strconv.ParseInt(memoryStr, 10, 32); err == nil {
		return int32(memory)
	}
	return 0
}

// isFractionalGPUCount checks if the GPU specification contains fractional GPU count
func isFractionalGPUCount(gpuSpec string) bool {
	// Check for common fractional patterns in Azure GPU specs
	fractionalPatterns := []string{
		"/",   // Direct fraction like "1/2", "1/3", "1/4", "1/6", "1/8"
		"th ", // Like "1/8th MI25"
	}

	for _, pattern := range fractionalPatterns {
		if strings.Contains(gpuSpec, pattern) {
			return true
		}
	}

	return false
}

// parseAzureGPUSpec parses Azure GPU specification like "8X V100 (NVlink)" or "1X H100"
// Returns (0, "") for fractional GPU counts to indicate they should be filtered out
func parseAzureGPUSpec(gpuSpec string) (int32, string) {
	// First check if this is a fractional GPU count that should be filtered out
	if isFractionalGPUCount(gpuSpec) {
		return 0, "" // Return 0 count to indicate this should be filtered
	}

	// Use regex to extract count and model
	re := regexp.MustCompile(`(\d+)[xX]\s*([^(]+)`)
	matches := re.FindStringSubmatch(gpuSpec)

	if len(matches) >= 3 {
		count, _ := strconv.ParseInt(matches[1], 10, 32)
		model := strings.TrimSpace(matches[2])

		// Clean up model name
		model = strings.ReplaceAll(model, " ", "")

		return int32(count), model
	}
	return 0, ""
}

// GetPricing gets the pricing for the instanceType, capacityType
func (p *StaticPricingProvider) GetPricing(instanceType string, capacityType tfv1.CapacityTypeEnum, region string) (float64, bool) {
	// TODO: region factor should be considered in future

	// Check AWS instances first
	if info, exists := globalAWSGPUInstanceData[instanceType]; exists {
		switch capacityType {
		case tfv1.CapacityTypeOnDemand:
			return info.onDemandPrice, true
		case tfv1.CapacityTypeReserved:
			return info.reservedPrice, true
		case tfv1.CapacityTypeSpot:
			return info.onDemandPrice, true // not support spot price for now
		}
	}

	// Check Azure instances
	if info, exists := globalAzureGPUInstanceData[instanceType]; exists {
		switch capacityType {
		case tfv1.CapacityTypeOnDemand:
			return info.onDemandPrice, true
		case tfv1.CapacityTypeReserved:
			return info.reservedPrice, true
		case tfv1.CapacityTypeSpot:
			return info.onDemandPrice, true // not support spot price for now
		}
	}

	return 0.0, false
}

// GetGPUNodeInstanceTypeInfoByInstance gets the gpu info for the instanceType, region
func (p *StaticPricingProvider) GetGPUNodeInstanceTypeInfoByInstance(instanceType string, region string) (types.GPUNodeInstanceInfo, bool) {
	<-readyCh
	var result types.GPUNodeInstanceInfo
	found := false

	// Check AWS instances first
	if info, exists := globalAWSGPUInstanceData[instanceType]; exists {
		if tflopsMap != nil {
			tflops := tflopsMap[info.GPUNodeInstanceInfo.GPUModel]
			info.GPUNodeInstanceInfo.FP16TFlopsPerGPU = tflops.Fp16TFlops.AsApproximateFloat64()
		}
		result = info.GPUNodeInstanceInfo
		found = true
	}

	// Check Azure instances
	if info, exists := globalAzureGPUInstanceData[instanceType]; exists {
		if tflopsMap != nil {
			tflops := tflopsMap[info.GPUNodeInstanceInfo.GPUModel]
			info.GPUNodeInstanceInfo.FP16TFlopsPerGPU = tflops.Fp16TFlops.AsApproximateFloat64()
		}
		result = info.GPUNodeInstanceInfo
		found = true
	}

	return result, found
}

// GetRegionalGPUNodeInstanceTypes implements PricingProvider interface
func (p *StaticPricingProvider) GetRegionalGPUNodeInstanceTypes(region string) ([]types.GPUNodeInstanceInfo, bool) {
	<-readyCh
	// Pre-allocate slice with estimated capacity
	instanceTypes := make([]types.GPUNodeInstanceInfo, 0, len(globalAWSGPUInstanceData))

	// Collect all instance types from AWS
	for instanceType := range globalAWSGPUInstanceData {
		instanceTypes = append(instanceTypes, globalAWSGPUInstanceData[instanceType].GPUNodeInstanceInfo)
	}

	// region only support aws now
	// for instanceType := range globalAzureGPUInstanceData {
	// 	instanceTypes = append(instanceTypes, globalAzureGPUInstanceData[instanceType].GPUNodeInstanceInfo)
	// }

	return instanceTypes, len(instanceTypes) > 0
}
