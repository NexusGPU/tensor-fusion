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
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"regexp"
	"runtime"

	"strconv"
	"strings"

	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	"sigs.k8s.io/yaml"
)

const (
	providerAWS   = "aws"
	providerAzure = "azure"
)

// GPU architecture mapping tables
var (
	nvidiaArchitectures = map[string]types.GPUArchitectureEnum{
		"V100":         types.GPUArchitectureNvidiaVolta,
		"TESLA V100":   types.GPUArchitectureNvidiaVolta,
		"T4":           types.GPUArchitectureNvidiaTuring,
		"RTX 20":       types.GPUArchitectureNvidiaTuring,
		"RTX20":        types.GPUArchitectureNvidiaTuring,
		"A100":         types.GPUArchitectureNvidiaAmpere,
		"A40":          types.GPUArchitectureNvidiaAmpere,
		"A10":          types.GPUArchitectureNvidiaAmpere,
		"A6000":        types.GPUArchitectureNvidiaAmpere,
		"RTX 30":       types.GPUArchitectureNvidiaAmpere,
		"RTX30":        types.GPUArchitectureNvidiaAmpere,
		"A800":         types.GPUArchitectureNvidiaAmpere,
		"L4":           types.GPUArchitectureNvidiaAdaLovelace,
		"L40":          types.GPUArchitectureNvidiaAdaLovelace,
		"RTX 40":       types.GPUArchitectureNvidiaAdaLovelace,
		"RTX40":        types.GPUArchitectureNvidiaAdaLovelace,
		"RTX 6000 ADA": types.GPUArchitectureNvidiaAdaLovelace,
		"H100":         types.GPUArchitectureNvidiaHopper,
		"H200":         types.GPUArchitectureNvidiaHopper,
		"H800":         types.GPUArchitectureNvidiaHopper,
		"H20":          types.GPUArchitectureNvidiaHopper,
		"B200":         types.GPUArchitectureNvidiaBlackwell,
		"RTX 50":       types.GPUArchitectureNvidiaBlackwell,
		"RTX50":        types.GPUArchitectureNvidiaBlackwell,
	}

	amdArchitectures = map[string]types.GPUArchitectureEnum{
		"MI25":  types.GPUArchitectureAMDCDNA1,
		"MI50":  types.GPUArchitectureAMDCDNA1,
		"MI60":  types.GPUArchitectureAMDCDNA1,
		"MI100": types.GPUArchitectureAMDCDNA2,
		"MI200": types.GPUArchitectureAMDCDNA2,
		"MI210": types.GPUArchitectureAMDCDNA2,
		"MI250": types.GPUArchitectureAMDCDNA2,
		"MI300": types.GPUArchitectureAMDCDNA3,
		"RX 6":  types.GPUArchitectureAMDRDNA2,
		"RX6":   types.GPUArchitectureAMDRDNA2,
		"RX 7":  types.GPUArchitectureAMDRDNA3,
		"RX7":   types.GPUArchitectureAMDRDNA3,
		"V520":  types.GPUArchitectureAMDRDNA2,
		"V620":  types.GPUArchitectureAMDRDNA2,
		"V710":  types.GPUArchitectureAMDRDNA2,
	}
)

// Global data initialized at package load time
var (
	globalAWSGPUInstanceData   map[string]GPUNodeInstanceInfoAndPrice
	globalAzureGPUInstanceData map[string]GPUNodeInstanceInfoAndPrice
	globalGPUModelToFP16TFlops map[string]int32
)

// init initializes the global GPU data when the package is loaded
func init() {
	globalAWSGPUInstanceData = make(map[string]GPUNodeInstanceInfoAndPrice)
	globalAzureGPUInstanceData = make(map[string]GPUNodeInstanceInfoAndPrice)
	globalGPUModelToFP16TFlops = make(map[string]int32)

	// Load GPU model to FP16TFlops mapping from YAML
	loadGPUInfoFromYAML("charts/tensor-fusion/templates/gpu-public-gpu-info.yaml")

	// Load AWS and Azure instance data from CSV files
	loadCSVInstanceData("internal/cloudprovider/pricing/aws_ec2.csv", providerAWS)
	loadCSVInstanceData("internal/cloudprovider/pricing/az.csv", providerAzure)
}

// PricingProvider provides pricing information and calculations for instance types
type PricingProvider interface {
	GetPringcing(instanceType, capacityType types.CapacityTypeEnum) (float64, bool)
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

// GPUInfo represents the structure of GPU info from YAML
type GPUInfo struct {
	Model         string  `yaml:"model"`
	FullModelName string  `yaml:"fullModelName"`
	Vendor        string  `yaml:"vendor"`
	CostPerHour   float64 `yaml:"costPerHour"`
	FP16TFlops    int32   `yaml:"fp16TFlops"`
}

func NewStaticPricingProvider() *StaticPricingProvider {
	return &StaticPricingProvider{}
}

// getProjectPath returns the absolute path to a file relative to project root
func getProjectPath(relativePath string) string {
	_, filename, _, _ := runtime.Caller(0)
	currentDir := filepath.Dir(filename)

	// Go up from internal/cloudprovider/pricing to project root
	projectRoot := filepath.Join(currentDir, "..", "..", "..")
	return filepath.Join(projectRoot, relativePath)
}

// loadGPUInfoFromYAML loads GPU model to FP16TFlops mapping from YAML file
func loadGPUInfoFromYAML(path string) {
	yamlFile := getProjectPath(path)

	file, err := os.Open(yamlFile)
	if err != nil {
		fmt.Printf("Error opening YAML file: %v\n", err)
		return
	}
	defer func() {
		if err := file.Close(); err != nil {
			fmt.Printf("Error closing YAML file: %v\n", err)
		}
	}()

	data, err := io.ReadAll(file)
	if err != nil {
		fmt.Printf("Error reading YAML file: %v\n", err)
		return
	}

	// Parse YAML content, extract data section
	var configMap map[string]any
	if err := yaml.Unmarshal(data, &configMap); err != nil {
		fmt.Printf("Error parsing YAML: %v\n", err)
		return
	}

	// Extract gpu-info.yaml content
	gpuInfoStr, ok := configMap["data"].(map[string]any)["gpu-info.yaml"].(string)
	if !ok {
		fmt.Printf("Error extracting gpu-info.yaml from ConfigMap\n")
		return
	}

	// Parse GPU info YAML
	var gpuInfos []GPUInfo
	if err := yaml.Unmarshal([]byte(gpuInfoStr), &gpuInfos); err != nil {
		fmt.Printf("Error parsing GPU info YAML: %v\n", err)
		return
	}

	// Build mapping from GPU model to FP16TFlops
	for _, gpu := range gpuInfos {
		globalGPUModelToFP16TFlops[gpu.Model] = gpu.FP16TFlops
		// Also add full model name as key for better matching
		globalGPUModelToFP16TFlops[gpu.FullModelName] = gpu.FP16TFlops
	}
}

// loadCSVInstanceData loads instance data from CSV files (unified for AWS and Azure)
func loadCSVInstanceData(relativePath, provider string) {
	csvFile := getProjectPath(relativePath)

	file, err := os.Open(csvFile)
	if err != nil {
		fmt.Printf("Error opening %s CSV file: %v\n", provider, err)
		return
	}
	defer func() {
		if err := file.Close(); err != nil {
			fmt.Printf("Error closing %s CSV file: %v\n", provider, err)
		}
	}()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Printf("Error reading %s CSV file: %v\n", provider, err)
		return
	}

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
		instanceInfoAndPrice := GPUNodeInstanceInfoAndPrice{
			GPUNodeInstanceInfo: instanceInfo,
			onDemandPrice:       prices[0],
			reservedPrice:       prices[1],
			spotPrice:           prices[2],
		}

		if provider == providerAWS {
			globalAWSGPUInstanceData[instanceInfo.InstanceType] = instanceInfoAndPrice
		} else {
			globalAzureGPUInstanceData[instanceInfo.InstanceType] = instanceInfoAndPrice
		}
	}
}

// parseAWSRecord parses a single AWS CSV record
func parseAWSRecord(record []string) (types.GPUNodeInstanceInfo, [3]float64) {
	instanceType := record[1]     // API Name
	memory := record[2]           // Instance Memory
	gpuCountStr := record[3]      // GPUs
	gpuModel := record[4]         // GPU model
	gpuMemory := record[5]        // GPU memory
	onDemandPriceStr := record[8] // On Demand
	reservedPriceStr := record[9] // Linux Reserved cost
	totalMemory := parseMemory(gpuMemory)
	gpuCount := parseGPUCount(gpuCountStr)
	var perGPUMemory int32
	if gpuCount != 0 {
		perGPUMemory = totalMemory / gpuCount
	}

	info := types.GPUNodeInstanceInfo{
		InstanceType:        instanceType,
		CostPerHour:         parsePrice(onDemandPriceStr),
		MemoryGiB:           parseMemory(memory),
		FP16TFlopsPerGPU:    getFP16TFlops(gpuModel),
		VRAMGigabytesPerGPU: perGPUMemory,
		GPUModel:            gpuModel,
		GPUCount:            gpuCount,
		GPUArchitecture:     parseGPUArchitecture(gpuModel),
	}

	prices := [3]float64{
		parsePrice(onDemandPriceStr),
		parsePrice(reservedPriceStr),
		0, // Spot price not available in current CSV
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
		CostPerHour:         parsePrice(onDemandPriceStr),
		MemoryGiB:           parseMemory(memory), // Now Azure has memory info
		FP16TFlopsPerGPU:    getFP16TFlops(gpuModel),
		VRAMGigabytesPerGPU: gpuMemoryInt, // Not provided in Azure CSV
		GPUModel:            gpuModel,
		GPUCount:            gpuCount,
		GPUArchitecture:     parseGPUArchitecture(gpuModel),
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

// parseGPUArchitecture converts GPU model to GPUArchitectureEnum
func parseGPUArchitecture(gpuModel string) types.GPUArchitectureEnum {
	gpuModel = strings.ToUpper(gpuModel)

	// Check NVIDIA architectures first
	if arch := parseNvidiaGPUArchitecture(gpuModel); arch != types.GPUArchitectureNvidiaAmpere {
		return arch
	}

	// Check AMD architectures
	if arch := parseAMDGPUArchitecture(gpuModel); arch != types.GPUArchitectureNvidiaAmpere {
		return arch
	}

	// Default to Ampere for unknown GPUs
	return types.GPUArchitectureNvidiaAmpere
}

// parseNvidiaGPUArchitecture parses NVIDIA GPU architectures
func parseNvidiaGPUArchitecture(gpuModel string) types.GPUArchitectureEnum {
	for model, arch := range nvidiaArchitectures {
		if strings.Contains(gpuModel, model) {
			return arch
		}
	}

	// Return default if no match found
	return types.GPUArchitectureNvidiaAmpere
}

// parseAMDGPUArchitecture parses AMD GPU architectures
func parseAMDGPUArchitecture(gpuModel string) types.GPUArchitectureEnum {
	for model, arch := range amdArchitectures {
		if strings.Contains(gpuModel, model) {
			return arch
		}
	}

	// Return default if no match found (indicates non-AMD GPU)
	return types.GPUArchitectureNvidiaAmpere
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

// getFP16TFlops gets FP16TFlops for a GPU model
func getFP16TFlops(gpuModel string) int32 {
	// Direct lookup
	if tflops, ok := globalGPUModelToFP16TFlops[gpuModel]; ok {
		return tflops
	}

	// Try common variations
	variations := []string{
		strings.ToUpper(gpuModel),
		strings.ToLower(gpuModel),
		strings.ReplaceAll(gpuModel, " ", ""),
		strings.ReplaceAll(gpuModel, "NVIDIA ", ""),
		strings.ReplaceAll(gpuModel, "Tesla ", ""),
		strings.ReplaceAll(gpuModel, "GRID ", ""),
		strings.ReplaceAll(gpuModel, "AWS ", ""),
	}

	for _, variation := range variations {
		if tflops, ok := globalGPUModelToFP16TFlops[variation]; ok {
			return tflops
		}
	}
	// Default values for common GPUs if not found in YAML
	return 0
}

// GetPringcing gets the pricing for the instanceType, capacityType
func (p *StaticPricingProvider) GetPringcing(instanceType string, capacityType types.CapacityTypeEnum) (float64, bool) {
	// Check AWS instances first
	if info, exists := globalAWSGPUInstanceData[instanceType]; exists {
		switch capacityType {
		case types.CapacityTypeOnDemand:
			return info.onDemandPrice, true
		case types.CapacityTypeReserved:
			return info.reservedPrice, true
		case types.CapacityTypeSpot:
			return info.onDemandPrice, true // not support spot price for now
		}
	}

	// Check Azure instances
	if info, exists := globalAzureGPUInstanceData[instanceType]; exists {
		switch capacityType {
		case types.CapacityTypeOnDemand:
			return info.onDemandPrice, true
		case types.CapacityTypeReserved:
			return info.reservedPrice, true
		case types.CapacityTypeSpot:
			return info.onDemandPrice, true // not support spot price for now
		}
	}

	return 0.0, false
}

// GetGPUNodeInstanceTypeInfoByInstance gets the gpu info for the instanceType, region
func (p *StaticPricingProvider) GetGPUNodeInstanceTypeInfoByInstance(instanceType string, region string) ([]types.GPUNodeInstanceInfo, bool) {
	var results []types.GPUNodeInstanceInfo

	// Check AWS instances first
	if info, exists := globalAWSGPUInstanceData[instanceType]; exists {
		results = append(results, info.GPUNodeInstanceInfo)
	}

	// Check Azure instances
	if info, exists := globalAzureGPUInstanceData[instanceType]; exists {
		results = append(results, info.GPUNodeInstanceInfo)
	}

	return results, len(results) > 0
}

// GetGPUNodeInstanceTypeInfo implements PricingProvider interface
func (p *StaticPricingProvider) GetGPUNodeInstanceTypeInfo(region string) ([]types.GPUNodeInstanceInfo, bool) {
	// Pre-allocate slice with estimated capacity
	instanceTypes := make([]types.GPUNodeInstanceInfo, 0, len(globalAWSGPUInstanceData)+len(globalAzureGPUInstanceData))

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
