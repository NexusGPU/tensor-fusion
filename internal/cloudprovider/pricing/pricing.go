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
	"errors"
	"fmt"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	providerAWS   = "aws"
	providerAzure = "azure"
	// providerAlibaba rows use the column layout of the AWS data.
	providerAlibaba = "alibaba"

	// DefaultFirstLoadTimeout bounds how long readers wait for the first load
	// of the pricing tables before they give up with ErrPricingNotLoaded.
	DefaultFirstLoadTimeout = 30 * time.Second

	// maxWarnedKeys caps the "unknown GPU model" log lines kept per table, so
	// lookups with arbitrary names can neither flood the log nor grow memory.
	maxWarnedKeys = 256
)

var (
	// ErrPricingNotLoaded is returned when no pricing table was loaded within
	// the first-load timeout.
	ErrPricingNotLoaded = errors.New("pricing data has not been loaded")
	// ErrInstanceTypeNotFound is returned for an instance type that is not in
	// the loaded tables: either it is not in the embedded instance data, or
	// its GPU model has no hardware metadata in any ProviderConfig.
	ErrInstanceTypeNotFound = errors.New("instance type not found in pricing data")
	// ErrGPUModelUnknown is returned for a GPU model without hardware metadata.
	ErrGPUModelUnknown = errors.New("GPU model is unknown to pricing data")
	// ErrUnsupportedCapacityType is returned for a capacity type other than
	// on-demand, spot and reserved.
	ErrUnsupportedCapacityType = errors.New("unsupported capacity type")
	// ErrPriceUnavailable is returned when the instance type is known but the
	// pricing data has no price for the requested capacity type.
	ErrPriceUnavailable = errors.New("no price available for capacity type")
)

var pricingLog = log.Log.WithName("pricing")

// PriceSource says where an hourly price comes from, so that a figure that was
// derived from an assumption is never passed on as a listed price.
type PriceSource string

const (
	// PriceSourceList is a price the pricing data lists for the instance type
	// and the capacity type.
	PriceSourceList PriceSource = "list"
	// PriceSourceSpotEstimate is not a listed price. It is the listed on-demand
	// price multiplied by an assumed spot factor, for an instance type whose
	// spot price the pricing data does not list.
	PriceSourceSpotEstimate PriceSource = "spot-estimate"
)

// AWSSpotAssumedPriceRatio is the share of the on-demand price an AWS spot
// instance is assumed to cost when the pricing data lists no spot price for
// its instance type. It is an assumption, not a price: actual spot prices
// differ per instance type, availability zone and hour. A price derived from
// it is reported as PriceSourceSpotEstimate.
//
// It has the value of constants.SpotInstanceAssumedDiscountRatio, which other
// code keeps using under that name.
const AWSSpotAssumedPriceRatio = constants.SpotInstanceAssumedDiscountRatio

// CompleteGPUInfo combines GpuInfo with VRAM information from instance data
type CompleteGPUInfo struct {
	*config.GpuInfo
	VRAMGigabytes int32
}

// PricingProvider provides pricing information and calculations for instance types
type PricingProvider interface {
	GetPricing(instanceType string, capacityType tfv1.CapacityTypeEnum, region string) (float64, bool)
	GetRegionalGPUNodeInstanceTypes(region string) ([]types.GPUNodeInstanceInfo, bool)
	GetGPUCapacityByModel(gpuModel string) (resource.Quantity, resource.Quantity, bool)
}

type GPUNodeInstanceInfoAndPrice struct {
	GPUNodeInstanceInfo types.GPUNodeInstanceInfo
	onDemandPrice       float64
	spotPrice           float64
	reservedPrice       float64
	// spotPriceEstimated says that spotPrice is not listed in the pricing data
	// but derived from onDemandPrice with an assumed factor.
	spotPriceEstimated bool
}

// StaticPricingProvider implements PricingProvider using static pricing data.
// The zero value (and a nil pointer) reads the process-wide tables that
// SetTflopsMapAndInitGPUPricingInfo maintains.
type StaticPricingProvider struct {
	// store is only set by tests that need tables isolated from the
	// process-wide ones.
	store *pricingStore
}

func NewStaticPricingProvider() *StaticPricingProvider {
	return &StaticPricingProvider{}
}

func (p *StaticPricingProvider) tables() *pricingStore {
	if p == nil || p.store == nil {
		return defaultStore
	}
	return p.store
}

//go:embed pricing-data/aws-gpu.csv
var awsCSV string

//go:embed pricing-data/azure-gpu.csv
var azureCSV string

// alibabaCSV lists Alibaba Cloud ECS GPU instance types with pay-as-you-go
// list prices of us-east-1, converted from CNY to USD. Spot prices are not
// listed, so a spot rate is an estimate, as for AWS.
//
//go:embed pricing-data/alibaba-gpu.csv
var alibabaCSV string

// csvInstance is one row of the embedded instance data, before it is joined
// with the GPU model info.
type csvInstance struct {
	info   types.GPUNodeInstanceInfo
	prices [3]float64 // onDemand, reserved, spot
	// spotEstimated says that the spot price is an estimate, not a listed one.
	spotEstimated bool
}

var (
	csvOnce    sync.Once
	csvAWS     []csvInstance
	csvAzure   []csvInstance
	csvAlibaba []csvInstance
)

// embeddedInstanceData parses the embedded CSV files once. The rows never
// change at runtime; only the GPU model info they are joined with does.
func embeddedInstanceData() (aws []csvInstance, azure []csvInstance) {
	csvOnce.Do(func() {
		csvAWS = parseCSVInstanceData([]byte(awsCSV), providerAWS)
		csvAzure = parseCSVInstanceData([]byte(azureCSV), providerAzure)
		csvAlibaba = parseCSVInstanceData([]byte(alibabaCSV), providerAlibaba)
	})
	return csvAWS, csvAzure
}

// embeddedAlibabaInstanceData returns the Alibaba Cloud rows of the embedded data.
func embeddedAlibabaInstanceData() []csvInstance {
	embeddedInstanceData()
	return csvAlibaba
}

// pricingSnapshot is one complete, immutable generation of the pricing tables.
// A reload builds a new snapshot and publishes it with a single pointer swap,
// so a reader works on one consistent generation and never sees a table that
// is still being filled.
type pricingSnapshot struct {
	aws     map[string]GPUNodeInstanceInfoAndPrice
	azure   map[string]GPUNodeInstanceInfoAndPrice
	alibaba map[string]GPUNodeInstanceInfoAndPrice
	// awsInstanceTypes holds the AWS instance types sorted by name.
	awsInstanceTypes []types.GPUNodeInstanceInfo
	// alibabaInstanceTypes holds the Alibaba Cloud instance types sorted by name.
	alibabaInstanceTypes []types.GPUNodeInstanceInfo
	// gpuInfo is keyed by both the short and the full GPU model name.
	gpuInfo map[string]*CompleteGPUInfo
	// skippedModels lists the GPU models of instance rows that were left out
	// because no hardware metadata describes them.
	skippedModels []string
	// signature identifies the GPU model info the snapshot was built from.
	signature string
	warned    *warnSet
}

// warnSet remembers which keys were already logged.
type warnSet struct {
	keys  sync.Map
	count atomic.Int32
}

// first reports whether key is seen for the first time.
func (w *warnSet) first(key string) bool {
	if w.count.Load() >= maxWarnedKeys {
		return false
	}
	if _, loaded := w.keys.LoadOrStore(key, struct{}{}); loaded {
		return false
	}
	w.count.Add(1)
	return true
}

// pricingStore owns the published snapshot. Readers load the pointer without
// taking a lock, so they neither wait for a reload nor delay one; reloadMu only
// serialises writers.
type pricingStore struct {
	current atomic.Pointer[pricingSnapshot]

	reloadMu sync.Mutex
	// onReload runs under reloadMu with the GPU info of every load.
	onReload func([]config.GpuInfo)

	// ready is closed by the first completed load.
	ready     chan struct{}
	readyOnce sync.Once

	// The wait for the first load is bounded once per store, starting when the
	// first reader arrives: after the deadline readers fail immediately instead
	// of each one blocking for the full timeout again.
	firstLoadTimeout time.Duration
	deadlineOnce     sync.Once
	deadline         time.Time
	notLoadedLogged  atomic.Bool
}

func newPricingStore(firstLoadTimeout time.Duration) *pricingStore {
	return &pricingStore{
		ready:            make(chan struct{}),
		firstLoadTimeout: firstLoadTimeout,
	}
}

// defaultStore backs the exported API of this package.
var defaultStore = func() *pricingStore {
	s := newPricingStore(DefaultFirstLoadTimeout)
	// Partition templates derive from the same GPU info and are refreshed
	// together with the pricing tables.
	s.onReload = gpuallocator.LoadPartitionTemplatesFromConfig
	return s
}()

// SetTflopsMapAndInitGPUPricingInfo rebuilds the pricing tables (and the
// partition templates) from the complete set of GPU model info. It is called
// at start-up and again whenever a ProviderConfig is created, changed or
// deleted; each call replaces the previous tables as a whole, so models that
// are no longer configured disappear and changed TFlops values take effect.
func SetTflopsMapAndInitGPUPricingInfo(ctx context.Context, gpuInfos *[]config.GpuInfo) {
	if gpuInfos == nil {
		log.FromContext(ctx).Info("gpuInfos is empty, check public-gpu-info config file")
		return
	}
	defaultStore.load(ctx, *gpuInfos)
}

// load builds a snapshot from gpuInfos and publishes it. An empty gpuInfos is
// a valid load (no ProviderConfig yet): it publishes empty tables, so readers
// get "not found" instead of waiting.
func (s *pricingStore) load(ctx context.Context, gpuInfos []config.GpuInfo) {
	s.reloadMu.Lock()
	defer s.reloadMu.Unlock()

	if s.onReload != nil {
		s.onReload(gpuInfos)
	}

	snap := buildSnapshot(gpuInfos)
	prev := s.current.Load()
	changed := prev == nil || prev.signature != snap.signature ||
		!slices.Equal(prev.skippedModels, snap.skippedModels)
	if !changed {
		// Same content as before: keep the record of what was already logged,
		// so periodic re-syncs do not repeat the "unknown GPU model" lines.
		snap.warned = prev.warned
	}
	s.current.Store(snap)
	s.readyOnce.Do(func() { close(s.ready) })

	logger := log.FromContext(ctx)
	if !changed {
		logger.V(6).Info("GPU pricing tables reloaded without changes")
		return
	}
	logger.Info("GPU pricing tables rebuilt",
		"gpuModels", len(gpuInfos), "awsInstanceTypes", len(snap.aws), "azureInstanceTypes", len(snap.azure),
		"alibabaInstanceTypes", len(snap.alibaba))
	if len(snap.skippedModels) > 0 {
		logger.Info("instance types skipped because their GPU model has no hardware metadata in any ProviderConfig",
			"gpuModels", snap.skippedModels)
	}
}

// snapshot returns the published tables. Before the first load it waits, up
// to the store's first-load deadline, and then fails with ErrPricingNotLoaded.
func (s *pricingStore) snapshot() (*pricingSnapshot, error) {
	if snap := s.current.Load(); snap != nil {
		return snap, nil
	}

	s.deadlineOnce.Do(func() { s.deadline = time.Now().Add(s.firstLoadTimeout) })
	if remaining := time.Until(s.deadline); remaining > 0 {
		timer := time.NewTimer(remaining)
		defer timer.Stop()
		select {
		case <-s.ready:
		case <-timer.C:
		}
	}

	if snap := s.current.Load(); snap != nil {
		return snap, nil
	}
	return nil, fmt.Errorf("%w: no load completed within %s of the first lookup", ErrPricingNotLoaded, s.firstLoadTimeout)
}

// logNotLoaded reports a lookup that failed because nothing was loaded. Only
// the callers of the bool-returning lookups need it (they cannot return the
// error), and one line is enough: every later lookup fails the same way.
func (s *pricingStore) logNotLoaded(err error) {
	if errors.Is(err, ErrPricingNotLoaded) && s.notLoadedLogged.CompareAndSwap(false, true) {
		pricingLog.Error(err, "pricing tables are not loaded, lookups find nothing until they are")
	}
}

// buildSnapshot joins the embedded instance data with the GPU model info.
func buildSnapshot(gpuInfos []config.GpuInfo) *pricingSnapshot {
	gpuInfoByModel := make(map[string]*CompleteGPUInfo, len(gpuInfos)*2)
	for i := range gpuInfos {
		gpuInfo := gpuInfos[i]
		if gpuInfoByModel[gpuInfo.FullModelName] != nil {
			continue
		}
		completeInfo := &CompleteGPUInfo{GpuInfo: &gpuInfo}
		gpuInfoByModel[gpuInfo.FullModelName] = completeInfo
		gpuInfoByModel[gpuInfo.Model] = completeInfo
	}

	awsRows, azureRows := embeddedInstanceData()
	skipped := map[string]struct{}{}
	snap := &pricingSnapshot{
		aws:     joinInstanceData(awsRows, gpuInfoByModel, skipped),
		azure:   joinInstanceData(azureRows, gpuInfoByModel, skipped),
		alibaba: joinInstanceData(embeddedAlibabaInstanceData(), gpuInfoByModel, skipped),
		gpuInfo: gpuInfoByModel,
		warned:  &warnSet{},
	}

	snap.awsInstanceTypes = sortedInstanceTypes(snap.aws)
	snap.alibabaInstanceTypes = sortedInstanceTypes(snap.alibaba)

	snap.skippedModels = make([]string, 0, len(skipped))
	for model := range skipped {
		snap.skippedModels = append(snap.skippedModels, model)
	}
	sort.Strings(snap.skippedModels)

	signature := make([]string, 0, len(gpuInfoByModel))
	for model, gpuInfo := range gpuInfoByModel {
		signature = append(signature, model+"="+gpuInfo.FullModelName+"/"+gpuInfo.Fp16TFlops.String())
	}
	sort.Strings(signature)
	snap.signature = strings.Join(signature, ";")
	return snap
}

func sortedInstanceTypes(table map[string]GPUNodeInstanceInfoAndPrice) []types.GPUNodeInstanceInfo {
	instanceTypes := make([]types.GPUNodeInstanceInfo, 0, len(table))
	for _, instance := range table {
		instanceTypes = append(instanceTypes, instance.GPUNodeInstanceInfo)
	}
	sort.Slice(instanceTypes, func(i, j int) bool {
		return instanceTypes[i].InstanceType < instanceTypes[j].InstanceType
	})
	return instanceTypes
}

// joinInstanceData keeps the instance rows whose GPU model is described by the
// GPU model info and records the models of the rows it leaves out.
func joinInstanceData(
	rows []csvInstance, gpuInfoByModel map[string]*CompleteGPUInfo, skipped map[string]struct{},
) map[string]GPUNodeInstanceInfoAndPrice {
	result := make(map[string]GPUNodeInstanceInfoAndPrice, len(rows))
	for _, row := range rows {
		instanceInfo := row.info
		gpuInfo := gpuInfoByModel[instanceInfo.GPUModel]
		if gpuInfo == nil || gpuInfo.GpuInfo == nil {
			// not configured, skip
			skipped[instanceInfo.GPUModel] = struct{}{}
			continue
		}
		instanceInfo.FP16TFlopsPerGPU = gpuInfo.Fp16TFlops.AsApproximateFloat64()

		// Fill VRAM information if not already set
		if gpuInfo.VRAMGigabytes == 0 {
			gpuInfo.VRAMGigabytes = instanceInfo.VRAMGigabytesPerGPU
		}

		result[instanceInfo.InstanceType] = GPUNodeInstanceInfoAndPrice{
			GPUNodeInstanceInfo: instanceInfo,
			onDemandPrice:       row.prices[0],
			reservedPrice:       row.prices[1],
			spotPrice:           row.prices[2],
			spotPriceEstimated:  row.spotEstimated,
		}
	}
	return result
}

// parseCSVInstanceData parses the instance data of one provider
func parseCSVInstanceData(data []byte, provider string) []csvInstance {
	reader := csv.NewReader(bytes.NewReader(data))
	records, err := reader.ReadAll()
	if err != nil {
		pricingLog.Error(err, "Error reading CSV file", "provider", provider)
		return nil
	}

	rows := make([]csvInstance, 0, len(records))
	// Parse CSV records (skip header)
	for i, record := range records {
		if i == 0 {
			continue // Skip header
		}

		// Determine required record length based on provider
		if (provider == providerAWS || provider == providerAlibaba) && len(record) < 13 {
			continue // AWS needs at least 13 fields
		}
		if provider == providerAzure && len(record) < 12 {
			continue // Azure needs at least 12 fields
		}

		var instanceInfo types.GPUNodeInstanceInfo
		var prices [3]float64 // onDemand, reserved, spot
		var spotEstimated bool

		switch provider {
		case providerAWS, providerAlibaba:
			instanceInfo, prices, spotEstimated = parseAWSRecord(record)
		case providerAzure:
			instanceInfo, prices = parseAzureRecord(record)
			// Filter out Azure instances with fractional GPU counts (GPUCount = 0)
			if instanceInfo.GPUCount == 0 {
				continue // Skip this record, don't store it in memory
			}
		default:
			continue
		}

		rows = append(rows, csvInstance{info: instanceInfo, prices: prices, spotEstimated: spotEstimated})
	}

	pricingLog.V(6).Info("Parsed GPU instance types", "count", len(rows), "provider", provider)
	return rows
}

// parseAWSRecord parses a single AWS CSV record. The last result says whether
// the spot price is an estimate rather than a price listed in the record.
func parseAWSRecord(record []string) (types.GPUNodeInstanceInfo, [3]float64, bool) {
	instanceType := record[1]      // API Name
	memory := record[2]            // Instance Memory
	cpuCountStr := record[3]       // vCPUs
	gpuCountStr := record[4]       // GPUs
	gpuModel := record[5]          // GPU model
	gpuMemory := record[6]         // GPU memory
	cpuArchStr := record[7]        // CPU architecture, x86_64 or arm64
	onDemandPriceStr := record[9]  // On Demand
	reservedPriceStr := record[10] // Linux Reserved cost
	spotPriceStr := record[11]     // Linux Spot Minimum cost
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

	// A spot price listed in the record is used as it is. The embedded AWS data
	// currently says "unavailable" in every row, so the spot rate is estimated
	// from the on-demand price and marked as an estimate.
	onDemandPrice := parsePrice(onDemandPriceStr)
	spotPrice := parsePrice(spotPriceStr)
	spotEstimated := false
	if spotPrice <= 0 {
		spotPrice = onDemandPrice * AWSSpotAssumedPriceRatio
		spotEstimated = spotPrice > 0
	}

	prices := [3]float64{
		onDemandPrice,
		parsePrice(reservedPriceStr),
		spotPrice,
	}

	return info, prices, spotEstimated
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

// lookupInstance finds an instance type: AWS first, then Azure, then Alibaba Cloud.
func (s *pricingStore) lookupInstance(instanceType string) (GPUNodeInstanceInfoAndPrice, error) {
	snap, err := s.snapshot()
	if err != nil {
		return GPUNodeInstanceInfoAndPrice{}, err
	}
	if info, exists := snap.aws[instanceType]; exists {
		return info, nil
	}
	if info, exists := snap.azure[instanceType]; exists {
		return info, nil
	}
	if info, exists := snap.alibaba[instanceType]; exists {
		return info, nil
	}
	return GPUNodeInstanceInfoAndPrice{}, fmt.Errorf("%w: %q", ErrInstanceTypeNotFound, instanceType)
}

// LookupPricing returns the price GetPricing reports, with the reason when
// there is none: ErrPricingNotLoaded, ErrInstanceTypeNotFound or
// ErrUnsupportedCapacityType.
//
// A spot request is answered with the on-demand list price, as node
// provisioning has always compared instance types by it; use
// HourlyPriceForCapacityType or HourlyPriceWithSource for the rate of the
// capacity type itself.
func (p *StaticPricingProvider) LookupPricing(instanceType string, capacityType tfv1.CapacityTypeEnum, region string) (float64, error) {
	// TODO: region factor should be considered in future
	info, err := p.tables().lookupInstance(instanceType)
	if err != nil {
		return 0.0, err
	}
	var price float64
	switch capacityType {
	case tfv1.CapacityTypeOnDemand:
		price = info.onDemandPrice
	case tfv1.CapacityTypeReserved:
		price = info.reservedPrice
	case tfv1.CapacityTypeSpot:
		price = info.onDemandPrice // not support spot price for now
	default:
		return 0.0, fmt.Errorf("%w: %q", ErrUnsupportedCapacityType, capacityType)
	}
	// A price the data lists as unavailable is not a price of zero: an
	// instance type without one must not win a comparison as the cheapest, it
	// is not offered for that capacity type at all.
	if price <= 0 {
		return 0.0, fmt.Errorf("%w: instance type %q, capacity type %q", ErrPriceUnavailable, instanceType, capacityType)
	}
	return price, nil
}

// GetPricing gets the pricing for the instanceType, capacityType
func (p *StaticPricingProvider) GetPricing(instanceType string, capacityType tfv1.CapacityTypeEnum, region string) (float64, bool) {
	price, err := p.LookupPricing(instanceType, capacityType, region)
	if err != nil {
		p.tables().logNotLoaded(err)
		return 0.0, false
	}
	return price, true
}

// HourlyPriceForCapacityType returns the hourly rate that applies to an
// instance type bought as capacityType: the on-demand, spot or reserved price.
// It fails with ErrPriceUnavailable when the pricing data has no price for
// that capacity type, so a caller never mistakes a missing price for a free
// instance and can decide on its own fallback.
//
// The rate is not always a listed price: see HourlyPriceWithSource, which
// also tells where the figure comes from.
func (p *StaticPricingProvider) HourlyPriceForCapacityType(instanceType string, capacityType tfv1.CapacityTypeEnum, region string) (float64, error) {
	price, _, err := p.HourlyPriceWithSource(instanceType, capacityType, region)
	return price, err
}

// HourlyPriceWithSource returns the rate HourlyPriceForCapacityType returns,
// together with where it comes from. On-demand and reserved rates are listed
// prices, and so is a spot rate the pricing data lists (the Azure data does).
// The AWS data lists no spot prices: an AWS spot rate is the on-demand price
// multiplied by AWSSpotAssumedPriceRatio and is reported as
// PriceSourceSpotEstimate. The source is empty when there is an error.
func (p *StaticPricingProvider) HourlyPriceWithSource(
	instanceType string, capacityType tfv1.CapacityTypeEnum, region string,
) (float64, PriceSource, error) {
	// TODO: region factor should be considered in future
	tables := p.tables()
	info, err := tables.lookupInstance(instanceType)
	if errors.Is(err, ErrInstanceTypeNotFound) {
		// A node names its instance type the way the cloud API does
		// ("Standard_NC24s_v3"); the Azure price list names the same type
		// "NC24s v3".
		if listed := azureListName(instanceType); listed != instanceType {
			info, err = tables.lookupInstance(listed)
		}
	}
	if err != nil {
		return 0.0, "", err
	}
	var price float64
	source := PriceSourceList
	switch capacityType {
	case tfv1.CapacityTypeOnDemand:
		price = info.onDemandPrice
	case tfv1.CapacityTypeReserved:
		price = info.reservedPrice
	case tfv1.CapacityTypeSpot:
		price = info.spotPrice
		if info.spotPriceEstimated {
			source = PriceSourceSpotEstimate
		}
	default:
		return 0.0, "", fmt.Errorf("%w: %q", ErrUnsupportedCapacityType, capacityType)
	}
	if price <= 0 {
		return 0.0, "", fmt.Errorf("%w: instance type %q, capacity type %q", ErrPriceUnavailable, instanceType, capacityType)
	}
	return price, source, nil
}

// azureListName turns an Azure VM size as the node.kubernetes.io/instance-type
// label spells it into the name the Azure price list uses for it.
func azureListName(instanceType string) string {
	trimmed, isAzure := strings.CutPrefix(instanceType, "Standard_")
	if !isAzure {
		return instanceType
	}
	return strings.ReplaceAll(trimmed, "_", " ")
}

// ParseCapacityType maps a capacity type as written on a node (the
// karpenter.sh/capacity-type label: on-demand, spot, reserved) or in a
// GPUNodeClaim (OnDemand, Spot, Reserved) to the API enum.
func ParseCapacityType(value string) (tfv1.CapacityTypeEnum, bool) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "on-demand", "ondemand":
		return tfv1.CapacityTypeOnDemand, true
	case "spot", "preemptive", "preemptible":
		return tfv1.CapacityTypeSpot, true
	case "reserved":
		return tfv1.CapacityTypeReserved, true
	}
	return "", false
}

// LookupGPUNodeInstanceTypeInfo gets the gpu info for the instanceType, region
func (p *StaticPricingProvider) LookupGPUNodeInstanceTypeInfo(instanceType string, region string) (types.GPUNodeInstanceInfo, error) {
	info, err := p.tables().lookupInstance(instanceType)
	if err != nil {
		return types.GPUNodeInstanceInfo{}, err
	}
	// FP16TFlopsPerGPU was filled when the snapshot was built, from the same
	// GPU model info generation as the rest of the snapshot.
	return info.GPUNodeInstanceInfo, nil
}

// GetGPUNodeInstanceTypeInfoByInstance gets the gpu info for the instanceType, region
func (p *StaticPricingProvider) GetGPUNodeInstanceTypeInfoByInstance(instanceType string, region string) (types.GPUNodeInstanceInfo, bool) {
	info, err := p.LookupGPUNodeInstanceTypeInfo(instanceType, region)
	if err != nil {
		p.tables().logNotLoaded(err)
		return types.GPUNodeInstanceInfo{}, false
	}
	return info, true
}

// LookupRegionalGPUNodeInstanceTypes returns the known instance types, sorted
// by name. The list is empty while no ProviderConfig describes any GPU model.
func (p *StaticPricingProvider) LookupRegionalGPUNodeInstanceTypes(region string) ([]types.GPUNodeInstanceInfo, error) {
	snap, err := p.tables().snapshot()
	if err != nil {
		return nil, err
	}
	// region only support aws now
	// The copy keeps callers from changing the published snapshot.
	return slices.Clone(snap.awsInstanceTypes), nil
}

// LookupAlibabaGPUNodeInstanceTypes returns the known Alibaba Cloud ECS
// instance types, sorted by name.
func (p *StaticPricingProvider) LookupAlibabaGPUNodeInstanceTypes() ([]types.GPUNodeInstanceInfo, error) {
	snap, err := p.tables().snapshot()
	if err != nil {
		return nil, err
	}
	return slices.Clone(snap.alibabaInstanceTypes), nil
}

// GetRegionalGPUNodeInstanceTypes implements PricingProvider interface
func (p *StaticPricingProvider) GetRegionalGPUNodeInstanceTypes(region string) ([]types.GPUNodeInstanceInfo, bool) {
	instanceTypes, err := p.LookupRegionalGPUNodeInstanceTypes(region)
	if err != nil {
		p.tables().logNotLoaded(err)
		return []types.GPUNodeInstanceInfo{}, false
	}
	return instanceTypes, len(instanceTypes) > 0
}

// LookupGPUCapacityByModel gets the full capacity (TFlops and VRAM) for a GPU
// model. A model without hardware metadata yields ErrGPUModelUnknown.
func (p *StaticPricingProvider) LookupGPUCapacityByModel(gpuModel string) (resource.Quantity, resource.Quantity, error) {
	snap, err := p.tables().snapshot()
	if err != nil {
		return resource.Quantity{}, resource.Quantity{}, err
	}

	gpuInfo := snap.gpuInfo[gpuModel]
	if gpuInfo == nil || gpuInfo.GpuInfo == nil {
		// Logged once per model and table generation: the same lookup is
		// repeated for every pod or node that uses the model.
		if snap.warned.first(gpuModel) {
			pricingLog.Info("GPU model has no hardware metadata in any ProviderConfig, treating it as unknown",
				"gpuModel", gpuModel)
		}
		return resource.Quantity{}, resource.Quantity{}, fmt.Errorf("%w: %q", ErrGPUModelUnknown, gpuModel)
	}

	tflops := gpuInfo.Fp16TFlops
	vram := *resource.NewQuantity(int64(gpuInfo.VRAMGigabytes)*constants.GiBToBytes, resource.BinarySI)

	return tflops, vram, nil
}

// GetGPUCapacityByModel gets the full capacity (TFlops and VRAM) for a GPU model
// Returns TFlops, VRAM, and whether found
func (p *StaticPricingProvider) GetGPUCapacityByModel(gpuModel string) (resource.Quantity, resource.Quantity, bool) {
	tflops, vram, err := p.LookupGPUCapacityByModel(gpuModel)
	if err != nil {
		p.tables().logNotLoaded(err)
		return resource.Quantity{}, resource.Quantity{}, false
	}
	return tflops, vram, true
}
