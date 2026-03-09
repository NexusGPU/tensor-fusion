package gpuresources

import (
	"math"
	"slices"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
)

const (
	defaultResourceScoreWeight      = 0.7
	defaultTopologyScoreWeight      = 0.3
	defaultSingleGPUProtectWeight   = 0.6
	defaultPairBandwidthTargetMBps  = 100000.0
	defaultMinValidCombinedWeight   = 1e-6
	defaultTopologyFallbackMaxScore = 100
)

func isNvLinkAwareEnabled(cfg *config.GPUFitConfig) bool {
	return cfg != nil && cfg.EnableNvLinkAware
}

func mergeResourceAndTopologyScore(resourceScore, topologyScore int, cfg *config.GPUFitConfig) int {
	if !isNvLinkAwareEnabled(cfg) {
		return clampScore(resourceScore)
	}
	resourceWeight, topologyWeight := scoreWeights(cfg)
	merged := resourceWeight*float64(resourceScore) + topologyWeight*float64(topologyScore)
	return clampScore(int(math.Round(merged)))
}

func selectPreferredGPUsWithTopology(
	validGPUs []*tfv1.GPU, needed int, strategy gpuallocator.Strategy, cfg *config.GPUFitConfig,
) ([]string, int, bool) {
	if !isNvLinkAwareEnabled(cfg) || needed <= 0 || needed > len(validGPUs) {
		return nil, 0, false
	}
	if len(validGPUs) == 0 {
		return nil, 0, false
	}

	resourceScoreByGPU := make(map[string]int, len(validGPUs))
	for _, gpu := range validGPUs {
		resourceScoreByGPU[gpu.Name] = clampScore(strategy.Score(gpu, false))
	}

	if needed == 1 {
		return selectSingleGPUWithTopology(validGPUs, resourceScoreByGPU, cfg)
	}
	return selectMultiGPUWithTopology(validGPUs, resourceScoreByGPU, needed, cfg)
}

func selectSingleGPUWithTopology(
	validGPUs []*tfv1.GPU,
	resourceScoreByGPU map[string]int,
	cfg *config.GPUFitConfig,
) ([]string, int, bool) {
	maxNodeConnectivity := 0.0
	for _, gpu := range validGPUs {
		maxNodeConnectivity = max(maxNodeConnectivity, gpuNvLinkBandwidth(gpu))
	}
	if maxNodeConnectivity <= 0 {
		return nil, 0, false
	}

	protectWeight := cfg.SingleGPUProtectWeight
	if protectWeight <= 0 || protectWeight >= 1 {
		protectWeight = defaultSingleGPUProtectWeight
	}

	bestGPUName := ""
	bestCombinedScore := -1.0
	bestTopologyScore := 0
	for _, gpu := range validGPUs {
		resourceScore := float64(resourceScoreByGPU[gpu.Name])
		topologyScore := defaultTopologyFallbackMaxScore
		if maxNodeConnectivity > 0 {
			connectivityRatio := gpuNvLinkBandwidth(gpu) / maxNodeConnectivity
			topologyScore = clampScore(int(math.Round((1.0 - connectivityRatio) * 100)))
		}

		combined := (1.0-protectWeight)*resourceScore + protectWeight*float64(topologyScore)
		if combined > bestCombinedScore || (combined == bestCombinedScore && gpu.Name < bestGPUName) {
			bestGPUName = gpu.Name
			bestCombinedScore = combined
			bestTopologyScore = topologyScore
		}
	}
	if bestGPUName == "" {
		return nil, 0, false
	}
	return []string{bestGPUName}, bestTopologyScore, true
}

func selectMultiGPUWithTopology(
	validGPUs []*tfv1.GPU,
	resourceScoreByGPU map[string]int,
	needed int,
	cfg *config.GPUFitConfig,
) ([]string, int, bool) {
	resourceWeight, topologyWeight := scoreWeights(cfg)

	indexes := make([]int, len(validGPUs))
	for i := range validGPUs {
		indexes[i] = i
	}

	bestFinalScore := -1.0
	bestTopologyScore := 0
	bestNames := []string(nil)
	hasTopologySignal := false
	for _, combo := range pickCombinations(indexes, needed) {
		resourceScore := avgResourceScore(validGPUs, combo, resourceScoreByGPU)
		topologyScore := topologyScoreOfCombo(validGPUs, combo)
		if topologyScore > 0 {
			hasTopologySignal = true
		}
		finalScore := resourceWeight*resourceScore + topologyWeight*topologyScore

		comboNames := make([]string, 0, len(combo))
		for _, idx := range combo {
			comboNames = append(comboNames, validGPUs[idx].Name)
		}
		slices.Sort(comboNames)

		if finalScore > bestFinalScore ||
			(finalScore == bestFinalScore && topologyScore > float64(bestTopologyScore)) ||
			(finalScore == bestFinalScore && topologyScore == float64(bestTopologyScore) && lessStringSlice(comboNames, bestNames)) {
			bestFinalScore = finalScore
			bestTopologyScore = clampScore(int(math.Round(topologyScore)))
			bestNames = comboNames
		}
	}
	if !hasTopologySignal {
		return nil, 0, false
	}
	if len(bestNames) == 0 {
		return nil, 0, false
	}
	return bestNames, bestTopologyScore, true
}

func avgResourceScore(validGPUs []*tfv1.GPU, combo []int, resourceScoreByGPU map[string]int) float64 {
	if len(combo) == 0 {
		return 0
	}
	sum := 0
	for _, idx := range combo {
		sum += resourceScoreByGPU[validGPUs[idx].Name]
	}
	return float64(sum) / float64(len(combo))
}

func topologyScoreOfCombo(validGPUs []*tfv1.GPU, combo []int) float64 {
	if len(combo) <= 1 {
		return 100
	}

	pairCount := 0
	sum := 0.0
	for i := 0; i < len(combo); i++ {
		for j := i + 1; j < len(combo); j++ {
			pairCount++
			bw := nvlinkBandwidthBetween(validGPUs[combo[i]], validGPUs[combo[j]])
			score := bw / defaultPairBandwidthTargetMBps * 100
			if score > 100 {
				score = 100
			}
			if score < 0 {
				score = 0
			}
			sum += score
		}
	}
	if pairCount == 0 {
		return 0
	}
	return sum / float64(pairCount)
}

func gpuNvLinkBandwidth(gpu *tfv1.GPU) float64 {
	if gpu == nil || gpu.Status.NvLink == nil {
		return 0
	}
	return float64(gpu.Status.NvLink.TotalBandwidthMBps)
}

func nvlinkBandwidthBetween(a, b *tfv1.GPU) float64 {
	if a == nil || b == nil {
		return 0
	}
	peerUUID := b.Status.UUID
	if peerUUID == "" {
		peerUUID = b.Name
	}
	reverseUUID := a.Status.UUID
	if reverseUUID == "" {
		reverseUUID = a.Name
	}

	forward := findPeerBandwidth(a.Status.NvLink, peerUUID, b.Name)
	reverse := findPeerBandwidth(b.Status.NvLink, reverseUUID, a.Name)
	return max(forward, reverse)
}

func findPeerBandwidth(nvlink *tfv1.GPUNvLinkStatus, peerUUID string, peerName string) float64 {
	if nvlink == nil {
		return 0
	}
	for _, peer := range nvlink.Peers {
		if strings.EqualFold(peer.PeerUUID, peerUUID) || strings.EqualFold(peer.PeerUUID, peerName) {
			return float64(peer.BandwidthMBps)
		}
	}
	return 0
}

func pickCombinations(input []int, choose int) [][]int {
	if choose <= 0 || choose > len(input) {
		return nil
	}
	result := make([][]int, 0)
	cur := make([]int, 0, choose)
	var dfs func(start int)
	dfs = func(start int) {
		if len(cur) == choose {
			result = append(result, append([]int(nil), cur...))
			return
		}
		for i := start; i < len(input); i++ {
			cur = append(cur, input[i])
			dfs(i + 1)
			cur = cur[:len(cur)-1]
		}
	}
	dfs(0)
	return result
}

func scoreWeights(cfg *config.GPUFitConfig) (float64, float64) {
	resourceWeight := cfg.ResourceScoreWeight
	topologyWeight := cfg.TopologyScoreWeight
	if resourceWeight < 0 {
		resourceWeight = 0
	}
	if topologyWeight < 0 {
		topologyWeight = 0
	}
	if resourceWeight+topologyWeight < defaultMinValidCombinedWeight {
		return defaultResourceScoreWeight, defaultTopologyScoreWeight
	}
	total := resourceWeight + topologyWeight
	return resourceWeight / total, topologyWeight / total
}

func clampScore(score int) int {
	if score < 0 {
		return 0
	}
	if score > 100 {
		return 100
	}
	return score
}

func lessStringSlice(left, right []string) bool {
	if len(right) == 0 {
		return true
	}
	for i := 0; i < len(left) && i < len(right); i++ {
		if left[i] == right[i] {
			continue
		}
		return left[i] < right[i]
	}
	return len(left) < len(right)
}
