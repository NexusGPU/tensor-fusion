package scheduler

import (
	"fmt"
	"sort"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// NUMAEvaluator evaluates GPU topology based solely on NUMA node affinity.
// It implements the NUMA-first strategy described in the design:
// - Multi-GPU: prefer combinations where all GPUs are on the same NUMA node
// - Single-GPU: prefer "least damage" GPUs that don't break high-quality clusters
type NUMAEvaluator struct {
	maxAllowedTier int
}

func NewNUMAEvaluator(maxAllowedTier int) *NUMAEvaluator {
	return &NUMAEvaluator{maxAllowedTier: maxAllowedTier}
}

func (e *NUMAEvaluator) Name() string {
	return "numa"
}

func (e *NUMAEvaluator) Evaluate(gpus []*tfv1.GPU, count uint, preferLeastDamage bool) (*NodeTopologyPlan, error) {
	if count == 0 {
		return &NodeTopologyPlan{
			CandidateGPUIds: []string{},
			Tier:            TierUnknown,
			Score:           0,
			ModeSatisfied:   false,
			Reason:          "zero GPU request",
		}, nil
	}

	candidateNames := make([]string, len(gpus))
	for i, g := range gpus {
		candidateNames[i] = g.Name
	}

	if count == 1 {
		return e.evaluateSingleGPU(gpus, candidateNames, preferLeastDamage)
	}
	return e.evaluateMultiGPU(gpus, candidateNames, count)
}

// numaGroup groups GPUs by their NUMA node ID.
type numaGroup struct {
	numaID int32
	gpus   []*tfv1.GPU
}

// groupByNUMA partitions GPUs by NUMA node. GPUs with nil or negative NUMA are grouped under -1.
func groupByNUMA(gpus []*tfv1.GPU) map[int32][]*tfv1.GPU {
	groups := make(map[int32][]*tfv1.GPU)
	for _, gpu := range gpus {
		numaID := int32(-1)
		if gpu.Status.NUMANode != nil && *gpu.Status.NUMANode >= 0 {
			numaID = *gpu.Status.NUMANode
		}
		groups[numaID] = append(groups[numaID], gpu)
	}
	return groups
}

// evaluateSingleGPU implements the least-damage strategy for single GPU requests.
// Priority (ascending preference):
// 1. GPU in a NUMA domain with only 1 remaining GPU ("orphan")
// 2. GPU in a smaller NUMA domain
// 3. GPU in a larger NUMA domain (avoid breaking large clusters)
func (e *NUMAEvaluator) evaluateSingleGPU(gpus []*tfv1.GPU, candidateNames []string, preferLeastDamage bool) (*NodeTopologyPlan, error) {
	if !preferLeastDamage || len(gpus) <= 1 {
		// No least-damage optimization needed; determine tier from the GPU's actual NUMA status
		tier := TierSameNUMA
		if gpus[0].Status.NUMANode == nil || *gpus[0].Status.NUMANode < 0 {
			tier = TierUnknown
		}
		return &NodeTopologyPlan{
			CandidateGPUIds: candidateNames,
			BestGPUIds:      []string{gpus[0].Name},
			Tier:            tier,
			Score:           TierBandedScore(tier, 100),
			ModeSatisfied:   int(tier) <= e.maxAllowedTier,
			Reason:          "single GPU, no topology optimization needed",
		}, nil
	}

	numaGroups := groupByNUMA(gpus)

	type gpuDamageScore struct {
		gpu          *tfv1.GPU
		domainSize   int
		numaID       int32
		isUnknownNUM bool
	}

	scores := make([]gpuDamageScore, 0, len(gpus))
	for _, gpu := range gpus {
		numaID := int32(-1)
		isUnknown := true
		if gpu.Status.NUMANode != nil && *gpu.Status.NUMANode >= 0 {
			numaID = *gpu.Status.NUMANode
			isUnknown = false
		}
		domainSize := len(numaGroups[numaID])
		scores = append(scores, gpuDamageScore{
			gpu:          gpu,
			domainSize:   domainSize,
			numaID:       numaID,
			isUnknownNUM: isUnknown,
		})
	}

	// Sort: prefer unknown NUMA → smaller domain → stable order by GPU name
	sort.SliceStable(scores, func(i, j int) bool {
		si, sj := scores[i], scores[j]
		// Unknown NUMA GPUs are least valuable topology-wise, use them first
		if si.isUnknownNUM != sj.isUnknownNUM {
			return si.isUnknownNUM
		}
		// Prefer GPUs from smaller NUMA domains (orphans first)
		if si.domainSize != sj.domainSize {
			return si.domainSize < sj.domainSize
		}
		// Stable tie-breaking by name
		return si.gpu.Name < sj.gpu.Name
	})

	best := scores[0]
	tier := TierSameNUMA
	if best.isUnknownNUM {
		tier = TierUnknown
	}

	return &NodeTopologyPlan{
		CandidateGPUIds: candidateNames,
		BestGPUIds:      []string{best.gpu.Name},
		Tier:            tier,
		Score:           TierBandedScore(tier, 100),
		ModeSatisfied:   int(tier) <= e.maxAllowedTier,
		Reason:          fmt.Sprintf("least-damage: selected GPU from NUMA %d (domain size %d)", best.numaID, best.domainSize),
	}, nil
}

// evaluateMultiGPU selects the best multi-GPU combination based on NUMA affinity.
// Uses a three-phase approach: prune → enumerate → sort.
func (e *NUMAEvaluator) evaluateMultiGPU(gpus []*tfv1.GPU, candidateNames []string, count uint) (*NodeTopologyPlan, error) {
	n := int(count)
	if len(gpus) < n {
		return &NodeTopologyPlan{
			CandidateGPUIds: candidateNames,
			Tier:            TierUnknown,
			Score:           0,
			ModeSatisfied:   false,
			Reason:          fmt.Sprintf("not enough candidate GPUs: need %d, have %d", n, len(gpus)),
		}, nil
	}

	numaGroups := groupByNUMA(gpus)

	// Phase 1: Try same-NUMA combinations first (best tier)
	bestPlan := e.findBestSameNUMACombination(numaGroups, candidateNames, n)
	if bestPlan != nil {
		return bestPlan, nil
	}

	// Phase 2: Cross-NUMA enumeration with complexity control
	return e.findBestCrossNUMACombination(gpus, candidateNames, n, numaGroups)
}

// findBestSameNUMACombination searches for the best all-same-NUMA combination.
func (e *NUMAEvaluator) findBestSameNUMACombination(numaGroups map[int32][]*tfv1.GPU, candidateNames []string, count int) *NodeTopologyPlan {
	// Collect NUMA domains that have enough GPUs
	var viableGroups []numaGroup
	for numaID, gpus := range numaGroups {
		if numaID < 0 {
			continue // skip unknown NUMA
		}
		if len(gpus) >= count {
			viableGroups = append(viableGroups, numaGroup{numaID: numaID, gpus: gpus})
		}
	}

	if len(viableGroups) == 0 {
		return nil
	}

	// Sort groups: prefer NUMA domains with fewer excess GPUs (compact allocation)
	sort.SliceStable(viableGroups, func(i, j int) bool {
		return len(viableGroups[i].gpus) < len(viableGroups[j].gpus)
	})

	// From the best group, pick the first `count` GPUs (sorted by name for stability)
	bestGroup := viableGroups[0]
	selected := make([]*tfv1.GPU, len(bestGroup.gpus))
	copy(selected, bestGroup.gpus)
	sort.SliceStable(selected, func(i, j int) bool {
		return selected[i].Name < selected[j].Name
	})

	bestGPUIds := make([]string, count)
	for i := 0; i < count; i++ {
		bestGPUIds[i] = selected[i].Name
	}

	// Intra-tier score: higher when the domain has less excess (tighter fit = better fragmentation)
	excess := len(bestGroup.gpus) - count
	intraScore := int64(100)
	if excess > 0 {
		intraScore = int64(100 - excess*SameNUMAExcessPenalty)
		if intraScore < SameNUMAMinIntraScore {
			intraScore = SameNUMAMinIntraScore
		}
	}

	return &NodeTopologyPlan{
		CandidateGPUIds: candidateNames,
		BestGPUIds:      bestGPUIds,
		Tier:            TierSameNUMA,
		Score:           TierBandedScore(TierSameNUMA, intraScore),
		ModeSatisfied:   int(TierSameNUMA) <= e.maxAllowedTier,
		Reason:          fmt.Sprintf("same-NUMA: all %d GPUs on NUMA %d (domain size %d)", count, bestGroup.numaID, len(bestGroup.gpus)),
	}
}

// findBestCrossNUMACombination finds the best cross-NUMA combination using
// a three-level strategy: full enumeration → domain-based pruning → heuristic.
func (e *NUMAEvaluator) findBestCrossNUMACombination(gpus []*tfv1.GPU, candidateNames []string, count int, numaGroups map[int32][]*tfv1.GPU) (*NodeTopologyPlan, error) {
	// Level 1: if full enumeration is feasible, use it directly
	if combinationCount(len(gpus), count) <= MaxCombinationSearch {
		return e.enumerateAndSelectBest(gpus, candidateNames, count, numaGroups)
	}

	// Level 2: domain-based pruning — keep only the top NUMA domains (largest
	// known domains first, unknown last) and check if the pruned set is small
	// enough to enumerate.
	pruned := e.pruneByDomain(gpus, count, numaGroups)
	if pruned != nil && combinationCount(len(pruned), count) <= MaxCombinationSearch {
		prunedNUMAGroups := groupByNUMA(pruned)
		return e.enumerateAndSelectBest(pruned, candidateNames, count, prunedNUMAGroups)
	}

	// Level 3: heuristic greedy selection
	return e.heuristicCrossNUMASelection(gpus, candidateNames, count, numaGroups)
}

// pruneByDomain reduces the candidate GPU set by keeping only GPUs from the
// top NUMA domains (sorted by size descending, known domains before unknown).
// It accumulates domains until the total GPU count >= requested count.
// Returns nil if pruning cannot reduce the set meaningfully (i.e. the pruned
// set is the same size as the original).
func (e *NUMAEvaluator) pruneByDomain(gpus []*tfv1.GPU, count int, numaGroups map[int32][]*tfv1.GPU) []*tfv1.GPU {
	// Sort domains: known domains by size descending, then unknown (-1) last
	type domainEntry struct {
		numaID int32
		gpus   []*tfv1.GPU
	}
	domains := make([]domainEntry, 0, len(numaGroups))
	for numaID, g := range numaGroups {
		domains = append(domains, domainEntry{numaID: numaID, gpus: g})
	}
	sort.SliceStable(domains, func(i, j int) bool {
		iUnknown := domains[i].numaID < 0
		jUnknown := domains[j].numaID < 0
		// Known domains come before unknown
		if iUnknown != jUnknown {
			return !iUnknown
		}
		// Larger domains first
		if len(domains[i].gpus) != len(domains[j].gpus) {
			return len(domains[i].gpus) > len(domains[j].gpus)
		}
		return domains[i].numaID < domains[j].numaID
	})

	// Accumulate top domains until we have enough GPUs
	var pruned []*tfv1.GPU
	for _, d := range domains {
		pruned = append(pruned, d.gpus...)
		if len(pruned) >= count {
			break
		}
	}

	// Only return pruned set if it's meaningfully smaller than the original
	if len(pruned) >= len(gpus) {
		return nil
	}
	return pruned
}

// combinationScore represents a scored GPU combination.
type combinationScore struct {
	gpuNames       []string
	tier           GPUAffinityTier
	numaCount      int   // number of distinct NUMA domains / clusters used
	affinityScore  int64 // normalized pairwise affinity (0-100)
	fragmentationP int   // total leftover GPUs in used NUMA domains / clusters
	totalBandwidth int64 // sum of pairwise bandwidth (bytes/sec), for tie-breaking
}

// enumerateAndSelectBest enumerates all C(M, N) combinations and returns the best one.
func (e *NUMAEvaluator) enumerateAndSelectBest(gpus []*tfv1.GPU, candidateNames []string, count int, numaGroups map[int32][]*tfv1.GPU) (*NodeTopologyPlan, error) {
	var bestCombo *combinationScore

	// Generate all combinations
	indices := make([]int, count)
	for i := range indices {
		indices[i] = i
	}

	for {
		// Score current combination
		combo := e.scoreCombination(gpus, indices, numaGroups)
		if bestCombo == nil || compareCombinations(combo, bestCombo) < 0 {
			bestCombo = combo
		}

		// Next combination (lexicographic)
		if !nextCombination(indices, len(gpus)) {
			break
		}
	}

	if bestCombo == nil {
		return &NodeTopologyPlan{
			CandidateGPUIds: candidateNames,
			Tier:            TierUnknown,
			Score:           0,
			ModeSatisfied:   false,
			Reason:          "no valid combination found",
		}, nil
	}

	satisfied := int(bestCombo.tier) <= e.maxAllowedTier

	return &NodeTopologyPlan{
		CandidateGPUIds: candidateNames,
		BestGPUIds:      bestCombo.gpuNames,
		Tier:            bestCombo.tier,
		Score:           TierBandedScore(bestCombo.tier, bestCombo.affinityScore),
		ModeSatisfied:   satisfied,
		Reason:          fmt.Sprintf("enumerated: tier=%d, numaCount=%d, score=%d", bestCombo.tier, bestCombo.numaCount, bestCombo.affinityScore),
	}, nil
}

// scoreCombination computes the topology score for a specific GPU combination.
func (e *NUMAEvaluator) scoreCombination(gpus []*tfv1.GPU, indices []int, numaGroups map[int32][]*tfv1.GPU) *combinationScore {
	names := make([]string, len(indices))
	numaSet := make(map[int32]int) // numaID → count of GPUs selected from this domain

	for i, idx := range indices {
		gpu := gpus[idx]
		names[i] = gpu.Name
		numaID := int32(-1)
		if gpu.Status.NUMANode != nil && *gpu.Status.NUMANode >= 0 {
			numaID = *gpu.Status.NUMANode
		}
		numaSet[numaID]++
	}

	// Determine tier
	tier := TierSameNUMA
	if len(numaSet) > 1 {
		tier = TierCrossNUMA
	}
	if _, hasUnknown := numaSet[-1]; hasUnknown {
		if len(numaSet) == 1 {
			tier = TierUnknown
		} else {
			tier = TierCrossNUMA
		}
	}

	// Calculate pairwise affinity score
	affinityScore := int64(0)
	for i := 0; i < len(indices); i++ {
		for j := i + 1; j < len(indices); j++ {
			numaI := getNUMAID(gpus[indices[i]])
			numaJ := getNUMAID(gpus[indices[j]])
			if numaI < 0 || numaJ < 0 {
				affinityScore += AffinityScoreUnknown
			} else if numaI == numaJ {
				affinityScore += AffinityScoreSameNUMA
			} else {
				affinityScore += AffinityScoreCrossNUMA
			}
		}
	}

	// Normalize score to 0-100 range with rounding
	maxPairs := len(indices) * (len(indices) - 1) / 2
	maxScore := int64(maxPairs * AffinityScoreSameNUMA)
	normalizedScore := int64(0)
	if maxScore > 0 {
		normalizedScore = (affinityScore*100 + maxScore/2) / maxScore
	}

	// Calculate fragmentation penalty
	fragmentationP := 0
	for numaID, selectedCount := range numaSet {
		if numaID < 0 {
			continue
		}
		totalInDomain := len(numaGroups[numaID])
		fragmentationP += totalInDomain - selectedCount
	}

	// Sort names for stable results
	sort.Strings(names)

	return &combinationScore{
		gpuNames:       names,
		tier:           tier,
		numaCount:      len(numaSet),
		affinityScore:  normalizedScore,
		fragmentationP: fragmentationP,
	}
}

// compareCombinations returns negative if a is better than b.
// Priority: smaller tier → higher score → lower fragmentation → higher bandwidth → lexicographic.
func compareCombinations(a, b *combinationScore) int {
	if a.tier != b.tier {
		return int(a.tier) - int(b.tier)
	}
	if a.affinityScore != b.affinityScore {
		return int(b.affinityScore - a.affinityScore) // higher is better
	}
	if a.fragmentationP != b.fragmentationP {
		return a.fragmentationP - b.fragmentationP // lower is better
	}
	if a.totalBandwidth != b.totalBandwidth {
		if b.totalBandwidth > a.totalBandwidth {
			return 1 // higher bandwidth is better
		}
		return -1
	}
	// Lexicographic tie-break
	for i := 0; i < len(a.gpuNames) && i < len(b.gpuNames); i++ {
		if a.gpuNames[i] < b.gpuNames[i] {
			return -1
		}
		if a.gpuNames[i] > b.gpuNames[i] {
			return 1
		}
	}
	return 0
}

// heuristicCrossNUMASelection uses a greedy heuristic when full enumeration is too expensive.
// Strategy: fill from the largest NUMA domain first to minimize cross-NUMA traffic.
func (e *NUMAEvaluator) heuristicCrossNUMASelection(_ []*tfv1.GPU, candidateNames []string, count int, numaGroups map[int32][]*tfv1.GPU) (*NodeTopologyPlan, error) {
	// Sort NUMA groups by size descending (fill from largest first)
	type groupEntry struct {
		numaID int32
		gpus   []*tfv1.GPU
	}
	groups := make([]groupEntry, 0, len(numaGroups))
	for numaID, g := range numaGroups {
		groups = append(groups, groupEntry{numaID: numaID, gpus: g})
	}
	sort.SliceStable(groups, func(i, j int) bool {
		if len(groups[i].gpus) != len(groups[j].gpus) {
			return len(groups[i].gpus) > len(groups[j].gpus)
		}
		return groups[i].numaID < groups[j].numaID
	})

	var selected []string
	numaUsed := 0
	hasUnknown := false
	for _, g := range groups {
		if len(selected) >= count {
			break
		}
		// Sort GPUs within group for stability
		sorted := make([]*tfv1.GPU, len(g.gpus))
		copy(sorted, g.gpus)
		sort.SliceStable(sorted, func(i, j int) bool {
			return sorted[i].Name < sorted[j].Name
		})

		for _, gpu := range sorted {
			if len(selected) >= count {
				break
			}
			selected = append(selected, gpu.Name)
		}
		if g.numaID < 0 {
			hasUnknown = true
		}
		numaUsed++
	}

	tier := TierCrossNUMA
	if numaUsed == 1 && !hasUnknown {
		tier = TierSameNUMA
	}
	if hasUnknown && numaUsed == 1 {
		tier = TierUnknown
	}

	// Intra-tier score: percentage of GPUs from the largest domain
	maxDomainCount := 0
	for _, g := range groups {
		if len(g.gpus) > maxDomainCount {
			maxDomainCount = len(g.gpus)
		}
	}
	intraScore := int64(50) // base score for heuristic
	if maxDomainCount >= count {
		intraScore = 80 // most GPUs can come from one domain
	}

	satisfied := int(tier) <= e.maxAllowedTier
	score := TierBandedScore(tier, intraScore)

	return &NodeTopologyPlan{
		CandidateGPUIds: candidateNames,
		BestGPUIds:      selected,
		Tier:            tier,
		Score:           score,
		ModeSatisfied:   satisfied,
		Reason:          fmt.Sprintf("heuristic: tier=%d, numaUsed=%d, score=%d (too many combinations for full enumeration)", tier, numaUsed, score),
	}, nil
}

// getNUMAID returns the NUMA node ID for a GPU, or -1 if unknown.
func getNUMAID(gpu *tfv1.GPU) int32 {
	if gpu.Status.NUMANode != nil && *gpu.Status.NUMANode >= 0 {
		return *gpu.Status.NUMANode
	}
	return -1
}

// nextCombination generates the next lexicographic combination of indices.
// Returns false when all combinations have been enumerated.
func nextCombination(indices []int, n int) bool {
	k := len(indices)
	for i := k - 1; i >= 0; i-- {
		if indices[i] < n-k+i {
			indices[i]++
			for j := i + 1; j < k; j++ {
				indices[j] = indices[j-1] + 1
			}
			return true
		}
	}
	return false
}

// combinationCount computes C(n, k) with overflow protection.
func combinationCount(n, k int) int {
	if k > n || k < 0 {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}
	if k > n-k {
		k = n - k
	}
	result := 1
	for i := 0; i < k; i++ {
		result = result * (n - i) / (i + 1)
		if result > MaxCombinationSearch*10 {
			return result // early exit for large values
		}
	}
	return result
}
