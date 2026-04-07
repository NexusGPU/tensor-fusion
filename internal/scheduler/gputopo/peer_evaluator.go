package scheduler

import (
	"fmt"
	"sort"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// Pairwise affinity scores for peer topology scoring.
// Finer-grained than NUMA: 4 tiers instead of 3.
const (
	PeerAffinityScoreTier0 int64 = 10 // Same interconnect (NVLink/HCCS)
	PeerAffinityScoreTier1 int64 = 6  // Same NUMA
	PeerAffinityScoreTier2 int64 = 2  // Cross NUMA
	PeerAffinityScoreTier3 int64 = 0  // Unknown
)

// PeerTopologyEvaluator evaluates GPU topology using vendor-provided peer link
// data (NVLink, HCCS, PCIe, etc.) from GPU.Status.Topology.Peers.
// It reads the normalized Tier values reported by the hypervisor and selects
// GPU combinations that maximize interconnect affinity.
type PeerTopologyEvaluator struct {
	maxAllowedTier int
}

func NewPeerTopologyEvaluator(maxAllowedTier int) *PeerTopologyEvaluator {
	return &PeerTopologyEvaluator{maxAllowedTier: maxAllowedTier}
}

func (e *PeerTopologyEvaluator) Name() string {
	return "peer"
}

func (e *PeerTopologyEvaluator) Evaluate(gpus []*tfv1.GPU, count uint, preferLeastDamage bool) (*NodeTopologyPlan, error) {
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

// peerTierMatrix holds pairwise tier and bandwidth values between GPUs.
type peerTierMatrix struct {
	tiers      [][]int32
	bandwidths [][]int64 // bytes/sec, 0 if unknown
	n          int
}

// buildPeerTierMatrix constructs the pairwise tier matrix from GPU.Status.Topology.Peers.
// For GPU pairs without explicit peer data, falls back to NUMA comparison.
func buildPeerTierMatrix(gpus []*tfv1.GPU) *peerTierMatrix {
	n := len(gpus)

	// UUID → index mapping for resolving PeerGPUUUID references
	uuidToIdx := make(map[string]int, n)
	for i, gpu := range gpus {
		if gpu.Status.UUID != "" {
			uuidToIdx[gpu.Status.UUID] = i
		}
	}

	// Initialize with TierUnknown / 0 bandwidth
	tiers := make([][]int32, n)
	bws := make([][]int64, n)
	for i := range tiers {
		tiers[i] = make([]int32, n)
		bws[i] = make([]int64, n)
		for j := range tiers[i] {
			tiers[i][j] = int32(TierUnknown)
		}
	}

	// Fill from Topology.Peers
	for i, gpu := range gpus {
		if gpu.Status.Topology == nil {
			continue
		}
		for _, peer := range gpu.Status.Topology.Peers {
			// Skip self-referencing peers (UUID matches own GPU)
			if peer.PeerGPUUUID == gpu.Status.UUID {
				continue
			}
			j, ok := uuidToIdx[peer.PeerGPUUUID]
			if !ok {
				continue // peer not in candidate set
			}
			if i != j {
				// Clamp invalid tier values to TierUnknown
				t := peer.Tier
				if t < 0 || t > int32(TierUnknown) {
					t = int32(TierUnknown)
				}
				tiers[i][j] = t
				bws[i][j] = peer.Bandwidth
			}
		}
	}

	// Symmetrize: use the better (lower non-unknown) tier if asymmetric;
	// for bandwidth, use the max of the two directions.
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			ti, tj := tiers[i][j], tiers[j][i]
			if ti != tj {
				better := ti
				if ti == int32(TierUnknown) {
					better = tj
				} else if tj != int32(TierUnknown) && tj < ti {
					better = tj
				}
				tiers[i][j] = better
				tiers[j][i] = better
			}
			// Bandwidth: use the higher reported value
			bw := bws[i][j]
			if bws[j][i] > bw {
				bw = bws[j][i]
			}
			bws[i][j] = bw
			bws[j][i] = bw
		}
	}

	// Fall back to NUMA for remaining unknown pairs
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if tiers[i][j] != int32(TierUnknown) {
				continue
			}
			numaI := getNUMAID(gpus[i])
			numaJ := getNUMAID(gpus[j])
			if numaI >= 0 && numaJ >= 0 {
				tier := int32(TierCrossNUMA)
				if numaI == numaJ {
					tier = int32(TierSameNUMA)
				}
				tiers[i][j] = tier
				tiers[j][i] = tier
			}
		}
	}

	return &peerTierMatrix{tiers: tiers, bandwidths: bws, n: n}
}

// buildTier0Clusters groups GPUs into connected components of tier-0 edges.
// Returns clusters as slices of GPU indices, sorted by size descending.
func buildTier0Clusters(matrix *peerTierMatrix) [][]int {
	n := matrix.n
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	union := func(x, y int) {
		px, py := find(x), find(y)
		if px != py {
			parent[px] = py
		}
	}

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if matrix.tiers[i][j] == int32(TierSameInterconnect) {
				union(i, j)
			}
		}
	}

	groups := make(map[int][]int)
	for i := 0; i < n; i++ {
		root := find(i)
		groups[root] = append(groups[root], i)
	}

	clusters := make([][]int, 0, len(groups))
	for _, indices := range groups {
		clusters = append(clusters, indices)
	}
	// Sort by size descending, then by first index for stability
	sort.SliceStable(clusters, func(i, j int) bool {
		if len(clusters[i]) != len(clusters[j]) {
			return len(clusters[i]) > len(clusters[j])
		}
		return clusters[i][0] < clusters[j][0]
	})
	return clusters
}

// evaluateSingleGPU selects the least-damage single GPU using peer topology clusters.
func (e *PeerTopologyEvaluator) evaluateSingleGPU(gpus []*tfv1.GPU, candidateNames []string, preferLeastDamage bool) (*NodeTopologyPlan, error) {
	if !preferLeastDamage || len(gpus) <= 1 {
		gpu := gpus[0]
		tier := singleGPUTier(gpu)
		return &NodeTopologyPlan{
			CandidateGPUIds: candidateNames,
			BestGPUIds:      []string{gpu.Name},
			Tier:            tier,
			Score:           TierBandedScore(tier, 100),
			ModeSatisfied:   int(tier) <= e.maxAllowedTier,
			Reason:          "single GPU, no topology optimization needed",
		}, nil
	}

	matrix := buildPeerTierMatrix(gpus)
	clusters := buildTier0Clusters(matrix)

	// Map GPU index → cluster size
	clusterSize := make([]int, len(gpus))
	for _, cluster := range clusters {
		for _, idx := range cluster {
			clusterSize[idx] = len(cluster)
		}
	}

	type gpuScore struct {
		gpu     *tfv1.GPU
		idx     int
		cSize   int
		noPeers bool // true if GPU has no vendor topology data
	}

	scores := make([]gpuScore, len(gpus))
	for i, gpu := range gpus {
		noPeers := gpu.Status.Topology == nil || len(gpu.Status.Topology.Peers) == 0
		scores[i] = gpuScore{gpu: gpu, idx: i, cSize: clusterSize[i], noPeers: noPeers}
	}

	// Sort: prefer GPUs without peer data → smaller cluster → stable name order
	sort.SliceStable(scores, func(i, j int) bool {
		si, sj := scores[i], scores[j]
		if si.noPeers != sj.noPeers {
			return si.noPeers // no-peer GPUs first (least valuable topology-wise)
		}
		if si.cSize != sj.cSize {
			return si.cSize < sj.cSize
		}
		return si.gpu.Name < sj.gpu.Name
	})

	best := scores[0]
	tier := singleGPUTier(best.gpu)

	return &NodeTopologyPlan{
		CandidateGPUIds: candidateNames,
		BestGPUIds:      []string{best.gpu.Name},
		Tier:            tier,
		Score:           TierBandedScore(tier, 100),
		ModeSatisfied:   int(tier) <= e.maxAllowedTier,
		Reason:          fmt.Sprintf("least-damage: selected GPU (cluster size %d)", best.cSize),
	}, nil
}

// singleGPUTier determines the tier for a single GPU based on available data.
func singleGPUTier(gpu *tfv1.GPU) GPUAffinityTier {
	if gpu.Status.Topology != nil && len(gpu.Status.Topology.Peers) > 0 {
		return TierSameInterconnect
	}
	if gpu.Status.NUMANode != nil && *gpu.Status.NUMANode >= 0 {
		return TierSameNUMA
	}
	return TierUnknown
}

// evaluateMultiGPU selects the best multi-GPU combination using peer topology data.
func (e *PeerTopologyEvaluator) evaluateMultiGPU(gpus []*tfv1.GPU, candidateNames []string, count uint) (*NodeTopologyPlan, error) {
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

	matrix := buildPeerTierMatrix(gpus)
	clusters := buildTier0Clusters(matrix)

	// Phase 1: try same-interconnect cluster (all pairs tier 0)
	bestPlan := e.findBestSameClusterCombination(gpus, candidateNames, n, matrix, clusters)
	if bestPlan != nil {
		return bestPlan, nil
	}

	// Phase 2: cross-cluster with complexity control
	return e.findBestCrossClusterCombination(gpus, candidateNames, n, matrix, clusters)
}

// findBestSameClusterCombination searches for a viable tier-0 cluster.
// A valid cluster must have all internal pairs at tier 0 and enough GPUs.
func (e *PeerTopologyEvaluator) findBestSameClusterCombination(gpus []*tfv1.GPU, candidateNames []string, count int, matrix *peerTierMatrix, clusters [][]int) *NodeTopologyPlan {
	var viableClusters [][]int
	for _, cluster := range clusters {
		if len(cluster) < count {
			continue
		}
		// Verify all pairs within cluster are actually tier 0
		// (connected components may over-group non-transitive edges)
		allTier0 := true
		for i := 0; i < len(cluster) && allTier0; i++ {
			for j := i + 1; j < len(cluster); j++ {
				if matrix.tiers[cluster[i]][cluster[j]] != int32(TierSameInterconnect) {
					allTier0 = false
					break
				}
			}
		}
		if allTier0 {
			viableClusters = append(viableClusters, cluster)
		}
	}

	if len(viableClusters) == 0 {
		return nil
	}

	// Prefer cluster with fewer excess GPUs (compact fit)
	sort.SliceStable(viableClusters, func(i, j int) bool {
		return len(viableClusters[i]) < len(viableClusters[j])
	})

	bestCluster := viableClusters[0]

	// Pick first `count` GPUs sorted by name
	selected := make([]*tfv1.GPU, len(bestCluster))
	for i, idx := range bestCluster {
		selected[i] = gpus[idx]
	}
	sort.SliceStable(selected, func(i, j int) bool {
		return selected[i].Name < selected[j].Name
	})
	bestGPUIds := make([]string, count)
	for i := 0; i < count; i++ {
		bestGPUIds[i] = selected[i].Name
	}

	excess := len(bestCluster) - count
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
		Tier:            TierSameInterconnect,
		Score:           TierBandedScore(TierSameInterconnect, intraScore),
		ModeSatisfied:   int(TierSameInterconnect) <= e.maxAllowedTier,
		Reason:          fmt.Sprintf("same-interconnect: all %d GPUs in tier-0 cluster (cluster size %d)", count, len(bestCluster)),
	}
}

// findBestCrossClusterCombination uses enumeration, pruning, or heuristic
// for combinations that span multiple interconnect clusters.
func (e *PeerTopologyEvaluator) findBestCrossClusterCombination(gpus []*tfv1.GPU, candidateNames []string, count int, matrix *peerTierMatrix, clusters [][]int) (*NodeTopologyPlan, error) {
	// Level 1: full enumeration
	if combinationCount(len(gpus), count) <= MaxCombinationSearch {
		return e.enumerateAndSelectBest(gpus, candidateNames, count, matrix)
	}

	// Level 2: cluster-based pruning
	pruned := pruneByClusters(gpus, count, clusters)
	if pruned != nil && combinationCount(len(pruned), count) <= MaxCombinationSearch {
		prunedMatrix := buildPeerTierMatrix(pruned)
		return e.enumerateAndSelectBest(pruned, candidateNames, count, prunedMatrix)
	}

	// Level 3: heuristic
	return e.heuristicSelection(gpus, candidateNames, count, matrix, clusters)
}

// pruneByClusters reduces the candidate set by keeping GPUs from the largest
// clusters first. Returns nil if pruning doesn't reduce the set.
func pruneByClusters(gpus []*tfv1.GPU, count int, clusters [][]int) []*tfv1.GPU {
	// clusters are already sorted by size descending
	var pruned []*tfv1.GPU
	for _, cluster := range clusters {
		for _, idx := range cluster {
			pruned = append(pruned, gpus[idx])
		}
		if len(pruned) >= count {
			break
		}
	}
	if len(pruned) >= len(gpus) {
		return nil
	}
	return pruned
}

// enumerateAndSelectBest scores all C(M,N) combinations using the peer tier matrix.
func (e *PeerTopologyEvaluator) enumerateAndSelectBest(gpus []*tfv1.GPU, candidateNames []string, count int, matrix *peerTierMatrix) (*NodeTopologyPlan, error) {
	var bestCombo *combinationScore

	indices := make([]int, count)
	for i := range indices {
		indices[i] = i
	}

	// Build cluster membership for the current candidate set
	clusters := buildTier0Clusters(matrix)
	gpuClusterID := make([]int, matrix.n) // gpu index → cluster ID
	clusterSizes := make(map[int]int)     // cluster ID → total size in candidate set
	for cid, cluster := range clusters {
		clusterSizes[cid] = len(cluster)
		for _, idx := range cluster {
			gpuClusterID[idx] = cid
		}
	}

	for {
		combo := scoreCombinationFromMatrix(gpus, indices, matrix, gpuClusterID, clusterSizes)
		if bestCombo == nil || compareCombinations(combo, bestCombo) < 0 {
			bestCombo = combo
		}
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
		Reason:          fmt.Sprintf("enumerated: tier=%d, clusters=%d, score=%d", bestCombo.tier, bestCombo.numaCount, bestCombo.affinityScore),
	}, nil
}

// scoreCombinationFromMatrix computes the topology score using pairwise tier data.
// gpuClusterID maps each GPU index to its tier-0 cluster ID.
// clusterSizes maps each cluster ID to the total number of GPUs in that cluster
// within the current candidate set.
func scoreCombinationFromMatrix(gpus []*tfv1.GPU, indices []int, matrix *peerTierMatrix, gpuClusterID []int, clusterSizes map[int]int) *combinationScore {
	names := make([]string, len(indices))
	for i, idx := range indices {
		names[i] = gpus[idx].Name
	}

	// Determine worst-case tier, pairwise affinity, and total bandwidth
	worstTier := int32(TierSameInterconnect)
	affinityScore := int64(0)
	totalBandwidth := int64(0)
	for i := 0; i < len(indices); i++ {
		for j := i + 1; j < len(indices); j++ {
			idxI, idxJ := indices[i], indices[j]
			pairTier := matrix.tiers[idxI][idxJ]
			if pairTier < 0 || pairTier > int32(TierUnknown) {
				pairTier = int32(TierUnknown)
			}
			if pairTier > worstTier {
				worstTier = pairTier
			}
			affinityScore += peerAffinityForTier(pairTier)
			totalBandwidth += matrix.bandwidths[idxI][idxJ]
		}
	}

	tier := GPUAffinityTier(worstTier)

	// Normalize to 0-100
	maxPairs := len(indices) * (len(indices) - 1) / 2
	maxScore := int64(maxPairs) * PeerAffinityScoreTier0
	normalizedScore := int64(0)
	if maxScore > 0 {
		normalizedScore = (affinityScore*100 + maxScore/2) / maxScore
	}

	// Fragmentation: count leftover GPUs in each used cluster
	clusterSelected := make(map[int]int) // cluster ID → number of GPUs selected
	for _, idx := range indices {
		clusterSelected[gpuClusterID[idx]]++
	}
	fragmentationP := 0
	for cid, selected := range clusterSelected {
		total := clusterSizes[cid]
		if total > selected {
			fragmentationP += total - selected
		}
	}

	sort.Strings(names)

	return &combinationScore{
		gpuNames:       names,
		tier:           tier,
		numaCount:      len(clusterSelected),
		affinityScore:  normalizedScore,
		fragmentationP: fragmentationP,
		totalBandwidth: totalBandwidth,
	}
}

// peerAffinityForTier returns the pairwise affinity score for a given tier.
func peerAffinityForTier(tier int32) int64 {
	switch GPUAffinityTier(tier) {
	case TierSameInterconnect:
		return PeerAffinityScoreTier0
	case TierSameNUMA:
		return PeerAffinityScoreTier1
	case TierCrossNUMA:
		return PeerAffinityScoreTier2
	default:
		return PeerAffinityScoreTier3
	}
}

// heuristicSelection fills from the largest tier-0 cluster first.
func (e *PeerTopologyEvaluator) heuristicSelection(gpus []*tfv1.GPU, candidateNames []string, count int, matrix *peerTierMatrix, clusters [][]int) (*NodeTopologyPlan, error) {
	// clusters already sorted by size descending
	var selected []string
	clustersUsed := 0
	hasUnknownCluster := false

	for _, cluster := range clusters {
		if len(selected) >= count {
			break
		}
		// Sort GPUs within cluster by name
		sorted := make([]int, len(cluster))
		copy(sorted, cluster)
		sort.SliceStable(sorted, func(i, j int) bool {
			return gpus[sorted[i]].Name < gpus[sorted[j]].Name
		})
		for _, idx := range sorted {
			if len(selected) >= count {
				break
			}
			selected = append(selected, gpus[idx].Name)
		}
		// Check if this cluster has no peer data (unknown topology)
		if len(cluster) == 1 {
			gpu := gpus[cluster[0]]
			if gpu.Status.Topology == nil || len(gpu.Status.Topology.Peers) == 0 {
				hasUnknownCluster = true
			}
		}
		clustersUsed++
	}

	// Determine tier from actual pairwise tiers of the selected GPUs
	nameToIdx := make(map[string]int, len(gpus))
	for i, gpu := range gpus {
		nameToIdx[gpu.Name] = i
	}
	tier := TierSameInterconnect
	for i := 0; i < len(selected); i++ {
		for j := i + 1; j < len(selected); j++ {
			idxI := nameToIdx[selected[i]]
			idxJ := nameToIdx[selected[j]]
			pairTier := matrix.tiers[idxI][idxJ]
			if pairTier < 0 || pairTier > int32(TierUnknown) {
				pairTier = int32(TierUnknown)
			}
			if GPUAffinityTier(pairTier) > tier {
				tier = GPUAffinityTier(pairTier)
			}
		}
	}
	if hasUnknownCluster && clustersUsed == 1 {
		tier = TierUnknown
	}

	// Intra-tier score
	maxClusterSize := 0
	if len(clusters) > 0 {
		maxClusterSize = len(clusters[0])
	}
	intraScore := int64(50)
	if maxClusterSize >= count {
		intraScore = 80
	}

	satisfied := int(tier) <= e.maxAllowedTier
	score := TierBandedScore(tier, intraScore)

	return &NodeTopologyPlan{
		CandidateGPUIds: candidateNames,
		BestGPUIds:      selected,
		Tier:            tier,
		Score:           score,
		ModeSatisfied:   satisfied,
		Reason:          fmt.Sprintf("heuristic: tier=%d, clustersUsed=%d, score=%d", tier, clustersUsed, score),
	}, nil
}
