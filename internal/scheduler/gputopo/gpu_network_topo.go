package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/scheduler/gpuresources"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
)

const Name = "GPUNetworkTopologyAware"

// Pod annotations for per-pod topology override.
const (
	AnnotationRequireTopology = "tensor-fusion.ai/require-gpu-topology"
	AnnotationTopologyMaxTier = "tensor-fusion.ai/gpu-topology-max-tier"
	AnnotationTopologySource  = "tensor-fusion.ai/gpu-topology-source"
)

// Event reasons for topology scheduling.
const (
	EventReasonTopologyUnsatisfied = "GPUInterconnectTopologyUnsatisfied"
	EventReasonTopologyFallback    = "GPUInterconnectTopologyFallback"
	EventReasonTopologySelected    = "GPUInterconnectTopologySelected"
)

var _ fwk.PreFilterPlugin = &GPUNetworkTopologyAware{}
var _ fwk.FilterPlugin = &GPUNetworkTopologyAware{}
var _ fwk.ScorePlugin = &GPUNetworkTopologyAware{}

type GPUNetworkTopologyAware struct {
	logger    *klog.Logger
	fh        fwk.Handle
	ctx       context.Context
	cfg       *config.GPUNetworkTopologyAwareConfig
	evaluator Evaluator
}

type PluginFactoryFunc func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error)

func New() PluginFactoryFunc {
	return func(ctx context.Context, obj runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
		target := &config.GPUNetworkTopologyAwareConfig{}
		if unknown, ok := obj.(*runtime.Unknown); ok {
			if err := json.Unmarshal(unknown.Raw, target); err != nil {
				return nil, err
			}
		}
		lh := klog.FromContext(ctx).WithValues("plugin", Name)

		evaluator := buildEvaluator(target)

		c := &GPUNetworkTopologyAware{
			logger:    &lh,
			fh:        handle,
			cfg:       target,
			ctx:       ctx,
			evaluator: evaluator,
		}
		return c, nil
	}
}

// buildEvaluator creates the appropriate evaluator based on configuration.
func buildEvaluator(cfg *config.GPUNetworkTopologyAwareConfig) Evaluator {
	maxTier := cfg.GetMaxAllowedTier()
	switch cfg.GetTopologySource() {
	case TopologySourceNUMA:
		return NewNUMAEvaluator(maxTier)
	case TopologySourceVendor:
		return NewPeerTopologyEvaluator(maxTier)
	default: // TopologySourceAuto
		return NewAutoEvaluator(maxTier)
	}
}

func (s *GPUNetworkTopologyAware) Name() string {
	return Name
}

// PreFilter evaluates GPU topology for all candidate nodes.
// It reads the GPU scheduling result from GPUResourcesFit and generates
// a topology plan for each node.
func (s *GPUNetworkTopologyAware) PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, _ []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	if !utils.IsTensorFusionWorker(pod) {
		return nil, fwk.NewStatus(fwk.Skip, "skip for non tensor-fusion pod")
	}

	start := time.Now()

	// Read candidate GPUs from GPUResourcesFit
	schedulingResultRaw, err := state.Read(gpuresources.CycleStateGPUSchedulingResult)
	if err != nil {
		s.logger.V(4).Info("no GPU scheduling result available, skipping topology evaluation", "pod", pod.Name)
		return nil, fwk.NewStatus(fwk.Skip, "no GPU scheduling result")
	}
	schedulingResult := schedulingResultRaw.(*gpuresources.GPUSchedulingStateData)

	// Read allocation request for GPU count
	allocRequestRaw, err := state.Read(gpuresources.CycleStateAllocateRequest)
	if err != nil {
		return nil, fwk.NewStatus(fwk.Skip, "no alloc request")
	}
	allocRequest := allocRequestRaw.(*tfv1.AllocRequest)
	gpuCount := allocRequest.Count

	if gpuCount == 0 {
		return nil, fwk.NewStatus(fwk.Skip, "zero GPU request")
	}

	// Resolve per-pod overrides
	mode, maxTier, source := s.resolvePerPodConfig(pod)
	preferLeastDamage := s.cfg.GetPreferLeastDamage()

	// Build evaluator with per-pod overrides if needed
	eval := s.evaluator
	if maxTier != s.cfg.GetMaxAllowedTier() || source != s.cfg.GetTopologySource() {
		overrideCfg := &config.GPUNetworkTopologyAwareConfig{
			TopologySource: source,
			MaxAllowedTier: maxTier,
		}
		eval = buildEvaluator(overrideCfg)
	}

	// Evaluate topology for each candidate node
	topoState := &GPUTopologyStateData{
		Plans: make(map[string]*NodeTopologyPlan),
	}

	for nodeName, gpus := range schedulingResult.NodeGPUs {
		if len(gpus) == 0 {
			continue
		}

		plan, evalErr := eval.Evaluate(gpus, gpuCount, preferLeastDamage)
		if evalErr != nil {
			s.logger.Error(evalErr, "topology evaluation failed", "pod", pod.Name, "node", nodeName)
			continue
		}

		plan.NodeName = nodeName
		// Update ModeSatisfied based on effective maxAllowedTier
		plan.ModeSatisfied = int(plan.Tier) <= maxTier

		topoState.Plans[nodeName] = plan

		s.logger.V(4).Info("topology evaluation",
			"pod", pod.Name,
			"node", nodeName,
			"tier", plan.Tier,
			"score", plan.Score,
			"bestGPUs", plan.BestGPUIds,
			"satisfied", plan.ModeSatisfied,
			"reason", plan.Reason,
		)
	}

	state.Write(CycleStateGPUTopologyResult, topoState)

	duration := time.Since(start)
	s.logger.V(2).Info("topology PreFilter completed",
		"pod", pod.Name,
		"mode", mode,
		"nodesEvaluated", len(topoState.Plans),
		"duration", duration,
	)

	// Emit metrics and events
	satisfiedCount := 0
	hasFallback := false
	hasDegraded := false
	for _, plan := range topoState.Plans {
		if plan.ModeSatisfied {
			satisfiedCount++
		}
		if plan.Tier == TierUnknown {
			hasFallback = true
		}
		if strings.Contains(plan.Reason, "heuristic") {
			hasDegraded = true
		}
	}

	poolName := allocRequest.PoolName
	metrics.SetTopologyMetrics(poolName, satisfiedCount > 0, hasFallback, hasDegraded)

	if hasFallback {
		s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeNormal, EventReasonTopologyFallback,
			"topology evaluation", "Topology data incomplete, fell back to NUMA or unknown for some nodes")
	}
	if satisfiedCount == 0 && len(topoState.Plans) > 0 && mode == TopologyModeHard {
		s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeWarning, EventReasonTopologyUnsatisfied,
			"topology evaluation", "No node satisfies GPU topology constraint (mode=hard, maxTier=%d)", maxTier)
	}

	return nil, fwk.NewStatus(fwk.Success, "")
}

// PreFilterExtensions returns nil since no additional extensions are needed.
func (s *GPUNetworkTopologyAware) PreFilterExtensions() fwk.PreFilterExtensions {
	return nil
}

// Filter rejects nodes that don't satisfy the topology constraint in hard mode.
func (s *GPUNetworkTopologyAware) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if !utils.IsTensorFusionWorker(pod) {
		return fwk.NewStatus(fwk.Success, "")
	}

	mode, maxTier, _ := s.resolvePerPodConfig(pod)

	// In soft mode, never reject nodes
	if mode != TopologyModeHard {
		return fwk.NewStatus(fwk.Success, "")
	}

	// Read topology state
	topoStateRaw, err := state.Read(CycleStateGPUTopologyResult)
	if err != nil {
		// No topology data — treat as TierUnknown
		return s.filterUnknownTopology(maxTier, "no topology data available")
	}

	topoState := topoStateRaw.(*GPUTopologyStateData)
	nodeName := nodeInfo.Node().Name

	plan, exists := topoState.Plans[nodeName]
	if !exists {
		// No plan for this node — treat as TierUnknown
		return s.filterUnknownTopology(maxTier, "no topology plan for node")
	}

	// When topology data is incomplete (TierUnknown), consult unknownTopologyPolicy
	if plan.Tier == TierUnknown {
		unknownPolicy := s.cfg.GetUnknownTopologyPolicy()
		if unknownPolicy == "reject" {
			return fwk.NewStatus(fwk.Unschedulable,
				fmt.Sprintf("GPU topology unknown and unknownTopologyPolicy=reject: %s", plan.Reason))
		}
		// treat-as-worst: allow if maxAllowedTier includes TierUnknown
	}

	if !plan.ModeSatisfied {
		return fwk.NewStatus(fwk.Unschedulable,
			fmt.Sprintf("GPU topology constraint not satisfied: %s (tier=%d, maxAllowed=%d)",
				plan.Reason, plan.Tier, s.cfg.GetMaxAllowedTier()))
	}

	return fwk.NewStatus(fwk.Success, "")
}

// Score returns a topology-based score for the node.
func (s *GPUNetworkTopologyAware) Score(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status) {
	nodeName := nodeInfo.Node().Name
	if !utils.IsTensorFusionWorker(pod) {
		return 0, fwk.NewStatus(fwk.Success, "")
	}

	topoStateRaw, err := state.Read(CycleStateGPUTopologyResult)
	if err != nil {
		return 0, fwk.NewStatus(fwk.Success, "")
	}

	topoState := topoStateRaw.(*GPUTopologyStateData)
	plan, exists := topoState.Plans[nodeName]
	if !exists {
		return 0, fwk.NewStatus(fwk.Success, "")
	}

	return plan.Score, fwk.NewStatus(fwk.Success, "")
}

// ScoreExtensions returns nil since no normalization extension is needed.
func (s *GPUNetworkTopologyAware) ScoreExtensions() fwk.ScoreExtensions {
	return nil
}

// filterUnknownTopology applies hard-mode filtering when topology data is missing.
// Under treat-as-worst, missing data is treated as TierUnknown and checked against maxTier.
// Under reject policy, it is always rejected.
func (s *GPUNetworkTopologyAware) filterUnknownTopology(maxTier int, reason string) *fwk.Status {
	unknownPolicy := s.cfg.GetUnknownTopologyPolicy()
	if unknownPolicy == "reject" {
		return fwk.NewStatus(fwk.Unschedulable,
			fmt.Sprintf("%s and unknownTopologyPolicy=reject", reason))
	}
	// treat-as-worst: check TierUnknown against maxAllowedTier
	if int(TierUnknown) > maxTier {
		return fwk.NewStatus(fwk.Unschedulable,
			fmt.Sprintf("%s: treated as TierUnknown (%d) which exceeds maxAllowedTier (%d)",
				reason, TierUnknown, maxTier))
	}
	return fwk.NewStatus(fwk.Success, "")
}

// resolvePerPodConfig returns the effective mode, maxAllowedTier, and topology
// source for a pod, considering pod annotation overrides.
func (s *GPUNetworkTopologyAware) resolvePerPodConfig(pod *v1.Pod) (string, int, string) {
	mode := s.cfg.GetMode()
	maxAllowedTier := s.cfg.GetMaxAllowedTier()
	topologySource := s.cfg.GetTopologySource()

	if pod.Annotations == nil {
		return mode, maxAllowedTier, topologySource
	}

	if v, ok := pod.Annotations[AnnotationRequireTopology]; ok {
		switch v {
		case AnnotationBoolTrue:
			mode = TopologyModeHard
		case AnnotationBoolFalse:
			mode = TopologyModeSoft
		}
	}

	if v, ok := pod.Annotations[AnnotationTopologyMaxTier]; ok {
		var tier int
		if _, err := fmt.Sscanf(v, "%d", &tier); err == nil && tier >= 0 {
			maxAllowedTier = tier
		}
	}

	if v, ok := pod.Annotations[AnnotationTopologySource]; ok {
		switch v {
		case TopologySourceAuto, TopologySourceNUMA, TopologySourceVendor:
			topologySource = v
		}
	}

	return mode, maxAllowedTier, topologySource
}
