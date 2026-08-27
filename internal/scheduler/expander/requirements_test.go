package expander

import (
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	karpv1 "sigs.k8s.io/karpenter/pkg/apis/v1"
)

func TestMergePodKarpenterRequirements(t *testing.T) {
	spec := &karpv1.NodeClaimSpec{
		Requirements: []karpv1.NodeSelectorRequirementWithMinValues{
			{
				Key:      "node.kubernetes.io/instance-type",
				Operator: corev1.NodeSelectorOpIn,
				Values:   []string{"g6.2xlarge", "g6.12xlarge"},
			},
			{
				Key:      "topology.kubernetes.io/zone",
				Operator: corev1.NodeSelectorOpIn,
				Values:   []string{"us-east-1a", "us-east-1b"},
			},
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "worker"},
		Spec: corev1.PodSpec{
			Affinity: &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
					NodeSelectorTerms: []corev1.NodeSelectorTerm{{MatchExpressions: []corev1.NodeSelectorRequirement{
						{Key: "node.kubernetes.io/instance-type", Operator: corev1.NodeSelectorOpIn, Values: []string{"g6.2xlarge", "g6.12xlarge"}},
						{Key: "topology.kubernetes.io/zone", Operator: corev1.NodeSelectorOpIn, Values: []string{"us-east-1a", "us-east-1b"}},
					}}},
				},
			}},
		},
	}

	preparedNode := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{
		"node.kubernetes.io/instance-type": "g6.12xlarge",
		"topology.kubernetes.io/zone":      "us-east-1a",
	}}}
	require.True(t, mergeKarpenterRequirements(spec, podKarpenterRequirementValues(pod, preparedNode)))

	require.ElementsMatch(t, []string{"g6.2xlarge", "g6.12xlarge"}, spec.Requirements[0].Values)
	require.ElementsMatch(t, []string{"us-east-1a", "us-east-1b"}, spec.Requirements[1].Values)
}

func TestMergePodKarpenterRequirementsPreservesPoolIntersection(t *testing.T) {
	spec := &karpv1.NodeClaimSpec{Requirements: []karpv1.NodeSelectorRequirementWithMinValues{
		{Key: "node.kubernetes.io/instance-type", Operator: corev1.NodeSelectorOpIn, Values: []string{"g6.2xlarge", "g6.12xlarge"}},
		{Key: "topology.kubernetes.io/zone", Operator: corev1.NodeSelectorOpIn, Values: []string{"us-east-1a", "us-east-1b"}},
	}}
	pod := &corev1.Pod{Spec: corev1.PodSpec{NodeSelector: map[string]string{
		"node.kubernetes.io/instance-type": "g6.12xlarge",
	}}}
	require.True(t, mergeKarpenterRequirements(spec, podKarpenterRequirementValues(pod, nil)))
	require.Equal(t, []string{"g6.12xlarge"}, spec.Requirements[0].Values)
	require.Equal(t, []string{"us-east-1a", "us-east-1b"}, spec.Requirements[1].Values)
}

func TestMergeKarpenterRequirementsRejectsEmptyIntersection(t *testing.T) {
	spec := &karpv1.NodeClaimSpec{Requirements: []karpv1.NodeSelectorRequirementWithMinValues{{
		Key: corev1.LabelInstanceTypeStable, Operator: corev1.NodeSelectorOpIn, Values: []string{"g6.12xlarge"},
	}}}

	require.False(t, mergeKarpenterRequirements(spec, map[string][]string{
		corev1.LabelInstanceTypeStable: {"g6.2xlarge"},
	}))
	require.Equal(t, []string{"g6.12xlarge"}, spec.Requirements[0].Values)
}

func TestMergeKarpenterRequirementsUsesKarpenterOperatorSemantics(t *testing.T) {
	spec := &karpv1.NodeClaimSpec{Requirements: []karpv1.NodeSelectorRequirementWithMinValues{
		{Key: "capacity", Operator: corev1.NodeSelectorOpNotIn, Values: []string{"reserved"}},
		{Key: "team", Operator: corev1.NodeSelectorOpExists},
	}}

	require.True(t, mergeKarpenterRequirements(spec, map[string][]string{
		"capacity": {"on-demand"},
		"team":     {"inference"},
	}))
	require.Equal(t, map[string][]string{
		"capacity": {"on-demand"},
		"team":     {"inference"},
	}, requirementValuesByKey(spec.Requirements))

	require.False(t, mergeKarpenterRequirements(spec, map[string][]string{
		"capacity": {"reserved"},
	}))
}

func TestMergeKarpenterRequirementsPreservesMinValues(t *testing.T) {
	minValues := 2
	spec := &karpv1.NodeClaimSpec{Requirements: []karpv1.NodeSelectorRequirementWithMinValues{{
		Key: corev1.LabelInstanceTypeStable, Operator: corev1.NodeSelectorOpIn,
		Values: []string{"g6.2xlarge", "g6.12xlarge"}, MinValues: &minValues,
	}}}

	require.False(t, mergeKarpenterRequirements(spec, map[string][]string{
		corev1.LabelInstanceTypeStable: {"g6.2xlarge"},
	}))
	require.ElementsMatch(t, []string{"g6.2xlarge", "g6.12xlarge"}, spec.Requirements[0].Values)
}

func requirementValuesByKey(requirements []karpv1.NodeSelectorRequirementWithMinValues) map[string][]string {
	values := make(map[string][]string, len(requirements))
	for _, requirement := range requirements {
		values[requirement.Key] = requirement.Values
	}
	return values
}

func TestPodKarpenterRequirementValuesIntersectsSelectorAndAffinity(t *testing.T) {
	pod := &corev1.Pod{Spec: corev1.PodSpec{
		NodeSelector: map[string]string{corev1.LabelInstanceTypeStable: "g6.12xlarge"},
		Affinity: &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{NodeSelectorTerms: []corev1.NodeSelectorTerm{{
				MatchExpressions: []corev1.NodeSelectorRequirement{{
					Key: corev1.LabelInstanceTypeStable, Operator: corev1.NodeSelectorOpIn, Values: []string{"g6.2xlarge", "g6.12xlarge"},
				}},
			}}},
		}},
	}}
	values := podKarpenterRequirementValues(pod, nil)
	require.Equal(t, []string{"g6.12xlarge"}, values[corev1.LabelInstanceTypeStable])
}

func TestPodKarpenterRequirementValuesIncludesRegion(t *testing.T) {
	pod := &corev1.Pod{Spec: corev1.PodSpec{
		NodeSelector: map[string]string{corev1.LabelTopologyRegion: "us-west-2"},
	}}

	values := podKarpenterRequirementValues(pod, nil)
	require.Equal(t, []string{"us-west-2"}, values[corev1.LabelTopologyRegion])
}

func TestPreferredExpansionCandidatesFollowWeight(t *testing.T) {
	pod := &corev1.Pod{Spec: corev1.PodSpec{Affinity: &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
		PreferredDuringSchedulingIgnoredDuringExecution: []corev1.PreferredSchedulingTerm{
			{Weight: 50, Preference: corev1.NodeSelectorTerm{MatchExpressions: []corev1.NodeSelectorRequirement{{
				Key: corev1.LabelInstanceTypeStable, Operator: corev1.NodeSelectorOpIn, Values: []string{"g6.12xlarge"},
			}}}},
			{Weight: 100, Preference: corev1.NodeSelectorTerm{MatchExpressions: []corev1.NodeSelectorRequirement{{
				Key: corev1.LabelInstanceTypeStable, Operator: corev1.NodeSelectorOpIn, Values: []string{"g6.2xlarge"},
			}}}},
			{Weight: 75, Preference: corev1.NodeSelectorTerm{MatchExpressions: []corev1.NodeSelectorRequirement{{
				Key: corev1.LabelInstanceTypeStable, Operator: corev1.NodeSelectorOpIn, Values: []string{"g6.2xlarge"},
			}}}},
		},
	}}}}

	candidates := preferredExpansionCandidates(pod)
	require.Len(t, candidates, 3)
	require.Equal(t, []string{"g6.2xlarge"}, candidates[0].requirements[corev1.LabelInstanceTypeStable])
	require.Equal(t, []string{"g6.12xlarge"}, candidates[1].requirements[corev1.LabelInstanceTypeStable])
	require.Equal(t, relaxedExpansionCandidate, candidates[2].candidate)

}

func TestPreferredExpansionCandidatesIncludeRegion(t *testing.T) {
	pod := &corev1.Pod{Spec: corev1.PodSpec{Affinity: &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
		PreferredDuringSchedulingIgnoredDuringExecution: []corev1.PreferredSchedulingTerm{
			{Weight: 100, Preference: corev1.NodeSelectorTerm{MatchExpressions: []corev1.NodeSelectorRequirement{{
				Key: corev1.LabelTopologyRegion, Operator: corev1.NodeSelectorOpIn, Values: []string{"us-west-2"},
			}}}},
		},
	}}}}

	candidates := preferredExpansionCandidates(pod)
	require.Len(t, candidates, 2)
	require.Equal(t, []string{"us-west-2"}, candidates[0].requirements[corev1.LabelTopologyRegion])
	require.Equal(t, relaxedExpansionCandidate, candidates[1].candidate)
}

func TestFailedExpansionCandidateState(t *testing.T) {
	expander := &NodeExpander{failedExpansionCandidates: make(map[string]map[string]struct{})}
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "worker", UID: types.UID("worker-uid")}}
	pod.Spec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
		PreferredDuringSchedulingIgnoredDuringExecution: []corev1.PreferredSchedulingTerm{
			{Weight: 100, Preference: corev1.NodeSelectorTerm{MatchExpressions: []corev1.NodeSelectorRequirement{{
				Key: corev1.LabelInstanceTypeStable, Operator: corev1.NodeSelectorOpIn, Values: []string{"g6.2xlarge"},
			}}}},
			{Weight: 50, Preference: corev1.NodeSelectorTerm{MatchExpressions: []corev1.NodeSelectorRequirement{{
				Key: corev1.LabelInstanceTypeStable, Operator: corev1.NodeSelectorOpIn, Values: []string{"g6.12xlarge"},
			}}}},
		},
	}}
	podKey := client.ObjectKeyFromObject(pod)
	remaining := expander.expansionPreferencesToTry(pod)
	require.Len(t, remaining, 3)
	high := remaining[0]
	require.Equal(t, []string{"g6.2xlarge"}, high.requirements[corev1.LabelInstanceTypeStable])

	expander.addFailedExpansionCandidate(podKey, pod.UID, high.candidate)
	remaining = expander.expansionPreferencesToTry(pod)
	require.Len(t, remaining, 2)
	low := remaining[0]
	require.Equal(t, []string{"g6.12xlarge"}, low.requirements[corev1.LabelInstanceTypeStable])

	expander.addFailedExpansionCandidate(podKey, pod.UID, low.candidate)
	remaining = expander.expansionPreferencesToTry(pod)
	require.Len(t, remaining, 1)
	relaxed := remaining[0]
	require.Equal(t, relaxedExpansionCandidate, relaxed.candidate)

	expander.addFailedExpansionCandidate(podKey, pod.UID, relaxed.candidate)
	require.Empty(t, expander.expansionPreferencesToTry(pod))
}

func TestFailedExpansionCandidateStateIsScopedByPodUID(t *testing.T) {
	expander := &NodeExpander{failedExpansionCandidates: make(map[string]map[string]struct{})}
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "worker", UID: types.UID("old-uid")}}
	pod.Spec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
		PreferredDuringSchedulingIgnoredDuringExecution: []corev1.PreferredSchedulingTerm{
			{
				Weight: 100,
				Preference: corev1.NodeSelectorTerm{MatchExpressions: []corev1.NodeSelectorRequirement{
					{
						Key:      corev1.LabelInstanceTypeStable,
						Operator: corev1.NodeSelectorOpIn,
						Values:   []string{"g6.2xlarge"},
					},
				}},
			},
		},
	}}

	first := expander.expansionPreferencesToTry(pod)[0]
	expander.addFailedExpansionCandidate(client.ObjectKeyFromObject(pod), pod.UID, first.candidate)

	recreated := pod.DeepCopy()
	recreated.UID = types.UID("new-uid")
	require.Equal(t, first.candidate, expander.expansionPreferencesToTry(recreated)[0].candidate)
	require.NotContains(t, expander.failedExpansionCandidates, expansionPodKey(pod.Namespace, pod.Name, pod.UID))
}

func TestClearFailedExpansionCandidatesForDeletedPod(t *testing.T) {
	expander := &NodeExpander{failedExpansionCandidates: make(map[string]map[string]struct{})}
	expander.addFailedExpansionCandidate(client.ObjectKey{Namespace: "default", Name: "worker"}, "old-uid", "old")
	expander.addFailedExpansionCandidate(client.ObjectKey{Namespace: "default", Name: "worker"}, "new-uid", "new")
	expander.addFailedExpansionCandidate(client.ObjectKey{Namespace: "default", Name: "other"}, "other-uid", "other")

	expander.ClearFailedExpansionCandidatesForPod("default", "worker")

	require.NotContains(t, expander.failedExpansionCandidates, expansionPodKey("default", "worker", "old-uid"))
	require.NotContains(t, expander.failedExpansionCandidates, expansionPodKey("default", "worker", "new-uid"))
	require.Contains(t, expander.failedExpansionCandidates, expansionPodKey("default", "other", "other-uid"))
}

func TestNodeScalerInfoExposesExpansionState(t *testing.T) {
	expander := &NodeExpander{
		inFlightNodes:             make(map[string][]*tfv1.GPU),
		failedExpansionCandidates: make(map[string]map[string]struct{}),
		preSchedulePods:           make(map[string]*tfv1.AllocRequest),
		preScheduleTimers:         make(map[string]*time.Timer),
	}
	claim := inFlightNodeClaim{
		podKey:    client.ObjectKey{Namespace: "default", Name: "worker"},
		podUID:    "worker-uid",
		candidate: "candidate-a",
	}
	expander.inFlightNodeClaims.Store("claim-a", claim)
	expander.addFailedExpansionCandidate(claim.podKey, claim.podUID, claim.candidate)

	info := expander.GetNodeScalerInfo().(map[string]any)
	inFlight := info["inFlightNodeClaims"].(map[string]any)
	require.Equal(t, map[string]string{
		"pod": "default/worker", "podUID": "worker-uid", "candidate": "candidate-a",
	}, inFlight["claim-a"])
	failed := info["failedExpansionCandidates"].(map[string][]string)
	require.Equal(t, []string{"candidate-a"}, failed[expansionPodKey("default", "worker", "worker-uid")])
}
