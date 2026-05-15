package workload

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// TestUpdateWorkloadState_DoesNotClobberRecommenderStatusWrites guards against
// the lost-write transaction race: while processSingleWorkload is mid-cycle
// (recommender already wrote Status.ActiveCronScalingRule / Recommendation /
// AppliedRecommendedReplicas / conditions into State, but UpdateWorkloadStatus
// has not yet patched the CR), a concurrent loadWorkloads tick must NOT
// overwrite the in-memory Status with the stale API view.
func TestUpdateWorkloadState_DoesNotClobberRecommenderStatusWrites(t *testing.T) {
	scheme := runtime.NewScheme()
	assert.NoError(t, corev1.AddToScheme(scheme))
	assert.NoError(t, tfv1.AddToScheme(scheme))

	apiStatus := tfv1.TensorFusionWorkloadStatus{
		WorkerCount:                3,
		ReadyWorkers:               3,
		Phase:                      tfv1.TensorFusionWorkloadPhaseRunning,
		PodTemplateHash:            "abc",
		Recommendation:             nil, // CR has no recommendation persisted yet
		AppliedRecommendedReplicas: 0,
		ActiveCronScalingRule:      nil,
	}
	wl := &tfv1.TensorFusionWorkload{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "wl"},
		Spec: tfv1.WorkloadProfileSpec{
			Resources: tfv1.Resources{
				Requests: tfv1.Resource{Tflops: resource.MustParse("10"), Vram: resource.MustParse("1Gi")},
				Limits:   tfv1.Resource{Tflops: resource.MustParse("20"), Vram: resource.MustParse("2Gi")},
			},
		},
		Status: apiStatus,
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(wl).WithStatusSubresource(wl).Build()
	h := NewHandler(c, nil)

	ws := NewWorkloadState()

	// First seed: state.Status is populated from the API CR.
	assert.NoError(t, h.UpdateWorkloadState(context.Background(), ws, wl))
	_, _, statusAfterSeed := ws.StatusSnapshot()
	assert.Equal(t, int32(3), statusAfterSeed.WorkerCount, "first load should seed Status from CR")
	assert.True(t, ws.statusSeeded, "statusSeeded flag should be flipped")

	// Recommender writes happen in-memory between Recommend and
	// UpdateWorkloadStatus. Simulate that.
	rec := &tfv1.Resources{
		Requests: tfv1.Resource{Tflops: resource.MustParse("99"), Vram: resource.MustParse("9Gi")},
		Limits:   tfv1.Resource{Tflops: resource.MustParse("99"), Vram: resource.MustParse("9Gi")},
	}
	cronRule := &tfv1.CronScalingRule{Name: "active-rule", Enable: true}
	ws.Mu.Lock()
	ws.Status.Recommendation = rec.DeepCopy()
	ws.Status.AppliedRecommendedReplicas = 2
	ws.Mu.Unlock()
	ws.SetActiveCronScalingRule(cronRule.DeepCopy())
	ws.UpsertStatusCondition(metav1.Condition{
		Type:    "ResourceUpdate",
		Status:  metav1.ConditionTrue,
		Reason:  "Updated",
		Message: "scaled up",
	})

	// loadWorkloads ticks concurrently — UpdateWorkloadState fires again with
	// the SAME stale API status (CR hasn't been patched yet). The fix must
	// preserve the in-memory writes above.
	assert.NoError(t, h.UpdateWorkloadState(context.Background(), ws, wl))

	_, _, statusAfter := ws.StatusSnapshot()
	assert.NotNil(t, statusAfter.Recommendation, "Recommendation must survive concurrent UpdateWorkloadState")
	assert.True(t, statusAfter.Recommendation.Equal(rec),
		"Recommendation value must match what recommender wrote")
	assert.Equal(t, int32(2), statusAfter.AppliedRecommendedReplicas,
		"AppliedRecommendedReplicas must survive")
	assert.NotNil(t, statusAfter.ActiveCronScalingRule, "ActiveCronScalingRule must survive")
	assert.Equal(t, "active-rule", statusAfter.ActiveCronScalingRule.Name)

	cond := findCondition(statusAfter.Conditions, "ResourceUpdate")
	assert.NotNil(t, cond, "Conditions written by recommender must survive")
	assert.Equal(t, metav1.ConditionTrue, cond.Status)
}

func findCondition(conds []metav1.Condition, t string) *metav1.Condition {
	for i := range conds {
		if conds[i].Type == t {
			return &conds[i]
		}
	}
	return nil
}

// TestSetRecommendation_VisibleToReaders guards that SetRecommendation makes
// the recommendation visible to GetCurrentResourcesSpec / LatestRecommendation
// / IsRecommendationAppliedToAllWorkers across reconcile rounds. The
// production path must call this after every successful GetRecommendation
// returning non-nil, otherwise (after the seed-once Status refactor)
// State.Status.Recommendation stays at its seeded value forever and:
//   - GetCurrentResourcesSpec falls back to original Spec.Resources, breaking
//     incremental scaling
//   - LatestRecommendation returns nil so the Apply retry path never recovers
//     a partially-applied recommendation
//   - IsRecommendationAppliedToAllWorkers short-circuits to true even when no
//     recommendation has been applied to any worker
func TestSetRecommendation_VisibleToReaders(t *testing.T) {
	ws := NewWorkloadState()
	ws.Spec.Resources = tfv1.Resources{
		Requests: tfv1.Resource{Tflops: resource.MustParse("10"), Vram: resource.MustParse("1Gi")},
		Limits:   tfv1.Resource{Tflops: resource.MustParse("20"), Vram: resource.MustParse("2Gi")},
	}

	// Fresh state: no recommendation, readers fall back to Spec.Resources.
	assert.Nil(t, ws.LatestRecommendation())
	assert.True(t, ws.GetCurrentResourcesSpec().Equal(&ws.Spec.Resources))

	rec := &tfv1.Resources{
		Requests: tfv1.Resource{Tflops: resource.MustParse("80"), Vram: resource.MustParse("8Gi")},
		Limits:   tfv1.Resource{Tflops: resource.MustParse("100"), Vram: resource.MustParse("10Gi")},
	}
	ws.SetRecommendation(rec)

	assert.NotNil(t, ws.LatestRecommendation())
	assert.True(t, ws.LatestRecommendation().Equal(rec),
		"LatestRecommendation must reflect the value set via SetRecommendation")
	assert.True(t, ws.GetCurrentResourcesSpec().Equal(rec),
		"GetCurrentResourcesSpec must prefer the in-memory recommendation over Spec.Resources "+
			"so the next reconcile round uses an incrementally correct baseline")

	// Mutating the caller's copy must not bleed into State (deep copy contract).
	rec.Requests.Tflops = resource.MustParse("999")
	stored := ws.LatestRecommendation()
	assert.NotEqual(t, "999", stored.Requests.Tflops.String(),
		"SetRecommendation must deep-copy to insulate State from caller mutations")

	// Clearing via nil resets to fallback behavior.
	ws.SetRecommendation(nil)
	assert.Nil(t, ws.LatestRecommendation())
	assert.True(t, ws.GetCurrentResourcesSpec().Equal(&ws.Spec.Resources))
}

// TestMatchesUID_FreshAndBoundStates documents the UID-tracking contract used
// by loadWorkloads to detect same-namespace/name workload recreation.
func TestMatchesUID_FreshAndBoundStates(t *testing.T) {
	ws := NewWorkloadState()
	// Fresh state: any UID matches so the first UpdateWorkloadState can claim it.
	assert.True(t, ws.MatchesUID("uid-A"))
	assert.True(t, ws.MatchesUID("uid-B"))
	assert.True(t, ws.MatchesUID(""))

	ws.Mu.Lock()
	ws.uid = "uid-A"
	ws.Mu.Unlock()

	// Bound state: only its own UID matches.
	assert.True(t, ws.MatchesUID("uid-A"))
	assert.False(t, ws.MatchesUID("uid-B"))
	assert.False(t, ws.MatchesUID(""))
}

// TestUpdateWorkloadState_RecordsUID guards that UpdateWorkloadState stores
// the workload UID on the State so loadWorkloads can detect recreation.
func TestUpdateWorkloadState_RecordsUID(t *testing.T) {
	scheme := runtime.NewScheme()
	assert.NoError(t, corev1.AddToScheme(scheme))
	assert.NoError(t, tfv1.AddToScheme(scheme))

	wl := &tfv1.TensorFusionWorkload{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "wl", UID: "uid-A"},
		Spec: tfv1.WorkloadProfileSpec{
			Resources: tfv1.Resources{
				Requests: tfv1.Resource{Tflops: resource.MustParse("10"), Vram: resource.MustParse("1Gi")},
				Limits:   tfv1.Resource{Tflops: resource.MustParse("20"), Vram: resource.MustParse("2Gi")},
			},
		},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(wl).WithStatusSubresource(wl).Build()
	h := NewHandler(c, nil)

	ws := NewWorkloadState()
	assert.NoError(t, h.UpdateWorkloadState(context.Background(), ws, wl))

	assert.True(t, ws.MatchesUID("uid-A"))
	assert.False(t, ws.MatchesUID("uid-B"),
		"after seeding with workload UID 'uid-A', a different UID must NOT match — "+
			"loadWorkloads relies on this to detect same-name workload recreation")
}
