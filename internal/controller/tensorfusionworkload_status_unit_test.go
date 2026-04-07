package controller

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDeriveWorkloadReadiness(t *testing.T) {
	t.Run("dynamic workload scaled to zero is pending and not ready", func(t *testing.T) {
		workload := &tfv1.TensorFusionWorkload{}
		phase, cond := deriveWorkloadReadiness(workload, 0, 0, 0)

		assert.Equal(t, tfv1.TensorFusionWorkloadPhasePending, phase)
		assert.Equal(t, constants.ConditionStatusTypeReady, cond.Type)
		assert.Equal(t, metav1.ConditionFalse, cond.Status)
		assert.Equal(t, "ScaledToZero", cond.Reason)
		assert.Equal(t, "Workload is scaled to zero", cond.Message)
	})

	t.Run("fixed workload scaled to zero is pending and not ready", func(t *testing.T) {
		replicas := int32(0)
		workload := &tfv1.TensorFusionWorkload{
			Spec: tfv1.WorkloadProfileSpec{
				Replicas: &replicas,
			},
		}
		phase, cond := deriveWorkloadReadiness(workload, 0, 0, 0)

		assert.Equal(t, tfv1.TensorFusionWorkloadPhasePending, phase)
		assert.Equal(t, metav1.ConditionFalse, cond.Status)
		assert.Equal(t, "ScaledToZero", cond.Reason)
	})

	t.Run("fixed workload with all replicas ready stays running", func(t *testing.T) {
		replicas := int32(2)
		workload := &tfv1.TensorFusionWorkload{
			Spec: tfv1.WorkloadProfileSpec{
				Replicas: &replicas,
			},
		}
		phase, cond := deriveWorkloadReadiness(workload, 2, 2, 0)

		assert.Equal(t, tfv1.TensorFusionWorkloadPhaseRunning, phase)
		assert.Equal(t, metav1.ConditionTrue, cond.Status)
		assert.Equal(t, "WorkloadReady", cond.Reason)
		assert.Equal(t, "All workers are running", cond.Message)
	})
}

func TestReconcileGangStatusInitialPendingReason(t *testing.T) {
	replicas := int32(3)
	workload := &tfv1.TensorFusionWorkload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "gang-workload",
			Namespace: "default",
		},
		Spec: tfv1.WorkloadProfileSpec{
			Replicas: &replicas,
			GangScheduling: &tfv1.GangSchedulingConfig{
				MinMembers: 2,
			},
		},
	}

	changed := reconcileGangStatus(context.Background(), nil, workload, 0, 0)
	assert.True(t, changed)
	if assert.NotNil(t, workload.Status.Gang) {
		assert.Equal(t, tfv1.GangSchedulingPhasePending, workload.Status.Gang.Phase)
		assert.Equal(t, "GangPending", workload.Status.Gang.Reason)
		assert.Equal(t, "Gang waiting for scheduling: 0/3 ready", workload.Status.Gang.Message)
	}

	phase, readyCond := deriveWorkloadReadiness(workload, replicas, 0, 0)
	effectivePhase, effectiveCond := reconcileWorkloadReadinessWithGang(workload, phase, readyCond)
	assert.Equal(t, tfv1.TensorFusionWorkloadPhasePending, effectivePhase)
	assert.Equal(t, metav1.ConditionFalse, effectiveCond.Status)
	assert.Equal(t, "GangPending", effectiveCond.Reason)
	assert.Equal(t, "Gang waiting for scheduling: 0/3 ready", effectiveCond.Message)
}
