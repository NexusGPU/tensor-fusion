package gpuallocator

import (
	"context"
	"testing"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/taints"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func nodeWithOptionalTaint(name string, withTaint bool) *corev1.Node {
	n := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}}
	if withTaint {
		n.Spec.Taints = []corev1.Taint{{
			Key:    constants.NodeUsedByTaintKey,
			Effect: corev1.TaintEffectPreferNoSchedule,
			Value:  constants.TensorFusionSystemName,
		}}
	}
	return n
}

func TestReconcileProgressiveTaint(t *testing.T) {
	orig := utils.IsProgressiveMigration()
	t.Cleanup(func() { utils.SetProgressiveMigration(orig) })
	utils.SetProgressiveMigration(true)

	scheme := runtime.NewScheme()
	assert.NoError(t, corev1.AddToScheme(scheme))

	check := func(t *testing.T, taintKey string, expectHasTaint bool, node *corev1.Node) {
		t.Helper()
		taint := &corev1.Taint{Key: taintKey, Effect: corev1.TaintEffectPreferNoSchedule, Value: constants.TensorFusionSystemName}
		assert.Equal(t, expectHasTaint, taints.TaintExists(node.Spec.Taints, taint))
	}

	t.Run("idle+hasTaint => remove", func(t *testing.T) {
		node := nodeWithOptionalTaint("n1", true)
		c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node).Build()

		assert.NoError(t, reconcileProgressiveTaint(context.Background(), c, node, true))

		got := &corev1.Node{}
		assert.NoError(t, c.Get(context.Background(), client.ObjectKey{Name: "n1"}, got))
		check(t, constants.NodeUsedByTaintKey, false, got)
	})

	t.Run("!idle+!hasTaint => add", func(t *testing.T) {
		node := nodeWithOptionalTaint("n2", false)
		c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node).Build()

		assert.NoError(t, reconcileProgressiveTaint(context.Background(), c, node, false))

		got := &corev1.Node{}
		assert.NoError(t, c.Get(context.Background(), client.ObjectKey{Name: "n2"}, got))
		check(t, constants.NodeUsedByTaintKey, true, got)
	})

	t.Run("idle+!hasTaint => no-op", func(t *testing.T) {
		node := nodeWithOptionalTaint("n3", false)
		c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node).Build()

		assert.NoError(t, reconcileProgressiveTaint(context.Background(), c, node, true))

		got := &corev1.Node{}
		assert.NoError(t, c.Get(context.Background(), client.ObjectKey{Name: "n3"}, got))
		assert.Equal(t, node.ResourceVersion, got.ResourceVersion, "no Update should have been issued")
	})

	t.Run("!idle+hasTaint => no-op", func(t *testing.T) {
		node := nodeWithOptionalTaint("n4", true)
		c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node).Build()

		assert.NoError(t, reconcileProgressiveTaint(context.Background(), c, node, false))

		got := &corev1.Node{}
		assert.NoError(t, c.Get(context.Background(), client.ObjectKey{Name: "n4"}, got))
		assert.Equal(t, node.ResourceVersion, got.ResourceVersion)
	})

	t.Run("progressive migration disabled => no-op", func(t *testing.T) {
		utils.SetProgressiveMigration(false)
		defer utils.SetProgressiveMigration(true)
		node := nodeWithOptionalTaint("n5", true)
		c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node).Build()

		assert.NoError(t, reconcileProgressiveTaint(context.Background(), c, node, true))

		got := &corev1.Node{}
		assert.NoError(t, c.Get(context.Background(), client.ObjectKey{Name: "n5"}, got))
		check(t, constants.NodeUsedByTaintKey, true, got)
	})

	t.Run("nil coreNode => no-op", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(scheme).Build()
		assert.NoError(t, reconcileProgressiveTaint(context.Background(), c, nil, true))
	})
}
