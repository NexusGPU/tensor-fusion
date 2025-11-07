package indexallocator

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestIndexAllocator_AssignIndex(t *testing.T) {
	scheme := fake.NewClientBuilder().WithScheme(fake.NewClientBuilder().Build().Scheme()).Build().Scheme()
	client := fake.NewClientBuilder().WithScheme(scheme).Build()

	ctx := context.Background()
	allocator, err := NewIndexAllocator(ctx, client)
	require.NoError(t, err)
	allocator.IsLeader = true

	// Test assigning first index
	index1, err := allocator.AssignIndex("pod-1")
	assert.NoError(t, err)
	assert.Equal(t, 1, index1)

	// Test assigning second index
	index2, err := allocator.AssignIndex("pod-2")
	assert.NoError(t, err)
	assert.Equal(t, 2, index2)

	// Test assigning multiple indices - verify ascending order
	for i := 3; i <= 10; i++ {
		index, err := allocator.AssignIndex("pod-" + string(rune(i)))
		assert.NoError(t, err)
		assert.Equal(t, i, index, "index should be assigned in ascending order")
	}
}

func TestIndexAllocator_AssignIndex_IncrementalOrder(t *testing.T) {
	scheme := fake.NewClientBuilder().WithScheme(fake.NewClientBuilder().Build().Scheme()).Build().Scheme()
	client := fake.NewClientBuilder().WithScheme(scheme).Build()

	ctx := context.Background()
	allocator, err := NewIndexAllocator(ctx, client)
	require.NoError(t, err)
	allocator.IsLeader = true

	// Test that indices are assigned in incremental order (1, 2, 3, ...)
	expectedIndex := 1
	for i := 0; i < 20; i++ {
		index, err := allocator.AssignIndex("pod-" + string(rune(i)))
		assert.NoError(t, err)
		assert.Equal(t, expectedIndex, index, "index should be assigned in ascending order")
		expectedIndex++
	}
}

func TestIndexAllocator_ReleaseIndex(t *testing.T) {
	scheme := fake.NewClientBuilder().WithScheme(fake.NewClientBuilder().Build().Scheme()).Build().Scheme()
	client := fake.NewClientBuilder().WithScheme(scheme).Build()

	ctx := context.Background()
	allocator, err := NewIndexAllocator(ctx, client)
	require.NoError(t, err)
	allocator.IsLeader = true

	// Assign an index
	index, err := allocator.AssignIndex("pod-1")
	require.NoError(t, err)

	// Release it immediately
	err = allocator.ReleaseIndex("pod-1", index, true)
	assert.NoError(t, err)

	// Should be able to assign the same index again (wrapped around)
	index2, err := allocator.AssignIndex("pod-2")
	assert.NoError(t, err)
	assert.Equal(t, index, index2, "released index should be available for reuse")
}

func TestIndexAllocator_InitBitmap(t *testing.T) {
	scheme := fake.NewClientBuilder().WithScheme(fake.NewClientBuilder().Build().Scheme()).Build().Scheme()

	// Create pods with index annotations
	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-1",
			Namespace: "default",
			Annotations: map[string]string{
				"tensor-fusion.ai/index": "1",
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}
	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-2",
			Namespace: "default",
			Annotations: map[string]string{
				"tensor-fusion.ai/index": "5",
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}

	client := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pod1, pod2).Build()

	ctx := context.Background()
	allocator, err := NewIndexAllocator(ctx, client)
	require.NoError(t, err)

	// Initialize bitmap
	allocator.initBitmap(ctx)

	// Check that indices 1 and 5 are marked as used
	assert.True(t, allocator.Bitmap[0]&(1<<0) != 0) // index 1
	assert.True(t, allocator.Bitmap[0]&(1<<4) != 0) // index 5

	// Next assignment should start from 2 (first available after 1 and 5)
	allocator.IsLeader = true
	index, err := allocator.AssignIndex("pod-3")
	assert.NoError(t, err)
	assert.Equal(t, 2, index, "should assign first available index (2) after 1 and 5 are used")
}
