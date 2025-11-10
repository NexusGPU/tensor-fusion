package indexallocator

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

func TestIndexAllocator_WrapAround(t *testing.T) {
	scheme := fake.NewClientBuilder().WithScheme(fake.NewClientBuilder().Build().Scheme()).Build().Scheme()
	client := fake.NewClientBuilder().WithScheme(scheme).Build()

	ctx := context.Background()
	allocator, err := NewIndexAllocator(ctx, client)
	require.NoError(t, err)
	allocator.IsLeader = true

	// Assign indices until we reach 512
	for i := 1; i <= 512; i++ {
		index, err := allocator.AssignIndex("pod-" + string(rune(i)))
		assert.NoError(t, err)
		assert.Equal(t, i, index)
	}

	// Next assignment should wrap around to 1
	index, err := allocator.AssignIndex("pod-513")
	assert.NoError(t, err)
	assert.Equal(t, 1, index, "index should wrap around from 512 to 1")
}
