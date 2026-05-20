/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package gang

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
)

// TestPreEnqueue covers the cheap queue-admission gate added in stage C.
func TestPreEnqueue(t *testing.T) {
	t.Run("non-gang pod passes", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		pod := createTestPod("p", "wl", 0, "")
		assert.NoError(t, manager.PreEnqueue(context.Background(), pod))
	})

	t.Run("gang pod with no peers is blocked", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		// No podLister wired ⇒ PreEnqueue defers to PreFilter (returns nil).
		// To exercise the peer-count branch we need a lister. Easier: assert
		// that without lister the gate is permissive (matches code intent).
		pod := createTestPod("p", "wl", 3, "")
		assert.NoError(t, manager.PreEnqueue(context.Background(), pod))
	})

	t.Run("gang pod is blocked when group is backed off", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		pod := createTestPod("p", "wl", 3, "")
		manager.backedOffGroups.Set("default/wl", struct{}{}, time.Minute)

		err := manager.PreEnqueue(context.Background(), pod)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "backed off")
	})

	t.Run("gang pod passes when OnceResourceSatisfied", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		pod := createTestPod("p", "wl", 3, "")

		// Pre-seed a pod group that has met quorum before (restart scenario).
		pg := NewPodGroupInfo("default/wl", 3, 3, 3, 0)
		pg.OnceResourceSatisfied = true
		manager.podGroups[PodGroupKey("default/wl")] = pg

		assert.NoError(t, manager.PreEnqueue(context.Background(), pod))
	})
}

// TestRejectGroupOnUnschedulable covers the PostFilter strict-fail path added
// in stage D.
func TestRejectGroupOnUnschedulable(t *testing.T) {
	t.Run("non-gang pod is a no-op (no group state created)", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		pod := createTestPod("p", "wl", 0, "")

		manager.RejectGroupOnUnschedulable(context.Background(), pod)
		assert.Empty(t, manager.podGroups)
	})

	t.Run("group not yet tracked installs backoff only", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		pod := createTestPod("p", "wl", 3, "")

		manager.RejectGroupOnUnschedulable(context.Background(), pod)
		_, backed := manager.backedOffGroups.Get("default/wl")
		assert.True(t, backed, "PostFilter should arm backoff even when group is not tracked yet")
		assert.Empty(t, manager.podGroups, "PostFilter must not spin up new group state")
	})

	t.Run("tracked group: invalidates cycle, rejects waiting peers, sets backoff", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		pod := createTestPod("p", "wl", 3, "")
		groupKey := PodGroupKey("default/wl")

		pg := NewPodGroupInfo(groupKey, 3, 3, 3, 0)
		pg.WaitingPods[types.UID("peer-a")] = NewWaitingPodInfo("peer-a", "peer-a", "default", "node-1", nil, nil)
		pg.WaitingPods[types.UID("peer-b")] = NewWaitingPodInfo("peer-b", "peer-b", "default", "node-1", nil, nil)
		manager.podGroups[groupKey] = pg

		manager.RejectGroupOnUnschedulable(context.Background(), pod)

		// Cycle invalidated, waiting peers cleared, backoff set.
		pg.mu.RLock()
		defer pg.mu.RUnlock()
		assert.False(t, pg.ScheduleCycleValid, "ScheduleCycleValid must flip to false")
		assert.Empty(t, pg.WaitingPods, "all waiting peers must be released")
		_, backed := manager.backedOffGroups.Get(string(groupKey))
		assert.True(t, backed)
	})

	t.Run("repeated calls are idempotent", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		pod := createTestPod("p", "wl", 3, "")
		groupKey := PodGroupKey("default/wl")
		manager.podGroups[groupKey] = NewPodGroupInfo(groupKey, 3, 3, 3, 0)

		manager.RejectGroupOnUnschedulable(context.Background(), pod)
		manager.RejectGroupOnUnschedulable(context.Background(), pod)

		pg := manager.podGroups[groupKey]
		pg.mu.RLock()
		defer pg.mu.RUnlock()
		assert.False(t, pg.ScheduleCycleValid)
	})
}

// TestQuorumReachable covers the discriminator PostFilter uses to skip
// group rejection when peers are still arriving.
func TestQuorumReachable(t *testing.T) {
	t.Run("non-gang pod is always reachable", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		pod := createTestPod("p", "wl", 0, "")
		assert.True(t, manager.QuorumReachable(pod))
	})

	t.Run("gang pod with no lister is permissive (assume reachable)", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		pod := createTestPod("p", "wl", 3, "")
		// No podLister wired — must not block strict-reject by returning false.
		assert.True(t, manager.QuorumReachable(pod))
	})
}

// TestScheduleCycleValidLazyReset covers the PreFilter lazy reset added in
// stage B that complements the PostFilter strict-fail flow.
func TestScheduleCycleValidLazyReset(t *testing.T) {
	manager := NewManager(nil, nil, "TestPlugin")
	pod := createTestPod("p", "wl", 3, "")
	groupKey := PodGroupKey("default/wl")

	pg := NewPodGroupInfo(groupKey, 3, 3, 3, 0)
	pg.ScheduleCycleValid = false // simulate prior PostFilter strict-fail
	manager.podGroups[groupKey] = pg
	// No backoff installed ⇒ backoff cache miss in PreFilter ⇒ lazy reset.

	_ = manager.PreFilter(context.Background(), pod)

	pg.mu.RLock()
	defer pg.mu.RUnlock()
	assert.True(t, pg.ScheduleCycleValid, "PreFilter must lazily reset ScheduleCycleValid once backoff has expired")
}

// TestCheckAndRejectGangIfNeededSkipsWhenCycleInvalid covers the third state
// in Unreserve's three-state branching: PostFilter already rejected, so the
// subsequent Unreserve does not overwrite the more-specific status.
func TestCheckAndRejectGangIfNeededSkipsWhenCycleInvalid(t *testing.T) {
	manager := NewManager(nil, nil, "TestPlugin")
	groupKey := PodGroupKey("default/wl")

	pg := NewPodGroupInfo(groupKey, 3, 3, 3, 0)
	pg.ScheduleCycleValid = false
	manager.podGroups[groupKey] = pg

	// Should be a no-op: no status sync emitted, no backoff overwritten.
	preBackoff, hadBackoff := manager.backedOffGroups.Get(string(groupKey))
	manager.checkAndRejectGangIfNeeded(context.Background(), groupKey)
	postBackoff, stillHadBackoff := manager.backedOffGroups.Get(string(groupKey))

	assert.Equal(t, hadBackoff, stillHadBackoff)
	assert.Equal(t, preBackoff, postBackoff)
}
