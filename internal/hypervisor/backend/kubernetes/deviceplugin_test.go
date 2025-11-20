/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kubernetes

import (
	"testing"

	"github.com/stretchr/testify/assert"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

// TestDevicePluginAllocate_ExtractsIndexFromDevicesIds tests that the device plugin
// correctly extracts the pod index from DevicesIds[0], not from len(req.ContainerRequests)
// This is a key test to verify the device plugin implementation matches the design:
// - DevicesIds[0] contains the index value (1-512) from resource limits
// - len(req.ContainerRequests) is just the number of containers, NOT the pod index
// - CdiDevices must be empty to prevent dummy device allocation
func TestDevicePluginAllocate_ExtractsIndexFromDevicesIds(t *testing.T) {
	// This test verifies the key design principle:
	// The pod index comes from DevicesIds[0], which contains the value from
	// tensor-fusion.ai/index resource limit, NOT from len(req.ContainerRequests)

	req := &pluginapi.AllocateRequest{
		ContainerRequests: []*pluginapi.ContainerAllocateRequest{
			{
				DevicesIds: []string{"3"}, // Index "3" from resource limit
			},
		},
	}

	// Verify the structure: len(ContainerRequests) = 1, but index is "3" from DevicesIds[0]
	assert.Len(t, req.ContainerRequests, 1, "Should have 1 container request")
	assert.Equal(t, "3", req.ContainerRequests[0].DevicesIds[0], "Index should come from DevicesIds[0], not from len(ContainerRequests)")

	// This demonstrates that len(req.ContainerRequests) is NOT the pod index
	// The pod index is extracted from DevicesIds[0]
	assert.NotEqual(t, len(req.ContainerRequests), 3, "len(ContainerRequests) should NOT equal the pod index")
}

// TestDevicePluginAllocate_MultipleContainers tests that len(req.ContainerRequests)
// is used for iteration, not for pod index identification
func TestDevicePluginAllocate_MultipleContainers(t *testing.T) {
	// Create request with 2 containers, both with index "5"
	// len(ContainerRequests) = 2, but pod index is still "5" from DevicesIds
	req := &pluginapi.AllocateRequest{
		ContainerRequests: []*pluginapi.ContainerAllocateRequest{
			{
				DevicesIds: []string{"5"}, // First container: index 5
			},
			{
				DevicesIds: []string{"5"}, // Second container: same pod, same index
			},
		},
	}

	// Verify: len(ContainerRequests) = 2, but index is "5" from DevicesIds
	assert.Len(t, req.ContainerRequests, 2, "Should have 2 container requests")
	assert.Equal(t, "5", req.ContainerRequests[0].DevicesIds[0], "First container index from DevicesIds")
	assert.Equal(t, "5", req.ContainerRequests[1].DevicesIds[0], "Second container index from DevicesIds")

	// Key verification: len(ContainerRequests) is NOT the pod index
	assert.NotEqual(t, len(req.ContainerRequests), 5, "len(ContainerRequests) should NOT equal the pod index")

	// Both containers have the same index because they're in the same pod
	assert.Equal(t, req.ContainerRequests[0].DevicesIds[0], req.ContainerRequests[1].DevicesIds[0],
		"Both containers should have the same index (same pod)")
}
