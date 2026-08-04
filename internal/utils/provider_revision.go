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

package utils

import (
	"encoding/json"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

// ProviderConfigRevisions returns a copy of the vendor revision map stored on
// a pool. Invalid or missing data is treated as empty so reconciliation can
// repair it on the next ProviderConfig event.
func ProviderConfigRevisions(pool *tfv1.GPUPool) map[string]string {
	revisions := map[string]string{}
	if pool == nil || pool.Annotations == nil {
		return revisions
	}
	raw := pool.Annotations[constants.ProviderConfigRevisionsAnnotation]
	if raw == "" {
		return revisions
	}
	if err := json.Unmarshal([]byte(raw), &revisions); err != nil {
		return map[string]string{}
	}
	return revisions
}

// SetProviderConfigRevision updates one vendor revision on a pool. An empty
// revision removes the vendor, which represents falling back after deletion.
func SetProviderConfigRevision(pool *tfv1.GPUPool, vendor, revision string) error {
	if pool.Annotations == nil {
		pool.Annotations = map[string]string{}
	}
	revisions := ProviderConfigRevisions(pool)
	key := normalizeProviderVendor(vendor)
	if revision == "" {
		delete(revisions, key)
	} else {
		revisions[key] = revision
	}
	if len(revisions) == 0 {
		delete(pool.Annotations, constants.ProviderConfigRevisionsAnnotation)
		return nil
	}
	raw, err := json.Marshal(revisions)
	if err != nil {
		return err
	}
	pool.Annotations[constants.ProviderConfigRevisionsAnnotation] = string(raw)
	return nil
}

func ProviderConfigRevision(pool *tfv1.GPUPool, vendor string) string {
	return ProviderConfigRevisions(pool)[normalizeProviderVendor(vendor)]
}

func normalizeProviderVendor(vendor string) string {
	return strings.ToLower(strings.TrimSpace(vendor))
}
