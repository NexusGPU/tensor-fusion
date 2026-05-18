package device

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	internalconfig "github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

type providerTemplateConfig struct {
	HardwareMetadata        []tfv1.HardwareModelInfo      `json:"hardwareMetadata"`
	VirtualizationTemplates []tfv1.VirtualizationTemplate `json:"virtualizationTemplates"`
}

var (
	providerTemplateConfigOnce sync.Once
	providerTemplateConfigData providerTemplateConfig
)

func resolveProviderTemplateID(vendor, model, requestedTemplateID string) string {
	requestedTemplateID = strings.TrimSpace(requestedTemplateID)
	if requestedTemplateID == "" {
		return requestedTemplateID
	}

	cfg := loadProviderTemplateConfig()
	if len(cfg.HardwareMetadata) == 0 || len(cfg.VirtualizationTemplates) == 0 {
		return requestedTemplateID
	}

	templateByID := make(map[string]tfv1.VirtualizationTemplate, len(cfg.VirtualizationTemplates))
	for _, tmpl := range cfg.VirtualizationTemplates {
		templateByID[tmpl.ID] = tmpl
	}

	for _, hw := range cfg.HardwareMetadata {
		if !providerTemplateMatchesModel(vendor, model, hw) {
			continue
		}
		for _, ref := range hw.PartitionTemplateRefs {
			tmpl, ok := templateByID[ref]
			if !ok {
				continue
			}
			if internalconfig.MatchTemplateIdentifier(tmpl.ID, tmpl.Name, requestedTemplateID) {
				return downstreamTemplateID(tmpl)
			}
		}
	}

	return requestedTemplateID
}

// downstreamTemplateID returns the identifier to forward to the vendor partition
// API (vgpu-provider → NVML for NVIDIA). When NVMLProfileID is set, return it as
// a decimal string so the same numeric profile id can be reused across cards
// even though chart-side `id` values must be unique. Otherwise fall back to the
// chart-side `id` (existing behavior — required for vendors like Ascend that
// already use friendly string ids).
func downstreamTemplateID(tmpl tfv1.VirtualizationTemplate) string {
	if tmpl.NVMLProfileID != nil {
		return strconv.FormatUint(uint64(*tmpl.NVMLProfileID), 10)
	}
	return tmpl.ID
}

func loadProviderTemplateConfig() providerTemplateConfig {
	providerTemplateConfigOnce.Do(func() {
		raw := strings.TrimSpace(os.Getenv(constants.TFProviderTemplateConfigEnv))
		if raw == "" {
			return
		}
		var cfg providerTemplateConfig
		if err := json.Unmarshal([]byte(raw), &cfg); err == nil {
			providerTemplateConfigData = cfg
		}
	})
	return providerTemplateConfigData
}

func providerTemplateMatchesModel(vendor, model string, hw tfv1.HardwareModelInfo) bool {
	model = strings.TrimSpace(model)
	if model == "" {
		return false
	}

	candidates := []string{model}
	vendor = strings.TrimSpace(vendor)
	if vendor != "" && len(model) >= len(vendor) && strings.EqualFold(model[:len(vendor)], vendor) {
		if trimmed := strings.TrimSpace(model[len(vendor):]); trimmed != "" {
			candidates = append(candidates, trimmed)
		}
	}

	for _, candidate := range candidates {
		if strings.EqualFold(hw.Model, candidate) || strings.EqualFold(hw.FullModelName, candidate) {
			return true
		}
	}
	return false
}

func resetProviderTemplateConfigForTest() {
	providerTemplateConfigOnce = sync.Once{}
	providerTemplateConfigData = providerTemplateConfig{}
}
