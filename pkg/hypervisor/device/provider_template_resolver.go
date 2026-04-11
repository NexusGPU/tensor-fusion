package device

import (
	"encoding/json"
	"os"
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
				return tmpl.ID
			}
		}
	}

	return requestedTemplateID
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
