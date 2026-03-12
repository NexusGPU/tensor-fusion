package device

import (
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/google/cel-go/cel"
	"k8s.io/klog/v2"
)

type providerDeviceMountEnv struct {
	Default *providerDeviceMountConfig           `json:"default,omitempty"`
	Models  map[string]providerDeviceMountConfig `json:"models,omitempty"`
}

type providerDeviceMountConfig struct {
	DeviceMountRule            string   `json:"deviceMountRule,omitempty"`
	PartitionedDeviceMountRule string   `json:"partitionedDeviceMountRule,omitempty"`
	SharedDevices              []string `json:"sharedDevices,omitempty"`
}

var (
	providerDeviceMountOnce    sync.Once
	providerDeviceMount        providerDeviceMountEnv
	providerDeviceMountEnabled bool
)

func getProviderDeviceMountEnv() *providerDeviceMountEnv {
	providerDeviceMountOnce.Do(func() {
		raw := strings.TrimSpace(os.Getenv(constants.TFProviderDeviceMountEnv))
		if raw == "" || raw == "{}" {
			return
		}
		if err := json.Unmarshal([]byte(raw), &providerDeviceMount); err != nil {
			klog.Warningf("invalid %s, fallback to builtin behavior: %v", constants.TFProviderDeviceMountEnv, err)
			return
		}

		normalized := make(map[string]providerDeviceMountConfig, len(providerDeviceMount.Models))
		for key, cfg := range providerDeviceMount.Models {
			normalized[normalizeProviderMountModelKey(key)] = cfg
		}
		providerDeviceMount.Models = normalized
		providerDeviceMountEnabled = providerDeviceMount.Default != nil || len(providerDeviceMount.Models) > 0
	})

	if !providerDeviceMountEnabled {
		return nil
	}
	return &providerDeviceMount
}

func resolveProviderDeviceMountConfig(model string) (providerDeviceMountConfig, bool) {
	envCfg := getProviderDeviceMountEnv()
	if envCfg == nil {
		return providerDeviceMountConfig{}, false
	}

	if key := normalizeProviderMountModelKey(model); key != "" {
		if cfg, ok := envCfg.Models[key]; ok {
			return cfg, true
		}
	}
	if envCfg.Default != nil {
		return *envCfg.Default, true
	}
	return providerDeviceMountConfig{}, false
}

func normalizeProviderMountModelKey(model string) string {
	return strings.ToLower(strings.TrimSpace(model))
}

func applyProviderDeviceMountPolicy(
	baseNodes map[string]string,
	vendor, model string,
	partitioned bool,
	envVars map[string]string,
) (map[string]string, bool) {
	mountCfg, ok := resolveProviderDeviceMountConfig(model)
	if !ok {
		return baseNodes, false
	}

	nodes := maps.Clone(baseNodes)
	rule := strings.TrimSpace(mountCfg.DeviceMountRule)
	if partitioned {
		rule = strings.TrimSpace(mountCfg.PartitionedDeviceMountRule)
	}
	if rule != "" {
		nodes = filterDeviceNodesByRule(rule, nodes, vendor, model, partitioned, envVars)
	}
	nodes = appendSharedDevices(nodes, mountCfg.SharedDevices)
	if len(nodes) == 0 {
		return nil, true
	}
	return nodes, true
}

func filterDeviceNodesByRule(
	rule string,
	deviceNodes map[string]string,
	vendor, model string,
	partitioned bool,
	envVars map[string]string,
) map[string]string {
	if len(deviceNodes) == 0 {
		return deviceNodes
	}

	celEnv, err := cel.NewEnv(
		cel.Variable("device", cel.MapType(cel.StringType, cel.StringType)),
		cel.Variable("vendor", cel.StringType),
		cel.Variable("model", cel.StringType),
		cel.Variable("partitioned", cel.BoolType),
		cel.Variable("env", cel.MapType(cel.StringType, cel.StringType)),
	)
	if err != nil {
		klog.Warningf("failed to initialize CEL env for device mount rule, skip rule: %v", err)
		return deviceNodes
	}

	ast, issues := celEnv.Compile(rule)
	if issues != nil && issues.Err() != nil {
		klog.Warningf("invalid device mount CEL rule %q, skip rule: %v", rule, issues.Err())
		return deviceNodes
	}

	program, err := celEnv.Program(ast)
	if err != nil {
		klog.Warningf("failed to create CEL program for rule %q, skip rule: %v", rule, err)
		return deviceNodes
	}

	result := make(map[string]string, len(deviceNodes))
	if envVars == nil {
		envVars = make(map[string]string)
	}
	for hostPath, guestPath := range deviceNodes {
		vars := map[string]any{
			"device": map[string]string{
				"hostPath":  hostPath,
				"guestPath": guestPath,
			},
			"vendor":      vendor,
			"model":       model,
			"partitioned": partitioned,
			"env":         envVars,
		}
		val, _, evalErr := program.Eval(vars)
		if evalErr != nil {
			klog.Warningf("failed to evaluate CEL rule %q for device %s, keep device: %v", rule, hostPath, evalErr)
			result[hostPath] = guestPath
			continue
		}
		keep, ok := val.Value().(bool)
		if !ok {
			klog.Warningf("CEL rule %q returns non-bool type %T, keep device %s", rule, val.Value(), hostPath)
			result[hostPath] = guestPath
			continue
		}
		if keep {
			result[hostPath] = guestPath
		}
	}
	return result
}

func appendSharedDevices(deviceNodes map[string]string, sharedPatterns []string) map[string]string {
	if len(sharedPatterns) == 0 {
		return deviceNodes
	}
	if deviceNodes == nil {
		deviceNodes = make(map[string]string)
	}

	for _, rawPattern := range sharedPatterns {
		pattern := strings.TrimSpace(rawPattern)
		if pattern == "" {
			continue
		}

		if strings.ContainsAny(pattern, "*?[") {
			matches, err := filepath.Glob(pattern)
			if err != nil {
				klog.Warningf("invalid shared device pattern %q: %v", pattern, err)
				continue
			}
			for _, matchedPath := range matches {
				addSharedDevicePath(deviceNodes, matchedPath)
			}
			continue
		}
		addSharedDevicePath(deviceNodes, pattern)
	}
	return deviceNodes
}

func addSharedDevicePath(deviceNodes map[string]string, path string) {
	info, err := os.Stat(path)
	if err != nil {
		return
	}
	if info.IsDir() {
		addDeviceNodesInDir(deviceNodes, path)
		return
	}
	deviceNodes[path] = path
}

func addDeviceNodesInDir(deviceNodes map[string]string, dirPath string) {
	entries, err := os.ReadDir(dirPath)
	if err != nil {
		return
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		fullDevicePath := filepath.Join(dirPath, entry.Name())
		deviceNodes[fullDevicePath] = fullDevicePath
	}
}
