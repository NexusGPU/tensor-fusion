package config

type GPUFitConfig struct {
	MaxWorkerPerNode int `json:"maxWorkerPerNode"`

	VramWeight   float64 `json:"vramWeight"`
	TflopsWeight float64 `json:"tflopsWeight"`
}

type GPUNetworkTopologyAwareConfig struct {
	// Mode controls the topology enforcement behavior.
	// "hard": reject nodes that don't satisfy the topology constraint.
	// "soft" (default): allow all nodes but prefer better topology via scoring.
	Mode string `json:"mode"`

	// TopologySource controls which data source the evaluator uses.
	// "auto" (default): prefer vendor topology if available, fall back to NUMA.
	// "numa": force NUMA-only evaluation.
	// "vendor": force vendor interconnect topology.
	TopologySource string `json:"topologySource"`

	// MaxAllowedTier is the maximum topology tier allowed (inclusive).
	// In hard mode, combinations exceeding this tier are rejected.
	// Default: 1 (same NUMA).
	MaxAllowedTier int `json:"maxAllowedTier"`

	// UnknownTopologyPolicy controls behavior when topology data is missing.
	// "treat-as-worst" (default): treat as worst tier.
	// "reject": treat as unsatisfied.
	UnknownTopologyPolicy string `json:"unknownTopologyPolicy"`

	// PreferLeastDamage controls whether single-GPU requests prefer GPUs
	// that cause the least damage to existing high-quality topology clusters.
	// Default: true.
	PreferLeastDamage *bool `json:"preferLeastDamage,omitempty"`
}

// GetMode returns the effective mode, defaulting to "soft".
func (c *GPUNetworkTopologyAwareConfig) GetMode() string {
	if c.Mode == "" {
		return "soft"
	}
	return c.Mode
}

// GetTopologySource returns the effective topology source, defaulting to "auto".
func (c *GPUNetworkTopologyAwareConfig) GetTopologySource() string {
	if c.TopologySource == "" {
		return "auto"
	}
	return c.TopologySource
}

// GetMaxAllowedTier returns the effective max allowed tier, defaulting to 1.
func (c *GPUNetworkTopologyAwareConfig) GetMaxAllowedTier() int {
	if c.MaxAllowedTier == 0 {
		return 1
	}
	return c.MaxAllowedTier
}

// GetUnknownTopologyPolicy returns the effective unknown topology policy.
func (c *GPUNetworkTopologyAwareConfig) GetUnknownTopologyPolicy() string {
	if c.UnknownTopologyPolicy == "" {
		return "treat-as-worst"
	}
	return c.UnknownTopologyPolicy
}

// GetPreferLeastDamage returns the effective prefer least damage setting.
func (c *GPUNetworkTopologyAwareConfig) GetPreferLeastDamage() bool {
	if c.PreferLeastDamage == nil {
		return true
	}
	return *c.PreferLeastDamage
}
