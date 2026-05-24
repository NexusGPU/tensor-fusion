package external_dp

// GenericDevicePluginDetector implements VendorDetector using configuration
// supplied via ProviderConfig.Spec.DevicePluginDetection. It is used for
// vendors that don't need any device-ID post-processing (unlike NVIDIA, which
// has to strip the HAMI "-<idx>" suffix to recover the real GPU UUID).
type GenericDevicePluginDetector struct {
	resourcePrefixes []string
	usedBySystem     string
}

// NewGenericDevicePluginDetector returns a detector that matches the given
// resource prefixes and reports the given usedBy system name. If usedBy is
// empty it falls back to "3rd-party-device-plugin" so we never write an empty
// string into gpu.Status.UsedBy.
func NewGenericDevicePluginDetector(resourcePrefixes []string, usedBy string) *GenericDevicePluginDetector {
	if usedBy == "" {
		usedBy = string(UsedBy3rdPartyDevicePlugin)
	}
	return &GenericDevicePluginDetector{
		resourcePrefixes: resourcePrefixes,
		usedBySystem:     usedBy,
	}
}

// GetResourceNamePrefixes returns the resource name prefixes this detector handles.
func (g *GenericDevicePluginDetector) GetResourceNamePrefixes() []string {
	return g.resourcePrefixes
}

// GetUsedBySystemAndRealDeviceID returns the configured usedBy system and the
// device ID unchanged. Vendors needing device-ID post-processing should ship
// their own detector instead of relying on this generic one.
func (g *GenericDevicePluginDetector) GetUsedBySystemAndRealDeviceID(
	deviceID,
	resourceName string,
) (system string, realDeviceID string) {
	return g.usedBySystem, deviceID
}
