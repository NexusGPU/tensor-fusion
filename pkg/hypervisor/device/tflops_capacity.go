package device

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"k8s.io/klog/v2"
)

// TensorFusion measures compute in vendor-published FP16 TFLOPS. Those figures
// live in the ProviderConfig hardware table (hardwareMetadata[].fp16TFlops); a
// provider library can only estimate a theoretical figure in its own terms.
//
// The device controller is where the two meet. It resolves every device's
// capacity once, here, and stores it on the device (api.DeviceInfo
// TflopsCapacity). GPU resource publication, node totals, the HTTP API, the TUI
// and the conversion of TFLOPS limits all read that stored value; none of them
// looks at the table or at the provider's estimate.

// hardwareTable is the hardware table of the provider this hypervisor serves.
// A nil *hardwareTable means that no table is configured at all, which is the
// case for backends without a ProviderConfig; an empty table is configured and
// simply lists no model.
type hardwareTable struct {
	models []tfv1.HardwareModelInfo
}

// fp16Tflops returns the vendor-published FP16 TFLOPS of a device model. The
// first entry naming the model decides; an entry without a positive figure
// does not list a capacity.
func (t *hardwareTable) fp16Tflops(vendor, model string) (float64, bool) {
	if t == nil {
		return 0, false
	}
	for _, hw := range t.models {
		if !providerTemplateMatchesModel(vendor, model, hw) {
			continue
		}
		if hw.Fp16TFlops.Sign() <= 0 {
			return 0, false
		}
		return hw.Fp16TFlops.AsApproximateFloat64(), true
	}
	return 0, false
}

// LoadHardwareMetadataFromEnv reads the hardware table that the operator passes
// to the hypervisor Pod from the ProviderConfig of the node's vendor.
func LoadHardwareMetadataFromEnv() ([]tfv1.HardwareModelInfo, error) {
	raw := strings.TrimSpace(os.Getenv(constants.TFProviderHardwareMetadataEnv))
	if raw == "" {
		return nil, fmt.Errorf("%s is not set", constants.TFProviderHardwareMetadataEnv)
	}
	var metadata []tfv1.HardwareModelInfo
	if err := json.Unmarshal([]byte(raw), &metadata); err != nil {
		return nil, fmt.Errorf("parse %s: %w", constants.TFProviderHardwareMetadataEnv, err)
	}
	return metadata, nil
}

// SetHardwareMetadata installs the hardware table that device capacities are
// resolved from, replacing any previous one. Devices already known are
// re-resolved against it right away; handlers receive OnUpdate for every device
// whose capacity changed, followed by OnDiscoveryComplete with the new totals.
//
// Calling it marks the table as configured, also with no entries: from then on
// a device whose model is not listed is reported (see TflopsCapacitySource and
// the warning logged once per device). A controller that never receives a table
// treats the provider's figure as the capacity without comment.
func (m *Controller) SetHardwareMetadata(metadata []tfv1.HardwareModelInfo) {
	type update struct{ old, new *api.DeviceInfo }
	var updates []update
	var nodeInfo *api.NodeInfo

	m.mu.Lock()
	m.hardwareTable = &hardwareTable{models: append([]tfv1.HardwareModelInfo(nil), metadata...)}
	// A device that still falls back under the new table is worth reporting again.
	m.estimateWarned = make(map[string]struct{})
	for uuid, device := range m.devices {
		if device == nil {
			continue
		}
		// Devices are shared with handlers and allocations once published and are
		// never modified in place.
		resolved := device.DeepCopy()
		if !m.resolveTflopsCapacityLocked(resolved) {
			continue
		}
		m.devices[uuid] = resolved
		// A partition is not announced on its own; it follows its parent.
		if resolved.ParentUUID == "" {
			updates = append(updates, update{old: device, new: resolved})
		}
	}
	if len(updates) > 0 {
		nodeInfo = m.AggregateNodeInfo()
	}
	m.mu.Unlock()

	for _, change := range updates {
		klog.Infof("TFLOPS capacity of device %s (%s) changed with the hardware table: %.2f (%s) -> %.2f (%s)",
			change.new.UUID, change.new.Model,
			change.old.TflopsCapacity, change.old.TflopsCapacitySource,
			change.new.TflopsCapacity, change.new.TflopsCapacitySource)
		m.notifyHandlers(func(handler framework.DeviceChangeHandler) {
			if handler.OnUpdate != nil {
				handler.OnUpdate(change.old, change.new)
			}
		})
	}
	if nodeInfo != nil {
		m.notifyHandlers(func(handler framework.DeviceChangeHandler) {
			if handler.OnDiscoveryComplete != nil {
				handler.OnDiscoveryComplete(nodeInfo)
			}
		})
	}
}

// resolveTflopsCapacityLocked resolves the capacity of a device that is not yet
// shared, and reports whether it changed. The caller must hold m.mu for writing.
func (m *Controller) resolveTflopsCapacityLocked(device *api.DeviceInfo) bool {
	tableTflops, listed := m.hardwareTable.fp16Tflops(device.Vendor, device.Model)
	changed := device.ResolveTflopsCapacity(tableTflops, listed)

	// Only a device of its own is reported, and only when a table exists that
	// should have listed it.
	if m.hardwareTable == nil || device.ParentUUID != "" {
		return changed
	}
	if !device.TflopsCapacityEstimated() {
		delete(m.estimateWarned, device.UUID)
		return changed
	}
	if _, warned := m.estimateWarned[device.UUID]; !warned {
		if m.estimateWarned == nil {
			m.estimateWarned = make(map[string]struct{})
		}
		m.estimateWarned[device.UUID] = struct{}{}
		klog.Warningf("Device %s: model %q (vendor %q) has no fp16TFlops entry in the ProviderConfig hardware table. "+
			"Its TFLOPS capacity falls back to the provider's estimate of %.2f, which is not vendor-published FP16 "+
			"TFLOPS, so TFLOPS requests and limits on this device are not comparable with listed models. "+
			"Add the model to ProviderConfig hardwareMetadata to fix this.",
			device.UUID, device.Model, device.Vendor, device.TflopsCapacity)
	}
	return changed
}
