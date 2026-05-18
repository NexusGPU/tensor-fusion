package device

import (
	"os"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

const (
	nvidiaProviderTemplateConfig = "" +
		`{"hardwareMetadata":[{"model":"A100_SXM_80G","fullModelName":"NVIDIA A100-SXM4-80GB",` +
		`"partitionTemplateRefs":["19","14"]}],"virtualizationTemplates":[{"id":"19","name":"1g.10gb"},` +
		`{"id":"14","name":"2g.20gb"}]}`
	ascendVir01TemplateID = "vir01"
)

func TestResolveProviderTemplateIDReturnsMappedProviderID(t *testing.T) {
	oldVal, hadOldVal := os.LookupEnv(constants.TFProviderTemplateConfigEnv)
	if err := os.Setenv(constants.TFProviderTemplateConfigEnv, nvidiaProviderTemplateConfig); err != nil {
		t.Fatalf("set %s: %v", constants.TFProviderTemplateConfigEnv, err)
	}
	resetProviderTemplateConfigForTest()
	t.Cleanup(func() {
		if hadOldVal {
			_ = os.Setenv(constants.TFProviderTemplateConfigEnv, oldVal)
		} else {
			_ = os.Unsetenv(constants.TFProviderTemplateConfigEnv)
		}
		resetProviderTemplateConfigForTest()
	})

	if got := resolveProviderTemplateID("NVIDIA", "NVIDIA A100-SXM4-80GB", "mig-1g-10gb"); got != "19" {
		t.Fatalf("expected provider template id 19, got %q", got)
	}
}

func TestResolveProviderTemplateIDPrefersNVMLProfileID(t *testing.T) {
	// Two cards (A30 4-slice, A100 7-slice) reusing the same NVML profile id 14
	// with different chart-side ids. The resolver must return NVMLProfileID
	// converted to string, not the chart-side id.
	const cfg = `{` +
		`"hardwareMetadata":[` +
		`{"model":"A30","fullModelName":"NVIDIA A30",` +
		`"partitionTemplateRefs":["a30-1g-6gb"]},` +
		`{"model":"A100_SXM_80G","fullModelName":"NVIDIA A100-SXM4-80GB",` +
		`"partitionTemplateRefs":["a100-80g-2g-20gb"]}` +
		`],` +
		`"virtualizationTemplates":[` +
		`{"id":"a30-1g-6gb","name":"1g.6gb","nvmlProfileId":14},` +
		`{"id":"a100-80g-2g-20gb","name":"2g.20gb","nvmlProfileId":14}` +
		`]}`

	oldVal, hadOldVal := os.LookupEnv(constants.TFProviderTemplateConfigEnv)
	if err := os.Setenv(constants.TFProviderTemplateConfigEnv, cfg); err != nil {
		t.Fatalf("set env: %v", err)
	}
	resetProviderTemplateConfigForTest()
	t.Cleanup(func() {
		if hadOldVal {
			_ = os.Setenv(constants.TFProviderTemplateConfigEnv, oldVal)
		} else {
			_ = os.Unsetenv(constants.TFProviderTemplateConfigEnv)
		}
		resetProviderTemplateConfigForTest()
	})

	if got := resolveProviderTemplateID("NVIDIA", "NVIDIA A30", "1g.6gb"); got != "14" {
		t.Fatalf("A30 1g.6gb: expected NVML profile id 14, got %q", got)
	}
	if got := resolveProviderTemplateID("NVIDIA", "NVIDIA A100-SXM4-80GB", "2g.20gb"); got != "14" {
		t.Fatalf("A100 2g.20gb: expected NVML profile id 14, got %q", got)
	}
}

func TestResolveProviderTemplateIDFallsBackToLogicalID(t *testing.T) {
	oldVal, hadOldVal := os.LookupEnv(constants.TFProviderTemplateConfigEnv)
	_ = os.Unsetenv(constants.TFProviderTemplateConfigEnv)
	resetProviderTemplateConfigForTest()
	t.Cleanup(func() {
		if hadOldVal {
			_ = os.Setenv(constants.TFProviderTemplateConfigEnv, oldVal)
		}
		resetProviderTemplateConfigForTest()
	})

	if got := resolveProviderTemplateID("Ascend", "Ascend 910B", ascendVir01TemplateID); got != ascendVir01TemplateID {
		t.Fatalf("expected fallback template id %s, got %q", ascendVir01TemplateID, got)
	}
}
