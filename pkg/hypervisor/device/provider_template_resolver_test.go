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
