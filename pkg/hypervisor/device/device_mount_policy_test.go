package device

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
)

func resetProviderDeviceMountPolicyForTest() {
	providerDeviceMountOnce = sync.Once{}
	providerDeviceMount = providerDeviceMountEnv{}
	providerDeviceMountEnabled = false
}

func setProviderDeviceMountEnvForTest(t *testing.T, cfg providerDeviceMountEnv) {
	t.Helper()

	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal provider mount config: %v", err)
	}

	oldVal, hadOldVal := os.LookupEnv(constants.TFProviderDeviceMountEnv)
	if err := os.Setenv(constants.TFProviderDeviceMountEnv, string(raw)); err != nil {
		t.Fatalf("set %s: %v", constants.TFProviderDeviceMountEnv, err)
	}
	resetProviderDeviceMountPolicyForTest()

	t.Cleanup(func() {
		if hadOldVal {
			_ = os.Setenv(constants.TFProviderDeviceMountEnv, oldVal)
		} else {
			_ = os.Unsetenv(constants.TFProviderDeviceMountEnv)
		}
		resetProviderDeviceMountPolicyForTest()
	})
}

func TestApplyProviderDeviceMountPolicyRuleAndShared(t *testing.T) {
	sharedDir := t.TempDir()
	sharedNode := filepath.Join(sharedDir, "uburma0")
	if err := os.WriteFile(sharedNode, []byte(""), 0o600); err != nil {
		t.Fatalf("create shared node: %v", err)
	}

	setProviderDeviceMountEnvForTest(t, providerDeviceMountEnv{
		Default: &providerDeviceMountConfig{
			DeviceMountRule: `device["hostPath"].endsWith("0")`,
			SharedDevices:   []string{filepath.Join(sharedDir, "uburma*")},
		},
	})

	nodes, applied := applyProviderDeviceMountPolicy(
		map[string]string{
			"/dev/davinci0": "/dev/davinci0",
			"/dev/davinci1": "/dev/davinci1",
		},
		"Ascend",
		"Ascend 910B",
		false,
		nil,
	)
	if !applied {
		t.Fatalf("expected provider mount policy to be applied")
	}
	if _, ok := nodes["/dev/davinci0"]; !ok {
		t.Fatalf("expected filtered compute device /dev/davinci0 to be kept")
	}
	if _, ok := nodes["/dev/davinci1"]; ok {
		t.Fatalf("expected /dev/davinci1 to be filtered out")
	}
	if _, ok := nodes[sharedNode]; !ok {
		t.Fatalf("expected shared device %s to be appended", sharedNode)
	}
}

func TestApplyProviderDeviceMountPolicyPartitionedRule(t *testing.T) {
	sharedDir := t.TempDir()
	sharedNode := filepath.Join(sharedDir, "hisi_hdc")
	if err := os.WriteFile(sharedNode, []byte(""), 0o600); err != nil {
		t.Fatalf("create shared node: %v", err)
	}

	setProviderDeviceMountEnvForTest(t, providerDeviceMountEnv{
		Default: &providerDeviceMountConfig{
			DeviceMountRule:            `true`,
			PartitionedDeviceMountRule: `false`,
			SharedDevices:              []string{sharedNode},
		},
	})

	base := map[string]string{"/dev/vdavinci0": "/dev/davinci0"}

	nonPartitioned, applied := applyProviderDeviceMountPolicy(base, "Ascend", "Ascend 910B", false, nil)
	if !applied {
		t.Fatalf("expected non-partitioned policy to be applied")
	}
	if _, ok := nonPartitioned["/dev/vdavinci0"]; !ok {
		t.Fatalf("expected device node to remain in non-partitioned mode")
	}
	if _, ok := nonPartitioned[sharedNode]; !ok {
		t.Fatalf("expected shared node to be appended in non-partitioned mode")
	}

	partitioned, applied := applyProviderDeviceMountPolicy(base, "Ascend", "Ascend 910B", true, nil)
	if !applied {
		t.Fatalf("expected partitioned policy to be applied")
	}
	if _, ok := partitioned["/dev/vdavinci0"]; ok {
		t.Fatalf("expected compute node to be filtered by partitionedDeviceMountRule")
	}
	if _, ok := partitioned[sharedNode]; !ok {
		t.Fatalf("expected shared node to be appended in partitioned mode")
	}
}

func TestApplyProviderDeviceMountPolicyModelOverride(t *testing.T) {
	defaultDir := t.TempDir()
	modelDir := t.TempDir()
	defaultShared := filepath.Join(defaultDir, "default")
	modelShared := filepath.Join(modelDir, "model")
	if err := os.WriteFile(defaultShared, []byte(""), 0o600); err != nil {
		t.Fatalf("create default shared node: %v", err)
	}
	if err := os.WriteFile(modelShared, []byte(""), 0o600); err != nil {
		t.Fatalf("create model shared node: %v", err)
	}

	setProviderDeviceMountEnvForTest(t, providerDeviceMountEnv{
		Default: &providerDeviceMountConfig{
			SharedDevices: []string{defaultShared},
		},
		Models: map[string]providerDeviceMountConfig{
			normalizeProviderMountModelKey("Ascend 910B"): {
				SharedDevices: []string{modelShared},
			},
		},
	})

	nodes, applied := applyProviderDeviceMountPolicy(
		map[string]string{"/dev/davinci0": "/dev/davinci0"},
		"Ascend",
		"Ascend 910B",
		false,
		nil,
	)
	if !applied {
		t.Fatalf("expected model override policy to be applied")
	}
	if _, ok := nodes[modelShared]; !ok {
		t.Fatalf("expected model-specific shared device to be used")
	}
	if _, ok := nodes[defaultShared]; ok {
		t.Fatalf("expected default shared device to be overridden by model config")
	}
}

func TestSplitDeviceEnvPartitionWithoutPolicyDoesNotInheritParentNode(t *testing.T) {
	oldAssignPartition := assignPartition
	assignPartition = makeAssignPartitionStub(
		PartitionTypeEnvironmentVariable,
		"partition-uuid-env",
		[]string{
			"ASCEND_VNPU_SPECS=vir01",
			"ASCEND_VISIBLE_DEVICES=0",
		},
	)
	t.Cleanup(func() {
		assignPartition = oldAssignPartition
	})

	oldVal, hadOldVal := os.LookupEnv(constants.TFProviderDeviceMountEnv)
	_ = os.Unsetenv(constants.TFProviderDeviceMountEnv)
	resetProviderDeviceMountPolicyForTest()
	t.Cleanup(func() {
		if hadOldVal {
			_ = os.Setenv(constants.TFProviderDeviceMountEnv, oldVal)
		}
		resetProviderDeviceMountPolicyForTest()
	})

	controller := &Controller{
		devices: map[string]*api.DeviceInfo{
			"device-0": {
				UUID:       "device-0",
				Vendor:     "Ascend",
				Model:      "Ascend 910B",
				DeviceNode: map[string]string{"/dev/davinci0": "/dev/davinci0"},
			},
		},
		accelerator: &AcceleratorInterface{},
	}

	partitioned, err := controller.SplitDevice("device-0", "vir01")
	if err != nil {
		t.Fatalf("split device failed: %v", err)
	}
	if _, ok := partitioned.DeviceNode["/dev/davinci0"]; ok {
		t.Fatalf("env partition should not inherit parent compute node")
	}
	if got := partitioned.DeviceEnv["ASCEND_VISIBLE_DEVICES"]; got != "0" {
		t.Fatalf("expected ASCEND_VISIBLE_DEVICES=0, got %q", got)
	}
	if got := partitioned.DeviceEnv["ASCEND_VNPU_SPECS"]; got != "vir01" {
		t.Fatalf("expected ASCEND_VNPU_SPECS=vir01, got %q", got)
	}
}

func TestSplitDeviceNodePartitionWithoutNodesDoesNotInheritParentNode(t *testing.T) {
	oldAssignPartition := assignPartition
	assignPartition = makeAssignPartitionStub(
		PartitionTypeDeviceNode,
		"partition-uuid-ascend-runtime",
		[]string{
			"ASCEND_VNPU_SPECS=vir01",
			"ASCEND_VISIBLE_DEVICES=0",
		},
	)
	t.Cleanup(func() {
		assignPartition = oldAssignPartition
	})

	controller := &Controller{
		devices: map[string]*api.DeviceInfo{
			"device-2": {
				UUID:       "device-2",
				Vendor:     "Ascend",
				Model:      "Ascend 910B",
				DeviceNode: map[string]string{"/dev/davinci9": "/dev/davinci9"},
			},
		},
		accelerator: &AcceleratorInterface{},
	}

	partitioned, err := controller.SplitDevice("device-2", "vir01")
	if err != nil {
		t.Fatalf("split device failed: %v", err)
	}
	if _, ok := partitioned.DeviceNode["/dev/davinci9"]; ok {
		t.Fatalf("partitioned device should not inherit parent compute node")
	}
	if len(partitioned.DeviceNode) != 0 {
		t.Fatalf("expected no partition device nodes when provider returns none")
	}
	if got := partitioned.DeviceEnv["ASCEND_VNPU_SPECS"]; got != "vir01" {
		t.Fatalf("expected ASCEND_VNPU_SPECS=vir01, got %q", got)
	}
}

func TestSplitDeviceNodePartitionWithDeviceNodesFallsBackToProviderNodes(t *testing.T) {
	oldAssignPartition := assignPartition
	assignPartition = makeAssignPartitionStubWithDeviceNodes(
		PartitionTypeDeviceNode,
		"partition-uuid-device-nodes",
		[]string{"ASCEND_VISIBLE_DEVICES=0"},
		[]string{"/dev/vdavinci100=/dev/davinci100"},
	)
	t.Cleanup(func() {
		assignPartition = oldAssignPartition
	})

	oldVal, hadOldVal := os.LookupEnv(constants.TFProviderDeviceMountEnv)
	_ = os.Unsetenv(constants.TFProviderDeviceMountEnv)
	resetProviderDeviceMountPolicyForTest()
	t.Cleanup(func() {
		if hadOldVal {
			_ = os.Setenv(constants.TFProviderDeviceMountEnv, oldVal)
		}
		resetProviderDeviceMountPolicyForTest()
	})

	controller := &Controller{
		devices: map[string]*api.DeviceInfo{
			"device-4": {
				UUID:       "device-4",
				Vendor:     "Ascend",
				Model:      "Ascend 910B",
				DeviceNode: map[string]string{"/dev/davinci4": "/dev/davinci4"},
			},
		},
		accelerator: &AcceleratorInterface{},
	}

	partitioned, err := controller.SplitDevice("device-4", "vir01")
	if err != nil {
		t.Fatalf("split device failed: %v", err)
	}
	if _, ok := partitioned.DeviceNode["/dev/davinci4"]; ok {
		t.Fatalf("partitioned device should not inherit parent compute node")
	}
	if got, ok := partitioned.DeviceNode["/dev/vdavinci100"]; !ok || got != "/dev/davinci100" {
		t.Fatalf("expected provider device node mapping to be kept, got %v", partitioned.DeviceNode)
	}
}

func TestSplitDeviceRejectsUnknownPartitionResultType(t *testing.T) {
	oldAssignPartition := assignPartition
	assignPartition = makeAssignPartitionStub(
		PartitionResultType(99),
		"partition-uuid-unknown",
		[]string{"ASCEND_VISIBLE_DEVICES=0"},
	)
	t.Cleanup(func() {
		assignPartition = oldAssignPartition
	})

	controller := &Controller{
		devices: map[string]*api.DeviceInfo{
			"device-3": {
				UUID:       "device-3",
				Vendor:     "Ascend",
				Model:      "Ascend 910B",
				DeviceNode: map[string]string{"/dev/davinci3": "/dev/davinci3"},
			},
		},
		accelerator: &AcceleratorInterface{},
	}

	if _, err := controller.SplitDevice("device-3", "vir99"); err == nil {
		t.Fatalf("expected error for unknown partition result type")
	}
}

func makeAssignPartitionStub(
	resultType PartitionResultType,
	partitionUUID string,
	envPairs []string,
) func(*byte, *byte, *PartitionResult) Result {
	return makeAssignPartitionStubWithDeviceNodes(resultType, partitionUUID, envPairs, nil)
}

func makeAssignPartitionStubWithDeviceNodes(
	resultType PartitionResultType,
	partitionUUID string,
	envPairs []string,
	deviceNodePairs []string,
) func(*byte, *byte, *PartitionResult) Result {
	return func(_ *byte, _ *byte, result *PartitionResult) Result {
		if result == nil {
			return ResultErrorInvalidParam
		}
		*result = PartitionResult{}
		result.Type = resultType
		copy(result.DeviceUUID[:], []byte(partitionUUID))
		if len(partitionUUID) < len(result.DeviceUUID) {
			result.DeviceUUID[len(partitionUUID)] = 0
		}
		for i, pair := range envPairs {
			if i >= len(result.EnvVars) {
				break
			}
			copy(result.EnvVars[i][:], []byte(pair))
			if len(pair) < len(result.EnvVars[i]) {
				result.EnvVars[i][len(pair)] = 0
			}
		}
		for i, pair := range deviceNodePairs {
			if i >= len(result.DeviceNodes) {
				break
			}
			copy(result.DeviceNodes[i][:], []byte(pair))
			if len(pair) < len(result.DeviceNodes[i]) {
				result.DeviceNodes[i][len(pair)] = 0
			}
		}
		return ResultSuccess
	}
}
