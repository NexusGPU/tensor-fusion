package external_dp

import (
	"context"
	"os"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// MockAPIServer is a mock implementation of APIServerInterface
type MockAPIServer struct {
	mock.Mock
}

func (m *MockAPIServer) GetGPU(uuid string) (*tfv1.GPU, error) {
	args := m.Called(uuid)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*tfv1.GPU), args.Error(1)
}

func (m *MockAPIServer) UpdateGPUStatus(gpu *tfv1.GPU) error {
	args := m.Called(gpu)
	return args.Error(0)
}

// MockKubeletClient is a mock implementation of KubeletClientInterface
type MockKubeletClient struct {
	mock.Mock
	pods map[string]interface{}
}

func (m *MockKubeletClient) GetAllPods() map[string]any {
	return m.pods
}

func TestReadCheckpointFile(t *testing.T) {
	// Create a temporary checkpoint file with test data
	testData := `{
  "Data": {
    "PodDeviceEntries": [
      {
        "PodUID": "a7461dc1-023a-4bd5-a403-c738bb1d7db4",
        "ContainerName": "web",
        "ResourceName": "nvidia.com/gpu",
        "DeviceIDs": {
          "-1": [
            "GPU-7d8429d5-531d-d6a6-6510-3b662081a75a"
          ]
        },
        "AllocResp": "CkIKFk5WSURJQV9WSVNJQkxFX0RFVklDRVMSKEdQVS03ZDg0MjlkNS01MzFkLWQ2YTYtNjUxMC0zYjY2MjA4MWE3NWEaJAoOL2Rldi9udmlkaWFjdGwSDi9kZXYvbnZpZGlhY3RsGgJydxomCg8vZGV2L252aWRpYS11dm0SDy9kZXYvbnZpZGlhLXV2bRoCcncaMgoVL2Rldi9udmlkaWEtdXZtLXRvb2xzEhUvZGV2L252aWRpYS11dm0tdG9vbHMaAnJ3Gi4KEy9kZXYvbnZpZGlhLW1vZGVzZXQSEy9kZXYvbnZpZGlhLW1vZGVzZXQaAnJ3GiAKDC9kZXYvbnZpZGlhMBIML2Rldi9udmlkaWEwGgJydw=="
      }
    ],
    "RegisteredDevices": {
      "nvidia.com/gpu": [
        "GPU-7d8429d5-531d-d6a6-6510-3b662081a75a"
      ]
    }
  },
  "Checksum": 2262205670
}`

	tmpFile, err := os.CreateTemp("", "checkpoint-*.json")
	assert.NoError(t, err)
	defer func() {
		_ = os.Remove(tmpFile.Name())
	}()

	_, err = tmpFile.WriteString(testData)
	assert.NoError(t, err)
	_ = tmpFile.Close()

	detector := &DevicePluginDetector{
		checkpointPath: tmpFile.Name(),
	}

	checkpoint, err := detector.readCheckpointFile()
	assert.NoError(t, err)
	assert.NotNil(t, checkpoint)
	assert.Len(t, checkpoint.Data.PodDeviceEntries, 1)
	assert.Equal(t, "a7461dc1-023a-4bd5-a403-c738bb1d7db4", checkpoint.Data.PodDeviceEntries[0].PodUID)
	assert.Equal(t, "nvidia.com/gpu", checkpoint.Data.PodDeviceEntries[0].ResourceName)
	assert.Contains(t, checkpoint.Data.RegisteredDevices, "nvidia.com/gpu")
}

func TestExtractDeviceIDs(t *testing.T) {
	checkpoint := &KubeletCheckpoint{
		Data: CheckpointData{
			PodDeviceEntries: []PodDeviceEntry{
				{
					ResourceName: "nvidia.com/gpu",
					DeviceIDs: map[string][]string{
						"-1": {"GPU-7d8429d5-531d-d6a6-6510-3b662081a75a"},
					},
				},
			},
			RegisteredDevices: map[string][]string{
				"nvidia.com/gpu": {"GPU-7d8429d5-531d-d6a6-6510-3b662081a75a"},
			},
		},
	}

	detector := &DevicePluginDetector{
		vendorDetectors: map[string]VendorDetector{
			"nvidia.com/gpu": NewNvidiaDevicePluginDetector(),
		},
	}

	allocated, registered := detector.extractDeviceIDs(checkpoint)
	assert.Contains(t, allocated, "gpu-7d8429d5-531d-d6a6-6510-3b662081a75a")
	assert.Contains(t, registered, "gpu-7d8429d5-531d-d6a6-6510-3b662081a75a")
}

func TestNvidiaDevicePluginDetector(t *testing.T) {
	detector := NewNvidiaDevicePluginDetector()
	assert.Equal(t, []string{"nvidia.com/gpu", "nvidia.com/mig"}, detector.GetResourceNamePrefixes())
	system, realDeviceID := detector.GetUsedBySystemAndRealDeviceID("GPU-8511dc03-7592-b8b7-1a92-582d40da52fb", "nvidia.com/gpu")
	assert.Equal(t, string(UsedByNvidiaDevicePlugin), system)
	assert.Equal(t, "GPU-8511dc03-7592-b8b7-1a92-582d40da52fb", realDeviceID)
	// External device plugin detection only works for nvidia.com/gpu resources with device IDs longer than 40 characters
	system, realDeviceID = detector.GetUsedBySystemAndRealDeviceID("GPU-422d6152-4d4b-5b0e-9d3a-b3b44e2742ea-1", "nvidia.com/gpu")
	assert.Equal(t, string(UsedBy3rdPartyDevicePlugin), system)
	assert.Equal(t, "GPU-422d6152-4d4b-5b0e-9d3a-b3b44e2742ea", realDeviceID)
	// nvidia.com/mig always returns nvidia-device-plugin
	system, realDeviceID = detector.GetUsedBySystemAndRealDeviceID("MIG-422d6152-4d4b-5b0e-9d3a-b3b44e2742ea", "nvidia.com/mig-1g.5gb")
	assert.Equal(t, string(UsedByNvidiaDevicePlugin), system)
	assert.Equal(t, "MIG-422d6152-4d4b-5b0e-9d3a-b3b44e2742ea", realDeviceID)
}

func TestProcessDeviceState_DeviceAdded(t *testing.T) {
	mockAPI := new(MockAPIServer)

	checkpointData := `{
  "Data": {
    "PodDeviceEntries": [
      {
        "PodUID": "a7461dc1-023a-4bd5-a403-c738bb1d7db4",
        "ContainerName": "web",
        "ResourceName": "nvidia.com/gpu",
        "DeviceIDs": {
          "-1": [
            "GPU-7d8429d5-531d-d6a6-6510-3b662081a75a"
          ]
        }
      }
    ],
    "RegisteredDevices": {
      "nvidia.com/gpu": [
        "GPU-7d8429d5-531d-d6a6-6510-3b662081a75a"
      ]
    }
  }
}`

	tmpFile, err := os.CreateTemp("", "checkpoint-*.json")
	assert.NoError(t, err)
	defer func() {
		_ = os.Remove(tmpFile.Name())
	}()

	_, err = tmpFile.WriteString(checkpointData)
	assert.NoError(t, err)
	_ = tmpFile.Close()

	// Mock GPU resource
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: "GPU-7d8429d5-531d-d6a6-6510-3b662081a75a",
		},
		Status: tfv1.GPUStatus{
			UsedBy: tfv1.UsedByTensorFusion,
		},
	}

	mockAPI.On("GetGPU", "gpu-7d8429d5-531d-d6a6-6510-3b662081a75a").Return(gpu, nil)
	mockAPI.On("UpdateGPUStatus", mock.MatchedBy(func(gpu *tfv1.GPU) bool {
		return gpu.Status.UsedBy == UsedByNvidiaDevicePlugin
	})).Return(nil)

	detector := &DevicePluginDetector{
		ctx:               context.Background(),
		checkpointPath:    tmpFile.Name(),
		apiClient:         mockAPI,
		vendorDetectors:   make(map[string]VendorDetector),
		previousDeviceIDs: make(map[string]string),
	}
	// Register vendor detectors properly - use the same pattern as registerVendorDetectors
	nvdpDetector := NewNvidiaDevicePluginDetector()
	for _, prefix := range nvdpDetector.GetResourceNamePrefixes() {
		detector.vendorDetectors[prefix] = nvdpDetector
	}

	// Verify checkpoint can be read and devices extracted
	checkpoint, err := detector.readCheckpointFile()
	assert.NoError(t, err)
	allocated, _ := detector.extractDeviceIDs(checkpoint)
	assert.Contains(t, allocated, "gpu-7d8429d5-531d-d6a6-6510-3b662081a75a", "Device should be in allocated map")

	err = detector.processDeviceState(false)
	assert.NoError(t, err)
	mockAPI.AssertExpectations(t)
}

func TestProcessDeviceState_DeviceRemoved(t *testing.T) {
	mockAPI := new(MockAPIServer)

	checkpointData := `{
  "Data": {
    "PodDeviceEntries": [],
    "RegisteredDevices": {
      "nvidia.com/gpu": [
        "GPU-7d8429d5-531d-d6a6-6510-3b662081a75a"
      ]
    }
  }
}`

	tmpFile, err := os.CreateTemp("", "checkpoint-*.json")
	assert.NoError(t, err)
	defer func() {
		_ = os.Remove(tmpFile.Name())
	}()

	_, err = tmpFile.WriteString(checkpointData)
	assert.NoError(t, err)
	_ = tmpFile.Close()

	// Mock GPU resource that was previously allocated
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: "GPU-7d8429d5-531d-d6a6-6510-3b662081a75a",
		},
		Status: tfv1.GPUStatus{
			UsedBy: UsedByNvidiaDevicePlugin,
		},
	}

	mockAPI.On("GetGPU", "gpu-7d8429d5-531d-d6a6-6510-3b662081a75a").Return(gpu, nil)
	mockAPI.On("UpdateGPUStatus", mock.MatchedBy(func(gpu *tfv1.GPU) bool {
		return gpu.Status.UsedBy == tfv1.UsedByTensorFusion
	})).Return(nil)

	detector := &DevicePluginDetector{
		ctx:               context.Background(),
		checkpointPath:    tmpFile.Name(),
		apiClient:         mockAPI,
		vendorDetectors:   make(map[string]VendorDetector),
		previousDeviceIDs: map[string]string{"gpu-7d8429d5-531d-d6a6-6510-3b662081a75a": "nvidia.com/gpu"},
	}
	// Register vendor detectors properly - use the same pattern as registerVendorDetectors
	nvdpDetector := NewNvidiaDevicePluginDetector()
	for _, prefix := range nvdpDetector.GetResourceNamePrefixes() {
		detector.vendorDetectors[prefix] = nvdpDetector
	}

	err = detector.processDeviceState(false)
	assert.NoError(t, err)
	mockAPI.AssertExpectations(t)
}

func TestFindEntryForDevice(t *testing.T) {
	checkpoint := &KubeletCheckpoint{
		Data: CheckpointData{
			PodDeviceEntries: []PodDeviceEntry{
				{
					ResourceName: "nvidia.com/gpu",
					DeviceIDs: map[string][]string{
						"-1": {"GPU-7d8429d5-531d-d6a6-6510-3b662081a75a"},
					},
				},
			},
		},
	}

	detector := &DevicePluginDetector{}
	entry := detector.findEntryForDevice(checkpoint, "GPU-7d8429d5-531d-d6a6-6510-3b662081a75a")
	assert.Equal(t, "nvidia.com/gpu", entry.ResourceName)
}
