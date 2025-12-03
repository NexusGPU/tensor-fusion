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
	assert.Equal(t, "nvidia.com/gpu", detector.GetResourceName())
	assert.Equal(t, string(tfv1.UsedByNvidiaDevicePlugin), detector.GetUsedBySystem())
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
	mockAPI.On("UpdateGPUStatus", mock.AnythingOfType("*v1.GPU")).Return(nil)

	detector := &DevicePluginDetector{
		ctx:               context.Background(),
		checkpointPath:    tmpFile.Name(),
		apiClient:         mockAPI,
		vendorDetectors:   map[string]VendorDetector{"nvidia.com/gpu": NewNvidiaDevicePluginDetector()},
		previousDeviceIDs: make(map[string]bool),
	}

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
			UsedBy: tfv1.UsedByNvidiaDevicePlugin,
		},
	}

	mockAPI.On("GetGPU", "gpu-7d8429d5-531d-d6a6-6510-3b662081a75a").Return(gpu, nil)
	mockAPI.On("UpdateGPUStatus", mock.AnythingOfType("*v1.GPU")).Return(nil)

	detector := &DevicePluginDetector{
		ctx:               context.Background(),
		checkpointPath:    tmpFile.Name(),
		apiClient:         mockAPI,
		vendorDetectors:   map[string]VendorDetector{"nvidia.com/gpu": NewNvidiaDevicePluginDetector()},
		previousDeviceIDs: map[string]bool{"gpu-7d8429d5-531d-d6a6-6510-3b662081a75a": true},
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
