package external_dp

import (
	"context"
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/stretchr/testify/mock"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
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
	pods map[string]any
}

func (m *MockKubeletClient) GetAllPods() map[string]any {
	return m.pods
}

var _ = Describe("DevicePluginDetector", func() {
	Describe("readCheckpointFile", func() {
		It("should read checkpoint file correctly", func() {
			//nolint:lll // test data contains long base64 string
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
			Expect(err).NotTo(HaveOccurred())
			defer func() { _ = os.Remove(tmpFile.Name()) }()

			_, err = tmpFile.WriteString(testData)
			Expect(err).NotTo(HaveOccurred())
			Expect(tmpFile.Close()).To(Succeed())

			detector := &DevicePluginDetector{
				checkpointPath: tmpFile.Name(),
			}

			checkpoint, err := detector.readCheckpointFile()
			Expect(err).NotTo(HaveOccurred())
			Expect(checkpoint).NotTo(BeNil())
			Expect(checkpoint.Data.PodDeviceEntries).To(HaveLen(1))
			Expect(checkpoint.Data.PodDeviceEntries[0].PodUID).To(Equal("a7461dc1-023a-4bd5-a403-c738bb1d7db4"))
			Expect(checkpoint.Data.PodDeviceEntries[0].ResourceName).To(Equal("nvidia.com/gpu"))
			Expect(checkpoint.Data.RegisteredDevices).To(HaveKey("nvidia.com/gpu"))
		})
	})

	Describe("extractDeviceIDs", func() {
		It("should extract device IDs correctly", func() {
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
			Expect(allocated).To(HaveKey("gpu-7d8429d5-531d-d6a6-6510-3b662081a75a"))
			Expect(registered).To(HaveKey("gpu-7d8429d5-531d-d6a6-6510-3b662081a75a"))
		})
	})

	Describe("NvidiaDevicePluginDetector", func() {
		var detector VendorDetector

		BeforeEach(func() {
			detector = NewNvidiaDevicePluginDetector()
		})

		It("should return correct resource name prefixes", func() {
			Expect(detector.GetResourceNamePrefixes()).To(Equal([]string{"nvidia.com/gpu", "nvidia.com/mig"}))
		})

		It("should identify nvidia device plugin for standard GPU", func() {
			system, realDeviceID := detector.GetUsedBySystemAndRealDeviceID(
				"GPU-8511dc03-7592-b8b7-1a92-582d40da52fb",
				"nvidia.com/gpu",
			)
			Expect(system).To(Equal(string(UsedByNvidiaDevicePlugin)))
			Expect(realDeviceID).To(Equal("GPU-8511dc03-7592-b8b7-1a92-582d40da52fb"))
		})

		It("should identify 3rd party device plugin for modified GPU ID", func() {
			system, realDeviceID := detector.GetUsedBySystemAndRealDeviceID(
				"GPU-422d6152-4d4b-5b0e-9d3a-b3b44e2742ea-1",
				"nvidia.com/gpu",
			)
			Expect(system).To(Equal(string(UsedBy3rdPartyDevicePlugin)))
			Expect(realDeviceID).To(Equal("GPU-422d6152-4d4b-5b0e-9d3a-b3b44e2742ea"))
		})

		It("should return nvidia-device-plugin for MIG", func() {
			system, realDeviceID := detector.GetUsedBySystemAndRealDeviceID(
				"MIG-422d6152-4d4b-5b0e-9d3a-b3b44e2742ea",
				"nvidia.com/mig-1g.5gb",
			)
			Expect(system).To(Equal(string(UsedByNvidiaDevicePlugin)))
			Expect(realDeviceID).To(Equal("MIG-422d6152-4d4b-5b0e-9d3a-b3b44e2742ea"))
		})
	})

	Describe("processDeviceState", func() {
		It("should handle device added", func() {
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
			Expect(err).NotTo(HaveOccurred())
			defer func() { _ = os.Remove(tmpFile.Name()) }()

			_, err = tmpFile.WriteString(checkpointData)
			Expect(err).NotTo(HaveOccurred())
			Expect(tmpFile.Close()).To(Succeed())

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
			nvdpDetector := NewNvidiaDevicePluginDetector()
			for _, prefix := range nvdpDetector.GetResourceNamePrefixes() {
				detector.vendorDetectors[prefix] = nvdpDetector
			}

			checkpoint, err := detector.readCheckpointFile()
			Expect(err).NotTo(HaveOccurred())
			allocated, _ := detector.extractDeviceIDs(checkpoint)
			Expect(allocated).To(HaveKey("gpu-7d8429d5-531d-d6a6-6510-3b662081a75a"))

			err = detector.processDeviceState(false)
			Expect(err).NotTo(HaveOccurred())
			Expect(mockAPI.AssertExpectations(GinkgoT())).To(BeTrue())
		})

		It("should handle device removed", func() {
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
			Expect(err).NotTo(HaveOccurred())
			defer func() { _ = os.Remove(tmpFile.Name()) }()

			_, err = tmpFile.WriteString(checkpointData)
			Expect(err).NotTo(HaveOccurred())
			Expect(tmpFile.Close()).To(Succeed())

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
			nvdpDetector := NewNvidiaDevicePluginDetector()
			for _, prefix := range nvdpDetector.GetResourceNamePrefixes() {
				detector.vendorDetectors[prefix] = nvdpDetector
			}

			err = detector.processDeviceState(false)
			Expect(err).NotTo(HaveOccurred())
			Expect(mockAPI.AssertExpectations(GinkgoT())).To(BeTrue())
		})
	})

	Describe("findEntryForDevice", func() {
		It("should find entry for device", func() {
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
			Expect(entry.ResourceName).To(Equal("nvidia.com/gpu"))
		})
	})
})
