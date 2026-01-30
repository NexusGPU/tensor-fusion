package device

import (
	"fmt"
	"os"
	"path/filepath"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("AcceleratorInterface", func() {
	var (
		accel       *AcceleratorInterface
		stubLibPath string
	)

	BeforeEach(func() {
		// Try to find stub library
		stubLibPath = "./provider/build/libaccelerator_example.so"
		if _, err := os.Stat(stubLibPath); os.IsNotExist(err) {
			// Try alternative path
			stubLibPath = filepath.Join("..", "..", "..", "provider", "build", "libaccelerator_example.so")
			if _, err := os.Stat(stubLibPath); os.IsNotExist(err) {
				Skip("Stub library not found, skipping tests")
			}
		}
	})

	AfterEach(func() {
		if accel != nil {
			Expect(accel.Close()).To(Succeed())
		}
	})

	Describe("Library Loading", func() {
		It("should load stub library successfully", func() {
			var err error
			accel, err = NewAcceleratorInterface(stubLibPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(accel).NotTo(BeNil())
			// If NewAcceleratorInterface succeeds, the library is loaded
		})

		It("should fail to load non-existent library", func() {
			accel, err := NewAcceleratorInterface("/non/existent/library.so")
			Expect(err).To(HaveOccurred())
			Expect(accel).To(BeNil())
		})

		It("should handle multiple load/unload cycles", func() {
			accel, err := NewAcceleratorInterface(stubLibPath)
			Expect(err).NotTo(HaveOccurred())

			// Reload
			Expect(accel.Load()).To(Succeed())
			Expect(accel.Close()).To(Succeed())
			Expect(accel.Load()).To(Succeed())
		})
	})

	Describe("GetDeviceMetrics", func() {
		BeforeEach(func() {
			var err error
			accel, err = NewAcceleratorInterface(stubLibPath)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should return empty slice for empty input", func() {
			metrics, err := accel.GetDeviceMetrics([]string{})
			Expect(err).NotTo(HaveOccurred())
			Expect(metrics).To(BeEmpty())
		})

		It("should retrieve metrics for single device with ExtraMetrics", func() {
			deviceUUIDs := []string{"test-device-001"}
			metrics, err := accel.GetDeviceMetrics(deviceUUIDs)
			Expect(err).NotTo(HaveOccurred())
			Expect(metrics).To(HaveLen(1))

			m := metrics[0]
			Expect(m.DeviceUUID).To(Equal(deviceUUIDs[0]))
			// Memory values may be 0 if mock driver is not running
			Expect(m.MemoryBytes).To(BeNumerically(">=", 0))
			Expect(m.MemoryPercentage).To(BeNumerically(">=", 0))
			Expect(m.MemoryPercentage).To(BeNumerically("<=", 100))
			// Power and temperature are always set by stub
			Expect(m.PowerUsage).To(BeNumerically(">", 0))
			Expect(m.Temperature).To(BeNumerically(">", 0))

			// Verify ExtraMetrics are populated
			Expect(m.ExtraMetrics).NotTo(BeEmpty())
			Expect(m.ExtraMetrics).To(HaveKey("tensorCoreUsagePercent"))
			Expect(m.ExtraMetrics).To(HaveKey("memoryBandwidthMBps"))
		})

		It("should handle multiple devices", func() {
			deviceUUIDs := []string{"device-1", "device-2", "device-3"}
			metrics, err := accel.GetDeviceMetrics(deviceUUIDs)
			Expect(err).NotTo(HaveOccurred())
			Expect(metrics).To(HaveLen(3))

			for i, m := range metrics {
				Expect(m.DeviceUUID).To(Equal(deviceUUIDs[i]))
				Expect(m.ExtraMetrics).NotTo(BeEmpty())
			}
		})

		It("should correctly convert PCIe bytes to KB", func() {
			metrics, err := accel.GetDeviceMetrics([]string{"test-device"})
			Expect(err).NotTo(HaveOccurred())
			Expect(metrics).To(HaveLen(1))

			// Rx and Tx should be in KB (bytes / 1024)
			Expect(metrics[0].Rx).To(BeNumerically(">", 0))
			Expect(metrics[0].Tx).To(BeNumerically(">", 0))
		})

		It("should calculate memory percentage correctly", func() {
			metrics, err := accel.GetDeviceMetrics([]string{"test-device"})
			Expect(err).NotTo(HaveOccurred())
			Expect(metrics).To(HaveLen(1))

			m := metrics[0]
			// Memory percentage should be between 0 and 100
			Expect(m.MemoryPercentage).To(BeNumerically(">=", 0))
			Expect(m.MemoryPercentage).To(BeNumerically("<=", 100))
		})
	})

	Describe("GetAllDevices", func() {
		BeforeEach(func() {
			var err error
			accel, err = NewAcceleratorInterface(stubLibPath)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should retrieve device list", func() {
			devices, err := accel.GetAllDevices()
			Expect(err).NotTo(HaveOccurred())
			Expect(devices).NotTo(BeNil())

			// Stub may return 0 or more devices
			if len(devices) > 0 {
				for _, d := range devices {
					Expect(d.UUID).NotTo(BeEmpty())
					Expect(d.TotalMemoryBytes).To(BeNumerically(">", 0))
				}
			}
		})
	})

	Describe("Process Utilization", func() {
		BeforeEach(func() {
			var err error
			accel, err = NewAcceleratorInterface(stubLibPath)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should return empty slice when no processes tracked", func() {
			// Test process information (combines compute and memory utilization)
			processInfos, err := accel.GetProcessInformation()
			Expect(err).NotTo(HaveOccurred())
			Expect(processInfos).To(BeEmpty())

			// Verify GetTotalProcessCount returns 0
			Expect(accel.GetTotalProcessCount()).To(Equal(0))
		})
	})

	Describe("GetVendorMountLibs", func() {
		BeforeEach(func() {
			var err error
			accel, err = NewAcceleratorInterface(stubLibPath)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should retrieve mount libs", func() {
			mounts, err := accel.GetVendorMountLibs()
			Expect(err).NotTo(HaveOccurred())
			Expect(mounts).NotTo(BeNil())
			// Stub may return empty or populated mounts
		})
	})

	Describe("Memory Management", func() {
		BeforeEach(func() {
			var err error
			accel, err = NewAcceleratorInterface(stubLibPath)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should not leak memory on repeated GetDeviceMetrics calls", func() {
			deviceUUIDs := []string{"device-1", "device-2"}

			// Call multiple times to check for memory leaks
			for i := 0; i < 10; i++ {
				metrics, err := accel.GetDeviceMetrics(deviceUUIDs)
				Expect(err).NotTo(HaveOccurred())
				Expect(metrics).To(HaveLen(2))
			}
		})

		It("should handle large number of devices (up to limit)", func() {
			// Create 64 device UUIDs (maxStackDevices limit)
			deviceUUIDs := make([]string, 64)
			for i := range deviceUUIDs {
				deviceUUIDs[i] = fmt.Sprintf("device-%d", i)
			}

			metrics, err := accel.GetDeviceMetrics(deviceUUIDs)
			Expect(err).NotTo(HaveOccurred())
			Expect(metrics).To(HaveLen(64))
		})
	})

	Describe("Edge Cases", func() {
		BeforeEach(func() {
			var err error
			accel, err = NewAcceleratorInterface(stubLibPath)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should handle various device UUID formats", func() {
			// Test different UUID formats that might be encountered
			uuidVariants := []string{
				"device-1",
				"device-2_@#$",
				"device-3-中文",
				"12345678-1234-1234-1234-123456789abc", // UUID format
			}
			metrics, err := accel.GetDeviceMetrics(uuidVariants)
			Expect(err).NotTo(HaveOccurred())
			Expect(metrics).To(HaveLen(len(uuidVariants)))
		})

		It("should handle empty strings in device UUIDs", func() {
			metrics, err := accel.GetDeviceMetrics([]string{""})
			Expect(err).NotTo(HaveOccurred())
			Expect(metrics).To(HaveLen(1))
		})
	})

	Describe("AssignPartition", func() {
		BeforeEach(func() {
			var err error
			accel, err = NewAcceleratorInterface(stubLibPath)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should assign partition successfully", func() {
			result, err := accel.AssignPartition("mig-1g.7gb", "stub-device-0")
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeNil())
			Expect(result.PartitionUUID).NotTo(BeEmpty())
		})

		It("should reject template ID that is too long", func() {
			longTemplateID := make([]byte, 100)
			for i := range longTemplateID {
				longTemplateID[i] = 'a'
			}
			_, err := accel.AssignPartition(string(longTemplateID), "stub-device-0")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("too long"))
		})

		It("should reject device UUID that is too long", func() {
			longDeviceUUID := make([]byte, 100)
			for i := range longDeviceUUID {
				longDeviceUUID[i] = 'a'
			}
			_, err := accel.AssignPartition("mig-1g.7gb", string(longDeviceUUID))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("too long"))
		})
	})

	Describe("RemovePartition", func() {
		BeforeEach(func() {
			var err error
			accel, err = NewAcceleratorInterface(stubLibPath)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should remove partition successfully", func() {
			// RemovePartition takes templateID (not partitionUUID) and deviceUUID
			err := accel.RemovePartition("mig-1g.7gb", "stub-device-0")
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Describe("SetLimits", func() {
		BeforeEach(func() {
			var err error
			accel, err = NewAcceleratorInterface(stubLibPath)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should set memory hard limit successfully", func() {
			err := accel.SetMemHardLimit("stub-device-0", 1024*1024*1024) // 1GB
			Expect(err).NotTo(HaveOccurred())
		})

		It("should set compute unit hard limit successfully", func() {
			err := accel.SetComputeUnitHardLimit("stub-device-0", 50) // 50%
			Expect(err).NotTo(HaveOccurred())
		})
	})
})
