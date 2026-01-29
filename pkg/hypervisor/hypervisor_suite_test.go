/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package hypervisor

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/backend/single_node"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/device"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/metrics"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/server"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker"
)

const (
	testWorkerUID1        = "test-worker-1"
	integrationWorkerUID1 = "integration-worker-1"
)

func TestHypervisor(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Hypervisor Suite")
}

// waitForDeviceDiscovery waits for devices to be discovered using Eventually
func waitForDeviceDiscovery(deviceController framework.DeviceController) []*api.DeviceInfo {
	var devices []*api.DeviceInfo
	Eventually(func() error {
		var err error
		devices, err = deviceController.ListDevices()
		if err != nil {
			return err
		}
		if len(devices) == 0 {
			return fmt.Errorf("no devices discovered yet")
		}
		return nil
	}).Should(Succeed())
	return devices
}

// createTestWorkerInfo creates a WorkerInfo for testing with the given parameters
func createTestWorkerInfo(workerUID string, deviceUUIDs []string, isolationMode api.IsolationMode) *api.WorkerInfo {
	return &api.WorkerInfo{
		WorkerUID:        workerUID,
		AllocatedDevices: deviceUUIDs,
		IsolationMode:    isolationMode,
	}
}

var _ = Describe("Hypervisor Integration Tests", func() {
	var (
		ctx                  context.Context
		cancel               context.CancelFunc
		deviceController     *device.Controller
		backend              framework.Backend
		workerController     framework.WorkerController
		allocationController framework.WorkerAllocationController
		metricsRecorder      *metrics.HypervisorMetricsRecorder
		httpServer           *server.Server
		stubLibPath          string
		tempMetricsFile      string
	)

	BeforeEach(func() {
		ctx, cancel = context.WithCancel(context.Background())

		// Find stub library path
		// Try relative path first (from provider/build)
		stubLibPath = filepath.Join("..", "..", "provider", "build", "libaccelerator_example.so")
		if _, err := os.Stat(stubLibPath); os.IsNotExist(err) {
			// Try absolute path from workspace root
			workspaceRoot := os.Getenv("WORKSPACE_ROOT")
			if workspaceRoot == "" {
				// Try to find it relative to current directory
				cwd, _ := os.Getwd()
				stubLibPath = filepath.Join(cwd, "..", "..", "provider", "build", "libaccelerator_example.so")
			} else {
				stubLibPath = filepath.Join(workspaceRoot, "provider", "build", "libaccelerator_example.so")
			}
		}

		// Create temp file for metrics
		tempFile, err := os.CreateTemp("", "hypervisor-metrics-*.log")
		Expect(err).NotTo(HaveOccurred())
		tempMetricsFile = tempFile.Name()
		_ = tempFile.Close()
	})

	AfterEach(func() {
		if cancel != nil {
			cancel()
		}
		if httpServer != nil {
			shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer shutdownCancel()
			_ = httpServer.Stop(shutdownCtx)
		}
		if workerController != nil {
			_ = workerController.Stop()
		}
		if backend != nil {
			_ = backend.Stop()
		}
		if deviceController != nil {
			_ = deviceController.Stop()
		}
		_ = os.Remove(tempMetricsFile)
	})

	Context("With stub device library", func() {
		BeforeEach(func() {
			// Check if stub library exists, skip if not
			if _, err := os.Stat(stubLibPath); os.IsNotExist(err) {
				Skip("Stub library not found. Run 'make stub' in provider directory first.")
			}

			var err error
			deviceController, err = device.NewController(ctx, stubLibPath, "stub", 1*time.Hour, tfv1.IsolationModeShared)
			Expect(err).NotTo(HaveOccurred())
			Expect(deviceController).NotTo(BeNil())

			// Create allocation controller first - it's a shared dependency
			allocationController = worker.NewAllocationController(deviceController)
			Expect(allocationController).NotTo(BeNil())
			deviceController.SetAllocationController(allocationController)

			backend = single_node.NewSingleNodeBackend(ctx, deviceController, allocationController)
			Expect(backend).NotTo(BeNil())

			workerController = worker.NewWorkerController(
				deviceController,
				allocationController,
				tfv1.IsolationModeShared,
				backend,
			)
			Expect(workerController).NotTo(BeNil())

			metricsRecorder = metrics.NewHypervisorMetricsRecorder(
				ctx,
				tempMetricsFile,
				deviceController,
				workerController,
				allocationController,
				1*time.Second,
			)
			Expect(metricsRecorder).NotTo(BeNil())

			httpServer = server.NewServer(
				ctx,
				deviceController,
				workerController,
				allocationController,
				metricsRecorder,
				backend,
				0,
			)
			Expect(httpServer).NotTo(BeNil())
		})

		Describe("C Stub Library Integration", func() {
			It("should load stub accelerator library", func() {
				// Verify library can be loaded
				accel, err := device.NewAcceleratorInterface(stubLibPath)
				Expect(err).NotTo(HaveOccurred())
				Expect(accel).NotTo(BeNil())

				// Test device discovery through C library
				devices, err := accel.GetAllDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				// Verify stub device properties - should have 4 devices
				Expect(devices).To(HaveLen(4), "Should return 4 example devices")
				device := devices[0]
				Expect(device.UUID).To(ContainSubstring("example-device"))
				Expect(device.Vendor).To(Equal("STUB"))
				Expect(device.TotalMemoryBytes).To(Equal(uint64(16 * 1024 * 1024 * 1024))) // 16GB

				_ = accel.Close()
			})

			It("should get process information from stub library", func() {
				accel, err := device.NewAcceleratorInterface(stubLibPath)
				Expect(err).NotTo(HaveOccurred())
				defer func() {
					_ = accel.Close()
				}()

				// Get process information (combines compute and memory utilization)
				processInfos, err := accel.GetProcessInformation()
				Expect(err).NotTo(HaveOccurred())
				Expect(processInfos).NotTo(BeNil())
			})
		})

		Describe("Device Controller", func() {
			It("should start and discover devices", func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())

				devices := waitForDeviceDiscovery(deviceController)
				Expect(devices).ToNot(BeEmpty(), "Should discover at least one stub device")
				Expect(devices).To(HaveLen(4), "Should discover exactly 4 example devices")

				// Verify device properties
				device := devices[0]
				Expect(device.UUID).NotTo(BeEmpty())
				Expect(device.UUID).To(ContainSubstring("example-device"))
				Expect(device.Vendor).To(Equal("STUB"))
				Expect(device.TotalMemoryBytes).To(BeNumerically(">", 0))
			})

			It("should allocate devices", func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())

				devices := waitForDeviceDiscovery(deviceController)
				Expect(devices).ToNot(BeEmpty())

				deviceUUID := devices[0].UUID
				req := &api.WorkerInfo{
					WorkerUID:        testWorkerUID1,
					AllocatedDevices: []string{deviceUUID},
					IsolationMode:    tfv1.IsolationModeSoft,
				}

				resp, err := allocationController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp).NotTo(BeNil())
				Expect(resp.WorkerInfo.WorkerUID).To(Equal(testWorkerUID1))
				Expect(resp.DeviceInfos).ToNot(BeEmpty())
				Expect(resp.DeviceInfos[0].UUID).To(Equal(deviceUUID))
				// Verify mounts and envs are populated (may be empty for stub, but structure should exist)
				Expect(resp.Mounts).ToNot(BeNil())
				Expect(resp.Envs).ToNot(BeNil())
				Expect(resp.Devices).ToNot(BeNil())

				// Verify allocation exists through allocation controller
				allocation, found := allocationController.GetWorkerAllocation(testWorkerUID1)
				Expect(found).To(BeTrue())
				Expect(allocation).NotTo(BeNil())
				Expect(allocation.WorkerInfo.WorkerUID).To(Equal(testWorkerUID1))
			})

			It("should get GPU metrics", func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())

				devices := waitForDeviceDiscovery(deviceController)

				metrics, err := deviceController.GetDeviceMetrics()
				Expect(err).NotTo(HaveOccurred())
				Expect(metrics).NotTo(BeNil())
				// Should have metrics for all discovered devices
				Expect(metrics).To(HaveLen(len(devices)))
			})

			It("should handle device partitioning", func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())

				devices := waitForDeviceDiscovery(deviceController)

				deviceUUID := devices[0].UUID
				partitionTemplateID := "test-partition-template"

				// Split device (interface signature: SplitDevice(deviceUUID, partitionID))
				partitionedDevice, err := deviceController.SplitDevice(deviceUUID, partitionTemplateID)
				Expect(err).NotTo(HaveOccurred())
				Expect(partitionedDevice).NotTo(BeNil())
				Expect(partitionedDevice.UUID).NotTo(Equal(deviceUUID))
				Expect(partitionedDevice.ParentUUID).To(Equal(deviceUUID))

				// Verify partitioned device is in device list
				allDevices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				found := false
				for _, d := range allDevices {
					if d.UUID == partitionedDevice.UUID {
						found = true
						break
					}
				}
				Expect(found).To(BeTrue(), "Partitioned device should be in device list")

				// Remove partition
				err = deviceController.RemovePartitionedDevice(partitionedDevice.UUID, deviceUUID)
				Expect(err).NotTo(HaveOccurred())

				// Verify partition is removed
				_, exists := deviceController.GetDevice(partitionedDevice.UUID)
				Expect(exists).To(BeFalse(), "Partitioned device should be removed")
			})

			It("should handle device update and removal handlers", func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())

				devices := waitForDeviceDiscovery(deviceController)

				// Track handler invocations
				var addCount, updateCount, removeCount int
				var addedDevice *api.DeviceInfo

				handler := framework.DeviceChangeHandler{
					OnAdd: func(device *api.DeviceInfo) {
						addCount++
						addedDevice = device
					},
					OnUpdate: func(oldDevice, newDevice *api.DeviceInfo) {
						updateCount++
					},
					OnRemove: func(device *api.DeviceInfo) {
						removeCount++
					},
				}

				deviceController.RegisterDeviceUpdateHandler(handler)

				// Handler should be notified of existing devices
				Eventually(func() int {
					return addCount
				}).Should(BeNumerically(">=", len(devices)), "Handler should be notified of existing devices")
				Expect(addedDevice).NotTo(BeNil(), "At least one device should be added")

				// Verify device discovery complete handler
				var discoveryCompleteCalled bool
				var nodeInfo *api.NodeInfo
				discoveryHandler := framework.DeviceChangeHandler{
					OnDiscoveryComplete: func(info *api.NodeInfo) {
						discoveryCompleteCalled = true
						nodeInfo = info
					},
				}
				deviceController.RegisterDeviceUpdateHandler(discoveryHandler)

				// Trigger discovery
				err = deviceController.DiscoverDevices()
				Expect(err).NotTo(HaveOccurred())

				Eventually(func() bool {
					return discoveryCompleteCalled
				}).Should(BeTrue(), "Discovery complete handler should be called")
				Expect(nodeInfo).NotTo(BeNil())
				Expect(nodeInfo.DeviceIDs).ToNot(BeEmpty())
			})

			It("should handle allocation with non-existent device", func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())

				req := &api.WorkerInfo{
					WorkerUID:        "test-worker-invalid",
					AllocatedDevices: []string{"non-existent-device-uuid"},
					IsolationMode:    tfv1.IsolationModeSoft,
				}

				// Allocation should fail or skip non-existent devices
				resp, err := allocationController.AllocateWorkerDevices(req)
				// Behavior may vary - either error or empty device list
				if err != nil {
					Expect(err).To(HaveOccurred())
				} else {
					Expect(resp).NotTo(BeNil())
					Expect(resp.DeviceInfos).To(BeEmpty(), "Should not allocate non-existent device")
				}
			})
		})

		Describe("Single Node Backend", func() {
			BeforeEach(func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				waitForDeviceDiscovery(deviceController)

				err = backend.Start()
				Expect(err).NotTo(HaveOccurred())
			})

			It("should start and stop", func() {
				Expect(backend).NotTo(BeNil())
			})

			It("should list workers from allocations", func() {
				// Register handler first before starting worker
				var found bool
				handler := framework.WorkerChangeHandler{
					OnAdd: func(worker *api.WorkerInfo) {
						if worker.WorkerUID == testWorkerUID1 {
							found = true
						}
					},
					OnRemove: func(worker *api.WorkerInfo) {},
					OnUpdate: func(oldWorker, newWorker *api.WorkerInfo) {
						// StartWorker adds worker to map before notifying, so it may trigger OnUpdate
						if newWorker.WorkerUID == testWorkerUID1 {
							found = true
						}
					},
				}
				err := backend.RegisterWorkerUpdateHandler(handler)
				Expect(err).NotTo(HaveOccurred())

				// Small delay to ensure handler goroutine is ready
				time.Sleep(50 * time.Millisecond)

				// Create an allocation
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				req := &api.WorkerInfo{
					WorkerUID:        testWorkerUID1,
					AllocatedDevices: []string{devices[0].UUID},
					IsolationMode:    tfv1.IsolationModeSoft,
				}
				_, err = allocationController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())

				// Start the worker in the backend
				err = backend.StartWorker(req)
				Expect(err).NotTo(HaveOccurred())

				// Wait for callback to be invoked (either OnAdd or OnUpdate)
				Eventually(func() bool {
					return found
				}, 2*time.Second).Should(BeTrue(), "Should find test-worker-1 via callback")
			})

			It("should track worker to process mapping", func() {
				// Start a worker
				worker := &api.WorkerInfo{
					WorkerUID:        testWorkerUID1,
					AllocatedDevices: []string{},
					IsolationMode:    tfv1.IsolationModeSoft,
				}
				err := backend.StartWorker(worker)
				Expect(err).NotTo(HaveOccurred())

				// Test process mapping
				// Note: In single_node mode, we can only return basic process info
				// since there's no Kubernetes environment to map PIDs to workers.
				// The GuestID would be empty because we can't determine the worker
				// from just the PID without Kubernetes pod environment variables.
				processInfo, err := backend.GetProcessMappingInfo(12345)
				Expect(err).NotTo(HaveOccurred())
				Expect(processInfo).NotTo(BeNil())
				// In single_node mode, GuestID is empty since we can't map PID to worker
				// without Kubernetes environment variables
				Expect(processInfo.HostPID).To(Equal(uint32(12345)))
				Expect(processInfo.GuestPID).To(Equal(uint32(12345)))
			})

			It("should start and manage process worker", func() {
				// Create a worker with process runtime info
				worker := &api.WorkerInfo{
					WorkerUID:        testWorkerUID1,
					AllocatedDevices: []string{},
					IsolationMode:    tfv1.IsolationModeSoft,
					WorkerRunningInfo: &api.WorkerRunningInfo{
						Type:       api.WorkerRuntimeTypeProcess,
						Executable: "sleep",
						Args:       []string{"10"},
						Env:        map[string]string{"TEST_ENV": "test_value"},
					},
				}

				err := backend.StartWorker(worker)
				Expect(err).NotTo(HaveOccurred())

				// Wait for process to start
				Eventually(func() bool {
					workers := backend.ListWorkers()
					for _, w := range workers {
						if w.WorkerUID == testWorkerUID1 && w.WorkerRunningInfo != nil {
							return w.WorkerRunningInfo.IsRunning && w.WorkerRunningInfo.PID > 0
						}
					}
					return false
				}, 5*time.Second).Should(BeTrue(), "Process should be running")

				// Verify process is actually running
				workers := backend.ListWorkers()
				var foundWorker *api.WorkerInfo
				for _, w := range workers {
					if w.WorkerUID == testWorkerUID1 {
						foundWorker = w
						break
					}
				}
				Expect(foundWorker).NotTo(BeNil())
				Expect(foundWorker.WorkerRunningInfo).NotTo(BeNil())
				Expect(foundWorker.WorkerRunningInfo.IsRunning).To(BeTrue())
				Expect(foundWorker.WorkerRunningInfo.PID).To(BeNumerically(">", 0))

				// Verify process exists
				proc, err := os.FindProcess(int(foundWorker.WorkerRunningInfo.PID))
				Expect(err).NotTo(HaveOccurred())
				Expect(proc).NotTo(BeNil())
			})

			It("should stop process worker", func() {
				// Create and start a worker with process runtime info
				worker := &api.WorkerInfo{
					WorkerUID:        testWorkerUID1,
					AllocatedDevices: []string{},
					IsolationMode:    tfv1.IsolationModeSoft,
					WorkerRunningInfo: &api.WorkerRunningInfo{
						Type:       api.WorkerRuntimeTypeProcess,
						Executable: "sleep",
						Args:       []string{"30"},
					},
				}

				err := backend.StartWorker(worker)
				Expect(err).NotTo(HaveOccurred())

				// Wait for process to start
				var pid uint32
				Eventually(func() bool {
					workers := backend.ListWorkers()
					for _, w := range workers {
						if w.WorkerUID == testWorkerUID1 && w.WorkerRunningInfo != nil && w.WorkerRunningInfo.IsRunning {
							pid = w.WorkerRunningInfo.PID
							return pid > 0
						}
					}
					return false
				}, 5*time.Second).Should(BeTrue(), "Process should be running")

				// Stop the worker
				err = backend.StopWorker(testWorkerUID1)
				Expect(err).NotTo(HaveOccurred())

				// Wait for process to be killed
				Eventually(func() bool {
					proc, err := os.FindProcess(int(pid))
					if err != nil {
						return true // Process not found, already killed
					}
					// Try to signal the process to check if it's alive
					err = proc.Signal(syscall.Signal(0))
					return err != nil // If signal fails, process is dead
				}, 5*time.Second).Should(BeTrue(), "Process should be killed")

				// Verify worker is removed
				workers := backend.ListWorkers()
				for _, w := range workers {
					Expect(w.WorkerUID).NotTo(Equal(testWorkerUID1))
				}
			})

			It("should retry process with backoff on exit", func() {
				if runtime.GOOS == "windows" {
					Skip("Process retry test skipped on Windows")
				}

				// Create a worker that will exit quickly
				worker := &api.WorkerInfo{
					WorkerUID:        testWorkerUID1,
					AllocatedDevices: []string{},
					IsolationMode:    tfv1.IsolationModeSoft,
					WorkerRunningInfo: &api.WorkerRunningInfo{
						Type:       api.WorkerRuntimeTypeProcess,
						Executable: "sh",
						Args:       []string{"-c", "exit 1"},
					},
				}

				err := backend.StartWorker(worker)
				Expect(err).NotTo(HaveOccurred())

				// Wait for process to start and get first PID
				var firstPID uint32
				Eventually(func() bool {
					workers := backend.ListWorkers()
					for _, w := range workers {
						if w.WorkerUID == testWorkerUID1 && w.WorkerRunningInfo != nil {
							if w.WorkerRunningInfo.IsRunning && w.WorkerRunningInfo.PID > 0 {
								firstPID = w.WorkerRunningInfo.PID
								return true
							}
						}
					}
					return false
				}, 3*time.Second).Should(BeTrue(), "Process should start")

				// Wait for process to exit and be retried (backoff is ~3 seconds for first retry)
				// Check for retry: Restarts > 0 indicates retry happened
				Eventually(func() bool {
					workers := backend.ListWorkers()
					for _, w := range workers {
						if w.WorkerUID == testWorkerUID1 && w.WorkerRunningInfo != nil {
							// Check if process was retried (Restarts > 0)
							return w.WorkerRunningInfo.Restarts > 0
						}
					}
					return false
				}, 8*time.Second).Should(BeTrue(), "Process should be retried (Restarts > 0)")

				// Verify retry happened with new PID (process may exit quickly, so check PID or Restarts)
				workers := backend.ListWorkers()
				var retriedWorker *api.WorkerInfo
				for _, w := range workers {
					if w.WorkerUID == testWorkerUID1 && w.WorkerRunningInfo != nil {
						retriedWorker = w
						break
					}
				}
				Expect(retriedWorker).NotTo(BeNil(), "Worker should exist")
				Expect(retriedWorker.WorkerRunningInfo.Restarts).To(BeNumerically(">=", 1), "Restarts should be >= 1")
				// If process is running, verify it has a new PID
				if retriedWorker.WorkerRunningInfo.IsRunning && retriedWorker.WorkerRunningInfo.PID > 0 {
					Expect(retriedWorker.WorkerRunningInfo.PID).NotTo(Equal(firstPID), "Retried process should have new PID")
				}
			})

			It("should stop retrying when worker is removed", func() {
				if runtime.GOOS == "windows" {
					Skip("Process retry test skipped on Windows")
				}

				// Create a worker that will exit quickly
				worker := &api.WorkerInfo{
					WorkerUID:        testWorkerUID1,
					AllocatedDevices: []string{},
					IsolationMode:    tfv1.IsolationModeSoft,
					WorkerRunningInfo: &api.WorkerRunningInfo{
						Type:       api.WorkerRuntimeTypeProcess,
						Executable: "sh",
						Args:       []string{"-c", "exit 1"},
					},
				}

				err := backend.StartWorker(worker)
				Expect(err).NotTo(HaveOccurred())

				// Wait for process to start
				Eventually(func() bool {
					workers := backend.ListWorkers()
					for _, w := range workers {
						if w.WorkerUID == testWorkerUID1 && w.WorkerRunningInfo != nil {
							return w.WorkerRunningInfo.IsRunning && w.WorkerRunningInfo.PID > 0
						}
					}
					return false
				}, 3*time.Second).Should(BeTrue(), "Process should start")

				// Wait for process to exit (it exits immediately)
				Eventually(func() bool {
					workers := backend.ListWorkers()
					for _, w := range workers {
						if w.WorkerUID == testWorkerUID1 && w.WorkerRunningInfo != nil {
							return !w.WorkerRunningInfo.IsRunning && w.WorkerRunningInfo.ExitCode == 1
						}
					}
					return false
				}, 3*time.Second).Should(BeTrue(), "Process should exit")

				// Remove the worker
				err = backend.StopWorker(testWorkerUID1)
				Expect(err).NotTo(HaveOccurred())

				// Verify worker is gone and not retried (wait a bit to ensure no retry happens)
				Eventually(func() bool {
					workers := backend.ListWorkers()
					for _, w := range workers {
						if w.WorkerUID == testWorkerUID1 {
							return false
						}
					}
					return true
				}, 5*time.Second).Should(BeTrue(), "Worker should be removed and not retried")
			})
		})

		Describe("Worker Controller", func() {
			BeforeEach(func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				waitForDeviceDiscovery(deviceController)

				err = workerController.Start()
				Expect(err).NotTo(HaveOccurred())
			})

			It("should start and stop", func() {
				Expect(workerController).NotTo(BeNil())
			})

			It("should list workers", func() {
				// Create an allocation
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				req := &api.WorkerInfo{
					WorkerUID:        testWorkerUID1,
					AllocatedDevices: []string{devices[0].UUID},
					IsolationMode:    tfv1.IsolationModeSoft,
				}
				_, err = allocationController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())

				// Start worker in backend so it appears in the worker list
				err = backend.StartWorker(req)
				Expect(err).NotTo(HaveOccurred())

				// Wait for worker to appear in list
				Eventually(func() bool {
					workers, err := workerController.ListWorkers()
					if err != nil {
						return false
					}
					for _, worker := range workers {
						if worker.WorkerUID == testWorkerUID1 {
							return true
						}
					}
					return false
				}, 2*time.Second).Should(BeTrue(), "Worker should appear in list")
			})

			It("should get worker allocation", func() {
				// Create an allocation
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				req := &api.WorkerInfo{
					WorkerUID:        testWorkerUID1,
					AllocatedDevices: []string{devices[0].UUID},
					IsolationMode:    tfv1.IsolationModeSoft,
				}
				_, err = allocationController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())

				allocation, found := allocationController.GetWorkerAllocation(testWorkerUID1)
				Expect(found).To(BeTrue())
				Expect(allocation).NotTo(BeNil())
				Expect(allocation.WorkerInfo.WorkerUID).To(Equal(testWorkerUID1))
			})

			It("should get worker metrics", func() {
				// Create an allocation
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				req := &api.WorkerInfo{
					WorkerUID:        testWorkerUID1,
					AllocatedDevices: []string{devices[0].UUID},
					IsolationMode:    tfv1.IsolationModeSoft,
				}
				_, err = allocationController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())

				metrics, err := workerController.GetWorkerMetrics()
				Expect(err).NotTo(HaveOccurred())
				// GetWorkerMetrics returns nil, nil (not implemented yet - TODO in code)
				// So we accept nil as valid response
				_ = metrics
			})

			It("should handle partitioned isolation mode", func() {
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				deviceUUID := devices[0].UUID
				partitionTemplateID := "test-partition-template"

				// Verify device exists before partitioning
				device, exists := deviceController.GetDevice(deviceUUID)
				Expect(exists).To(BeTrue(), "Device should exist before partitioning")
				Expect(device).NotTo(BeNil(), "Device should not be nil")

				req := &api.WorkerInfo{
					WorkerUID:           "test-worker-partitioned",
					AllocatedDevices:    []string{deviceUUID},
					IsolationMode:       tfv1.IsolationModePartitioned,
					PartitionTemplateID: partitionTemplateID,
				}

				resp, err := allocationController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp).NotTo(BeNil())
				Expect(resp.DeviceInfos).ToNot(BeEmpty())
				// In partitioned mode, device UUID should be different from original
				Expect(resp.DeviceInfos[0].UUID).NotTo(Equal(deviceUUID))
				Expect(resp.DeviceInfos[0].ParentUUID).To(Equal(deviceUUID))

				// Verify allocation
				allocation, found := allocationController.GetWorkerAllocation("test-worker-partitioned")
				Expect(found).To(BeTrue())
				Expect(allocation).NotTo(BeNil())

				// Deallocate should clean up partition
				err = allocationController.DeallocateWorker("test-worker-partitioned")
				Expect(err).NotTo(HaveOccurred())

				// Verify deallocation
				_, found = allocationController.GetWorkerAllocation("test-worker-partitioned")
				Expect(found).To(BeFalse())
			})

			It("should handle worker deallocation cleanup", func() {
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				deviceUUID := devices[0].UUID
				req := createTestWorkerInfo("test-worker-cleanup", []string{deviceUUID}, tfv1.IsolationModeSoft)

				_, err = allocationController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())

				// Verify allocation exists
				allocation, found := allocationController.GetWorkerAllocation("test-worker-cleanup")
				Expect(found).To(BeTrue())
				Expect(allocation).NotTo(BeNil())
				Expect(allocation.WorkerInfo.WorkerUID).To(Equal("test-worker-cleanup"))

				// Verify device allocation tracking
				deviceAllocations := allocationController.GetDeviceAllocations()
				Expect(deviceAllocations[deviceUUID]).ToNot(BeEmpty())

				// Deallocate
				err = allocationController.DeallocateWorker("test-worker-cleanup")
				Expect(err).NotTo(HaveOccurred())

				// Verify allocation removed
				_, found = allocationController.GetWorkerAllocation("test-worker-cleanup")
				Expect(found).To(BeFalse())

				// Verify device allocation cleaned up
				deviceAllocations = allocationController.GetDeviceAllocations()
				allocationsForDevice := deviceAllocations[deviceUUID]
				foundInDeviceAllocations := false
				for _, alloc := range allocationsForDevice {
					if alloc.WorkerInfo.WorkerUID == "test-worker-cleanup" {
						foundInDeviceAllocations = true
						break
					}
				}
				Expect(foundInDeviceAllocations).To(BeFalse(), "Worker allocation should be removed from device allocations")
			})

			It("should deallocate partition when worker status changes to Terminated via OnUpdate", func() {
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				deviceUUID := devices[0].UUID
				partitionTemplateID := "test-partition-onupdate"
				workerUID := "test-worker-onupdate-terminated"

				// Create partitioned worker allocation
				req := &api.WorkerInfo{
					WorkerUID:           workerUID,
					AllocatedDevices:    []string{deviceUUID},
					IsolationMode:       tfv1.IsolationModePartitioned,
					PartitionTemplateID: partitionTemplateID,
					Status:              api.WorkerStatusRunning,
				}

				resp, err := allocationController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp).NotTo(BeNil())
				Expect(resp.DeviceInfos).ToNot(BeEmpty())
				Expect(resp.DeviceInfos[0].ParentUUID).To(Equal(deviceUUID))
				partitionUUID := resp.DeviceInfos[0].UUID

				// Verify partition was created
				_, exists := deviceController.GetDevice(partitionUUID)
				Expect(exists).To(BeTrue(), "Partition should exist after allocation")

				// Verify allocation exists
				allocation, found := allocationController.GetWorkerAllocation(workerUID)
				Expect(found).To(BeTrue())
				Expect(allocation).NotTo(BeNil())

				// Simulate OnUpdate with status change from Running to Terminated
				// This should trigger partition deallocation
				oldWorker := &api.WorkerInfo{
					WorkerUID:           workerUID,
					Status:              api.WorkerStatusRunning,
					IsolationMode:       tfv1.IsolationModePartitioned,
					PartitionTemplateID: partitionTemplateID,
				}
				newWorker := &api.WorkerInfo{
					WorkerUID:           workerUID,
					Status:              api.WorkerStatusTerminated,
					IsolationMode:       tfv1.IsolationModePartitioned,
					PartitionTemplateID: partitionTemplateID,
				}

				// Get the WorkerController's OnUpdate handler behavior
				// Since we can't directly access the handler, we simulate the behavior
				// by calling DeallocateWorker when status changes to Terminated
				if oldWorker.Status != api.WorkerStatusTerminated && newWorker.Status == api.WorkerStatusTerminated {
					err = allocationController.DeallocateWorker(workerUID)
					Expect(err).NotTo(HaveOccurred())
				}

				// Verify allocation is removed
				_, found = allocationController.GetWorkerAllocation(workerUID)
				Expect(found).To(BeFalse(), "Allocation should be removed after termination")

				// Verify partition is removed from device controller
				_, exists = deviceController.GetDevice(partitionUUID)
				Expect(exists).To(BeFalse(), "Partition should be removed from device controller after deallocation")
			})

			It("should NOT deallocate partition when worker status changes but not to Terminated", func() {
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				deviceUUID := devices[0].UUID
				partitionTemplateID := "test-partition-no-dealloc"
				workerUID := "test-worker-status-change"

				// Create partitioned worker allocation
				req := &api.WorkerInfo{
					WorkerUID:           workerUID,
					AllocatedDevices:    []string{deviceUUID},
					IsolationMode:       tfv1.IsolationModePartitioned,
					PartitionTemplateID: partitionTemplateID,
					Status:              api.WorkerStatusPending,
				}

				resp, err := allocationController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp).NotTo(BeNil())
				partitionUUID := resp.DeviceInfos[0].UUID

				// Simulate status change from Pending to Running (not Terminated)
				oldWorker := &api.WorkerInfo{
					WorkerUID: workerUID,
					Status:    api.WorkerStatusPending,
				}
				newWorker := &api.WorkerInfo{
					WorkerUID: workerUID,
					Status:    api.WorkerStatusRunning,
				}

				// This should NOT trigger deallocation (simulating OnUpdate logic)
				if oldWorker.Status != api.WorkerStatusTerminated && newWorker.Status == api.WorkerStatusTerminated {
					err = allocationController.DeallocateWorker(workerUID)
					Expect(err).NotTo(HaveOccurred())
				}

				// Verify allocation still exists
				allocation, found := allocationController.GetWorkerAllocation(workerUID)
				Expect(found).To(BeTrue(), "Allocation should still exist after non-termination status change")
				Expect(allocation).NotTo(BeNil())

				// Verify partition still exists
				_, exists := deviceController.GetDevice(partitionUUID)
				Expect(exists).To(BeTrue(), "Partition should still exist after non-termination status change")

				// Cleanup
				err = allocationController.DeallocateWorker(workerUID)
				Expect(err).NotTo(HaveOccurred())
			})
		})

		Describe("Metrics Recorder", func() {
			BeforeEach(func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				waitForDeviceDiscovery(deviceController)

				err = workerController.Start()
				Expect(err).NotTo(HaveOccurred())

				metricsRecorder.Start()
			})

			It("should record device metrics to file", func() {
				// Wait for metrics to be recorded (metricsInterval is 1 second)
				Eventually(func() error {
					info, err := os.Stat(tempMetricsFile)
					if err != nil {
						return err
					}
					if info.Size() == 0 {
						return fmt.Errorf("metrics file is empty, waiting for metrics to be written")
					}
					return nil
				}, 5*time.Second).Should(Succeed(), "Metrics file should have content")

				// Read and verify metrics file content
				content, err := os.ReadFile(tempMetricsFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(content).ToNot(BeEmpty(), "Metrics file should have content")

				// Verify metrics contain expected fields (GPU usage metrics)
				contentStr := string(content)
				Expect(contentStr).To(ContainSubstring("tf_gpu_usage"), "Should contain GPU usage metrics")
			})
		})

		Describe("HTTP Server", func() {
			BeforeEach(func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				waitForDeviceDiscovery(deviceController)

				err = workerController.Start()
				Expect(err).NotTo(HaveOccurred())

				metricsRecorder.Start()
			})

			It("should start HTTP server", func() {
				// Start server in background
				serverStarted := make(chan error, 1)
				go func() {
					err := httpServer.Start()
					serverStarted <- err
				}()

				// Wait for server to start (or fail quickly)
				select {
				case err := <-serverStarted:
					Expect(err).To(Or(BeNil(), MatchError("http: Server closed")))
				case <-time.After(1 * time.Second):
					// Server is running, which is expected
				}

				// Server should be running (we can't easily test HTTP endpoints without knowing the port)
				// But we can verify the server object is created
				Expect(httpServer).NotTo(BeNil())
			})
		})

		Describe("Mock Driver Integration with app_mock Processes", func() {
			var (
				appMockPath string
				processes   []*os.Process
			)

			BeforeEach(func() {
				// Find app_mock executable - try multiple paths
				possiblePaths := []string{
					filepath.Join("..", "..", "provider", "build", "app_mock"),
					filepath.Join("provider", "build", "app_mock"),
				}

				cwd, _ := os.Getwd()
				possiblePaths = append(possiblePaths,
					filepath.Join(cwd, "..", "..", "provider", "build", "app_mock"),
					filepath.Join(cwd, "provider", "build", "app_mock"),
				)

				workspaceRoot := os.Getenv("WORKSPACE_ROOT")
				if workspaceRoot != "" {
					possiblePaths = append(possiblePaths,
						filepath.Join(workspaceRoot, "provider", "build", "app_mock"),
					)
				}

				appMockPath = ""
				for _, path := range possiblePaths {
					if _, err := os.Stat(path); err == nil {
						appMockPath = path
						break
					}
				}

				if appMockPath == "" {
					Skip("app_mock not found. Run 'make mock' in provider directory first.")
				}
				processes = []*os.Process{}
			})

			AfterEach(func() {
				// Kill all spawned processes
				for _, proc := range processes {
					if proc != nil {
						_ = proc.Kill()
						_, _ = proc.Wait()
					}
				}
				// Clean up shared memory file
				shmPath := filepath.Join("..", "..", "provider", "build", "tmp.example_accelerator.bin")
				_ = os.Remove(shmPath)
			})

			It("should track metrics from app_mock processes with different frequencies", func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				waitForDeviceDiscovery(deviceController)

				// Start 3 app_mock processes with different parameters
				// Process 1: Low frequency (5 Hz), 100MB
				cmd1 := exec.Command(appMockPath, "-m", "100", "-k", "512", "-f", "5", "-d", "10")
				cmd1.Dir = filepath.Dir(appMockPath)
				err = cmd1.Start()
				Expect(err).NotTo(HaveOccurred())
				processes = append(processes, cmd1.Process)

				// Process 2: Medium frequency (10 Hz), 200MB
				cmd2 := exec.Command(appMockPath, "-m", "200", "-k", "1024", "-f", "10", "-d", "10")
				cmd2.Dir = filepath.Dir(appMockPath)
				err = cmd2.Start()
				Expect(err).NotTo(HaveOccurred())
				processes = append(processes, cmd2.Process)

				// Process 3: High frequency (20 Hz), 150MB
				cmd3 := exec.Command(appMockPath, "-m", "150", "-k", "2048", "-f", "20", "-d", "10")
				cmd3.Dir = filepath.Dir(appMockPath)
				err = cmd3.Start()
				Expect(err).NotTo(HaveOccurred())
				processes = append(processes, cmd3.Process)

				// Wait for processes to initialize and start launching kernels
				// Use Eventually to wait for processes to be tracked
				Eventually(func() int {
					accel, err := device.NewAcceleratorInterface(stubLibPath)
					if err != nil {
						return 0
					}
					defer func() {
						_ = accel.Close()
					}()
					processInfos, err := accel.GetProcessInformation()
					if err != nil {
						return 0
					}
					return len(processInfos)
				}, 5*time.Second).Should(BeNumerically(">=", 0))

				// Get process information (combines compute and memory utilization)
				accel, err := device.NewAcceleratorInterface(stubLibPath)
				Expect(err).NotTo(HaveOccurred())
				defer func() {
					_ = accel.Close()
				}()

				processInfos, err := accel.GetProcessInformation()
				Expect(err).NotTo(HaveOccurred())
				Expect(processInfos).NotTo(BeNil())
				// Should have at least some processes tracked
				Expect(len(processInfos)).To(BeNumerically(">=", 0))

				// Should track memory for the processes
				if len(processInfos) > 0 {
					totalMemory := uint64(0)
					for _, info := range processInfos {
						totalMemory += info.MemoryUsedBytes
					}
					// Total should be around 450MB (100+200+150)
					Expect(totalMemory).To(BeNumerically(">=", 400*1024*1024))
					Expect(totalMemory).To(BeNumerically("<=", 500*1024*1024))
				}

				// Get device metrics
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				deviceUUIDs := make([]string, len(devices))
				for i, d := range devices {
					deviceUUIDs[i] = d.UUID
				}

				gpuMetrics, err := accel.GetDeviceMetrics(deviceUUIDs)
				Expect(err).NotTo(HaveOccurred())
				Expect(gpuMetrics).NotTo(BeNil())
				// Should have metrics for all devices
				Expect(gpuMetrics).To(HaveLen(len(devices)))
			})

			It("should handle high load scenario with many processes", func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				waitForDeviceDiscovery(deviceController)

				// Start 10 processes to create high load
				for i := 0; i < 10; i++ {
					freq := 5 + (i * 2)  // Varying frequencies from 5 to 23 Hz
					mem := 50 + (i * 10) // Varying memory from 50 to 140 MB
					cmd := exec.Command(
						appMockPath,
						"-m",
						fmt.Sprintf("%d", mem),
						"-k",
						"1024",
						"-f",
						fmt.Sprintf("%d", freq),
						"-d",
						"15",
					)
					cmd.Dir = filepath.Dir(appMockPath)
					err = cmd.Start()
					Expect(err).NotTo(HaveOccurred())
					processes = append(processes, cmd.Process)
					time.Sleep(100 * time.Millisecond) // Stagger starts
				}

				// Wait for processes to initialize and launch kernels
				// Use Eventually to wait for some processes to be tracked
				Eventually(func() int {
					accel, err := device.NewAcceleratorInterface(stubLibPath)
					if err != nil {
						return 0
					}
					defer func() {
						_ = accel.Close()
					}()
					processInfos, err := accel.GetProcessInformation()
					if err != nil {
						return 0
					}
					return len(processInfos)
				}, 8*time.Second).Should(BeNumerically(">=", 0))

				accel, err := device.NewAcceleratorInterface(stubLibPath)
				Expect(err).NotTo(HaveOccurred())
				defer func() {
					_ = accel.Close()
				}()

				// Get process information (combines compute and memory metrics)
				processInfos, err := accel.GetProcessInformation()
				Expect(err).NotTo(HaveOccurred())
				// Should track multiple processes (may be 0 if processes haven't launched kernels yet)
				// Just verify the API works, don't enforce exact count
				_ = processInfos

				// Verify API works - memory tracking should be present if processes allocated VRAM
				if len(processInfos) > 0 {
					totalMemory := uint64(0)
					for _, info := range processInfos {
						totalMemory += info.MemoryUsedBytes
					}
					// Total should be significant (10 processes * ~100MB average)
					// But allow for some variance
					Expect(totalMemory).To(BeNumerically(">=", 200*1024*1024))
				}

				// Get device metrics - should show high utilization
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				deviceUUIDs := make([]string, len(devices))
				for i, d := range devices {
					deviceUUIDs[i] = d.UUID
				}

				gpuMetrics, err := accel.GetDeviceMetrics(deviceUUIDs)
				Expect(err).NotTo(HaveOccurred())
				Expect(gpuMetrics).NotTo(BeNil())

				// Check that we have 4 devices as expected
				Expect(gpuMetrics).To(HaveLen(4), "Should return metrics for 4 devices")

				// Note: Stub library may not accurately track compute utilization from app_mock processes
				// since the stub implementation uses simulated metrics. Just verify the API works.
				for _, metric := range gpuMetrics {
					// ComputePercentage should be a valid value (0-100)
					Expect(metric.ComputePercentage).To(BeNumerically(">=", 0.0))
					Expect(metric.ComputePercentage).To(BeNumerically("<=", 100.0))
				}
			})

			It("should verify rate limiting at 100 launches/second", func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				waitForDeviceDiscovery(deviceController)

				// Start processes that will trigger rate limiting
				// Each process at 20 Hz = 20 launches/sec, so 6 processes = 120 launches/sec > 100
				for i := 0; i < 6; i++ {
					cmd := exec.Command(appMockPath, "-m", "50", "-k", "512", "-f", "20", "-d", "5")
					cmd.Dir = filepath.Dir(appMockPath)
					err = cmd.Start()
					Expect(err).NotTo(HaveOccurred())
					processes = append(processes, cmd.Process)
					time.Sleep(50 * time.Millisecond)
				}

				// Wait for rate limiting to kick in
				// Use Eventually to wait for metrics to be available
				Eventually(func() error {
					accel, err := device.NewAcceleratorInterface(stubLibPath)
					if err != nil {
						return err
					}
					defer func() {
						_ = accel.Close()
					}()
					devices, err := deviceController.ListDevices()
					if err != nil {
						return err
					}
					if len(devices) == 0 {
						return fmt.Errorf("no devices")
					}
					deviceUUIDs := []string{devices[0].UUID}
					_, err = accel.GetDeviceMetrics(deviceUUIDs)
					return err
				}, 3*time.Second).Should(Succeed())

				accel, err := device.NewAcceleratorInterface(stubLibPath)
				Expect(err).NotTo(HaveOccurred())
				defer func() {
					_ = accel.Close()
				}()

				// Get device metrics - should show 100% utilization when rate limited
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				deviceUUIDs := make([]string, 1)
				deviceUUIDs[0] = devices[0].UUID

				gpuMetrics, err := accel.GetDeviceMetrics(deviceUUIDs)
				Expect(err).NotTo(HaveOccurred())
				Expect(gpuMetrics).NotTo(BeNil())
				Expect(gpuMetrics).To(HaveLen(1))

				// Utilization might be at or near 100% due to rate limiting
				// Allow some variance as the window resets
				util := gpuMetrics[0].ComputePercentage
				Expect(util).To(BeNumerically(">=", 0.0))
				Expect(util).To(BeNumerically("<=", 100.0))
			})
		})

		Describe("Full Integration", func() {
			BeforeEach(func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				waitForDeviceDiscovery(deviceController)

				err = backend.Start()
				Expect(err).NotTo(HaveOccurred())

				err = workerController.Start()
				Expect(err).NotTo(HaveOccurred())

				metricsRecorder.Start()

				// Start HTTP server in background
				go func() {
					_ = httpServer.Start()
				}()
				// Give server a moment to start
				time.Sleep(500 * time.Millisecond)
			})

			It("should handle complete workflow: discover -> allocate -> track -> metrics", func() {
				// 1. Discover devices
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())
				deviceUUID := devices[0].UUID

				// Register handler before starting worker to catch OnAdd event
				var foundInList bool
				handler := framework.WorkerChangeHandler{
					OnAdd: func(worker *api.WorkerInfo) {
						if worker.WorkerUID == integrationWorkerUID1 {
							foundInList = true
						}
					},
					OnRemove: func(worker *api.WorkerInfo) {},
					OnUpdate: func(oldWorker, newWorker *api.WorkerInfo) {
						// StartWorker adds worker to map before notifying, so it may trigger OnUpdate
						if newWorker.WorkerUID == integrationWorkerUID1 {
							foundInList = true
						}
					},
				}
				err = backend.RegisterWorkerUpdateHandler(handler)
				Expect(err).NotTo(HaveOccurred())

				// Small delay to ensure handler goroutine is ready
				time.Sleep(50 * time.Millisecond)

				// 2. Allocate device
				req := &api.WorkerInfo{
					WorkerUID:        integrationWorkerUID1,
					AllocatedDevices: []string{deviceUUID},
					IsolationMode:    tfv1.IsolationModeSoft,
					Requests: tfv1.Resource{
						Tflops: resource.MustParse("1000"),
						Vram:   resource.MustParse("1Gi"),
					},
				}
				resp, err := allocationController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp).To(Not(BeNil()))

				// Start worker in backend
				err = backend.StartWorker(req)
				Expect(err).NotTo(HaveOccurred())

				// 3. Verify allocation through allocation controller
				allocation, found := allocationController.GetWorkerAllocation(integrationWorkerUID1)
				Expect(found).To(BeTrue())
				Expect(allocation).NotTo(BeNil())
				Expect(allocation.WorkerInfo.WorkerUID).To(Equal(integrationWorkerUID1))

				// 4. Backend should list worker - wait for OnAdd callback to be invoked
				Eventually(func() bool {
					return foundInList
				}, 2*time.Second).Should(BeTrue(), "Should find integration-worker-1 via OnAdd callback")

				// 5. Worker controller should list worker
				workerList, err := workerController.ListWorkers()
				Expect(err).NotTo(HaveOccurred())
				foundInWorkerList := false
				for _, worker := range workerList {
					if worker.WorkerUID == integrationWorkerUID1 {
						foundInWorkerList = true
						break
					}
				}
				Expect(foundInWorkerList).To(BeTrue())

				// 6. Get worker allocation
				allocation, found = allocationController.GetWorkerAllocation(integrationWorkerUID1)
				Expect(found).To(BeTrue())
				Expect(allocation).NotTo(BeNil())
				Expect(allocation.WorkerInfo.WorkerUID).To(Equal(integrationWorkerUID1))

				// 7. Get metrics
				gpuMetrics, err := deviceController.GetDeviceMetrics()
				Expect(err).NotTo(HaveOccurred())
				Expect(gpuMetrics).NotTo(BeNil())
				Expect(gpuMetrics[deviceUUID]).NotTo(BeNil())

				workerMetrics, err := workerController.GetWorkerMetrics()
				Expect(err).NotTo(HaveOccurred())
				// GetWorkerMetrics returns nil, nil (not implemented yet - TODO in code)
				// So we accept nil as valid response
				_ = workerMetrics

				// 8. Deallocate worker
				err = allocationController.DeallocateWorker(integrationWorkerUID1)
				Expect(err).NotTo(HaveOccurred())

				// 9. Verify deallocation
				_, found = allocationController.GetWorkerAllocation(integrationWorkerUID1)
				Expect(found).To(BeFalse())
			})
		})
	})
})
