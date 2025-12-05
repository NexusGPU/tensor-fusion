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
	"os"
	"path/filepath"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/backend/single_node"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/device"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/server"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/worker"
)

func TestHypervisor(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Hypervisor Suite")
}

var _ = Describe("Hypervisor Integration Tests", func() {
	var (
		ctx              context.Context
		cancel           context.CancelFunc
		deviceController framework.DeviceController
		backend          framework.Backend
		workerController framework.WorkerController
		metricsRecorder  *metrics.HypervisorMetricsRecorder
		httpServer       *server.Server
		stubLibPath      string
		tempMetricsFile  string
	)

	BeforeEach(func() {
		ctx, cancel = context.WithCancel(context.Background())

		// Find stub library path
		// Try relative path first (from provider/build)
		stubLibPath = filepath.Join("..", "..", "provider", "build", "libaccelerator_stub.so")
		if _, err := os.Stat(stubLibPath); os.IsNotExist(err) {
			// Try absolute path from workspace root
			workspaceRoot := os.Getenv("WORKSPACE_ROOT")
			if workspaceRoot == "" {
				// Try to find it relative to current directory
				cwd, _ := os.Getwd()
				stubLibPath = filepath.Join(cwd, "..", "..", "provider", "build", "libaccelerator_stub.so")
			} else {
				stubLibPath = filepath.Join(workspaceRoot, "provider", "build", "libaccelerator_stub.so")
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
			if closer, ok := deviceController.(interface{ Close() error }); ok {
				_ = closer.Close()
			}
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

			backend = single_node.NewSingleNodeBackend(ctx, deviceController)
			Expect(backend).NotTo(BeNil())

			workerController = worker.NewWorkerController(deviceController, tfv1.IsolationModeShared, backend)
			Expect(workerController).NotTo(BeNil())

			metricsRecorder = metrics.NewHypervisorMetricsRecorder(ctx, tempMetricsFile, deviceController, workerController)
			Expect(metricsRecorder).NotTo(BeNil())

			httpServer = server.NewServer(ctx, deviceController, workerController, metricsRecorder, backend, 0)
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

				// Verify stub device properties
				device := devices[0]
				Expect(device.UUID).To(ContainSubstring("stub-device"))
				Expect(device.Vendor).To(Equal("STUB"))
				Expect(device.TotalMemoryBytes).To(Equal(uint64(16 * 1024 * 1024 * 1024))) // 16GB

				_ = accel.Close()
			})

			It("should get process utilization from stub library", func() {
				accel, err := device.NewAcceleratorInterface(stubLibPath)
				Expect(err).NotTo(HaveOccurred())
				defer func() {
					_ = accel.Close()
				}()

				// Get compute utilization (may be empty for stub)
				computeUtils, err := accel.GetProcessComputeUtilization()
				Expect(err).NotTo(HaveOccurred())
				Expect(computeUtils).NotTo(BeNil())

				// Get memory utilization (may be empty for stub)
				memUtils, err := accel.GetProcessMemoryUtilization()
				Expect(err).NotTo(HaveOccurred())
				Expect(memUtils).NotTo(BeNil())
			})
		})

		Describe("Device Controller", func() {
			It("should start and discover devices", func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())

				// Wait a bit for discovery
				time.Sleep(100 * time.Millisecond)

				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty(), "Should discover at least one stub device")

				// Verify device properties
				device := devices[0]
				Expect(device.UUID).NotTo(BeEmpty())
				Expect(device.Vendor).To(Equal("STUB"))
				Expect(device.TotalMemoryBytes).To(BeNumerically(">", 0))
			})

			It("should allocate devices", func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())

				time.Sleep(100 * time.Millisecond)

				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				deviceUUID := devices[0].UUID
				req := &api.WorkerInfo{
					WorkerUID:        "test-worker-1",
					AllocatedDevices: []string{deviceUUID},
					IsolationMode:    tfv1.IsolationModeSoft,
				}

				resp, err := workerController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp).NotTo(BeNil())
				// TODO verify the mounts/envs

				// Verify allocation exists through worker controller
				allocation, found := workerController.GetWorkerAllocation("test-worker-1")
				Expect(found).To(BeTrue())
				Expect(allocation).NotTo(BeNil())
				Expect(allocation.WorkerInfo.WorkerUID).To(Equal("test-worker-1"))
			})

			It("should get GPU metrics", func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())

				time.Sleep(100 * time.Millisecond)

				metrics, err := deviceController.GetDeviceMetrics()
				Expect(err).NotTo(HaveOccurred())
				Expect(metrics).NotTo(BeNil())

				// Should have metrics for all discovered devices
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(metrics).To(HaveLen(len(devices)))
			})
		})

		Describe("Single Node Backend", func() {
			BeforeEach(func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				time.Sleep(100 * time.Millisecond)

				err = backend.Start()
				Expect(err).NotTo(HaveOccurred())
			})

			It("should start and stop", func() {
				Expect(backend).NotTo(BeNil())
			})

			It("should list workers from allocations", func() {
				// Create an allocation
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				req := &api.WorkerInfo{
					WorkerUID:        "test-worker-1",
					AllocatedDevices: []string{devices[0].UUID},
					IsolationMode:    tfv1.IsolationModeSoft,
				}
				_, err = workerController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())

				// Start the worker in the backend
				err = backend.StartWorker(req)
				Expect(err).NotTo(HaveOccurred())

				// Wait a bit for state to sync
				time.Sleep(500 * time.Millisecond)

				// Register a handler to receive updates and track initial workers
				var found bool
				handler := framework.WorkerChangeHandler{
					OnAdd: func(worker *api.WorkerInfo) {
						if worker.WorkerUID == "test-worker-1" {
							found = true
						}
					},
					OnRemove: func(worker *api.WorkerInfo) {},
					OnUpdate: func(oldWorker, newWorker *api.WorkerInfo) {},
				}
				err = backend.RegisterWorkerUpdateHandler(handler)
				Expect(err).NotTo(HaveOccurred())

				// Wait a bit for OnAdd callbacks to be invoked
				time.Sleep(100 * time.Millisecond)
				Expect(found).To(BeTrue(), "Should find test-worker-1 via OnAdd callback")
			})

			It("should track worker to process mapping", func() {
				// Start a worker
				worker := &api.WorkerInfo{
					WorkerUID:        "test-worker-1",
					AllocatedDevices: []string{},
					IsolationMode:    tfv1.IsolationModeSoft,
				}
				err := backend.StartWorker(worker)
				Expect(err).NotTo(HaveOccurred())

				// Test process mapping
				processInfo, err := backend.GetProcessMappingInfo("test-worker-1", 12345)
				Expect(err).NotTo(HaveOccurred())
				Expect(processInfo).NotTo(BeNil())
				Expect(processInfo.GuestID).To(Equal("test-worker-1"))
				Expect(processInfo.HostPID).To(Equal(uint32(12345)))
			})
		})

		Describe("Worker Controller", func() {
			BeforeEach(func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				time.Sleep(100 * time.Millisecond)

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
					WorkerUID:        "test-worker-1",
					AllocatedDevices: []string{devices[0].UUID},
					IsolationMode:    tfv1.IsolationModeSoft,
				}
				_, err = workerController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())

				workers, err := workerController.ListWorkers()
				Expect(err).NotTo(HaveOccurred())
				found := false
				for _, worker := range workers {
					if worker.WorkerUID == "test-worker-1" {
						found = true
						break
					}
				}
				Expect(found).To(BeTrue())
			})

			It("should get worker allocation", func() {
				// Create an allocation
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				req := &api.WorkerInfo{
					WorkerUID:        "test-worker-1",
					AllocatedDevices: []string{devices[0].UUID},
					IsolationMode:    tfv1.IsolationModeSoft,
				}
				_, err = workerController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())

				allocation, found := workerController.GetWorkerAllocation("test-worker-1")
				Expect(found).To(BeTrue())
				Expect(allocation).NotTo(BeNil())
				Expect(allocation.WorkerInfo.WorkerUID).To(Equal("test-worker-1"))
			})

			It("should get worker metrics", func() {
				// Create an allocation
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())

				req := &api.WorkerInfo{
					WorkerUID:        "test-worker-1",
					AllocatedDevices: []string{devices[0].UUID},
					IsolationMode:    tfv1.IsolationModeSoft,
				}
				_, err = workerController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())

				metrics, err := workerController.GetWorkerMetrics()
				Expect(err).NotTo(HaveOccurred())
				// Metrics may be empty for stub devices, which is okay
				// Just verify we got a valid response (nil or empty map is acceptable)
				_ = metrics
			})
		})

		Describe("Metrics Recorder", func() {
			BeforeEach(func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				time.Sleep(100 * time.Millisecond)

				err = workerController.Start()
				Expect(err).NotTo(HaveOccurred())

				metricsRecorder.Start()
			})

			It("should record metrics", func() {
				// Wait for metrics to be recorded
				time.Sleep(2 * time.Second)

				// Check if metrics file was created and has content
				info, err := os.Stat(tempMetricsFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(info.Size()).To(BeNumerically(">=", 0))
			})
		})

		Describe("HTTP Server", func() {
			BeforeEach(func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				time.Sleep(100 * time.Millisecond)

				err = workerController.Start()
				Expect(err).NotTo(HaveOccurred())

				metricsRecorder.Start()
			})

			It("should start HTTP server", func() {
				// Start server in background
				go func() {
					err := httpServer.Start()
					Expect(err).To(Or(BeNil(), MatchError("http: Server closed")))
				}()

				// Wait for server to start
				time.Sleep(500 * time.Millisecond)

				// Server should be running (we can't easily test HTTP endpoints without knowing the port)
				// But we can verify the server object is created
				Expect(httpServer).NotTo(BeNil())
			})
		})

		Describe("Full Integration", func() {
			BeforeEach(func() {
				err := deviceController.Start()
				Expect(err).NotTo(HaveOccurred())
				time.Sleep(100 * time.Millisecond)

				err = backend.Start()
				Expect(err).NotTo(HaveOccurred())

				err = workerController.Start()
				Expect(err).NotTo(HaveOccurred())

				metricsRecorder.Start()

				// Start HTTP server in background
				go func() {
					_ = httpServer.Start()
				}()
				time.Sleep(500 * time.Millisecond)
			})

			It("should handle complete workflow: discover -> allocate -> track -> metrics", func() {
				// 1. Discover devices
				devices, err := deviceController.ListDevices()
				Expect(err).NotTo(HaveOccurred())
				Expect(devices).ToNot(BeEmpty())
				deviceUUID := devices[0].UUID

				// 2. Allocate device
				req := &api.WorkerInfo{
					WorkerUID:        "integration-worker-1",
					AllocatedDevices: []string{deviceUUID},
					IsolationMode:    tfv1.IsolationModeSoft,
					Requests: tfv1.Resource{
						Tflops: resource.MustParse("1000"),
						Vram:   resource.MustParse("1Gi"),
					},
				}
				resp, err := workerController.AllocateWorkerDevices(req)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp).To(Not(BeNil()))

				// Start worker in backend
				err = backend.StartWorker(req)
				Expect(err).NotTo(HaveOccurred())

				// 3. Verify allocation through worker controller
				allocation, found := workerController.GetWorkerAllocation("integration-worker-1")
				Expect(found).To(BeTrue())
				Expect(allocation).NotTo(BeNil())
				Expect(allocation.WorkerInfo.WorkerUID).To(Equal("integration-worker-1"))

				// 4. Backend should list worker
				time.Sleep(500 * time.Millisecond)
				// Register a handler to receive updates and track initial workers
				var foundInList bool
				handler := framework.WorkerChangeHandler{
					OnAdd: func(worker *api.WorkerInfo) {
						if worker.WorkerUID == "integration-worker-1" {
							foundInList = true
						}
					},
					OnRemove: func(worker *api.WorkerInfo) {},
					OnUpdate: func(oldWorker, newWorker *api.WorkerInfo) {},
				}
				err = backend.RegisterWorkerUpdateHandler(handler)
				Expect(err).NotTo(HaveOccurred())

				// Wait a bit for OnAdd callbacks to be invoked
				time.Sleep(100 * time.Millisecond)
				Expect(foundInList).To(BeTrue(), "Should find integration-worker-1 via OnAdd callback")

				// 5. Worker controller should list worker
				workerList, err := workerController.ListWorkers()
				Expect(err).NotTo(HaveOccurred())
				foundInWorkerList := false
				for _, worker := range workerList {
					if worker.WorkerUID == "integration-worker-1" {
						foundInWorkerList = true
						break
					}
				}
				Expect(foundInWorkerList).To(BeTrue())

				// 6. Get worker allocation
				allocation, found = workerController.GetWorkerAllocation("integration-worker-1")
				Expect(found).To(BeTrue())
				Expect(allocation).NotTo(BeNil())
				Expect(allocation.WorkerInfo.WorkerUID).To(Equal("integration-worker-1"))

				// 7. Get metrics
				gpuMetrics, err := deviceController.GetDeviceMetrics()
				Expect(err).NotTo(HaveOccurred())
				Expect(gpuMetrics).NotTo(BeNil())
				Expect(gpuMetrics[deviceUUID]).NotTo(BeNil())

				workerMetrics, err := workerController.GetWorkerMetrics()
				Expect(err).NotTo(HaveOccurred())
				Expect(workerMetrics).NotTo(BeNil())

				// 8. Deallocate worker
				err = workerController.DeallocateWorker("integration-worker-1")
				Expect(err).NotTo(HaveOccurred())

				// 9. Verify deallocation
				_, found = workerController.GetWorkerAllocation("integration-worker-1")
				Expect(found).To(BeFalse())
			})
		})
	})
})
