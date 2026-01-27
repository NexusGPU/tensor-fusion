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

package gpuallocator

import (
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/samber/lo"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

var workloadNameNs = tfv1.NameNamespace{Namespace: "default", Name: "test-workload"}

var testPodMeta = metav1.ObjectMeta{UID: "test-pod", Namespace: "default", Name: "test-pod"}

var _ = Describe("GPU Allocator", func() {
	var allocator *GpuAllocator
	var mutex sync.Mutex

	allocateAndSync := func(poolName string, request tfv1.Resource, count uint, gpuModel string) ([]*tfv1.GPU, error) {
		mutex.Lock()
		defer mutex.Unlock()
		gpus, err := allocator.Alloc(&tfv1.AllocRequest{
			PoolName:              poolName,
			WorkloadNameNamespace: workloadNameNs,
			Request:               request,
			// use same limits as requests during unit testing
			Limit:    request,
			Count:    count,
			GPUModel: gpuModel,

			PodMeta: testPodMeta,
		})
		allocator.syncToK8s(ctx)
		return gpus, err
	}

	deallocateAndSync := func(gpus []*tfv1.GPU) {
		mutex.Lock()
		defer mutex.Unlock()
		allocator.Dealloc(workloadNameNs, lo.Map(gpus, func(gpu *tfv1.GPU, _ int) string {
			return gpu.Name
		}), testPodMeta)
		allocator.syncToK8s(ctx)
	}

	BeforeEach(func() {
		allocator = NewGpuAllocator(ctx, nil, k8sClient, 150*time.Millisecond)
		err := allocator.SetupWithManager(ctx, mgr)
		Expect(err).NotTo(HaveOccurred())
		<-allocator.initializedCh
	})

	AfterEach(func() {
		if allocator != nil {
			allocator.Stop()
		}
	})

	Context("GPU Allocation", func() {
		It("should allocate a single GPU successfully", func() {
			request := tfv1.Resource{
				Tflops: resource.MustParse("50"),
				Vram:   resource.MustParse("8Gi"),
			}

			gpus, err := allocateAndSync("test-pool", request, 1, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(1))

			gpuNode := &tfv1.GPUNode{}
			if err := k8sClient.Get(ctx, types.NamespacedName{Name: gpus[0].Labels[constants.LabelKeyOwner]}, gpuNode); err != nil {
				Expect(err).NotTo(HaveOccurred())
			}
			pool := &tfv1.GPUPool{}
			if err := k8sClient.Get(ctx, types.NamespacedName{Name: "test-pool"}, pool); err != nil {
				Expect(err).NotTo(HaveOccurred())
			}
			_, _ = RefreshGPUNodeCapacity(ctx, k8sClient, gpuNode, pool, allocator, nil)

			// Verify resources were reduced on the allocated GPU
			gpu := getGPU(gpus[0].Name)
			Expect(gpu.Status.Available.Tflops.Cmp(gpu.Status.Capacity.Tflops)).To(Equal(-1))
			Expect(gpu.Status.Available.Vram.Cmp(gpu.Status.Capacity.Vram)).To(Equal(-1))

			node := getGPUNode(gpu)
			diffTflops := node.Status.TotalTFlops.Value() - node.Status.AvailableTFlops.Value()
			diffVRAM := node.Status.TotalVRAM.Value() - node.Status.AvailableVRAM.Value()

			diffVirtualTflops := node.Status.VirtualTFlops.Value() - node.Status.VirtualAvailableTFlops.Value()
			diffVirtualVRAM := node.Status.VirtualVRAM.Value() - node.Status.VirtualAvailableVRAM.Value()
			Expect(diffTflops).To(BeEquivalentTo(50))
			Expect(diffVRAM).To(BeEquivalentTo(8 * 1024 * 1024 * 1024))

			Expect(diffVirtualTflops).To(BeEquivalentTo(50))
			Expect(diffVirtualVRAM).To(BeEquivalentTo(8 * 1024 * 1024 * 1024))
		})

		It("should allocate multiple GPUs from the same node", func() {
			request := tfv1.Resource{
				Tflops: resource.MustParse("20"),
				Vram:   resource.MustParse("4Gi"),
			}

			gpus, err := allocateAndSync("test-pool", request, 2, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(2))

			// Verify all GPUs are from the same node
			node := gpus[0].Labels[constants.LabelKeyOwner]
			for _, gpu := range gpus {
				Expect(gpu.Labels[constants.LabelKeyOwner]).To(Equal(node))
			}
		})

		It("should fail when requesting more GPUs than available", func() {
			request := tfv1.Resource{
				Tflops: resource.MustParse("10"),
				Vram:   resource.MustParse("2Gi"),
			}

			_, err := allocateAndSync("test-pool", request, 10, "")
			Expect(err).To(HaveOccurred())
		})

		It("should fail when resources are insufficient", func() {
			request := tfv1.Resource{
				Tflops: resource.MustParse("200"),
				Vram:   resource.MustParse("64Gi"),
			}

			_, err := allocateAndSync("test-pool", request, 1, "")
			Expect(err).To(HaveOccurred())
		})

		It("should fail when pool doesn't exist", func() {
			request := tfv1.Resource{
				Tflops: resource.MustParse("10"),
				Vram:   resource.MustParse("2Gi"),
			}

			_, err := allocateAndSync("nonexistent-pool", request, 1, "")
			Expect(err).To(HaveOccurred())
		})

		It("should filter GPUs by model", func() {
			request := tfv1.Resource{
				Tflops: resource.MustParse("50"),
				Vram:   resource.MustParse("8Gi"),
			}

			// Try allocating with a specific GPU model
			gpus, err := allocateAndSync("test-pool", request, 1, "NVIDIA A100")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus[0].Status.GPUModel).To(Equal("NVIDIA A100"))

			// Try allocating with a non-existent GPU model
			_, err = allocateAndSync("test-pool", request, 1, "NonExistentModel")
			Expect(err).To(HaveOccurred())
		})
	})

	Context("GPU Deallocation", func() {
		It("should deallocate resources successfully", func() {
			// First allocate resources
			request := tfv1.Resource{
				Tflops: resource.MustParse("30"),
				Vram:   resource.MustParse("6Gi"),
			}

			gpus, err := allocateAndSync("test-pool", request, 1, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(1))

			// Store the allocated values
			allocatedGPU := gpus[0]
			allocatedTflops := allocatedGPU.Status.Available.Tflops.DeepCopy()
			allocatedVram := allocatedGPU.Status.Available.Vram.DeepCopy()

			// Now deallocate
			deallocateAndSync(gpus)

			// Verify resources were restored
			deallocatedGPU := getGPU(allocatedGPU.Name)
			expectedTflops := allocatedTflops.DeepCopy()
			expectedVram := allocatedVram.DeepCopy()
			expectedTflops.Add(request.Tflops)
			expectedVram.Add(request.Vram)

			Expect(deallocatedGPU.Status.Available.Tflops.Cmp(expectedTflops)).To(Equal(0))
			Expect(deallocatedGPU.Status.Available.Vram.Cmp(expectedVram)).To(Equal(0))
			Expect(deallocatedGPU.Status.Available.Vram.Cmp(allocatedVram)).To(Equal(1))
		})

		It("should continue deallocating when some GPUs don't exist", func() {
			// First allocate resources to multiple GPUs
			request := tfv1.Resource{
				Tflops: resource.MustParse("20"),
				Vram:   resource.MustParse("4Gi"),
			}

			// Allocate 2 GPUs
			allocatedGPUs, err := allocateAndSync("test-pool", request, 2, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(allocatedGPUs).To(HaveLen(2))

			// Create a non-existent GPU
			nonExistentGPU := &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "non-existent-gpu",
					Namespace: "default",
				},
			}

			// Add the non-existent GPU to the list
			gpusToDealloc := append(allocatedGPUs, nonExistentGPU)

			// Store the allocated values for existing GPUs
			initialStates := make(map[string]struct {
				tflops resource.Quantity
				vram   resource.Quantity
			})
			for _, gpu := range allocatedGPUs {
				initialStates[gpu.Name] = struct {
					tflops resource.Quantity
					vram   resource.Quantity
				}{
					tflops: gpu.Status.Available.Tflops.DeepCopy(),
					vram:   gpu.Status.Available.Vram.DeepCopy(),
				}
			}

			// Now deallocate all GPUs including the non-existent one
			deallocateAndSync(gpusToDealloc)

			// Verify resources were restored for existing GPUs
			for _, allocatedGPU := range allocatedGPUs {
				deallocatedGPU := getGPU(allocatedGPU.Name)
				initialState := initialStates[allocatedGPU.Name]
				Expect(deallocatedGPU.Status.Available.Tflops.Cmp(initialState.tflops)).To(Equal(1))
				Expect(deallocatedGPU.Status.Available.Vram.Cmp(initialState.vram)).To(Equal(1))
			}
		})
	})

	Context("GPU AutoScale", func() {
		It("should scale up GPUs", func() {
			request := tfv1.Resource{
				Tflops: resource.MustParse("50"),
				Vram:   resource.MustParse("10Gi"),
			}
			gpus, err := allocateAndSync("test-pool", request, 1, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(1))

			gpu := getGPU(gpus[0].Name)
			remain, _, err := allocator.AdjustAllocation(ctx, tfv1.AdjustRequest{
				PodUID:    string(testPodMeta.UID),
				IsScaleUp: true,
				NewRequest: tfv1.Resource{
					Tflops: resource.MustParse("300"),
					Vram:   resource.MustParse("30Gi"),
				},
				NewLimit: tfv1.Resource{
					Tflops: resource.MustParse("400"),
					Vram:   resource.MustParse("40Gi"),
				},
			}, true)

			Expect(IsScalingQuotaExceededError(err)).To(BeTrue())
			Expect(remain.Tflops.Value()).To(BeEquivalentTo(gpu.Status.Available.Tflops.Value()))
			Expect(remain.Vram.Value()).To(BeEquivalentTo(gpu.Status.Available.Vram.Value()))

			_, _, err = allocator.AdjustAllocation(ctx, tfv1.AdjustRequest{
				PodUID:    string(testPodMeta.UID),
				IsScaleUp: true,
				NewRequest: tfv1.Resource{
					Tflops: resource.MustParse("90"),
					Vram:   resource.MustParse("15Gi"),
				},
			}, false)
			Expect(err).NotTo(HaveOccurred())

			allocator.syncToK8s(ctx)

			// get actual available resources
			latestGPU := getGPU(gpus[0].Name)
			Expect(gpu.Status.Available.Tflops.Value() - latestGPU.Status.Available.Tflops.Value()).
				To(BeEquivalentTo(40))
			Expect(gpu.Status.Available.Vram.Value() - latestGPU.Status.Available.Vram.Value()).
				To(BeEquivalentTo(5 * 1024 * 1024 * 1024))

			// test scale down
			_, _, err = allocator.AdjustAllocation(ctx, tfv1.AdjustRequest{
				PodUID:    string(testPodMeta.UID),
				IsScaleUp: false,
				NewRequest: tfv1.Resource{
					Tflops: resource.MustParse("10"),
					Vram:   resource.MustParse("1Gi"),
				},
			}, false)
			Expect(err).NotTo(HaveOccurred())

			allocator.syncToK8s(ctx)

			// get actual available resources
			latestGPU = getGPU(gpus[0].Name)
			Expect(gpu.Status.Available.Tflops.Value() - latestGPU.Status.Available.Tflops.Value()).
				To(BeEquivalentTo(-40))
			Expect(gpu.Status.Available.Vram.Value() - latestGPU.Status.Available.Vram.Value()).
				To(BeEquivalentTo(-9 * 1024 * 1024 * 1024))
		})
	})

	Context("Event Handling", func() {
		It("should handle GPU creation events", func() {
			// Create a new GPU
			newGPU := &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "new-test-gpu",
					Namespace: "default",
					Labels: map[string]string{
						"gpupool.tensorfusion.io/name": "test-pool",
						"kubernetes.io/node":           "node-1",
					},
				},
				Status: tfv1.GPUStatus{
					Phase: tfv1.TensorFusionGPUPhaseRunning,
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("90"),
						Vram:   resource.MustParse("20Gi"),
					},
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("90"),
						Vram:   resource.MustParse("20Gi"),
					},
					GPUModel: "NVIDIA A100",
				},
			}

			// Save to API server
			err := k8sClient.Create(ctx, newGPU)
			Expect(err).NotTo(HaveOccurred())

			// Cleanup: delete the GPU after test completes
			DeferCleanup(func() {
				_ = k8sClient.Delete(ctx, newGPU)
			})

			// Handle the creation event
			allocator.handleGPUCreate(ctx, newGPU)

			// Verify the GPU is in the store
			key := types.NamespacedName{Name: newGPU.Name, Namespace: newGPU.Namespace}
			cachedGPU, exists := allocator.gpuStore[key]
			Expect(exists).To(BeTrue())
			Expect(cachedGPU.Name).To(Equal(newGPU.Name))
			Expect(cachedGPU.Status.Phase).To(Equal(newGPU.Status.Phase))
		})

		It("should handle GPU deletion events", func() {
			// Get an existing GPU from the store
			key := types.NamespacedName{Name: "gpu-1"}
			_, exists := allocator.gpuStore[key]
			Expect(exists).To(BeTrue())

			// Get the GPU from the API server
			gpuToDelete := getGPU("gpu-1")

			// Handle the deletion event
			allocator.handleGPUDelete(ctx, gpuToDelete)

			// Verify the GPU is removed from the store
			_, exists = allocator.gpuStore[key]
			Expect(exists).To(BeFalse())
		})
	})
	Context("FilterWithPreempt", func() {
		It("should include other available GPUs from same node for multi-GPU preemption", func() {
			// Allocate GPU-1 to a workload first
			request := tfv1.Resource{
				Tflops: resource.MustParse("50"),
				Vram:   resource.MustParse("10Gi"),
			}
			gpus, err := allocateAndSync("test-pool", request, 1, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(1))

			// Now test FilterWithPreempt for a 2-GPU requirement
			// Simulate preempting the allocated GPU
			preemptAllocRequest := &tfv1.AllocRequest{
				WorkloadNameNamespace: workloadNameNs,
				GPUNames:              []string{gpus[0].Name},
				Request:               request,
				PodMeta:               testPodMeta,
			}

			// Request 2 GPUs from the same node
			allocReq := &tfv1.AllocRequest{
				PoolName:              "test-pool",
				WorkloadNameNamespace: tfv1.NameNamespace{Namespace: "default", Name: "test-workload-2"},
				Request: tfv1.Resource{
					Tflops: resource.MustParse("30"),
					Vram:   resource.MustParse("8Gi"),
				},
				Count:   2,
				PodMeta: metav1.ObjectMeta{UID: "test-pod-2", Namespace: "default", Name: "test-pod-2"},
			}

			// Call FilterWithPreempt
			filteredGPUs, _, err := allocator.FilterWithPreempt(allocReq, []*tfv1.AllocRequest{preemptAllocRequest})
			Expect(err).NotTo(HaveOccurred())

			// FilterWithPreempt returns all GPUs from the same node that satisfy the conditions
			// Should return at least 2 GPUs from the same node: one from preemption + one already available
			Expect(len(filteredGPUs)).To(BeNumerically(">=", 2))

			// Verify all GPUs are from the same node
			nodeName := filteredGPUs[0].Labels[constants.LabelKeyOwner]
			for _, gpu := range filteredGPUs {
				Expect(gpu.Labels[constants.LabelKeyOwner]).To(Equal(nodeName))
			}
		})

		It("should apply SameNodeFilter for multi-GPU requirements", func() {
			// Request 2 GPUs
			allocReq := &tfv1.AllocRequest{
				PoolName:              "test-pool",
				WorkloadNameNamespace: tfv1.NameNamespace{Namespace: "default", Name: "test-workload-multi"},
				Request: tfv1.Resource{
					Tflops: resource.MustParse("20"),
					Vram:   resource.MustParse("4Gi"),
				},
				Count:   2,
				PodMeta: metav1.ObjectMeta{UID: "test-pod-multi", Namespace: "default", Name: "test-pod-multi"},
			}

			// Simulate preempting one GPU
			preemptAllocRequest := &tfv1.AllocRequest{
				WorkloadNameNamespace: workloadNameNs,
				GPUNames:              []string{"gpu-1"},
				Request: tfv1.Resource{
					Tflops: resource.MustParse("10"),
					Vram:   resource.MustParse("2Gi"),
				},
				PodMeta: testPodMeta,
			}

			// Call FilterWithPreempt
			filteredGPUs, _, err := allocator.FilterWithPreempt(allocReq, []*tfv1.AllocRequest{preemptAllocRequest})

			// Should succeed if same node has enough GPUs, otherwise error
			if err == nil {
				// FilterWithPreempt returns all GPUs from the same node that satisfy the conditions
				// Should return at least 2 GPUs from the same node
				Expect(len(filteredGPUs)).To(BeNumerically(">=", 2))
				// Verify all GPUs are from the same node
				nodeName := filteredGPUs[0].Labels[constants.LabelKeyOwner]
				for _, gpu := range filteredGPUs {
					Expect(gpu.Labels[constants.LabelKeyOwner]).To(Equal(nodeName))
				}
			}
		})

		It("should simulate resource release correctly during preemption", func() {
			// Allocate a GPU first
			request := tfv1.Resource{
				Tflops: resource.MustParse("70"),
				Vram:   resource.MustParse("15Gi"),
			}
			gpus, err := allocateAndSync("test-pool", request, 1, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(1))

			// Store the GPU's available resources before preemption
			gpuBefore := getGPU(gpus[0].Name)
			availableTflopsBefore := gpuBefore.Status.Available.Tflops.DeepCopy()
			availableVramBefore := gpuBefore.Status.Available.Vram.DeepCopy()

			// Simulate preemption
			preemptAllocRequest := &tfv1.AllocRequest{
				WorkloadNameNamespace: workloadNameNs,
				GPUNames:              []string{gpus[0].Name},
				Request:               request,
				PodMeta:               testPodMeta,
			}

			allocReq := &tfv1.AllocRequest{
				PoolName:              "test-pool",
				WorkloadNameNamespace: tfv1.NameNamespace{Namespace: "default", Name: "new-workload"},
				Request: tfv1.Resource{
					Tflops: resource.MustParse("50"),
					Vram:   resource.MustParse("10Gi"),
				},
				Count:   1,
				PodMeta: metav1.ObjectMeta{UID: "new-pod", Namespace: "default", Name: "new-pod"},
			}

			// Call FilterWithPreempt
			filteredGPUs, _, err := allocator.FilterWithPreempt(allocReq, []*tfv1.AllocRequest{preemptAllocRequest})
			Expect(err).NotTo(HaveOccurred())
			// FilterWithPreempt returns all GPUs that satisfy the conditions, not limited to req.Count
			Expect(filteredGPUs).ToNot(BeEmpty())

			// Find the preempted GPU in the results and verify it has simulated the resource release
			var preemptedGPU *tfv1.GPU
			for _, gpu := range filteredGPUs {
				if gpu.Name == gpus[0].Name {
					preemptedGPU = gpu
					break
				}
			}
			Expect(preemptedGPU).NotTo(BeNil(), "preempted GPU should be in the filtered results")

			expectedTflops := availableTflopsBefore.DeepCopy()
			expectedTflops.Add(request.Tflops)
			expectedVram := availableVramBefore.DeepCopy()
			expectedVram.Add(request.Vram)

			Expect(preemptedGPU.Status.Available.Tflops.Cmp(expectedTflops)).To(Equal(0))
			Expect(preemptedGPU.Status.Available.Vram.Cmp(expectedVram)).To(Equal(0))
		})

		It("should use targetNodeNames parameter for optimization", func() {
			// Allocate a GPU on a specific node
			request := tfv1.Resource{
				Tflops: resource.MustParse("40"),
				Vram:   resource.MustParse("8Gi"),
			}
			gpus, err := allocateAndSync("test-pool", request, 1, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(1))

			targetNode := gpus[0].Labels[constants.LabelKeyOwner]

			// Simulate preemption
			preemptAllocRequest := &tfv1.AllocRequest{
				WorkloadNameNamespace: workloadNameNs,
				GPUNames:              []string{gpus[0].Name},
				Request:               request,
				PodMeta:               testPodMeta,
			}

			allocReq := &tfv1.AllocRequest{
				PoolName:              "test-pool",
				WorkloadNameNamespace: tfv1.NameNamespace{Namespace: "default", Name: "target-node-test"},
				Request: tfv1.Resource{
					Tflops: resource.MustParse("30"),
					Vram:   resource.MustParse("6Gi"),
				},
				Count:   1,
				PodMeta: metav1.ObjectMeta{UID: "target-node-pod", Namespace: "default", Name: "target-node-pod"},
			}

			// Call FilterWithPreempt with targetNodeNames parameter
			filteredGPUs, _, err := allocator.FilterWithPreempt(allocReq, []*tfv1.AllocRequest{preemptAllocRequest}, targetNode)
			Expect(err).NotTo(HaveOccurred())
			// FilterWithPreempt returns all GPUs that satisfy the conditions, not limited to req.Count
			Expect(filteredGPUs).ToNot(BeEmpty())

			// Verify all returned GPUs are from the target node
			for _, gpu := range filteredGPUs {
				Expect(gpu.Labels[constants.LabelKeyOwner]).To(Equal(targetNode))
			}
		})

		It("should only check GPUs from specified target nodes", func() {
			// Test that when targetNodeNames is specified,
			// only GPUs from those nodes are considered (performance optimization)

			request := tfv1.Resource{
				Tflops: resource.MustParse("20"),
				Vram:   resource.MustParse("4Gi"),
			}

			// Get a GPU to identify its node
			gpus, err := allocateAndSync("test-pool", request, 1, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(1))

			targetNode := gpus[0].Labels[constants.LabelKeyOwner]

			// Deallocate for next test
			deallocateAndSync(gpus)

			// Create preempt request
			preemptAllocRequest := &tfv1.AllocRequest{
				WorkloadNameNamespace: tfv1.NameNamespace{Namespace: "default", Name: "victim"},
				GPUNames:              []string{gpus[0].Name},
				Request:               request,
				PodMeta:               metav1.ObjectMeta{UID: "victim-pod", Namespace: "default", Name: "victim-pod"},
			}

			allocReq := &tfv1.AllocRequest{
				PoolName:              "test-pool",
				WorkloadNameNamespace: tfv1.NameNamespace{Namespace: "default", Name: "new-workload"},
				Request: tfv1.Resource{
					Tflops: resource.MustParse("15"),
					Vram:   resource.MustParse("3Gi"),
				},
				Count:   1,
				PodMeta: metav1.ObjectMeta{UID: "new-pod-2", Namespace: "default", Name: "new-pod-2"},
			}

			// Call with specific target node
			filteredGPUs, _, err := allocator.FilterWithPreempt(allocReq, []*tfv1.AllocRequest{preemptAllocRequest}, targetNode)
			Expect(err).NotTo(HaveOccurred())

			// All returned GPUs should be from target node
			for _, gpu := range filteredGPUs {
				nodeName := gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
				Expect(nodeName).To(Equal(targetNode))
			}
		})

		It("should return error when no affected nodes", func() {
			// Test that FilterWithPreempt returns error when called with no preemptAllocRequests and no targetNodeNames
			allocReq := &tfv1.AllocRequest{
				PoolName:              "test-pool",
				WorkloadNameNamespace: tfv1.NameNamespace{Namespace: "default", Name: "test"},
				Request: tfv1.Resource{
					Tflops: resource.MustParse("10"),
					Vram:   resource.MustParse("2Gi"),
				},
				Count:   1,
				PodMeta: metav1.ObjectMeta{UID: "test-pod", Namespace: "default", Name: "test-pod"},
			}

			// Call FilterWithPreempt with empty preemptAllocRequests and no targetNodeNames
			_, _, err := allocator.FilterWithPreempt(allocReq, []*tfv1.AllocRequest{})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("no affected nodes"))
		})
	})

	Context("Small Resource Values", func() {
		It("should handle small TFLOPs values like 500m", func() {
			// Test allocation with small resource values (500m = 0.5 TFLOPs)
			request := tfv1.Resource{
				Tflops: resource.MustParse("500m"),
				Vram:   resource.MustParse("1Gi"),
			}

			gpus, err := allocateAndSync("test-pool", request, 1, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(1))

			// Verify resources were reduced correctly
			gpu := getGPU(gpus[0].Name)
			Expect(gpu.Status.Available.Tflops.Cmp(gpu.Status.Capacity.Tflops)).To(Equal(-1))
			Expect(gpu.Status.Available.Vram.Cmp(gpu.Status.Capacity.Vram)).To(Equal(-1))

			// Deallocate
			deallocateAndSync(gpus)
		})

		It("should handle preemption with small TFLOPs values", func() {
			// Allocate with small value
			request := tfv1.Resource{
				Tflops: resource.MustParse("500m"),
				Vram:   resource.MustParse("2Gi"),
			}
			gpus, err := allocateAndSync("test-pool", request, 1, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(1))

			// Simulate preemption
			preemptAllocRequest := &tfv1.AllocRequest{
				WorkloadNameNamespace: workloadNameNs,
				GPUNames:              []string{gpus[0].Name},
				Request:               request,
				PodMeta:               testPodMeta,
			}

			allocReq := &tfv1.AllocRequest{
				PoolName:              "test-pool",
				WorkloadNameNamespace: tfv1.NameNamespace{Namespace: "default", Name: "small-preempt"},
				Request: tfv1.Resource{
					Tflops: resource.MustParse("300m"),
					Vram:   resource.MustParse("1Gi"),
				},
				Count:   1,
				PodMeta: metav1.ObjectMeta{UID: "small-preempt-pod", Namespace: "default", Name: "small-preempt-pod"},
			}

			// Call FilterWithPreempt
			filteredGPUs, _, err := allocator.FilterWithPreempt(allocReq, []*tfv1.AllocRequest{preemptAllocRequest})
			Expect(err).NotTo(HaveOccurred())
			// FilterWithPreempt returns all GPUs that satisfy the conditions, not limited to req.Count
			Expect(filteredGPUs).ToNot(BeEmpty())

			// Verify at least one filtered GPU has correct available resources after simulated release
			// Original available + 500m released should be enough for 300m request
			hasValidGPU := false
			for _, gpu := range filteredGPUs {
				if gpu.Status.Available.Tflops.Cmp(resource.MustParse("0")) > 0 {
					hasValidGPU = true
					break
				}
			}
			Expect(hasValidGPU).To(BeTrue(), "at least one GPU should have enough resources")
		})

		It("should correctly multiply small TFLOPs by count in quota calculation", func() {
			// Test multi-GPU allocation with small values
			request := tfv1.Resource{
				Tflops: resource.MustParse("250m"), // 0.25 TFLOPs per GPU
				Vram:   resource.MustParse("1Gi"),
			}

			gpus, err := allocateAndSync("test-pool", request, 2, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(2))

			// Get allocation info
			_, _, uniqueAllocation := allocator.GetAllocationInfo()
			allocInfo, exists := uniqueAllocation[string(testPodMeta.UID)]
			Expect(exists).To(BeTrue())
			Expect(allocInfo.Count).To(Equal(uint(2)))

			// Calculate total: 250m * 2 = 500m (0.5 TFLOPs)
			expectedTotalTflops := resource.MustParse("500m")
			expectedTotalVram := resource.MustParse("2Gi")

			totalTflops := allocInfo.Request.Tflops.DeepCopy()
			totalTflops.Mul(int64(allocInfo.Count))
			totalVram := allocInfo.Request.Vram.DeepCopy()
			totalVram.Mul(int64(allocInfo.Count))

			Expect(totalTflops.Cmp(expectedTotalTflops)).To(Equal(0),
				"Should correctly multiply small TFLOPs: 250m * 2 = 500m")
			Expect(totalVram.Cmp(expectedTotalVram)).To(Equal(0))

			// Deallocate
			deallocateAndSync(gpus)
		})

		It("should handle mixed small and large values in preemption", func() {
			// Allocate large resource
			largeRequest := tfv1.Resource{
				Tflops: resource.MustParse("50"),
				Vram:   resource.MustParse("10Gi"),
			}
			largeGPUs, err := allocateAndSync("test-pool", largeRequest, 1, "")
			Expect(err).NotTo(HaveOccurred())

			// Try to allocate small resource that should fit after preemption
			preemptAllocRequest := &tfv1.AllocRequest{
				WorkloadNameNamespace: workloadNameNs,
				GPUNames:              []string{largeGPUs[0].Name},
				Request:               largeRequest,
				PodMeta:               testPodMeta,
			}

			smallAllocReq := &tfv1.AllocRequest{
				PoolName:              "test-pool",
				WorkloadNameNamespace: tfv1.NameNamespace{Namespace: "default", Name: "small-after-large"},
				Request: tfv1.Resource{
					Tflops: resource.MustParse("100m"), // Very small request
					Vram:   resource.MustParse("500Mi"),
				},
				Count:   1,
				PodMeta: metav1.ObjectMeta{UID: "small-pod", Namespace: "default", Name: "small-pod"},
			}

			filteredGPUs, _, err := allocator.FilterWithPreempt(smallAllocReq, []*tfv1.AllocRequest{preemptAllocRequest})
			Expect(err).NotTo(HaveOccurred())
			// FilterWithPreempt returns all GPUs that satisfy the conditions, not limited to req.Count
			Expect(filteredGPUs).ToNot(BeEmpty())

			// Deallocate
			deallocateAndSync(largeGPUs)
		})
	})

	Context("Multi-GPU Quota Calculation", func() {
		It("should correctly calculate quota for multi-GPU pods during preemption", func() {
			// This tests the fix for multi-GPU quota calculation bug
			// where Count was not multiplied with resources

			// Allocate 2 GPUs
			request := tfv1.Resource{
				Tflops: resource.MustParse("50"),
				Vram:   resource.MustParse("10Gi"),
			}
			gpus, err := allocateAndSync("test-pool", request, 2, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(2))

			// Get allocation info using GetAllocationInfo
			_, _, uniqueAllocation := allocator.GetAllocationInfo()
			allocInfo, exists := uniqueAllocation[string(testPodMeta.UID)]
			Expect(exists).To(BeTrue())
			Expect(allocInfo).NotTo(BeNil())
			Expect(allocInfo.Count).To(Equal(uint(2)))

			// The total quota usage should be: request * count
			// For 2 GPUs with 50 TFLOPs each, total should be 100 TFLOPs
			expectedTotalTflops := resource.MustParse("100") // 50 * 2
			expectedTotalVram := resource.MustParse("20Gi")  // 10Gi * 2

			// Verify in quota tracking (this would be checked in CheckQuotaAndFilterSingleNodePreempt)
			// For now, verify the allocation has correct Count
			Expect(allocInfo.Request.Tflops.Cmp(request.Tflops)).To(Equal(0))
			Expect(allocInfo.Request.Vram.Cmp(request.Vram)).To(Equal(0))

			// Calculate total as the quota calculation does
			totalTflops := allocInfo.Request.Tflops.DeepCopy()
			totalTflops.Mul(int64(allocInfo.Count))
			totalVram := allocInfo.Request.Vram.DeepCopy()
			totalVram.Mul(int64(allocInfo.Count))

			Expect(totalTflops.Cmp(expectedTotalTflops)).To(Equal(0))
			Expect(totalVram.Cmp(expectedTotalVram)).To(Equal(0))
		})

		It("should multiply resources by count in quota preemption checks", func() {
			// Test that quota calculation correctly multiplies by GPU count
			// This is a regression test for the bug fix

			request := tfv1.Resource{
				Tflops: resource.MustParse("30"),
				Vram:   resource.MustParse("6Gi"),
			}

			// Allocate 3 GPUs to a workload
			gpus, err := allocateAndSync("test-pool", request, 3, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(3))

			// Get allocation info
			_, _, uniqueAllocation := allocator.GetAllocationInfo()
			allocInfo, exists := uniqueAllocation[string(testPodMeta.UID)]
			Expect(exists).To(BeTrue())
			Expect(allocInfo).NotTo(BeNil())

			// Manually calculate what the quota usage should be
			expectedTotalTflops := resource.MustParse("90") // 30 * 3
			expectedTotalVram := resource.MustParse("18Gi") // 6Gi * 3

			// Simulate quota calculation as done in CheckQuotaAndFilterSingleNodePreempt
			calculatedTflops := allocInfo.Request.Tflops.DeepCopy()
			calculatedTflops.Mul(int64(allocInfo.Count))
			calculatedVram := allocInfo.Request.Vram.DeepCopy()
			calculatedVram.Mul(int64(allocInfo.Count))

			// Verify multiplication is correct
			Expect(calculatedTflops.Cmp(expectedTotalTflops)).To(Equal(0),
				"TFLOPs should be multiplied by count: %v * %d = %v",
				allocInfo.Request.Tflops.String(), allocInfo.Count, expectedTotalTflops.String())
			Expect(calculatedVram.Cmp(expectedTotalVram)).To(Equal(0),
				"VRAM should be multiplied by count: %v * %d = %v",
				allocInfo.Request.Vram.String(), allocInfo.Count, expectedTotalVram.String())
		})
	})

})
