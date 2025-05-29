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
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/samber/lo"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var _ = Describe("GPU Allocator", func() {
	var allocator *GpuAllocator

	BeforeEach(func() {
		allocator = NewGpuAllocator(ctx, k8sClient, 3*time.Second)
		readyCh, err := allocator.SetupWithManager(ctx, mgr)
		Expect(err).NotTo(HaveOccurred())

		var ready bool
		select {
		case <-readyCh:
			ready = true
		case <-time.After(10 * time.Second):
			ready = false
		}
		Expect(ready).To(BeTrue(), "allocator failed to become ready within timeout")
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

			gpus, err := allocator.Alloc(ctx, "test-pool", request, 1, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(1))

			// Explicitly call syncToK8s to persist changes before verification
			allocator.syncToK8s(ctx)

			// Verify resources were reduced on the allocated GPU
			gpu := getGPU(gpus[0].Name, gpus[0].Namespace)
			Expect(gpu.Status.Available.Tflops.Cmp(gpu.Status.Capacity.Tflops)).To(Equal(-1))
			Expect(gpu.Status.Available.Vram.Cmp(gpu.Status.Capacity.Vram)).To(Equal(-1))
		})

		It("should allocate multiple GPUs from the same node", func() {
			request := tfv1.Resource{
				Tflops: resource.MustParse("20"),
				Vram:   resource.MustParse("4Gi"),
			}

			gpus, err := allocator.Alloc(ctx, "test-pool", request, 2, "")
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

			_, err := allocator.Alloc(ctx, "test-pool", request, 10, "")
			Expect(err).To(HaveOccurred())
		})

		It("should fail when resources are insufficient", func() {
			request := tfv1.Resource{
				Tflops: resource.MustParse("200"),
				Vram:   resource.MustParse("64Gi"),
			}

			_, err := allocator.Alloc(ctx, "test-pool", request, 1, "")
			Expect(err).To(HaveOccurred())
		})

		It("should fail when pool doesn't exist", func() {
			request := tfv1.Resource{
				Tflops: resource.MustParse("10"),
				Vram:   resource.MustParse("2Gi"),
			}

			_, err := allocator.Alloc(ctx, "nonexistent-pool", request, 1, "")
			Expect(err).To(HaveOccurred())
		})

		It("should filter GPUs by model", func() {
			request := tfv1.Resource{
				Tflops: resource.MustParse("50"),
				Vram:   resource.MustParse("8Gi"),
			}

			// Try allocating with a specific GPU model
			gpus, err := allocator.Alloc(ctx, "test-pool", request, 1, "NVIDIA A100")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(1))
			Expect(gpus[0].Status.GPUModel).To(Equal("NVIDIA A100"))

			// Try allocating with a non-existent GPU model
			_, err = allocator.Alloc(ctx, "test-pool", request, 1, "NonExistentModel")
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

			gpus, err := allocator.Alloc(ctx, "test-pool", request, 1, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(gpus).To(HaveLen(1))

			// Store the allocated values
			allocatedGPU := gpus[0]
			allocatedTflops := allocatedGPU.Status.Available.Tflops.DeepCopy()
			allocatedVram := allocatedGPU.Status.Available.Vram.DeepCopy()

			// Now deallocate
			err = allocator.Dealloc(ctx, request, []types.NamespacedName{client.ObjectKeyFromObject(gpus[0])})
			Expect(err).NotTo(HaveOccurred())

			// Verify resources were restored
			deallocatedGPU := getGPU(allocatedGPU.Name, allocatedGPU.Namespace)
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
			allocatedGPUs, err := allocator.Alloc(ctx, "test-pool", request, 2, "")
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
			gpusToDeallocKeys := lo.Map(gpusToDealloc, func(gpu *tfv1.GPU, _ int) types.NamespacedName {
				return client.ObjectKeyFromObject(gpu)
			})
			// Now deallocate all GPUs including the non-existent one
			err = allocator.Dealloc(ctx, request, gpusToDeallocKeys)
			Expect(err).NotTo(HaveOccurred())

			// Verify resources were restored for existing GPUs
			for _, allocatedGPU := range allocatedGPUs {
				deallocatedGPU := getGPU(allocatedGPU.Name, allocatedGPU.Namespace)
				initialState := initialStates[allocatedGPU.Name]

				expectedTflops := initialState.tflops.DeepCopy()
				expectedVram := initialState.vram.DeepCopy()
				expectedTflops.Add(request.Tflops)
				expectedVram.Add(request.Vram)

				Expect(deallocatedGPU.Status.Available.Tflops.Cmp(initialState.tflops)).To(Equal(1))
				Expect(deallocatedGPU.Status.Available.Vram.Cmp(initialState.vram)).To(Equal(1))
			}
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
			gpuToDelete := getGPU("gpu-1", "")

			// Handle the deletion event
			allocator.handleGPUDelete(ctx, gpuToDelete)

			// Verify the GPU is removed from the store
			_, exists = allocator.gpuStore[key]
			Expect(exists).To(BeFalse())
		})
	})

})
