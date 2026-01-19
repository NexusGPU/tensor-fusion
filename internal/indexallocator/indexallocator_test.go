package indexallocator

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

var _ = Describe("IndexAllocator", func() {
	var (
		allocator *IndexAllocator
		ctx       context.Context
	)

	BeforeEach(func() {
		scheme := fake.NewClientBuilder().WithScheme(fake.NewClientBuilder().Build().Scheme()).Build().Scheme()
		client := fake.NewClientBuilder().WithScheme(scheme).Build()

		ctx = context.Background()
		var err error
		allocator, err = NewIndexAllocator(ctx, client)
		Expect(err).NotTo(HaveOccurred())
		allocator.IsLeader = true
	})

	Describe("AssignIndex", func() {
		It("should assign first index as 1", func() {
			index1, err := allocator.AssignIndex("pod-1")
			Expect(err).NotTo(HaveOccurred())
			Expect(index1).To(Equal(1))
		})

		It("should assign second index as 2", func() {
			_, err := allocator.AssignIndex("pod-1")
			Expect(err).NotTo(HaveOccurred())

			index2, err := allocator.AssignIndex("pod-2")
			Expect(err).NotTo(HaveOccurred())
			Expect(index2).To(Equal(2))
		})

		It("should assign indices in ascending order", func() {
			for i := 1; i <= 10; i++ {
				index, err := allocator.AssignIndex("pod-" + string(rune(i)))
				Expect(err).NotTo(HaveOccurred())
				Expect(index).To(Equal(i), "index should be assigned in ascending order")
			}
		})
	})

	Describe("AssignIndex incremental order", func() {
		It("should assign indices in incremental order (1, 2, 3, ...)", func() {
			expectedIndex := 1
			for i := 0; i < 20; i++ {
				index, err := allocator.AssignIndex("pod-" + string(rune(i)))
				Expect(err).NotTo(HaveOccurred())
				Expect(index).To(Equal(expectedIndex), "index should be assigned in ascending order")
				expectedIndex++
			}
		})
	})

	Describe("WrapAround", func() {
		It("should wrap around from max index to 1", func() {
			// Max index is IndexModLength * IndexKeyLength = 8 * 16 = 128
			maxIndex := constants.IndexModLength * constants.IndexKeyLength
			// Assign indices until we reach maxIndex (128)
			for i := 1; i <= maxIndex; i++ {
				index, err := allocator.AssignIndex("pod-" + string(rune(i)))
				Expect(err).NotTo(HaveOccurred())
				Expect(index).To(Equal(i))
			}

			// Next assignment should wrap around to 1
			index, err := allocator.AssignIndex("pod-wrap")
			Expect(err).NotTo(HaveOccurred())
			Expect(index).To(Equal(1), "index should wrap around from 128 to 1")
		})
	})
})
