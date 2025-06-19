package autoscaler

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Recommender", func() {
	Context("when get recommeded resource", func() {
		It("should generate recommended resource based on histogram", func() {
			recommender := NewRecommender()
			Expect(recommender.GetRecommendedResources(nil)).To(BeNil())
		})
		It("should gererate recommended resource with safety margin", func() {
		})
		It("should gererate recommended resource with confidence multiplier", func() {
		})
	})
})
