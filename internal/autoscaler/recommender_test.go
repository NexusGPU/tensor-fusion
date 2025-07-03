package autoscaler

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Resource Recommender", func() {
	Context("when getting recommended resource", func() {
		It("should return correct RecommendedResources based on WorkloadState and config", func() {
			ws := NewWorkloadState("test")
			rr := resourceRecommender{}
		})
	})
})
