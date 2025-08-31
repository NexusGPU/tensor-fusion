package recommender

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"
)

var _ = Describe("Recommender", func() {
	It("should merge recomendations based on a larger request value", func() {
		recs := map[string]*Recommendation{
			"rec1": {
				Resources: tfv1.Resources{
					Requests: tfv1.Resource{
						Tflops: resource.MustParse("10"),
						Vram:   resource.MustParse("10Gi"),
					},
					Limits: tfv1.Resource{
						Tflops: resource.MustParse("15"),
						Vram:   resource.MustParse("15Gi"),
					},
				},
				HasApplied:       false,
				ScaleDownLocking: false,
			},
			"rec2": {
				Resources: tfv1.Resources{
					Requests: tfv1.Resource{
						Tflops: resource.MustParse("5"),
						Vram:   resource.MustParse("15Gi"),
					},
					Limits: tfv1.Resource{
						Tflops: resource.MustParse("20"),
						Vram:   resource.MustParse("20Gi"),
					},
				},
				HasApplied:       false,
				ScaleDownLocking: false,
			},
		}

		final := getResourcesFromRecommendations(recs)
		Expect(final.Equal(&tfv1.Resources{
			Requests: tfv1.Resource{
				Tflops: resource.MustParse("10"),
				Vram:   resource.MustParse("15Gi"),
			},
			Limits: tfv1.Resource{
				Tflops: resource.MustParse("15"),
				Vram:   resource.MustParse("20Gi"),
			},
		})).To(BeTrue())
	})

	It("should not reduce resources if scale down is locked", func() {
		recs := map[string]*Recommendation{
			"rec1": {
				Resources: tfv1.Resources{
					Requests: tfv1.Resource{
						Tflops: resource.MustParse("50"),
						Vram:   resource.MustParse("50Gi"),
					},
					Limits: tfv1.Resource{
						Tflops: resource.MustParse("50"),
						Vram:   resource.MustParse("50Gi"),
					},
				},
				HasApplied:       true,
				ScaleDownLocking: true,
			},
			"rec2": {
				Resources: tfv1.Resources{
					Requests: tfv1.Resource{
						Tflops: resource.MustParse("10"),
						Vram:   resource.MustParse("10Gi"),
					},
					Limits: tfv1.Resource{
						Tflops: resource.MustParse("10"),
						Vram:   resource.MustParse("10Gi"),
					},
				},
				HasApplied:       false,
				ScaleDownLocking: false,
			},
		}

		Expect(getResourcesFromRecommendations(recs)).To(BeNil())
	})
})
