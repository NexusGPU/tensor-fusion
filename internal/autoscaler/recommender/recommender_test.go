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
				Applied:          false,
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
				Applied:          false,
				ScaleDownLocking: false,
			},
		}

		final := MergeRecommendations(recs)
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

	It("should be excluded from megring operations if recommendations have been applied", func() {
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
				Applied:          false,
				ScaleDownLocking: false,
			},
			"rec2": {
				Resources: tfv1.Resources{
					Requests: tfv1.Resource{
						Tflops: resource.MustParse("100"),
						Vram:   resource.MustParse("150Gi"),
					},
					Limits: tfv1.Resource{
						Tflops: resource.MustParse("100"),
						Vram:   resource.MustParse("150Gi"),
					},
				},
				Applied:          true,
				ScaleDownLocking: false,
			},
		}

		final := MergeRecommendations(recs)
		Expect(final.Equal(&tfv1.Resources{
			Requests: tfv1.Resource{
				Tflops: resource.MustParse("10"),
				Vram:   resource.MustParse("10Gi"),
			},
			Limits: tfv1.Resource{
				Tflops: resource.MustParse("15"),
				Vram:   resource.MustParse("15Gi"),
			},
		})).To(BeTrue())
	})

	It("should not scale down when merging recomendations if scale down is locked", func() {
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
				Applied:          true,
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
				Applied:          false,
				ScaleDownLocking: false,
			},
		}

		final := MergeRecommendations(recs)
		Expect(final.Equal(&tfv1.Resources{
			Requests: tfv1.Resource{
				Tflops: resource.MustParse("50"),
				Vram:   resource.MustParse("50Gi"),
			},
			Limits: tfv1.Resource{
				Tflops: resource.MustParse("50"),
				Vram:   resource.MustParse("50Gi"),
			},
		})).To(BeTrue())
	})
})
