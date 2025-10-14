package recommender

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"
)

var _ = Describe("Recommender", func() {
	It("should merge recomendations based on a larger request value", func() {
		recs := map[string]*RecResult{
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

		final := recommenderToRecResult(recs).generateRecommendation()
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
		recs := map[string]*RecResult{
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

		Expect(recommenderToRecResult(recs).generateRecommendation()).To(BeNil())
	})

	It("should return recommendation that replaced with the current maximum allowable GPU resource", func() {
		recommendation := tfv1.Resources{
			Requests: tfv1.Resource{
				Tflops: resource.MustParse("200"),
				Vram:   resource.MustParse("200Gi"),
			},
			Limits: tfv1.Resource{
				Tflops: resource.MustParse("400"),
				Vram:   resource.MustParse("400Gi"),
			},
		}
		expectedRec := tfv1.Resources{
			Requests: tfv1.Resource{
				Tflops: resource.MustParse("100"),
				Vram:   resource.MustParse("100Gi"),
			},
			Limits: tfv1.Resource{
				Tflops: resource.MustParse("200"),
				Vram:   resource.MustParse("200Gi"),
			},
		}
		maxAllowedRes := tfv1.Resource{
			Tflops: resource.MustParse("100"),
			Vram:   resource.MustParse("100Gi"),
		}
		workload := workload.NewWorkloadState()
		processor := &recommendationProcessor{&fakeWorkloadHandler{Resource: maxAllowedRes}}
		got, msg, _ := processor.Apply(context.Background(), workload, &recommendation)
		Expect(got.Equal(&expectedRec)).To(BeTrue())
		Expect(msg).To(Equal("TFLOPS reduced due to target (200) exceed max allowed (100), VRAM reduced due to target (200Gi) exceed max allowed (100Gi)"))
	})

	It("should return the original recommendation if it does not exceed maximum allowable GPU resource", func() {
		recommendation := tfv1.Resources{
			Requests: tfv1.Resource{
				Tflops: resource.MustParse("200"),
				Vram:   resource.MustParse("200Gi"),
			},
			Limits: tfv1.Resource{
				Tflops: resource.MustParse("400"),
				Vram:   resource.MustParse("400Gi"),
			},
		}
		maxAllowedRes := tfv1.Resource{
			Tflops: resource.MustParse("300"),
			Vram:   resource.MustParse("300Gi"),
		}
		workload := workload.NewWorkloadState()
		processor := &recommendationProcessor{&fakeWorkloadHandler{Resource: maxAllowedRes}}
		got, msg, _ := processor.Apply(context.Background(), workload, &recommendation)
		Expect(got.Equal(&recommendation)).To(BeTrue())
		Expect(msg).To(BeEmpty())
	})
})

type fakeWorkloadHandler struct {
	tfv1.Resource
	workload.Handler
}

func (f *fakeWorkloadHandler) GetMaxAllowedResourcesSpec(workload *workload.State) (*tfv1.Resource, error) {
	return &f.Resource, nil
}

type fakeRecommendationProcessor struct {
	tfv1.Resources
}

func (r *fakeRecommendationProcessor) Apply(
	ctx context.Context,
	workload *workload.State,
	rec *tfv1.Resources) (tfv1.Resources, string, error) {
	return r.Resources, "fake message", nil
}
