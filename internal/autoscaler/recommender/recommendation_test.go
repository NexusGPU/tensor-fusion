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
				Tflops: resource.MustParse("400"),   // Limits are not modified by processor
				Vram:   resource.MustParse("400Gi"), // Limits are not modified by processor
			},
		}
		maxAllowedRes := tfv1.Resource{
			Tflops: resource.MustParse("100"),
			Vram:   resource.MustParse("100Gi"),
		}
		workload := workload.NewWorkloadState()
		// Set current resources to be less than recommendation to trigger scale-up check
		workload.Spec.Resources = tfv1.Resources{
			Requests: tfv1.Resource{
				Tflops: resource.MustParse("50"),
				Vram:   resource.MustParse("50Gi"),
			},
			Limits: tfv1.Resource{
				Tflops: resource.MustParse("100"),
				Vram:   resource.MustParse("100Gi"),
			},
		}
		processor := &recommendationProcessor{&fakeWorkloadHandler{Resource: maxAllowedRes}}
		got, msg, _ := processor.Apply(context.Background(), workload, &recommendation)
		Expect(got.Equal(&expectedRec)).To(BeTrue())
		Expect(msg).To(Equal("TFlops request set to max allowed: (100), VRAM request set to max allowed: (100Gi)"))
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

	It("should cap requests and limits at the capacity of the GPUs held, whatever the direction of the change", func() {
		// The reported case: a limit of 82 TFLOPS on a GPU of 71.2.
		recommendation := tfv1.Resources{
			Requests: tfv1.Resource{Tflops: resource.MustParse("10"), Vram: resource.MustParse("4Gi")},
			Limits:   tfv1.Resource{Tflops: resource.MustParse("82"), Vram: resource.MustParse("30Gi")},
		}
		state := workload.NewWorkloadState()
		// The request goes down, so nothing asks for the room left on the GPUs.
		state.Spec.Resources = tfv1.Resources{
			Requests: tfv1.Resource{Tflops: resource.MustParse("20"), Vram: resource.MustParse("8Gi")},
			Limits:   tfv1.Resource{Tflops: resource.MustParse("67"), Vram: resource.MustParse("20Gi")},
		}
		processor := &recommendationProcessor{&fakeWorkloadHandler{capacity: &workload.DeviceCapacity{
			PerGPU:        tfv1.Resource{Tflops: resource.MustParse("71.2"), Vram: resource.MustParse("24Gi")},
			GPUsPerWorker: 1,
		}}}
		got, msg, err := processor.Apply(context.Background(), state, &recommendation)
		Expect(err).ToNot(HaveOccurred())
		Expect(got.Limits.Tflops.String()).To(Equal("71"))
		Expect(got.Limits.Vram.String()).To(Equal("24Gi"))
		Expect(got.Requests.Tflops.String()).To(Equal("10"))
		Expect(got.Requests.Vram.String()).To(Equal("4Gi"))
		Expect(msg).To(Equal("bounds: Compute limit: recommended 82, applied 71 (device capacity: 71200m per GPU held); " +
			"VRAM limit: recommended 30Gi, applied 24Gi (device capacity: 24Gi per GPU held)"))
	})

	It("should cap a request above the capacity before fitting it to the room left", func() {
		recommendation := tfv1.Resources{
			Requests: tfv1.Resource{Tflops: resource.MustParse("90"), Vram: resource.MustParse("4Gi")},
			Limits:   tfv1.Resource{Tflops: resource.MustParse("90"), Vram: resource.MustParse("8Gi")},
		}
		state := workload.NewWorkloadState()
		state.Spec.Resources = tfv1.Resources{
			Requests: tfv1.Resource{Tflops: resource.MustParse("20"), Vram: resource.MustParse("4Gi")},
			Limits:   tfv1.Resource{Tflops: resource.MustParse("40"), Vram: resource.MustParse("8Gi")},
		}
		processor := &recommendationProcessor{&fakeWorkloadHandler{
			// 50 TFLOPS of the GPU are left for this workload.
			Resource: tfv1.Resource{Tflops: resource.MustParse("50"), Vram: resource.MustParse("24Gi")},
			capacity: &workload.DeviceCapacity{
				PerGPU:        tfv1.Resource{Tflops: resource.MustParse("71.2"), Vram: resource.MustParse("24Gi")},
				GPUsPerWorker: 1,
			},
		}}
		got, msg, err := processor.Apply(context.Background(), state, &recommendation)
		Expect(err).ToNot(HaveOccurred())
		Expect(got.Requests.Tflops.String()).To(Equal("50"))
		Expect(got.Limits.Tflops.String()).To(Equal("71"))
		Expect(msg).To(ContainSubstring("Compute request: recommended 90, applied 71 (device capacity"))
		Expect(msg).To(ContainSubstring("TFlops request set to max allowed: (50)"))
	})
})

type fakeWorkloadHandler struct {
	tfv1.Resource
	workload.Handler
	capacity *workload.DeviceCapacity
}

func (f *fakeWorkloadHandler) GetWorkerDeviceCapacity(workload *workload.State) *workload.DeviceCapacity {
	return f.capacity
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
