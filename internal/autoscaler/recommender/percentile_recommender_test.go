package recommender

import (
	"context"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
)

var _ = Describe("Percentile Recommender", func() {
	Context("when recommending resources", func() {
		ctx := context.Background()
		var estimations EstimatedResources
		var recommender *PercentileRecommender
		var ws *workload.State
		BeforeEach(func() {
			estimations = EstimatedResources{
				LowerBoundTflops: resource.MustParse("100"),
				TargetTflops:     resource.MustParse("200"),
				UpperBoundTflops: resource.MustParse("300"),
				LowerBoundVram:   resource.MustParse("100Gi"),
				TargetVram:       resource.MustParse("200Gi"),
				UpperBoundVram:   resource.MustParse("300Gi"),
			}
			recommender = &PercentileRecommender{
				&fakeResourcesEstimator{&estimations},
				nil,
			}
			ws = workload.NewWorkloadState()
		})

		It("should scale up if current resources below lower bounds", func() {
			curRes := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("20"),
					Vram:   resource.MustParse("20Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("40"),
					Vram:   resource.MustParse("40Gi"),
				},
			}
			expectRes := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("200"),
					Vram:   resource.MustParse("200Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("400"),
					Vram:   resource.MustParse("400Gi"),
				},
			}

			ws.Spec.Resources = curRes
			got, _ := recommender.Recommend(ctx, ws)
			Expect(got.Resources.Equal(&expectRes)).To(BeTrue())
			condition := meta.FindStatusCondition(ws.Status.Conditions, constants.ConditionStatusTypeRecommendationProvided)
			Expect(condition.Message).To(Equal("TFLOPS scaled up due to (20) below lower bound (100), VRAM scaled up due to (20Gi) below lower bound (100Gi)"))
		})

		It("should scale down if current resources above upper bounds", func() {
			curRes := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("400"),
					Vram:   resource.MustParse("400Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("800"),
					Vram:   resource.MustParse("800Gi"),
				},
			}
			expectRes := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("200"),
					Vram:   resource.MustParse("200Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("400"),
					Vram:   resource.MustParse("400Gi"),
				},
			}

			ws.Spec.Resources = curRes
			got, _ := recommender.Recommend(ctx, ws)
			Expect(got.Resources.Equal(&expectRes)).To(BeTrue())
			condition := meta.FindStatusCondition(ws.Status.Conditions, constants.ConditionStatusTypeRecommendationProvided)
			Expect(condition.Message).To(Equal("TFLOPS scaled down due to (400) above upper bound (300), VRAM scaled down due to (400Gi) above upper bound (300Gi)"))
		})

		It("should return nil if current resources within estimated bounds", func() {
			curRes := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("150"),
					Vram:   resource.MustParse("150Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("200"),
					Vram:   resource.MustParse("200Gi"),
				},
			}

			ws.Spec.Resources = curRes
			got, _ := recommender.Recommend(ctx, ws)
			Expect(got).To(BeNil())
		})

		It("should correctly apply recommendation processor", func() {
			curRes := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("20"),
					Vram:   resource.MustParse("20Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("40"),
					Vram:   resource.MustParse("40Gi"),
				},
			}
			expectRes := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("100"),
					Vram:   resource.MustParse("100Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("200"),
					Vram:   resource.MustParse("200Gi"),
				},
			}

			recommender = &PercentileRecommender{
				&fakeResourcesEstimator{&estimations},
				&fakeRecommendationProcessor{expectRes},
			}
			ws.Spec.Resources = curRes
			got, _ := recommender.Recommend(ctx, ws)
			Expect(got.Resources.Equal(&expectRes)).To(BeTrue())
			condition := meta.FindStatusCondition(ws.Status.Conditions, constants.ConditionStatusTypeRecommendationProvided)
			Expect(condition.Message).To(Equal("TFLOPS scaled up due to (20) below lower bound (100), VRAM scaled up due to (20Gi) below lower bound (100Gi), fake message"))
		})
	})

	Context("when parsing AutoScalingConfig", func() {
		It("should return default config when no AutoScalingConfig is set", func() {
			cfg := getPercentileConfig(nil)
			Expect(cfg).ToNot(BeNil())
			Expect(*cfg).To(Equal(defaultPercentileConfig))
		})

		It("should parse float fields from AutoSetResources", func() {
			asr := &tfv1.AutoSetResources{
				TargetTflopsPercentile:     "0.8",
				LowerBoundTflopsPercentile: "0.1",
				UpperBoundTflopsPercentile: "0.95",
				TargetVramPercentile:       "0.7",
				LowerBoundVramPercentile:   "0.2",
				UpperBoundVramPercentile:   "0.9",
				RequestMarginFraction:      "0.15",
			}
			cfg := getPercentileConfig(asr)
			Expect(cfg.TargetTflopsPercentile).To(Equal(0.8))
			Expect(cfg.LowerBoundTflopsPercentile).To(Equal(0.1))
			Expect(cfg.UpperBoundTflopsPercentile).To(Equal(0.95))
			Expect(cfg.TargetVramPercentile).To(Equal(0.7))
			Expect(cfg.LowerBoundVramPercentile).To(Equal(0.2))
			Expect(cfg.UpperBoundVramPercentile).To(Equal(0.9))
			Expect(cfg.RequestMarginFraction).To(Equal(0.15))
		})

		It("should ignore invalid float fields and keep defaults", func() {
			asr := &tfv1.AutoSetResources{
				TargetTflopsPercentile:     "not-a-float",
				LowerBoundTflopsPercentile: "",
				UpperBoundTflopsPercentile: "0.99",
			}
			cfg := getPercentileConfig(asr)
			Expect(cfg.TargetTflopsPercentile).To(Equal(defaultPercentileConfig.TargetTflopsPercentile))
			Expect(cfg.LowerBoundTflopsPercentile).To(Equal(defaultPercentileConfig.LowerBoundTflopsPercentile))
			Expect(cfg.UpperBoundTflopsPercentile).To(Equal(0.99))
		})

		It("should parse ConfidenceInterval if valid", func() {
			asr := &tfv1.AutoSetResources{
				ConfidenceInterval: "30m",
			}
			cfg := getPercentileConfig(asr)
			Expect(cfg.ConfidenceInterval).To(Equal(30 * time.Minute))
		})

		It("should ignore invalid ConfidenceInterval and keep default", func() {
			asr := &tfv1.AutoSetResources{
				ConfidenceInterval: "not-a-duration",
			}
			cfg := getPercentileConfig(asr)
			Expect(cfg.ConfidenceInterval).To(Equal(defaultPercentileConfig.ConfidenceInterval))
		})
	})
})

type fakeResourcesEstimator struct {
	*EstimatedResources
}

func (f *fakeResourcesEstimator) GetResourcesEstimation(workoad *workload.State) *EstimatedResources {
	return f.EstimatedResources
}
