package recommender

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

var _ = Describe("CronRecommender", func() {
	It("should return recommended resource based on active cron scaling rule", func() {
		tflopsRequest := resource.MustParse("10")
		vramRequest := resource.MustParse("8Gi")
		tflopsLimit := resource.MustParse("20")
		vramLimit := resource.MustParse("16Gi")

		workload := workload.NewWorkloadState("test")
		workload.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{
				{
					Enable: true,
					Name:   "test",
					Start:  "0 0 * * *",
					End:    "59 23 * * *",
					DesiredResources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: tflopsRequest,
							Vram:   vramRequest,
						},
						Limits: tfv1.Resource{
							Tflops: tflopsLimit,
							Vram:   vramLimit,
						},
					},
				},
			},
		}

		recommender := NewCronRecommender()
		recommendation, _ := recommender.Recommend(workload)
		Expect(*recommendation).To(Equal(tfv1.RecommendedResources{
			LowerBoundTflops: tflopsRequest,
			TargetTflops:     tflopsRequest,
			UpperBoundTflops: tflopsRequest,
			LowerBoundVram:   vramRequest,
			TargetVram:       vramRequest,
			UpperBoundVram:   vramRequest,
		}))
	})

	FIt("should return recommended resource based on last resource annotations", func() {
		tflopsRequest := resource.MustParse("10")
		vramRequest := resource.MustParse("8Gi")
		// tflopsLimit := resource.MustParse("20")
		// vramLimit := resource.MustParse("16Gi")

		workload := workload.NewWorkloadState("test")
		workload.Annotations = map[string]string{
			constants.LastTFLOPSRequestAnnotation: tflopsRequest.String(),
			constants.LastVRAMRequestAnnotation:   vramRequest.String(),
			// constants.LastTFLOPSLimitAnnotation:   tflopsLimit.String(),
			// constants.LastVRAMLimitAnnotation:     vramLimit.String(),
		}
		workload.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{
				{
					Enable: true,
					Name:   "test",
					Start:  "",
					End:    "",
				},
			},
		}

		recommender := NewCronRecommender()
		recommendation, _ := recommender.Recommend(workload)
		Expect(recommendation.Equal(&tfv1.RecommendedResources{
			LowerBoundTflops: tflopsRequest,
			TargetTflops:     tflopsRequest,
			UpperBoundTflops: tflopsRequest,
			LowerBoundVram:   vramRequest,
			TargetVram:       vramRequest,
			UpperBoundVram:   vramRequest,
		})).To(BeTrue())
	})

	It("should return error if getting multiple active rules", func() {
		workload := workload.NewWorkloadState("test")
		workload.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{
				{
					Enable: true,
					Name:   "test",
					Start:  "0 0 * * *",
					End:    "59 23 * * *",
				},
				{
					Enable: true,
					Name:   "test",
					Start:  "0 0 * * *",
					End:    "59 23 * * *",
				},
			},
		}
		recommender := NewCronRecommender()
		_, err := recommender.Recommend(workload)
		Expect(err).To(HaveOccurred())
	})

	It("should not return cron scaling rule if no config or disable", func() {
		asc := tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{},
		}
		Expect(NewCronRecommender().getActiveCronScalingRule(&asc)).To(BeNil())
		asc = tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{
				{Enable: false},
			},
		}
		Expect(NewCronRecommender().getActiveCronScalingRule(&asc)).To(BeNil())
	})

	It("should return the active cron scaling rule if the current time falls within its scheduled interval", func() {
		asc := tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{
				{
					Enable: true,
					Name:   "test",
					Start:  "0 0 * * *",
					End:    "59 23 * * *",
				},
			},
		}
		rule, _ := NewCronRecommender().getActiveCronScalingRule(&asc)
		Expect(rule).NotTo(BeNil())
	})
})
