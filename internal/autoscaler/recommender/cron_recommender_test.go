package recommender

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
)

var _ = Describe("CronRecommender", func() {
	ctx := context.TODO()
	res := tfv1.Resources{
		Requests: tfv1.Resource{
			Tflops: resource.MustParse("10"),
			Vram:   resource.MustParse("8Gi"),
		},
		Limits: tfv1.Resource{
			Tflops: resource.MustParse("20"),
			Vram:   resource.MustParse("16Gi"),
		},
	}

	It("should return recommendation based on the active cron scaling rule", func() {
		workload := workload.NewWorkloadState("test")
		workload.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{
				{
					Enable:           true,
					Name:             "test",
					Start:            "0 0 * * *",
					End:              "59 23 * * *",
					DesiredResources: res,
				},
			},
		}

		recommender := NewCronRecommender(nil)
		recommendation, _ := recommender.Recommend(ctx, workload)
		Expect(recommendation.Equal(&res)).To(BeTrue())
	})

	It("should not return recommendation if there is no active cron scaling rule", func() {
		workload := workload.NewWorkloadState("test")
		workload.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{
				{
					Enable:           true,
					Name:             "test",
					Start:            "",
					End:              "",
					DesiredResources: res,
				},
			},
		}

		recommender := NewCronRecommender(nil)
		recommendation, _ := recommender.Recommend(ctx, workload)
		Expect(recommendation).To(BeNil())
	})

	It("should not return recommendation if the active cron scaling rule remains unchanged", func() {
		workload := workload.NewWorkloadState("test")
		workload.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{
				{
					Enable:           true,
					Name:             "test",
					Start:            "0 0 * * *",
					End:              "59 23 * * *",
					DesiredResources: res,
				},
			},
		}

		recommender := NewCronRecommender(nil)
		recommendation, _ := recommender.Recommend(ctx, workload)
		Expect(recommendation.Equal(&res)).To(BeTrue())

		workload.Annotations = cronScalingResourcesToAnnotations(&res)

		recommendation, _ = recommender.Recommend(ctx, workload)
		Expect(recommendation).To(BeNil())
	})

	It("should revert the resources to those specified in the workload spec if the active cron scaling finished", func() {
		workload := workload.NewWorkloadState("test")
		workload.Spec.Resources = tfv1.Resources{
			Requests: tfv1.Resource{
				Tflops: resource.MustParse("5"),
				Vram:   resource.MustParse("4Gi"),
			},
			Limits: tfv1.Resource{
				Tflops: resource.MustParse("10"),
				Vram:   resource.MustParse("8Gi"),
			},
		}
		workload.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{
				{
					Enable:           true,
					Name:             "test",
					Start:            "0 0 * * *",
					End:              "59 23 * * *",
					DesiredResources: res,
				},
			},
		}

		recommender := NewCronRecommender(nil)
		recommendation, _ := recommender.Recommend(ctx, workload)
		Expect(recommendation.Equal(&res)).To(BeTrue())

		workload.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{
				{
					Enable:           true,
					Name:             "test",
					Start:            "",
					End:              "",
					DesiredResources: res,
				},
			},
		}

		workload.Annotations = cronScalingResourcesToAnnotations(&res)
		recommendation, _ = recommender.Recommend(ctx, workload)
		Expect(recommendation.Equal(&workload.Spec.Resources)).To(BeTrue())

		workload.Annotations = cronScalingResourcesToAnnotations(&tfv1.Resources{})
		recommendation, _ = recommender.Recommend(ctx, workload)
		Expect(recommendation).To(BeNil())
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
		recommender := NewCronRecommender(nil)
		_, err := recommender.Recommend(ctx, workload)
		Expect(err).To(HaveOccurred())
	})

	It("should not return cron scaling rule if no config or disable", func() {
		asc := tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{},
		}
		Expect(NewCronRecommender(nil).getActiveCronScalingRule(&asc)).To(BeNil())
		asc = tfv1.AutoScalingConfig{
			CronScalingRules: []tfv1.CronScalingRule{
				{Enable: false},
			},
		}
		Expect(NewCronRecommender(nil).getActiveCronScalingRule(&asc)).To(BeNil())
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
		rule, _ := NewCronRecommender(nil).getActiveCronScalingRule(&asc)
		Expect(rule).NotTo(BeNil())
	})
})
