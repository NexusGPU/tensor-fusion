package recommender

import (
	"context"
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

var _ = Describe("CronRecommender", func() {
	ctx := context.TODO()
	recommender := NewCronRecommender(nil)

	Context("When an active rule is present", func() {
		var activeRule *tfv1.CronScalingRule
		var ws *workload.State
		BeforeEach(func() {
			ws = workload.NewWorkloadState()
			defaultRes := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("10"),
					Vram:   resource.MustParse("8Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("20"),
					Vram:   resource.MustParse("16Gi"),
				},
			}
			activeRule = &tfv1.CronScalingRule{
				Enable:           true,
				Name:             "test",
				Start:            "0 0 * * *",
				End:              "59 23 * * *",
				DesiredResources: defaultRes,
			}
			ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
				CronScalingRules: []tfv1.CronScalingRule{*activeRule},
			}
			verifyCronRecommendationStatus(ctx, recommender, ws, activeRule)
		})

		It("should return recommendation based on the newest active cron scaling rule", func() {
			newRes := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("5"),
					Vram:   resource.MustParse("4Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("10"),
					Vram:   resource.MustParse("8Gi"),
				},
			}

			activeRule.Name = "updatedName"
			activeRule.DesiredResources = newRes
			ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
				CronScalingRules: []tfv1.CronScalingRule{*activeRule},
			}

			verifyCronRecommendationStatus(ctx, recommender, ws, activeRule)
		})

		It("should return recResult with correct fields if the active cron scaling rule remains unchanged", func() {
			recResult, _ := recommender.Recommend(ctx, ws)
			Expect(recResult).ToNot(BeNil())
			Expect(recResult.HasApplied).To(BeTrue())
			Expect(recResult.ScaleDownLocking).To(BeTrue())
			Expect(recResult.Resources.Equal(&activeRule.DesiredResources)).To(BeTrue())
		})

		It("should revert the resources to those specified in the workload spec if the active cron scaling became inactive", func() {
			ws.Spec.Resources = tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("5"),
					Vram:   resource.MustParse("4Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("10"),
					Vram:   resource.MustParse("8Gi"),
				},
			}

			// invalid the activeRule
			activeRule.Enable = false
			ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
				CronScalingRules: []tfv1.CronScalingRule{*activeRule},
			}

			recResult, _ := recommender.Recommend(ctx, ws)
			Expect(recResult.Resources.Equal(&ws.Spec.Resources)).To(BeTrue())
			Expect(recResult.HasApplied).To(BeFalse())
			Expect(recResult.ScaleDownLocking).To(BeTrue())
			Expect(ws.Status.ActiveCronScalingRule).To(BeNil())
			condition := meta.FindStatusCondition(ws.Status.Conditions, constants.ConditionStatusTypeRecommendationProvided)
			expectedMsg := fmt.Sprintf("Cron scaling rule %q is inactive", activeRule.Name)
			Expect(condition.Message).To(Equal(expectedMsg))

			recResult, _ = recommender.Recommend(ctx, ws)
			Expect(recResult).To(BeNil())
		})

		It("should correctly apply recommendation processor", func() {
			newRes := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("200"),
					Vram:   resource.MustParse("200Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("400"),
					Vram:   resource.MustParse("400Gi"),
				},
			}
			activeRule = &tfv1.CronScalingRule{
				Enable:           true,
				Name:             "test",
				Start:            "0 0 * * *",
				End:              "59 23 * * *",
				DesiredResources: newRes,
			}
			ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
				CronScalingRules: []tfv1.CronScalingRule{*activeRule},
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
			recommender = NewCronRecommender(&fakeRecommendationProcessor{expectRes})
			recResult, _ := recommender.Recommend(ctx, ws)
			Expect(recResult.Resources.Equal(&expectRes)).To(BeTrue())
			condition := meta.FindStatusCondition(ws.Status.Conditions, constants.ConditionStatusTypeRecommendationProvided)
			expectedMsg := fmt.Sprintf("Cron scaling rule %q is active, fake message", activeRule.Name)
			Expect(condition.Message).To(Equal(expectedMsg))
		})
	})

	Context("When getting active rule", func() {
		ws := workload.NewWorkloadState()
		It("should return error if getting multiple active rules", func() {
			ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
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
			_, err := recommender.Recommend(ctx, ws)
			Expect(err).To(HaveOccurred())
		})

		It("should not return cron scaling rule if no config or disable", func() {
			asc := tfv1.AutoScalingConfig{
				CronScalingRules: []tfv1.CronScalingRule{},
			}
			Expect(recommender.getActiveCronScalingRule(&asc)).To(BeNil())
			asc = tfv1.AutoScalingConfig{
				CronScalingRules: []tfv1.CronScalingRule{
					{Enable: false},
				},
			}
			Expect(recommender.getActiveCronScalingRule(&asc)).To(BeNil())
		})

		It("should return the active cron scaling rule if the current time falls within its scheduled interval", func() {
			rec := NewCronRecommender(nil)
			weekDay := time.Now().Weekday().String()
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
			rule, _ := rec.getActiveCronScalingRule(&asc)
			Expect(rule).NotTo(BeNil())

			asc = tfv1.AutoScalingConfig{
				CronScalingRules: []tfv1.CronScalingRule{
					{
						Enable: true,
						Name:   "test",
						Start:  "0 0 * * Thu",   // Thursday at 00:00
						End:    "59 23 * * Thu", // Thursday at 23:59
					},
				},
			}
			rule, _ = rec.getActiveCronScalingRule(&asc)
			if weekDay == "Thursday" {
				Expect(rule).NotTo(BeNil())
			} else {
				Expect(rule).To(BeNil())
			}
		})

		It("should return error if the cron expressions is not valid or the same", func() {
			configs := []tfv1.AutoScalingConfig{
				{
					CronScalingRules: []tfv1.CronScalingRule{
						{
							Enable: true,
							Name:   "test",
							Start:  "-30 0 * * *", // invalid
							End:    "59 23 * * *",
						},
					},
				},
				{
					CronScalingRules: []tfv1.CronScalingRule{
						{
							Enable: true,
							Name:   "test",
							Start:  "30 0 * * *",
							End:    "59 -23 * * *", // invalid
						},
					},
				},
				{
					CronScalingRules: []tfv1.CronScalingRule{
						{
							Enable: true,
							Name:   "test",
							Start:  "30 0 * * *",
							End:    "59 23 * -3 *", // invalid
						},
					},
				},
				{
					CronScalingRules: []tfv1.CronScalingRule{
						{
							Enable: true,
							Name:   "test",
							Start:  "30 0 * * *", // same with end
							End:    "30 0 * * *",
						},
					},
				},
			}

			rec := NewCronRecommender(nil)
			for _, config := range configs {
				rule, err := rec.getActiveCronScalingRule(&config)
				Expect(err).To(HaveOccurred())
				Expect(rule).To(BeNil())
			}
		})
	})
})

func verifyCronRecommendationStatus(ctx context.Context, recommender *CronRecommender, w *workload.State, rule *tfv1.CronScalingRule) {
	GinkgoHelper()
	recommendation, _ := recommender.Recommend(ctx, w)
	if rule != nil {
		// verify resource of recommendation
		Expect(recommendation.Resources.Equal(&rule.DesiredResources)).To(BeTrue())
		Expect(recommendation.HasApplied).To(BeFalse())
		Expect(recommendation.ScaleDownLocking).To(BeTrue())
		// verify workload status
		Expect(equality.Semantic.DeepEqual(w.Status.ActiveCronScalingRule, rule)).To(BeTrue())
		condition := meta.FindStatusCondition(w.Status.Conditions, constants.ConditionStatusTypeRecommendationProvided)
		expectedMsg := fmt.Sprintf("Cron scaling rule %q is active", rule.Name)
		Expect(condition.Message).To(Equal(expectedMsg))
	} else {
		Expect(recommendation).To(BeNil())
		Expect(w.Status.ActiveCronScalingRule).To(BeNil())
		condition := meta.FindStatusCondition(w.Status.Conditions, constants.ConditionStatusTypeRecommendationProvided)
		Expect(condition).To(BeNil())
	}
}
