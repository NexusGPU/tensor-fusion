package recommender

import (
	"context"
	"fmt"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/robfig/cron/v3"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type CronRecommender struct {
	parser                  cron.Parser
	recommendationProcessor RecommendationProcessor
}

func NewCronRecommender(recommendationProcessor RecommendationProcessor) *CronRecommender {
	return &CronRecommender{
		parser:                  cron.NewParser(cron.Minute | cron.Hour | cron.Dom | cron.Month | cron.Dow),
		recommendationProcessor: recommendationProcessor,
	}
}

func (c *CronRecommender) Name() string {
	return "cron"
}

func (c *CronRecommender) Recommend(ctx context.Context, w *workload.State) (*RecResult, error) {
	activeRule, err := c.getActiveCronScalingRule(&w.Spec.AutoScalingConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get active cron scaling rule %w", err)
	}

	currentRule := w.Status.ActiveCronScalingRule

	if activeRule == nil && currentRule == nil {
		return nil, nil
	}

	var recommendation tfv1.Resources
	var reason, message string
	if activeRule == nil {
		// Revert the resources to those specified in the workload spec
		recommendation = *w.GetOriginalResourcesSpec()
		reason = "RuleInactive"
		message = fmt.Sprintf("Cron scaling rule %q is inactive", currentRule.Name)
		log.FromContext(ctx).Info("cron scaling rule inactive",
			"rule", currentRule.Name, "workload", w.Name, "resources", recommendation)
	} else {
		recommendation = activeRule.DesiredResources
		if currentRule == nil || !recommendation.Equal(&currentRule.DesiredResources) {
			reason = "RuleActive"
			message = fmt.Sprintf("Cron scaling rule %q is active", activeRule.Name)
			log.FromContext(ctx).Info("cron scaling rule active",
				"rule", activeRule.Name, "workload", w.Name, "resources", recommendation)
			if c.recommendationProcessor != nil {
				var err error
				var msg string
				recommendation, msg, err = c.recommendationProcessor.Apply(ctx, w, &recommendation)
				if err != nil {
					return nil, fmt.Errorf("failed to apply recommendation processor: %v", err)
				}
				if msg != "" {
					message += fmt.Sprintf(", %s", msg)
					log.FromContext(ctx).Info("recommendation processor applied", "message", message)
				}
			}
		}
	}

	if len(reason) > 0 {
		w.Status.ActiveCronScalingRule = activeRule.DeepCopy()
		meta.SetStatusCondition(&w.Status.Conditions, metav1.Condition{
			Type:               constants.ConditionStatusTypeRecommendationProvided,
			Status:             metav1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
			Reason:             reason,
			Message:            message,
		})
	}

	return &RecResult{
		Resources:        recommendation,
		HasApplied:       len(reason) == 0,
		ScaleDownLocking: true,
	}, nil
}

func (c *CronRecommender) getActiveCronScalingRule(config *tfv1.AutoScalingConfig) (*tfv1.CronScalingRule, error) {
	activeRules := []*tfv1.CronScalingRule{}

	currentTime := time.Now()

	for _, rule := range config.CronScalingRules {
		if !rule.Enable || rule.Name == "" ||
			rule.Start == "" || rule.End == "" {
			continue
		}

		if rule.Start == rule.End {
			return nil, fmt.Errorf("start and end can not same")
		}

		startSchedule, err := c.parser.Parse(rule.Start)
		if err != nil {
			return nil, fmt.Errorf("failed to parse cron rule %s start: %w", rule.Name, err)
		}
		endSchedule, err := c.parser.Parse(rule.End)
		if err != nil {
			return nil, fmt.Errorf("failed to parse cron rule %s end: %w", rule.Name, err)
		}

		nextStartTime := startSchedule.Next(currentTime)
		nextEndTime := endSchedule.Next(currentTime)

		isActive := false
		if nextStartTime.Before(nextEndTime) {
			isActive = currentTime.After(nextStartTime) && currentTime.Before(nextEndTime)
		} else {
			isActive = currentTime.After(nextStartTime) || currentTime.Before(nextEndTime)
		}

		if isActive {
			activeRules = append(activeRules, &rule)
		}
	}

	if len(activeRules) > 1 {
		return nil, fmt.Errorf("only one active cron scaling rule is permitted at any given time")
	}

	if len(activeRules) == 0 {
		return nil, nil
	}

	return activeRules[0], nil
}
