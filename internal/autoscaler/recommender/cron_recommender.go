package recommender

import (
	"fmt"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/robfig/cron/v3"
	"k8s.io/apimachinery/pkg/api/resource"
)

type CronRecommender struct {
	parser cron.Parser
}

func NewCronRecommender() *CronRecommender {
	return &CronRecommender{
		parser: cron.NewParser(cron.Minute | cron.Hour | cron.Dom | cron.Month | cron.Dow),
	}
}

func (c *CronRecommender) Name() string {
	return "cron"
}

func (c *CronRecommender) Recommend(w *workload.State) (*tfv1.RecommendedResources, error) {
	activeRule, err := c.getActiveCronScalingRule(&w.Spec.AutoScalingConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get active cron scaling rule %w", err)
	}

	var tflopsRequest, vramRequest resource.Quantity
	if activeRule == nil {
		// if no active rule, return last resources if annotations exists
		resources, err := w.GetLastResourcesFromAnnotations()
		if err != nil {
			return nil, fmt.Errorf("failed to get last resources: %w", err)
		}
		// no annotations
		if resources == nil {
			return nil, nil
		}
		tflopsRequest = resources.Requests.Tflops
		vramRequest = resources.Requests.Vram
	} else {
		tflopsRequest = activeRule.DesiredResources.Requests.Tflops
		vramRequest = activeRule.DesiredResources.Requests.Vram
	}

	return &tfv1.RecommendedResources{
		LowerBoundTflops: tflopsRequest,
		TargetTflops:     tflopsRequest,
		UpperBoundTflops: tflopsRequest,
		LowerBoundVram:   vramRequest,
		TargetVram:       vramRequest,
		UpperBoundVram:   vramRequest,
	}, nil
}

func (c *CronRecommender) getActiveCronScalingRule(config *tfv1.AutoScalingConfig) (*tfv1.CronScalingRule, error) {
	activeRules := []*tfv1.CronScalingRule{}

	currentTime := time.Now()

	for _, rule := range config.CronScalingRules {
		if !rule.Enable || rule.Start == "" || rule.End == "" {
			continue
		}

		if rule.Start == rule.End {
			return nil, fmt.Errorf("start and end can not same")
		}

		startSchedule, err := c.parser.Parse(rule.Start)
		if err != nil {
			return nil, fmt.Errorf("failed to parse start: %w", err)
		}
		endSchedule, err := c.parser.Parse(rule.End)
		if err != nil {
			return nil, fmt.Errorf("failed to parse end: %w", err)
		}

		nextStartTime := startSchedule.Next(time.Now())
		nextEndTime := endSchedule.Next(time.Now())

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
