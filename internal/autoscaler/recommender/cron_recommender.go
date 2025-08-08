package recommender

import (
	"context"
	"fmt"
	"maps"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/robfig/cron/v3"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Utilize these annotations to determine if the configuration has changed
const (
	CronScalingTFLOPSRequestAnnotation = constants.Domain + "/cron-scaling-tflops-request"
	CronScalingVRAMRequestAnnotation   = constants.Domain + "/cron-scaling-vram-request"
	CronScalingTFLOPSLimitAnnotation   = constants.Domain + "/cron-scaling-tflops-limit"
	CronScalingVRAMLimitAnnotation     = constants.Domain + "/cron-scaling-vram-limit"
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

func (c *CronRecommender) Recommend(ctx context.Context, w *workload.State) (*Recommendation, error) {
	log := log.FromContext(ctx)
	activeRule, err := c.getActiveCronScalingRule(&w.Spec.AutoScalingConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get active cron scaling rule %w", err)
	}

	curRes, err := cronScalingResourcesFromAnnotations(w.Annotations)
	if err != nil {
		return nil, fmt.Errorf("failed to get current resources from workload %s: %v", w.Name, err)
	}

	var targetRes *tfv1.Resources
	if activeRule == nil {
		if curRes == nil {
			return nil, nil
		}
		// revert the resources to those specified in the workload spec
		targetRes = w.GetResourcesSpec()
		maps.Copy(w.ScalingAnnotations, cronScalingResourcesToAnnotations(&tfv1.Resources{}))
		log.Info("cron scaling finished", "workload", w.Name, "resources", targetRes)
	} else {
		targetRes = &activeRule.DesiredResources
		maps.Copy(w.ScalingAnnotations, cronScalingResourcesToAnnotations(targetRes))
		log.Info("cron scaling rule matched", "workload", w.Name, "rule", activeRule.Name, "resources", targetRes)
	}

	return &Recommendation{
		Resources:        *targetRes,
		Applied:          curRes != nil && targetRes.Equal(curRes),
		ScaleDownLocking: true,
	}, nil
}

func cronScalingResourcesToAnnotations(resources *tfv1.Resources) map[string]string {
	return map[string]string{
		CronScalingTFLOPSRequestAnnotation: resources.Requests.Tflops.String(),
		CronScalingTFLOPSLimitAnnotation:   resources.Limits.Tflops.String(),
		CronScalingVRAMRequestAnnotation:   resources.Requests.Vram.String(),
		CronScalingVRAMLimitAnnotation:     resources.Limits.Vram.String(),
	}
}

func cronScalingResourcesFromAnnotations(annotations map[string]string) (*tfv1.Resources, error) {
	result := tfv1.Resources{}
	resInfo := []struct {
		key string
		dst *resource.Quantity
	}{
		{CronScalingTFLOPSRequestAnnotation, &result.Requests.Tflops},
		{CronScalingTFLOPSLimitAnnotation, &result.Limits.Tflops},
		{CronScalingVRAMRequestAnnotation, &result.Requests.Vram},
		{CronScalingVRAMLimitAnnotation, &result.Limits.Vram},
	}
	for _, info := range resInfo {
		annotation, ok := annotations[info.key]
		if !ok {
			continue
		}
		q, err := resource.ParseQuantity(annotation)
		if err != nil {
			return nil, fmt.Errorf("failed to parse %s: %v", info.key, err)
		}
		*info.dst = q
	}

	if result.IsZero() {
		return nil, nil
	}

	return &result, nil
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
