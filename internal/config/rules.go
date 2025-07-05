package config

import (
	"bytes"
	"fmt"
	"hash/fnv"
	"sort"
	"text/template"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

// offer API for managing user configured alert rules, stored in configMap
// offer mem synced rules for evaluation routine to use

type AlertRule struct {
	Name                string  `yaml:"name"`
	Query               string  `yaml:"query"`
	Threshold           float64 `yaml:"threshold"`
	EvaluationInterval  string  `yaml:"evaluationInterval"`
	ConsecutiveCount    int     `yaml:"consecutiveCount"`
	Severity            string  `yaml:"severity"`
	Summary             string  `yaml:"summary"`
	Description         string  `yaml:"description"`
	RunBookURL          string  `yaml:"runbookURL"`
	AlertTargetInstance string  `yaml:"alertTargetInstance"`

	summaryTmplParsed     *template.Template
	descriptionTmplParsed *template.Template
	instanceTmplParsed    *template.Template

	// when the rule is in test mode, it will not send alerts to alert manager
	TestMode bool

	FiringAlerts map[string]*FiringAlertCache
}

type FiringAlertCache struct {
	Alert PostableAlert
	Count int
}

type Alert struct {
	Labels       LabelSet `json:"labels"`
	GeneratorURL string   `json:"generatorURL,omitempty"`
}

type LabelSet map[string]string

type PostableAlert struct {
	Alert
	StartsAt    time.Time `json:"startsAt,omitempty"`
	EndsAt      time.Time `json:"endsAt,omitempty"`
	Annotations LabelSet  `json:"annotations,omitempty"`
}

func (r *AlertRule) String() string {
	return fmt.Sprintf("Rule{Name: %s, Query: %s, Threshold: %f, EvaluationInterval: %s, ConsecutiveCount: %d, Severity: %s}",
		r.Name, r.Query, r.Threshold, r.EvaluationInterval, r.ConsecutiveCount, r.Severity)
}

func (r *AlertRule) AddFiringAlertAndCheckResolved(alertQueryResult map[string]interface{}) (*PostableAlert, bool, string) {
	if r.FiringAlerts == nil {
		r.FiringAlerts = make(map[string]*FiringAlertCache)
	}

	alert := r.toPostableAlert(alertQueryResult, time.Now(), false)

	// calculate hash based on labels as fingerprint, for counting consecutive alerts
	hasher := fnv.New64a()
	labels := make([]string, 0, len(alert.Labels))
	for label := range alert.Labels {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	for _, label := range labels {
		hasher.Write([]byte(label))
		hasher.Write([]byte(alert.Labels[label]))
	}
	alertHash := fmt.Sprintf("%x", hasher.Sum(nil))

	if _, ok := r.FiringAlerts[alertHash]; !ok {
		r.FiringAlerts[alertHash] = &FiringAlertCache{alert, 0}
	}
	r.FiringAlerts[alertHash].Count++
	if r.FiringAlerts[alertHash].Count >= r.ConsecutiveCount {
		// should fire alert
		return &r.FiringAlerts[alertHash].Alert, true, alertHash
	}
	return &r.FiringAlerts[alertHash].Alert, false, alertHash
}

func (r *AlertRule) CheckAndRemoveFiringAlerts(firingAlertSet map[string]struct{}) []PostableAlert {
	if r.FiringAlerts == nil {
		return nil
	}

	alerts := []PostableAlert{}
	for alertHash, alertVal := range r.FiringAlerts {
		if _, ok := firingAlertSet[alertHash]; ok {
			continue
		}

		if alertVal.Count < r.ConsecutiveCount {
			// still not enough consecutive alerts
			// delete it and do not send resolved alert
			delete(r.FiringAlerts, alertHash)
			continue
		}

		resolvedAlert := alertVal.Alert
		resolvedAlert.EndsAt = time.Now()
		alerts = append(alerts, resolvedAlert)
		delete(r.FiringAlerts, alertHash)
	}
	return alerts

}

func (r *AlertRule) IsTestMode() bool {
	return r.TestMode
}

func (r *AlertRule) toPostableAlert(alertQueryResult map[string]interface{}, startsAt time.Time, isResolved bool) PostableAlert {
	summary, description, instance, err := r.renderAlertContentTemplate(alertQueryResult)

	if err != nil {
		return PostableAlert{}
	}
	// MUST be unique, otherwise alert manager's fingerprint mechanism will not work
	labels := LabelSet{
		"alertname": r.Name,
		"severity":  r.Severity,
		"job":       constants.AlertJobName,
		"instance":  instance,
	}
	annotations := LabelSet{
		"summary":     summary,
		"description": description,
		"runbook_url": r.RunBookURL,
	}
	alert := CreateAlertData(r.Name, summary, description, labels, annotations, startsAt)
	if isResolved {
		alert.EndsAt = time.Now()
	}
	return alert
}

func (rule *AlertRule) renderAlertContentTemplate(data interface{}) (string, string, string, error) {
	if rule.summaryTmplParsed == nil {
		summaryTmplParsed, err := template.New("summary").Parse(rule.Summary)
		rule.summaryTmplParsed = summaryTmplParsed
		if err != nil {
			return "", "", "", fmt.Errorf("failed to parse summary template: %w", err)
		}
	}
	if rule.descriptionTmplParsed == nil {
		descriptionTmplParsed, err := template.New("description").Parse(rule.Description)
		rule.descriptionTmplParsed = descriptionTmplParsed
		if err != nil {
			return "", "", "", fmt.Errorf("failed to parse description template: %w", err)
		}
	}
	if rule.instanceTmplParsed == nil {
		instanceTmplParsed, err := template.New("instance").Parse(rule.AlertTargetInstance)
		rule.instanceTmplParsed = instanceTmplParsed
		if err != nil {
			return "", "", "", fmt.Errorf("failed to parse instance template: %w", err)
		}
	}
	var summaryBuf bytes.Buffer
	if err := rule.summaryTmplParsed.Execute(&summaryBuf, data); err != nil {
		return "", "", "", fmt.Errorf("failed to execute summary template: %w", err)
	}
	var descriptionBuf bytes.Buffer
	if err := rule.descriptionTmplParsed.Execute(&descriptionBuf, data); err != nil {
		return "", "", "", fmt.Errorf("failed to execute description template: %w", err)
	}

	var instanceBuf bytes.Buffer
	if err := rule.instanceTmplParsed.Execute(&instanceBuf, data); err != nil {
		return "", "", "", fmt.Errorf("failed to execute instance template: %w", err)
	}

	return summaryBuf.String(), descriptionBuf.String(), instanceBuf.String(), nil
}

func CreateAlertData(name, summary, description string, labels LabelSet, annotations LabelSet, startsAt time.Time) PostableAlert {
	return PostableAlert{
		Alert: Alert{
			Labels: labels,
		},
		StartsAt:    startsAt,
		Annotations: annotations,
	}
}
