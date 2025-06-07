package alert

import "fmt"

// offer API for managing user configured alert rules, stored in configMap
// offer mem synced rules for evaluation routine to use

type Rule struct {
	Name               string
	Query              string
	Threshold          float64
	EvaluationInterval string
	ConsecutiveCount   int
	Severity           string
	Summary            string
	Description        string
}

func (r Rule) String() string {
	return fmt.Sprintf("Rule{Name: %s, Query: %s, Threshold: %f, EvaluationInterval: %s, ConsecutiveCount: %d, Severity: %s}",
		r.Name, r.Query, r.Threshold, r.EvaluationInterval, r.ConsecutiveCount, r.Severity)
}

func (r Rule) ToPostableAlert(labels LabelSet, annotations LabelSet, isResolved bool) PostableAlert {
	return CreateAlert(r.Name, r.Summary, r.Description, labels, annotations, isResolved)
}
