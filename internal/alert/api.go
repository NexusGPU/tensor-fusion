package alert

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/config"
)

var alertManagerHttpClient = &http.Client{
	Timeout: 10 * time.Second,
}

type LabelSet map[string]string

type Matcher struct {
	Name    string `json:"name"`
	Value   string `json:"value"`
	IsRegex bool   `json:"isRegex"`
	IsEqual bool   `json:"isEqual,omitempty"`
}

type Receiver struct {
	Name string `json:"name"`
}

type AlertStatus struct {
	State       string   `json:"state"`
	SilencedBy  []string `json:"silencedBy"`
	InhibitedBy []string `json:"inhibitedBy"`
	MutedBy     []string `json:"mutedBy"`
}

type GettableAlert struct {
	config.Alert
	Annotations LabelSet    `json:"annotations"`
	Receivers   []Receiver  `json:"receivers"`
	Fingerprint string      `json:"fingerprint"`
	StartsAt    time.Time   `json:"startsAt"`
	UpdatedAt   time.Time   `json:"updatedAt"`
	EndsAt      time.Time   `json:"endsAt"`
	Status      AlertStatus `json:"status"`
}

func SendAlert(ctx context.Context, alertManagerURL string, alerts []config.PostableAlert) error {
	if len(alerts) == 0 {
		return nil
	}
	if alertManagerURL[len(alertManagerURL)-1] != '/' {
		alertManagerURL += "/"
	}
	alertManagerURL += "api/v2/alerts"

	jsonData, err := json.Marshal(alerts)
	if err != nil {
		return fmt.Errorf("error marshaling alerts: %w", err)
	}

	req, err := http.NewRequest("POST", alertManagerURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("error creating alert request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := alertManagerHttpClient.Do(req)
	if err != nil {
		return fmt.Errorf("error sending alert request: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code when sending alert: %d", resp.StatusCode)
	}
	return nil
}
