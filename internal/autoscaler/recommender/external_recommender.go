package recommender

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type ExternalRecommender struct {
	client                  client.Client
	recommendationProcessor RecommendationProcessor
	httpClient              *http.Client
}

func NewExternalRecommender(client client.Client, recommendationProcessor RecommendationProcessor) *ExternalRecommender {
	return &ExternalRecommender{
		client:                  client,
		recommendationProcessor: recommendationProcessor,
		httpClient:              &http.Client{Timeout: 10 * time.Second},
	}
}

func (e *ExternalRecommender) Name() string {
	return "external"
}

func (e *ExternalRecommender) Recommend(ctx context.Context, workloadState *workload.State) (*RecResult, error) {
	log := log.FromContext(ctx)
	config := workloadState.Spec.AutoScalingConfig.ExternalScaler

	if config == nil || !config.Enable {
		return nil, nil
	}

	// Check InitialDelayPeriod
	initialDelay := 30 * time.Minute
	if config.InitialDelayPeriod != "" {
		if d, parseErr := time.ParseDuration(config.InitialDelayPeriod); parseErr == nil {
			initialDelay = d
		} else {
			log.Error(parseErr, "failed to parse initial delay period, using default")
		}
	}

	timeSinceCreation := time.Since(workloadState.CreationTimestamp.Time)
	if timeSinceCreation < initialDelay {
		meta.SetStatusCondition(&workloadState.Status.Conditions, metav1.Condition{
			Type:               constants.ConditionStatusTypeResourceUpdate,
			Status:             metav1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
			Reason:             "LowConfidence",
			Message:            fmt.Sprintf("Workload created %v ago, less than InitialDelayPeriod %v, no update performed", timeSinceCreation, initialDelay),
		})
		return &RecResult{
			Resources:        tfv1.Resources{},
			HasApplied:       true,
			ScaleDownLocking: false,
		}, nil
	}

	// Prepare request
	curRes := workloadState.GetCurrentResourcesSpec()
	request := tfv1.ExternalScalerRequest{
		WorkloadName:     workloadState.Name,
		Namespace:        workloadState.Namespace,
		CurrentResources: *curRes,
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", config.URL, bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Add API key if configured
	if config.APIKeySecretRef != nil {
		apiKey, err := e.getAPIKey(ctx, config.APIKeySecretRef)
		if err != nil {
			return nil, fmt.Errorf("failed to get API key: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	// Send request
	resp, err := e.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Error(err, "failed to close response body")
		}
	}()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("external scaler returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var response tfv1.ExternalScalerResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// If no scaling needed, return nil
	if !response.NeedScaleUp && !response.NeedScaleDown {
		meta.SetStatusCondition(&workloadState.Status.Conditions, metav1.Condition{
			Type:               constants.ConditionStatusTypeResourceUpdate,
			Status:             metav1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
			Reason:             "NoScalingNeeded",
			Message:            response.Reason,
		})
		return &RecResult{
			Resources:        tfv1.Resources{},
			HasApplied:       true,
			ScaleDownLocking: false,
		}, nil
	}

	recommendation := response.RecommendedResources
	if recommendation.IsZero() {
		return nil, nil
	}

	// Apply recommendation processor
	if e.recommendationProcessor != nil {
		var err error
		var msg string
		recommendation, msg, err = e.recommendationProcessor.Apply(ctx, workloadState, &recommendation)
		if err != nil {
			return nil, fmt.Errorf("failed to apply recommendation processor: %v", err)
		}
		if msg != "" {
			log.Info("recommendation processor applied", "message", msg)
		}
	}

	hasApplied := recommendation.Equal(curRes)
	if !hasApplied {
		reason := "Updated"
		if response.Reason != "" {
			reason = response.Reason
		}
		meta.SetStatusCondition(&workloadState.Status.Conditions, metav1.Condition{
			Type:               constants.ConditionStatusTypeResourceUpdate,
			Status:             metav1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
			Reason:             reason,
			Message:            fmt.Sprintf("External scaler recommendation: %s", response.Reason),
		})
	}

	return &RecResult{
		Resources:        recommendation,
		HasApplied:       hasApplied,
		ScaleDownLocking: false,
	}, nil
}

func (e *ExternalRecommender) getAPIKey(ctx context.Context, secretRef *corev1.SecretReference) (string, error) {
	secret := &corev1.Secret{}
	key := client.ObjectKey{
		Namespace: secretRef.Namespace,
		Name:      secretRef.Name,
	}
	if err := e.client.Get(ctx, key, secret); err != nil {
		return "", fmt.Errorf("failed to get secret: %w", err)
	}

	// Look for common API key field names
	apiKeyFields := []string{"apiKey", "token", "key"}
	for _, field := range apiKeyFields {
		if val, ok := secret.Data[field]; ok {
			return string(val), nil
		}
	}

	return "", fmt.Errorf("API key not found in secret %s/%s", secretRef.Namespace, secretRef.Name)
}
