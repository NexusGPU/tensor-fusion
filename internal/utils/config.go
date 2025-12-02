package utils

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	constants "github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/lithammer/shortuuid/v4"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"
)

const (
	WatchConfigFileChangesInterval = 15 * time.Second

	ServiceAccountTokenPath = "/var/run/secrets/kubernetes.io/serviceaccount/token"
)

var selfServiceAccountName string

func InitServiceAccountConfig() {
	if os.Getenv("IMPERSONATE_SERVICE_ACCOUNT") != "" {
		selfServiceAccountName = os.Getenv("IMPERSONATE_SERVICE_ACCOUNT")
		ctrl.Log.Info("impersonate service account mode detected", "name", selfServiceAccountName)
		return
	}
	data, err := os.ReadFile(ServiceAccountTokenPath)
	if err != nil {
		ctrl.Log.Info("service account token not found, run outside of Kubernetes cluster")
		return
	}
	tokenParts := strings.Split(string(data), ".")
	if len(tokenParts) != 3 {
		ctrl.Log.Error(err, "failed to parse service account token")
		return
	}
	// resolve JWT token to get service account name
	decodedToken, err := base64.RawURLEncoding.DecodeString(tokenParts[1])
	if err != nil {
		ctrl.Log.Error(err, "failed to decode service account token")
		return
	}
	var jwt map[string]any
	if err := json.Unmarshal(decodedToken, &jwt); err != nil {
		ctrl.Log.Error(err, "failed to parse service account token")
		return
	}
	selfServiceAccountName = jwt["sub"].(string)
	ctrl.Log.Info("in-cluster mode detected, service account resolved", "name", selfServiceAccountName)
}

func LoadConfigFromFile[T any](filename string, target *T) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	return yaml.Unmarshal(data, target)
}

// WatchConfigFileChanges watches a file for changes and sends the file content through a channel when changes are detected.
// The channel will receive the raw file content as []byte whenever the file is modified.
// The watch interval is set to 15 seconds by default.
func WatchConfigFileChanges(ctx context.Context, filename string) (<-chan []byte, error) {
	ch := make(chan []byte, 1)
	var lastModTime time.Time

	if _, err := os.Stat(filename); err != nil {
		return nil, err
	}
	go func() {
		ticker := time.NewTicker(WatchConfigFileChangesInterval)
		defer ticker.Stop()
		defer close(ch)

		for {
			select {
			case <-ctx.Done():
				ctrl.Log.Info("stopping config file watcher", "filename", filename)
				return
			case <-ticker.C:
				lastModTime = checkFileUpdated(filename, lastModTime, ch)
			}
		}
	}()

	return ch, nil
}

func checkFileUpdated(filename string, lastModTime time.Time, ch chan []byte) time.Time {
	fileInfo, err := os.Stat(filename)
	if err != nil {
		ctrl.Log.Error(err, "unable to stat config file", "filename", filename)
		return lastModTime
	}

	currentModTime := fileInfo.ModTime()
	if currentModTime.After(lastModTime) {
		ctrl.Log.Info("load config", "filename", filename)

		data, err := os.ReadFile(filename)
		if err != nil {
			ctrl.Log.Error(err, "unable to read config file", "filename", filename)
			return lastModTime
		}

		ch <- data
		ctrl.Log.Info("config file loaded/reloaded", "filename", filename)
		return currentModTime
	}
	return lastModTime
}

func GetEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// PodWorkerInfo contains extracted worker information from pod annotations
type PodWorkerInfo struct {
	DeviceUUIDs       []string
	IsolationMode     string
	MemoryLimitBytes  uint64
	ComputeLimitUnits uint32
	TemplateID        string
}

// ExtractPodWorkerInfo extracts worker information from pod annotations
// This is a common utility function used by both GpuAllocator and PodCacheManager
func ExtractPodWorkerInfo(pod *corev1.Pod) PodWorkerInfo {
	info := PodWorkerInfo{}

	// Extract GPU device IDs
	if gpuIDsStr, exists := pod.Annotations[constants.GPUDeviceIDsAnnotation]; exists {
		ids := strings.Split(gpuIDsStr, ",")
		info.DeviceUUIDs = make([]string, 0, len(ids))
		for _, id := range ids {
			id = strings.TrimSpace(id)
			if id != "" {
				info.DeviceUUIDs = append(info.DeviceUUIDs, id)
			}
		}
	}

	// Extract isolation mode
	if isolationMode, exists := pod.Annotations[constants.IsolationModeAnnotation]; exists {
		info.IsolationMode = isolationMode
	} else {
		info.IsolationMode = string(tfv1.IsolationModeSoft) // default
	}

	// Extract memory limit (VRAM)
	if vramLimit, exists := pod.Annotations[constants.VRAMLimitAnnotation]; exists {
		if qty, err := resource.ParseQuantity(vramLimit); err == nil {
			info.MemoryLimitBytes = uint64(qty.Value())
		}
	}

	// Extract compute limit (compute percent)
	if computeLimit, exists := pod.Annotations[constants.ComputeLimitAnnotation]; exists {
		if qty, err := resource.ParseQuantity(computeLimit); err == nil {
			// Convert to percentage units (e.g., "50" -> 50, "100" -> 100)
			percent := qty.AsApproximateFloat64()
			info.ComputeLimitUnits = uint32(percent)
		}
	}

	// Extract template ID (for partitioned mode)
	// First check PartitionTemplateIDAnnotation (set by scheduler)
	if templateID, exists := pod.Annotations[constants.PartitionTemplateIDAnnotation]; exists {
		info.TemplateID = templateID
	} else if templateID, exists := pod.Annotations[constants.WorkloadProfileAnnotation]; exists {
		// Fallback to WorkloadProfileAnnotation
		info.TemplateID = templateID
	}

	return info
}

func GetGPUResource(pod *corev1.Pod, isRequest bool) (tfv1.Resource, error) {
	tflopsKey := constants.TFLOPSRequestAnnotation
	vramKey := constants.VRAMRequestAnnotation
	computePercentKey := constants.ComputeRequestAnnotation
	if !isRequest {
		tflopsKey = constants.TFLOPSLimitAnnotation
		vramKey = constants.VRAMLimitAnnotation
		computePercentKey = constants.ComputeLimitAnnotation
	}

	tflops, tflopsErr := resource.ParseQuantity(pod.Annotations[tflopsKey])
	computePercent, percentErr := resource.ParseQuantity(pod.Annotations[computePercentKey])
	if tflopsErr == nil && percentErr == nil {
		return tfv1.Resource{}, fmt.Errorf("tflops and compute-percent are mutually exclusive, please specify only one")
	} else if tflopsErr != nil && percentErr != nil {
		ctrl.Log.Info("failed to parse tflops and compute-percent, no computing limit/request set", "pod", pod.Name, "namespace", pod.Namespace)
	}

	vram, vramErr := resource.ParseQuantity(pod.Annotations[vramKey])
	if vramErr != nil {
		ctrl.Log.Info("failed to parse vram, annotation not found", "pod", pod.Name, "namespace", pod.Namespace, "annotation", vramKey)
	}

	return tfv1.Resource{
		Tflops:         tflops,
		Vram:           vram,
		ComputePercent: computePercent,
	}, nil
}

func NewShortID(length int) string {
	id := shortuuid.NewWithAlphabet(constants.ShortUUIDAlphabet)
	if length >= len(id) {
		return id
	}
	return id[:length]
}

func ReadServiceAccountToken() string {
	data, err := os.ReadFile(ServiceAccountTokenPath)
	if err != nil {
		return ""
	}
	return string(data)
}

func GetSelfServiceAccountNameFull() string {
	return selfServiceAccountName
}

func GetSelfServiceAccountNameShort() string {
	parts := strings.Split(selfServiceAccountName, ":")
	return parts[len(parts)-1]
}

var nvidiaOperatorProgressiveMigrationEnv = os.Getenv(constants.NvidiaOperatorProgressiveMigrationEnv) == constants.TrueStringValue

var isLicensedEnv = os.Getenv(constants.UsingCommercialComponentEnv) == constants.TrueStringValue

func init() {
	if isLicensedEnv {
		ctrl.Log.Info("Enabling none open source components, please make sure you are in trial stage or have bought commercial license. Contact us: support@tensor-fusion.com")
	}
}

func IsLicensed() bool {
	return isLicensedEnv
}

func IsProgressiveMigration() bool {
	return nvidiaOperatorProgressiveMigrationEnv
}

// For test purpose only
func SetProgressiveMigration(isProgressiveMigration bool) {
	nvidiaOperatorProgressiveMigrationEnv = isProgressiveMigration
}

// GetLeaderIP retrieves the leader IP from the ConfigMap
// This is used by both PortAllocator and IndexAllocator for distributed allocation
func GetLeaderIP(client client.Client) string {
	leaderInfo := &corev1.ConfigMap{}
	err := client.Get(context.Background(), types.NamespacedName{
		Name:      constants.LeaderInfoConfigMapName,
		Namespace: CurrentNamespace(),
	}, leaderInfo)
	if err != nil {
		ctrl.Log.V(5).Info("Failed to get leader IP info from ConfigMap", "error", err)
		return ""
	}
	if leaderInfo.Data == nil {
		return ""
	}
	return leaderInfo.Data[constants.LeaderInfoConfigMapLeaderIPKey]
}

// only for local development, won't set KUBECONFIG env var in none local environments
func NormalizeKubeConfigEnv() {
	cfgPath := os.Getenv("KUBECONFIG")
	if cfgPath != "" && strings.HasPrefix(cfgPath, "~") {
		home, err := os.UserHomeDir()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		_ = os.Setenv("KUBECONFIG", strings.Replace(cfgPath, "~", home, 1))
	}
}

func CleanUpExistingIndexAnnotationOnPod(pod *corev1.Pod) {
	for key := range pod.Annotations {
		if strings.HasPrefix(key, constants.PodIndexAnnotation) {
			delete(pod.Annotations, key)
		}
	}
}
