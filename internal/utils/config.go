package utils

import (
	context "context"
	"os"
	"time"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/yaml"
)

const (
	WatchConfigFileChangesInterval = 15 * time.Second
)

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
	ch := make(chan []byte)
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
				ctrl.Log.Info("stopping config file watcher for %s", filename)
				return
			case <-ticker.C:
				fileInfo, err := os.Stat(filename)
				if err != nil {
					ctrl.Log.Error(err, "unable to stat config file %s", filename)
					continue
				}

				currentModTime := fileInfo.ModTime()
				if currentModTime.After(lastModTime) {
					ctrl.Log.Info("config file %s modified, reloading", filename)

					data, err := os.ReadFile(filename)
					if err != nil {
						ctrl.Log.Error(err, "unable to read config file %s", filename)
						continue
					}

					ch <- data
					lastModTime = currentModTime
					ctrl.Log.Info("config file %s reloaded successfully", filename)
				}
			}
		}
	}()

	return ch, nil
}

func GetEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
