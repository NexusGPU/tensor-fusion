package metrics

import (
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/config"
)

func TestEncoderRefactor(t *testing.T) {
	tests := []struct {
		name        string
		encoderType string
		expectError bool
	}{
		{
			name:        "influx encoder",
			encoderType: config.MetricsFormatInflux,
			expectError: false,
		},
		{
			name:        "json encoder",
			encoderType: config.MetricsFormatJson,
			expectError: false,
		},
		{
			name:        "otel encoder",
			encoderType: config.MetricsFormatOTel,
			expectError: false,
		},
		{
			name:        "unknown encoder defaults to json",
			encoderType: "unknown",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoder := NewEncoder(tt.encoderType)
			if encoder == nil {
				t.Fatal("encoder should not be nil")
			}

			// Test basic encoding operations
			encoder.StartLine("test_measurement")
			encoder.AddTag("tag1", "value1")
			encoder.AddField("field1", 42.0)
			encoder.EndLine(time.Now())

			// Check for errors
			if err := encoder.Err(); err != nil && !tt.expectError {
				t.Errorf("unexpected error: %v", err)
			}

			// Check that bytes are generated
			bytes := encoder.Bytes()
			if len(bytes) == 0 && !tt.expectError {
				t.Error("expected non-empty bytes output")
			}
		})
	}
}

func BenchmarkEncoderPerformance(b *testing.B) {
	encoders := map[string]string{
		"influx": config.MetricsFormatInflux,
		"json":   config.MetricsFormatJson,
		"otel":   config.MetricsFormatOTel,
	}

	for name, encoderType := range encoders {
		b.Run(name, func(b *testing.B) {
			encoder := NewEncoder(encoderType)
			timestamp := time.Now()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				encoder.StartLine("benchmark_measurement")
				encoder.AddTag("worker", "worker-1")
				encoder.AddTag("namespace", "default")
				encoder.AddField("cpu_usage", 75.5)
				encoder.AddField("memory_usage", int64(1024))
				encoder.EndLine(timestamp)
				_ = encoder.Bytes()
			}
		})
	}
}
