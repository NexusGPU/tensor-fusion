package utils

import (
	"math"
	"testing"
)

func TestTflopsToComputePercent(t *testing.T) {
	tests := []struct {
		name     string
		tflops   float64
		capacity float64
		want     uint32
		wantOK   bool
	}{
		{"40 of an RTX 3090 (71.2 FP16)", 40, 71.2, 57, true},
		{"40 of an RTX 3090 by the provider's FP32 estimate", 40, 44.08, 91, true},
		{"100 of an A100 (312 FP16)", 100, 312, 33, true},
		{"10 of an A100", 10, 312, 4, true},
		{"20 of a T4 (65 FP16)", 20, 65, 31, true},
		{"100 of an H100 (989 FP16)", 100, 989, 11, true},
		{"exact quarter", 78, 312, 25, true},
		{"exact half", 82.58, 165.16, 50, true},
		{"whole device", 312, 312, 100, true},
		{"more than the device", 400, 312, 100, true},
		{"just above a whole percent rounds up", 78.01, 312, 26, true},
		{"just below a whole percent rounds up to it", 77.99, 312, 25, true},
		{"tiny limit is never zero", 0.001, 989, 1, true},
		// 7/25*100 is 28.000000000000004 in floating point.
		{"whole percent despite floating-point noise", 7, 25, 28, true},
		{"whole percent despite floating-point noise, thirds", 0.3, 1, 30, true},
		{"zero limit", 0, 312, 0, false},
		{"negative limit", -5, 312, 0, false},
		{"unknown capacity", 40, 0, 0, false},
		{"negative capacity", 40, -1, 0, false},
		{"NaN limit", math.NaN(), 312, 0, false},
		{"NaN capacity", 40, math.NaN(), 0, false},
		{"infinite limit", math.Inf(1), 312, 0, false},
		{"infinite capacity", 40, math.Inf(1), 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := TflopsToComputePercent(tt.tflops, tt.capacity)
			if got != tt.want || ok != tt.wantOK {
				t.Fatalf("TflopsToComputePercent(%v, %v) = %d, %v; want %d, %v",
					tt.tflops, tt.capacity, got, ok, tt.want, tt.wantOK)
			}
		})
	}
}

// The percentage is the smallest whole share that covers the limit.
func TestTflopsToComputePercentCoversTheLimit(t *testing.T) {
	for _, capacity := range []float64{8.1408, 19.49184, 44.08, 65, 71.2, 121, 165.16, 312, 989} {
		for step := 1; step <= 400; step++ {
			tflops := capacity * float64(step) / 400
			percent, ok := TflopsToComputePercent(tflops, capacity)
			if !ok {
				t.Fatalf("no result for %v of %v", tflops, capacity)
			}
			granted := capacity * float64(percent) / 100
			if granted < tflops*(1-1e-9) {
				t.Fatalf("%d%% of %v grants %v, less than the limit %v", percent, capacity, granted, tflops)
			}
			if percent > 1 && capacity*float64(percent-1)/100 >= tflops*(1+1e-9) {
				t.Fatalf("%d%% of %v is more than needed for %v", percent, capacity, tflops)
			}
		}
	}
}
