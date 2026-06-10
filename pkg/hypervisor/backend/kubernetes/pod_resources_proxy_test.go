/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package kubernetes

import (
	"reflect"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestNormalizeNVMLUUID(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		// Lowercased TF kubernetes-name form must be uppercased so DCGM's
		// string match against NVML output succeeds.
		{"gpu-f5d00867-cc01-0631-8d51-14394632047c", "GPU-f5d00867-cc01-0631-8d51-14394632047c"},
		// Already canonical — passthrough.
		{"GPU-f5d00867-cc01-0631-8d51-14394632047c", "GPU-f5d00867-cc01-0631-8d51-14394632047c"},
		// MIG instance — DCGM already expects this form, do not touch.
		{"MIG-ab12cd34-5678-90ef-1234-567890abcdef", "MIG-ab12cd34-5678-90ef-1234-567890abcdef"},
		// Empty — keep empty (filterEmpty drops it upstream).
		{"", ""},
		// Unknown prefix — leave unchanged rather than mangling.
		{"weird-id", "weird-id"},
	}
	for _, c := range cases {
		got := normalizeNVMLUUID(c.in)
		if got != c.want {
			t.Errorf("normalizeNVMLUUID(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestIsTFDevicePluginResource(t *testing.T) {
	prefix := constants.PodIndexAnnotation + constants.PodIndexDelimiter
	cases := []struct {
		name string
		want bool
	}{
		{prefix + "0", true},
		{prefix + "a", true},
		{prefix + "f", true},
		{"nvidia.com/gpu", false},
		{"cpu", false},
		{constants.PodIndexAnnotation, false}, // missing delimiter
		{"", false},
	}
	for _, c := range cases {
		if got := isTFDevicePluginResource(c.name); got != c.want {
			t.Errorf("isTFDevicePluginResource(%q) = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestParsePartitionUUIDs(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []string
	}{
		{
			name: "single MIG instance",
			in:   "MIG-aaa:GPU-parent",
			want: []string{"MIG-aaa"},
		},
		{
			name: "multiple MIG instances",
			in:   "MIG-aaa:GPU-parent1,MIG-bbb:GPU-parent2",
			want: []string{"MIG-aaa", "MIG-bbb"},
		},
		{
			name: "tolerates whitespace and empty segments",
			in:   " MIG-aaa:GPU-x , , MIG-bbb:GPU-y ",
			want: []string{"MIG-aaa", "MIG-bbb"},
		},
		{
			name: "missing parent — still emits the partition uuid",
			in:   "MIG-aaa",
			want: []string{"MIG-aaa"},
		},
		{
			name: "empty input",
			in:   "",
			want: []string{},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := parsePartitionUUIDs(c.in)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("parsePartitionUUIDs(%q) = %v, want %v", c.in, got, c.want)
			}
		})
	}
}

func TestFilterEmpty(t *testing.T) {
	got := filterEmpty([]string{"a", "", " ", "b", "  c  "})
	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("filterEmpty = %v, want %v", got, want)
	}
}

func TestContainerGPUIDs_Precedence(t *testing.T) {
	const (
		uuidA = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
		uuidB = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
		mig1  = "MIG-11111111-1111-1111-1111-111111111111"
	)
	cases := []struct {
		name      string
		ann       map[string]string
		container string
		want      []string
	}{
		{
			name: "container-gpus wins over partition-uuids and gpu-ids",
			ann: map[string]string{
				constants.ContainerGPUsAnnotation:  `{"worker":["` + uuidA + `"]}`,
				constants.PartitionUUIDsAnnotation: mig1 + ":GPU-parent",
				constants.GPUDeviceIDsAnnotation:   uuidB,
			},
			container: "worker",
			want:      []string{uuidA},
		},
		{
			name: "container-gpus map miss falls through to partition-uuids",
			ann: map[string]string{
				constants.ContainerGPUsAnnotation:  `{"other":["` + uuidA + `"]}`,
				constants.PartitionUUIDsAnnotation: mig1 + ":GPU-parent",
			},
			container: "worker",
			want:      []string{mig1},
		},
		{
			name: "partition-uuids wins over gpu-ids",
			ann: map[string]string{
				constants.PartitionUUIDsAnnotation: mig1 + ":GPU-parent",
				constants.GPUDeviceIDsAnnotation:   uuidB,
			},
			container: "worker",
			want:      []string{mig1},
		},
		{
			name: "gpu-ids fallback is comma-split and trimmed",
			ann: map[string]string{
				constants.GPUDeviceIDsAnnotation: " " + uuidA + " ,  " + uuidB + " ",
			},
			container: "worker",
			want:      []string{uuidA, uuidB},
		},
		{
			name:      "no annotations returns nil",
			ann:       map[string]string{},
			container: "worker",
			want:      nil,
		},
		{
			name: "malformed container-gpus JSON falls through",
			ann: map[string]string{
				constants.ContainerGPUsAnnotation: "not-json",
				constants.GPUDeviceIDsAnnotation:  uuidA,
			},
			container: "worker",
			want:      []string{uuidA},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "p",
					Namespace:   "ns",
					Annotations: c.ann,
				},
			}
			got := containerGPUIDs(pod, c.container)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("containerGPUIDs = %v, want %v", got, c.want)
			}
		})
	}
}

func TestContainerGPUIDs_NilPod(t *testing.T) {
	if got := containerGPUIDs(nil, "worker"); got != nil {
		t.Errorf("containerGPUIDs(nil) = %v, want nil", got)
	}
	if got := containerGPUIDs(&corev1.Pod{}, "worker"); got != nil {
		t.Errorf("containerGPUIDs(no annotations) = %v, want nil", got)
	}
}

func TestMetricsExporterResourceNamePerVendor(t *testing.T) {
	tests := []struct {
		vendor string
		want   string
	}{
		{constants.AcceleratorVendorNvidia, "nvidia.com/gpu"},
		{constants.AcceleratorVendorMThreads, "mthreads.com/vgpu"},
		{constants.AcceleratorVendorHuaweiAscendNPU, "huawei.com/npu"},
		{constants.AcceleratorVendorAlibabaPPU, "aliyun.com/ppu"},
		{"", "nvidia.com/gpu"},        // undeclared vendor: backward compat
		{"Unknown", "nvidia.com/gpu"}, // unenumerated vendor: backward compat
	}
	for _, tt := range tests {
		if got := constants.GetMetricsExporterResourceName(tt.vendor); got != tt.want {
			t.Errorf("GetMetricsExporterResourceName(%q) = %q, want %q", tt.vendor, got, tt.want)
		}
	}
}
