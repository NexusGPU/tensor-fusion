/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"testing"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

const (
	// DefaultConcurrency is the default number of concurrent goroutines for parallel benchmarks
	DefaultConcurrency = 10
)

// createAdmissionRequest creates an AdmissionRequest for the given pod
func createAdmissionRequest(pod *corev1.Pod) *admissionv1.AdmissionRequest {
	podBytes, _ := json.Marshal(pod)

	return &admissionv1.AdmissionRequest{
		UID: types.UID("test-uid"),
		Kind: metav1.GroupVersionKind{
			Group:   "",
			Version: "v1",
			Kind:    "Pod",
		},
		Resource: metav1.GroupVersionResource{
			Group:    "",
			Version:  "v1",
			Resource: "pods",
		},
		Name:      pod.Name,
		Namespace: pod.Namespace,
		Operation: admissionv1.Create,
		Object: runtime.RawExtension{
			Raw: podBytes,
		},
	}
}

// makeWebhookRequestWithVerification makes an HTTP request with pre-marshaled data and verifies response
func makeWebhookRequestWithVerification(client *http.Client, serverURL string, reqBytes []byte) error {
	resp, err := client.Post(serverURL, "application/json", bytes.NewBuffer(reqBytes))
	if err != nil {
		return err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	// Only verify response status code
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return nil
}

// BenchmarkPodWebhookQPS measures QPS for each code path via HTTP requests
func BenchmarkPodWebhookQPS(b *testing.B) {
	// Setup test environment with proper webhook framework
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = tfv1.AddToScheme(scheme)

	// Create test GPU pool
	gpuPool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-pool",
		},
		Spec: tfv1.GPUPoolSpec{
			ComponentConfig: &tfv1.ComponentConfig{
				Client: &tfv1.ClientConfig{
					OperatorEndpoint: "http://localhost:8080",
					PatchToPod:       &runtime.RawExtension{Raw: []byte("{}")},
					PatchToContainer: &runtime.RawExtension{Raw: []byte("{}")},
				},
			},
		},
	}

	// Create test workload
	workload := &tfv1.TensorFusionWorkload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-workload",
			Namespace: "default",
		},
		Spec: tfv1.WorkloadProfileSpec{
			PoolName: "test-pool",
			Resources: tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("1"),
					Vram:   resource.MustParse("1Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("2"),
					Vram:   resource.MustParse("2Gi"),
				},
			},
			GPUCount:   1,
			IsLocalGPU: false,
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(gpuPool, workload).
		Build()

	portAllocator := &portallocator.PortAllocator{
		IsLeader: true,
	}

	mutator := &TensorFusionPodMutator{
		Client:        fakeClient,
		decoder:       admission.NewDecoder(scheme),
		portAllocator: portAllocator,
	}

	// Create HTTP server using controller-runtime webhook framework
	mux := http.NewServeMux()
	mux.Handle("/mutate-v1-pod", &webhook.Admission{Handler: mutator})

	listener, err := net.Listen("tcp", ":0")
	if err != nil {
		b.Fatalf("Failed to create listener: %v", err)
	}
	defer func() {
		_ = listener.Close()
	}()

	server := &http.Server{
		Handler: mux,
		// Add timeouts to prevent connection buildup
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
		IdleTimeout:  30 * time.Second,
	}

	// Start server in goroutine
	serverErr := make(chan error, 1)
	go func() {
		if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
			serverErr <- err
		}
	}()

	// Wait for server to be ready and check for startup errors
	time.Sleep(100 * time.Millisecond)
	select {
	case err := <-serverErr:
		b.Fatalf("Server failed to start: %v", err)
	default:
	}

	serverURL := fmt.Sprintf("http://%s/mutate-v1-pod", listener.Addr().String())

	// Ensure proper cleanup
	defer func() {
		_ = server.Close()
	}()

	testCases := []struct {
		name     string
		setupFn  func() *corev1.Pod
		setupEnv func() func() // returns cleanup function
	}{
		{
			name: "NonTensorFusionPod_NoGPU_HTTP_QPS",
			setupFn: func() *corev1.Pod {
				return &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-pod",
						Namespace: "default",
					},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{
								Name:  "test-container",
								Image: "nginx",
							},
						},
					},
				}
			},
			setupEnv: func() func() { return func() {} },
		},
		{
			name: "NonTensorFusionPod_WithGPU_ProgressiveMigration_HTTP_QPS",
			setupFn: func() *corev1.Pod {
				return &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-pod-gpu",
						Namespace: "default",
					},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{
								Name:  "test-container",
								Image: "nvidia/cuda:11.8-runtime-ubuntu20.04",
								Resources: corev1.ResourceRequirements{
									Requests: corev1.ResourceList{
										"nvidia.com/gpu": resource.MustParse("1"),
									},
								},
							},
						},
					},
				}
			},
			setupEnv: func() func() {
				originalValue := utils.IsProgressiveMigration()
				utils.SetProgressiveMigration(true)
				return func() { utils.SetProgressiveMigration(originalValue) }
			},
		},
		{
			name: "FullTensorFusionPod_HTTP_QPS",
			setupFn: func() *corev1.Pod {
				return &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-tf-pod",
						Namespace: "default",
						Labels: map[string]string{
							constants.TensorFusionEnabledLabelKey: constants.TrueStringValue,
						},
						Annotations: map[string]string{
							constants.GpuPoolKey:                "test-pool",
							constants.TFLOPSRequestAnnotation:   "1",
							constants.VRAMRequestAnnotation:     "1Gi",
							constants.TFLOPSLimitAnnotation:     "2",
							constants.VRAMLimitAnnotation:       "2Gi",
							constants.InjectContainerAnnotation: "test-container",
						},
					},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{
								Name:  "test-container",
								Image: "nvidia/cuda:11.8-runtime-ubuntu20.04",
								Resources: corev1.ResourceRequirements{
									Requests: corev1.ResourceList{
										"nvidia.com/gpu": resource.MustParse("1"),
									},
								},
							},
						},
					},
				}
			},
			setupEnv: func() func() { return func() {} },
		},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			cleanup := tc.setupEnv()
			defer cleanup()

			// Pre-create HTTP client with connection pooling to avoid overhead
			client := &http.Client{
				Timeout: 5 * time.Second,
				Transport: &http.Transport{
					MaxIdleConns:        100,
					MaxIdleConnsPerHost: 100,
					IdleConnTimeout:     90 * time.Second,
				},
			}

			// Create proper admission request for the pod
			pod := tc.setupFn()
			admissionRequest := createAdmissionRequest(pod)

			// Pre-marshal admission review outside parallel section
			admissionReview := &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "admission.k8s.io/v1",
					Kind:       "AdmissionReview",
				},
				Request: admissionRequest,
			}
			reqBytes, err := json.Marshal(admissionReview)
			if err != nil {
				b.Fatalf("Failed to marshal admission review: %v", err)
			}

			b.ResetTimer()

			// Run sequential requests to avoid connection exhaustion
			for b.Loop() {
				if err := makeWebhookRequestWithVerification(client, serverURL, reqBytes); err != nil {
					b.Errorf("Request failed: %v", err)
				}
			}
		})
	}
}
