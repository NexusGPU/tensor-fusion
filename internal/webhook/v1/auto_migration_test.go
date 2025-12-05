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
	"context"

	"github.com/NexusGPU/tensor-fusion/internal/config"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var _ = Describe("AutoMigrationRules", func() {
	var (
		ctx            context.Context
		pod            *corev1.Pod
		namespace      *corev1.Namespace
		originalConfig *config.GlobalConfig
	)

	// Create namespace once before all tests
	BeforeEach(func() {
		ctx = context.Background()
		originalConfig = config.GetGlobalConfig()

		// Mock pod - no k8s creation
		pod = &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-pod",
				Namespace: "auto-migrate",
				Labels: map[string]string{
					"app": "test",
				},
			},
		}

		// Create namespace once
		namespace = &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: "auto-migrate",
				Labels: map[string]string{
					"environment": "production",
				},
			},
		}
		err := k8sClient.Create(ctx, namespace)
		if err != nil && !errors.IsAlreadyExists(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	})

	// Delete namespace once after all tests
	AfterEach(func() {
		config.SetGlobalConfig(originalConfig)
		if namespace != nil {
			_ = k8sClient.Delete(ctx, namespace)
		}
	})

	It("should return false when config is nil", func() {
		config.SetGlobalConfig(nil)
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeFalse())
	})

	It("should return false when AutoMigration is nil", func() {
		config.SetGlobalConfig(&config.GlobalConfig{})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeFalse())
	})

	It("should return false when enable is false", func() {
		config.SetGlobalConfig(&config.GlobalConfig{
			AutoMigration: &config.AutoMigrationConfig{
				Enable: false,
			},
		})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeFalse())
	})

	It("should return true when enable is true and scope is nil", func() {
		config.SetGlobalConfig(&config.GlobalConfig{
			AutoMigration: &config.AutoMigrationConfig{
				Enable: true,
			},
		})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeTrue())
	})

	It("should return true when pod namespace matches includes namespaceNames", func() {
		config.SetGlobalConfig(&config.GlobalConfig{
			AutoMigration: &config.AutoMigrationConfig{
				Enable: true,
				Scope: &config.AutoMigrationScope{
					Includes: &config.AutoMigrationRules{
						NamespaceNames: []string{"auto-migrate", "kube-system"},
					},
				},
			},
		})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeTrue())
	})

	It("should return false when pod namespace does not match includes namespaceNames", func() {
		config.SetGlobalConfig(&config.GlobalConfig{
			AutoMigration: &config.AutoMigrationConfig{
				Enable: true,
				Scope: &config.AutoMigrationScope{
					Includes: &config.AutoMigrationRules{
						NamespaceNames: []string{"kube-system"},
					},
				},
			},
		})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeFalse())
	})

	It("should return true when pod matches includes namespaceSelector", func() {
		config.SetGlobalConfig(&config.GlobalConfig{
			AutoMigration: &config.AutoMigrationConfig{
				Enable: true,
				Scope: &config.AutoMigrationScope{
					Includes: &config.AutoMigrationRules{
						NamespaceSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"environment": "production",
							},
						},
					},
				},
			},
		})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeTrue())
	})

	It("should return true when pod matches includes podSelector", func() {
		config.SetGlobalConfig(&config.GlobalConfig{
			AutoMigration: &config.AutoMigrationConfig{
				Enable: true,
				Scope: &config.AutoMigrationScope{
					Includes: &config.AutoMigrationRules{
						PodSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"app": "test",
							},
						},
					},
				},
			},
		})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeTrue())
	})

	It("should return false when pod does not match includes podSelector", func() {
		config.SetGlobalConfig(&config.GlobalConfig{
			AutoMigration: &config.AutoMigrationConfig{
				Enable: true,
				Scope: &config.AutoMigrationScope{
					Includes: &config.AutoMigrationRules{
						PodSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"app": "other",
							},
						},
					},
				},
			},
		})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeFalse())
	})

	It("should return false when pod matches excludes", func() {
		config.SetGlobalConfig(&config.GlobalConfig{
			AutoMigration: &config.AutoMigrationConfig{
				Enable: true,
				Scope: &config.AutoMigrationScope{
					Excludes: &config.AutoMigrationRules{
						NamespaceNames: []string{"auto-migrate"},
					},
				},
			},
		})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeFalse())
	})

	It("should return true when pod matches includes but not excludes", func() {
		config.SetGlobalConfig(&config.GlobalConfig{
			AutoMigration: &config.AutoMigrationConfig{
				Enable: true,
				Scope: &config.AutoMigrationScope{
					Includes: &config.AutoMigrationRules{
						NamespaceNames: []string{"auto-migrate"},
					},
					Excludes: &config.AutoMigrationRules{
						PodSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"app": "excluded",
							},
						},
					},
				},
			},
		})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeTrue())
	})

	It("should return false when pod matches both includes and excludes", func() {
		config.SetGlobalConfig(&config.GlobalConfig{
			AutoMigration: &config.AutoMigrationConfig{
				Enable: true,
				Scope: &config.AutoMigrationScope{
					Includes: &config.AutoMigrationRules{
						NamespaceNames: []string{"auto-migrate"},
					},
					Excludes: &config.AutoMigrationRules{
						PodSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"app": "test",
							},
						},
					},
				},
			},
		})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeFalse())
	})

	It("should return true when excludes is set but pod does not match", func() {
		config.SetGlobalConfig(&config.GlobalConfig{
			AutoMigration: &config.AutoMigrationConfig{
				Enable: true,
				Scope: &config.AutoMigrationScope{
					Excludes: &config.AutoMigrationRules{
						NamespaceNames: []string{"kube-system"},
					},
				},
			},
		})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(BeTrue())
	})

	It("should return error when namespace not found for namespaceSelector", func() {
		pod.Namespace = "non-existent"
		config.SetGlobalConfig(&config.GlobalConfig{
			AutoMigration: &config.AutoMigrationConfig{
				Enable: true,
				Scope: &config.AutoMigrationScope{
					Includes: &config.AutoMigrationRules{
						NamespaceSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"environment": "production",
							},
						},
					},
				},
			},
		})
		result, err := ShouldAutoMigrateGPUPod(ctx, k8sClient, pod)
		Expect(err).To(HaveOccurred())
		Expect(result).To(BeFalse())
		Expect(err.Error()).To(ContainSubstring("failed to get namespace"))
	})
})
