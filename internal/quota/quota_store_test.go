package quota

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// Test constants
const (
	TestNamespace = "test-ns"
	TestPoolName  = "test-pool"
	TestWorkload  = "test-workload"
	TestQuotaName = "test-quota"
)

func createTestQuota(tflops, vram int64, workers int32) *tfv1.GPUResourceQuota {
	return &tfv1.GPUResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      TestQuotaName,
			Namespace: TestNamespace,
		},
		Spec: tfv1.GPUResourceQuotaSpec{
			Total: tfv1.GPUResourceQuotaTotal{
				Requests: &tfv1.Resource{
					Tflops: *resource.NewQuantity(tflops, resource.DecimalSI),
					Vram:   *resource.NewQuantity(vram, resource.BinarySI),
				},
				Limits: &tfv1.Resource{
					Tflops: *resource.NewQuantity(tflops, resource.DecimalSI),
					Vram:   *resource.NewQuantity(vram, resource.BinarySI),
				},
				MaxWorkers: &workers,
			},
		},
	}
}

func createZeroUsage() *tfv1.GPUResourceUsage {
	calc := NewCalculator()
	return calc.CreateZeroUsage()
}

func createAllocRequest(namespace string, tflopsReq, vramReq, tflopsLim, vramLim int64, gpuCount uint) *tfv1.AllocRequest {
	return &tfv1.AllocRequest{
		WorkloadNameNamespace: tfv1.NameNamespace{
			Name:      TestWorkload,
			Namespace: namespace,
		},
		Request: tfv1.Resource{
			Tflops: *resource.NewQuantity(tflopsReq, resource.DecimalSI),
			Vram:   *resource.NewQuantity(vramReq, resource.BinarySI),
		},
		Limit: tfv1.Resource{
			Tflops: *resource.NewQuantity(tflopsLim, resource.DecimalSI),
			Vram:   *resource.NewQuantity(vramLim, resource.BinarySI),
		},
		Count: gpuCount,
	}
}

func createQuotaWithSingleLimits(maxTflops, maxVram int64, maxGPUCount *int32) *tfv1.GPUResourceQuota {
	quota := createTestQuota(1000, 10000, 100)
	quota.Spec.Single.MaxLimits = &tfv1.Resource{
		Tflops: *resource.NewQuantity(maxTflops, resource.DecimalSI),
		Vram:   *resource.NewQuantity(maxVram, resource.BinarySI),
	}
	quota.Spec.Single.MaxGPUCount = maxGPUCount
	return quota
}

var _ = Describe("QuotaCalculator", func() {
	var calc *Calculator

	BeforeEach(func() {
		calc = NewCalculator()
	})

	Describe("Edge Cases", func() {
		It("should handle zero division safely", func() {
			quota := createTestQuota(0, 0, 0)
			usage := createZeroUsage()

			percent := calc.CalculateAvailablePercent(quota, usage)
			Expect(percent.RequestsTFlops).To(Equal("100"))
			Expect(percent.RequestsVRAM).To(Equal("100"))
			Expect(percent.Workers).To(Equal("100"))
		})

		It("should protect against negative usage", func() {
			usage := createZeroUsage()
			usage.Requests.Tflops.Set(10)

			toSubtract := *resource.NewQuantity(20, resource.DecimalSI)
			calc.SafeSub(&usage.Requests.Tflops, toSubtract, 1)

			Expect(usage.Requests.Tflops.Value()).To(Equal(int64(0)))
		})

		It("should handle nil quantity without panic", func() {
			var nilQty *resource.Quantity
			testQty := *resource.NewQuantity(10, resource.DecimalSI)

			// Should not panic
			Expect(func() { calc.SafeAdd(nilQty, testQty, 1) }).NotTo(Panic())
			Expect(func() { calc.SafeSub(nilQty, testQty, 1) }).NotTo(Panic())
		})
	})

	Describe("IncreaseUsage", func() {
		It("should increase usage correctly", func() {
			usage := createZeroUsage()
			req := createAllocRequest(TestNamespace, 100, 1000, 150, 1500, 2)

			calc.IncreaseUsage(usage, req)

			Expect(usage.Requests.Tflops.Value()).To(Equal(int64(200)))
			Expect(usage.Requests.Vram.Value()).To(Equal(int64(2000)))
			Expect(usage.Limits.Tflops.Value()).To(Equal(int64(300)))
			Expect(usage.Limits.Vram.Value()).To(Equal(int64(3000)))
			Expect(usage.Workers).To(Equal(int32(1)))
		})

		It("should be no-op with nil allocation", func() {
			usage := createZeroUsage()
			calc.IncreaseUsage(usage, nil)
			Expect(usage.Requests.Tflops.Value()).To(Equal(int64(0)))
		})
	})

	Describe("DecreaseUsage", func() {
		It("should decrease usage correctly", func() {
			usage := createZeroUsage()
			usage.Requests.Tflops = *resource.NewQuantity(200, resource.DecimalSI)
			usage.Requests.Vram = *resource.NewQuantity(2000, resource.BinarySI)
			usage.Limits.Tflops = *resource.NewQuantity(300, resource.DecimalSI)
			usage.Limits.Vram = *resource.NewQuantity(3000, resource.BinarySI)
			usage.Workers = 2

			req := createAllocRequest(TestNamespace, 100, 1000, 150, 1500, 2)

			calc.DecreaseUsage(usage, req)

			Expect(usage.Requests.Tflops.Value()).To(Equal(int64(0)))
			Expect(usage.Requests.Vram.Value()).To(Equal(int64(0)))
			Expect(usage.Limits.Tflops.Value()).To(Equal(int64(0)))
			Expect(usage.Limits.Vram.Value()).To(Equal(int64(0)))
			Expect(usage.Workers).To(Equal(int32(1)))
		})
	})

	Describe("CalculateAvailablePercent", func() {
		It("should calculate 50% usage correctly", func() {
			quota := createTestQuota(100, 1000, 10)
			usage := createZeroUsage()
			usage.Requests.Tflops = *resource.NewQuantity(50, resource.DecimalSI)
			usage.Requests.Vram = *resource.NewQuantity(500, resource.BinarySI)
			usage.Workers = 5

			percent := calc.CalculateAvailablePercent(quota, usage)

			Expect(percent.RequestsTFlops).To(Equal("50.00"))
			Expect(percent.RequestsVRAM).To(Equal("50.00"))
			Expect(percent.Workers).To(Equal("50.00"))
		})

		It("should return 100% for nil inputs", func() {
			percent := calc.CalculateAvailablePercent(nil, nil)
			Expect(percent.RequestsTFlops).To(Equal("100"))
			Expect(percent.Workers).To(Equal("100"))
		})
	})

	Describe("IsAlertThresholdReached", func() {
		DescribeTable("determines alert threshold correctly",
			func(availablePercent string, threshold int32, expected bool) {
				result := calc.IsAlertThresholdReached(availablePercent, threshold)
				Expect(result).To(Equal(expected))
			},
			Entry("below min available - alert reached", "4", int32(95), true),
			Entry("above min available - no alert", "10", int32(95), false),
			Entry("at min available - no alert", "5", int32(95), false),
			Entry("empty string defaults to alert", "", int32(95), true),
			Entry("high threshold low available", "8", int32(90), true),
		)
	})
})

var _ = Describe("QuotaStore", func() {
	var qs *QuotaStore

	BeforeEach(func() {
		qs = NewQuotaStore(nil, context.Background())
	})

	Describe("ValidationRules", func() {
		It("should accept valid quota configuration", func() {
			quota := createTestQuota(100, 1000, 10)
			err := qs.validateQuotaConfig(quota)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should reject negative values", func() {
			quota := createTestQuota(-10, 1000, 10)
			err := qs.validateQuotaConfig(quota)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("requests.tflops cannot be negative"))
		})

		It("should reject limits less than requests", func() {
			quota := createTestQuota(100, 1000, 10)
			quota.Spec.Total.Limits.Tflops = *resource.NewQuantity(50, resource.DecimalSI)

			err := qs.validateQuotaConfig(quota)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("limits.tflops cannot be less than requests.tflops"))
		})

		It("should reject invalid alert threshold", func() {
			quota := createTestQuota(100, 1000, 10)
			invalidThreshold := int32(150)
			quota.Spec.Total.AlertThresholdPercent = &invalidThreshold

			err := qs.validateQuotaConfig(quota)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("alertThresholdPercent must be between 0 and 100"))
		})
	})

	Describe("CheckSingleQuotas", func() {
		It("should allow when no quota for namespace", func() {
			req := createAllocRequest("unknown-ns", 100, 1000, 100, 1000, 1)
			err := qs.CheckSingleQuotaAvailable(req)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should reject when tflops limit exceeded", func() {
			maxGPU := int32(4)
			quota := createQuotaWithSingleLimits(50, 10000, &maxGPU)
			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: createZeroUsage(),
			}

			req := createAllocRequest(TestNamespace, 100, 1000, 100, 1000, 1)
			err := qs.CheckSingleQuotaAvailable(req)

			Expect(err).To(HaveOccurred())
			quotaErr, ok := err.(*QuotaExceededError)
			Expect(ok).To(BeTrue())
			Expect(quotaErr.Resource).To(Equal(MaxTFlopsLimitResource))
			Expect(quotaErr.Unresolvable).To(BeTrue())
		})

		It("should reject when vram limit exceeded", func() {
			maxGPU := int32(4)
			quota := createQuotaWithSingleLimits(1000, 500, &maxGPU)
			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: createZeroUsage(),
			}

			req := createAllocRequest(TestNamespace, 100, 1000, 100, 1000, 1)
			err := qs.CheckSingleQuotaAvailable(req)

			Expect(err).To(HaveOccurred())
			quotaErr, ok := err.(*QuotaExceededError)
			Expect(ok).To(BeTrue())
			Expect(quotaErr.Resource).To(Equal(MaxVRAMLimitResource))
			Expect(quotaErr.Unresolvable).To(BeTrue())
		})

		It("should reject when gpu count exceeded", func() {
			maxGPU := int32(2)
			quota := createQuotaWithSingleLimits(1000, 10000, &maxGPU)
			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: createZeroUsage(),
			}

			req := createAllocRequest(TestNamespace, 100, 1000, 100, 1000, 3)
			err := qs.CheckSingleQuotaAvailable(req)

			Expect(err).To(HaveOccurred())
			quotaErr, ok := err.(*QuotaExceededError)
			Expect(ok).To(BeTrue())
			Expect(quotaErr.Resource).To(Equal(MaxGPULimitResource))
			Expect(quotaErr.Unresolvable).To(BeTrue())
		})

		It("should allow when within limits", func() {
			maxGPU := int32(4)
			quota := createQuotaWithSingleLimits(1000, 10000, &maxGPU)
			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: createZeroUsage(),
			}

			req := createAllocRequest(TestNamespace, 100, 1000, 100, 1000, 2)
			err := qs.CheckSingleQuotaAvailable(req)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Describe("CheckTotalQuotas", func() {
		It("should reject when total tflops exceeded", func() {
			quota := createTestQuota(100, 10000, 10)
			usage := createZeroUsage()
			usage.Requests.Tflops = *resource.NewQuantity(80, resource.DecimalSI)

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: usage,
			}

			req := createAllocRequest(TestNamespace, 30, 1000, 30, 1000, 1)
			err := qs.CheckQuotaAvailable(req)

			Expect(err).To(HaveOccurred())
			quotaErr, ok := err.(*QuotaExceededError)
			Expect(ok).To(BeTrue())
			Expect(quotaErr.Resource).To(Equal(TotalMaxTFlopsRequestResource))
		})

		It("should reject when total vram exceeded", func() {
			quota := createTestQuota(1000, 1000, 10)
			usage := createZeroUsage()
			usage.Requests.Vram = *resource.NewQuantity(800, resource.BinarySI)

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: usage,
			}

			req := createAllocRequest(TestNamespace, 10, 300, 10, 300, 1)
			err := qs.CheckQuotaAvailable(req)

			Expect(err).To(HaveOccurred())
			quotaErr, ok := err.(*QuotaExceededError)
			Expect(ok).To(BeTrue())
			Expect(quotaErr.Resource).To(Equal(TotalMaxVRAMRequestResource))
		})

		It("should reject when total workers exceeded", func() {
			maxWorkers := int32(5)
			quota := &tfv1.GPUResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
					Name:      TestQuotaName,
					Namespace: TestNamespace,
				},
				Spec: tfv1.GPUResourceQuotaSpec{
					Total: tfv1.GPUResourceQuotaTotal{
						MaxWorkers: &maxWorkers,
					},
				},
			}
			usage := createZeroUsage()
			usage.Workers = 5

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: usage,
			}

			req := createAllocRequest(TestNamespace, 10, 100, 10, 100, 1)
			err := qs.CheckQuotaAvailable(req)

			Expect(err).To(HaveOccurred())
			quotaErr, ok := err.(*QuotaExceededError)
			Expect(ok).To(BeTrue())
			Expect(quotaErr.Resource).To(Equal(TotalMaxWorkersLimitResource))
		})

		It("should multiply gpu count for total request", func() {
			quota := createTestQuota(100, 10000, 10)
			usage := createZeroUsage()
			usage.Requests.Tflops = *resource.NewQuantity(50, resource.DecimalSI)

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: usage,
			}

			// Request 20 tflops * 3 GPUs = 60 tflops, total would be 110 > 100
			req := createAllocRequest(TestNamespace, 20, 1000, 20, 1000, 3)
			err := qs.CheckQuotaAvailable(req)

			Expect(err).To(HaveOccurred())
			quotaErr, ok := err.(*QuotaExceededError)
			Expect(ok).To(BeTrue())
			Expect(quotaErr.Resource).To(Equal(TotalMaxTFlopsRequestResource))
		})

		It("should pass relaxed check with release", func() {
			quota := createTestQuota(100, 10000, 10)
			usage := createZeroUsage()
			usage.Requests.Tflops = *resource.NewQuantity(80, resource.DecimalSI)

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: usage,
			}

			req := createAllocRequest(TestNamespace, 30, 1000, 30, 1000, 1)

			// Without release - should fail
			err := qs.CheckQuotaAvailable(req)
			Expect(err).To(HaveOccurred())

			// With release - should pass
			toRelease := &tfv1.GPUResourceUsage{
				Requests: tfv1.Resource{
					Tflops: *resource.NewQuantity(20, resource.DecimalSI),
				},
			}
			err = qs.CheckTotalQuotaRelaxed(req, toRelease)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should fail check with pre-scheduled", func() {
			quota := createTestQuota(100, 10000, 10)
			usage := createZeroUsage()
			usage.Requests.Tflops = *resource.NewQuantity(50, resource.DecimalSI)

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: usage,
			}

			req := createAllocRequest(TestNamespace, 20, 1000, 20, 1000, 1)

			toAdd := &tfv1.GPUResourceUsage{
				Requests: tfv1.Resource{
					Tflops: *resource.NewQuantity(40, resource.DecimalSI),
				},
			}
			err := qs.CheckTotalQuotaWithPreScheduled(req, toAdd)
			Expect(err).To(HaveOccurred())
		})
	})

	Describe("AllocateAndDeallocate", func() {
		It("should update usage and mark dirty on allocate", func() {
			quota := createTestQuota(1000, 10000, 100)
			usage := createZeroUsage()

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: usage,
			}

			req := createAllocRequest(TestNamespace, 100, 1000, 150, 1500, 2)
			qs.AllocateQuota(TestNamespace, req)

			entry := qs.QuotaStore[TestNamespace]
			Expect(entry.CurrentUsage.Requests.Tflops.Value()).To(Equal(int64(200)))
			Expect(entry.CurrentUsage.Requests.Vram.Value()).To(Equal(int64(2000)))
			Expect(entry.CurrentUsage.Limits.Tflops.Value()).To(Equal(int64(300)))
			Expect(entry.CurrentUsage.Limits.Vram.Value()).To(Equal(int64(3000)))
			Expect(entry.CurrentUsage.Workers).To(Equal(int32(1)))

			qs.dirtyQuotaLock.Lock()
			_, isDirty := qs.dirtyQuotas[TestNamespace]
			qs.dirtyQuotaLock.Unlock()
			Expect(isDirty).To(BeTrue())
		})

		It("should decrease usage on deallocate", func() {
			quota := createTestQuota(1000, 10000, 100)
			usage := createZeroUsage()
			usage.Requests.Tflops = *resource.NewQuantity(200, resource.DecimalSI)
			usage.Requests.Vram = *resource.NewQuantity(2000, resource.BinarySI)
			usage.Limits.Tflops = *resource.NewQuantity(300, resource.DecimalSI)
			usage.Limits.Vram = *resource.NewQuantity(3000, resource.BinarySI)
			usage.Workers = 2

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: usage,
			}

			req := createAllocRequest(TestNamespace, 100, 1000, 150, 1500, 2)
			qs.DeallocateQuota(TestNamespace, req)

			entry := qs.QuotaStore[TestNamespace]
			Expect(entry.CurrentUsage.Requests.Tflops.Value()).To(Equal(int64(0)))
			Expect(entry.CurrentUsage.Requests.Vram.Value()).To(Equal(int64(0)))
			Expect(entry.CurrentUsage.Limits.Tflops.Value()).To(Equal(int64(0)))
			Expect(entry.CurrentUsage.Limits.Vram.Value()).To(Equal(int64(0)))
			Expect(entry.CurrentUsage.Workers).To(Equal(int32(1)))
		})

		It("should be no-op for non-existent namespace", func() {
			req := createAllocRequest("non-existent", 100, 1000, 100, 1000, 1)

			// Should not panic
			Expect(func() { qs.AllocateQuota("non-existent", req) }).NotTo(Panic())
			Expect(func() { qs.DeallocateQuota("non-existent", req) }).NotTo(Panic())
		})
	})

	Describe("AdjustQuota", func() {
		It("should add deltas correctly", func() {
			quota := createTestQuota(1000, 10000, 100)
			usage := createZeroUsage()
			usage.Requests.Tflops = *resource.NewQuantity(100, resource.DecimalSI)

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: usage,
			}

			reqDelta := tfv1.Resource{
				Tflops: *resource.NewQuantity(50, resource.DecimalSI),
				Vram:   *resource.NewQuantity(500, resource.BinarySI),
			}
			limitDelta := tfv1.Resource{
				Tflops: *resource.NewQuantity(75, resource.DecimalSI),
				Vram:   *resource.NewQuantity(750, resource.BinarySI),
			}

			qs.AdjustQuota(TestNamespace, reqDelta, limitDelta)

			entry := qs.QuotaStore[TestNamespace]
			Expect(entry.CurrentUsage.Requests.Tflops.Value()).To(Equal(int64(150)))
			Expect(entry.CurrentUsage.Requests.Vram.Value()).To(Equal(int64(500)))
			Expect(entry.CurrentUsage.Limits.Tflops.Value()).To(Equal(int64(75)))
			Expect(entry.CurrentUsage.Limits.Vram.Value()).To(Equal(int64(750)))
		})

		It("should be no-op for non-existent namespace", func() {
			reqDelta := tfv1.Resource{
				Tflops: *resource.NewQuantity(50, resource.DecimalSI),
			}
			limitDelta := tfv1.Resource{}

			Expect(func() { qs.AdjustQuota("non-existent", reqDelta, limitDelta) }).NotTo(Panic())
		})
	})

	Describe("GetQuotaStatus", func() {
		It("should return existing quota status", func() {
			quota := createTestQuota(1000, 10000, 100)
			usage := createZeroUsage()
			usage.Requests.Tflops = *resource.NewQuantity(100, resource.DecimalSI)
			usage.Workers = 5

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: usage,
			}

			status, exists := qs.GetQuotaStatus(TestNamespace)
			Expect(exists).To(BeTrue())
			Expect(status).NotTo(BeNil())
			Expect(status.Requests.Tflops.Value()).To(Equal(int64(100)))
			Expect(status.Workers).To(Equal(int32(5)))
		})

		It("should return false for non-existent namespace", func() {
			status, exists := qs.GetQuotaStatus("non-existent")
			Expect(exists).To(BeFalse())
			Expect(status).To(BeNil())
		})

		It("should return a deep copy", func() {
			quota := createTestQuota(1000, 10000, 100)
			usage := createZeroUsage()
			usage.Requests.Tflops = *resource.NewQuantity(100, resource.DecimalSI)

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: usage,
			}

			status, _ := qs.GetQuotaStatus(TestNamespace)
			status.Requests.Tflops = *resource.NewQuantity(999, resource.DecimalSI)

			entry := qs.QuotaStore[TestNamespace]
			Expect(entry.CurrentUsage.Requests.Tflops.Value()).To(Equal(int64(100)))
		})
	})

	Describe("ReconcileQuotaStore", func() {
		It("should reset and rebuild usage", func() {
			ctx := context.Background()
			quota := createTestQuota(1000, 10000, 100)
			usage := createZeroUsage()
			usage.Requests.Tflops = *resource.NewQuantity(500, resource.DecimalSI)

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: usage,
			}

			allocations := map[string]*tfv1.AllocRequest{
				"pod1": createAllocRequest(TestNamespace, 100, 1000, 100, 1000, 1),
				"pod2": createAllocRequest(TestNamespace, 50, 500, 50, 500, 2),
			}

			qs.ReconcileQuotaStore(ctx, allocations)

			entry := qs.QuotaStore[TestNamespace]
			Expect(entry.CurrentUsage.Requests.Tflops.Value()).To(Equal(int64(200)))
			Expect(entry.CurrentUsage.Workers).To(Equal(int32(2)))
		})

		It("should be no-op with empty store", func() {
			ctx := context.Background()
			qs := NewQuotaStore(nil, ctx)

			allocations := map[string]*tfv1.AllocRequest{
				"pod1": createAllocRequest(TestNamespace, 100, 1000, 100, 1000, 1),
			}

			Expect(func() { qs.ReconcileQuotaStore(ctx, allocations) }).NotTo(Panic())
		})

		It("should ignore allocations without quota", func() {
			ctx := context.Background()
			quota := createTestQuota(1000, 10000, 100)

			qs.QuotaStore[TestNamespace] = &QuotaStoreEntry{
				Quota:        quota,
				CurrentUsage: createZeroUsage(),
			}

			allocations := map[string]*tfv1.AllocRequest{
				"pod1": createAllocRequest(TestNamespace, 100, 1000, 100, 1000, 1),
				"pod2": createAllocRequest("other-ns", 50, 500, 50, 500, 1),
			}

			qs.ReconcileQuotaStore(ctx, allocations)

			entry := qs.QuotaStore[TestNamespace]
			Expect(entry.CurrentUsage.Requests.Tflops.Value()).To(Equal(int64(100)))
			Expect(entry.CurrentUsage.Workers).To(Equal(int32(1)))
		})
	})

	Describe("DirtyQuotaManagement", func() {
		It("should mark and clear dirty correctly", func() {
			qs.markQuotaDirty(TestNamespace)

			qs.dirtyQuotaLock.Lock()
			_, isDirty := qs.dirtyQuotas[TestNamespace]
			qs.dirtyQuotaLock.Unlock()
			Expect(isDirty).To(BeTrue())

			qs.clearQuotaDirty(TestNamespace)

			qs.dirtyQuotaLock.Lock()
			_, isDirty = qs.dirtyQuotas[TestNamespace]
			qs.dirtyQuotaLock.Unlock()
			Expect(isDirty).To(BeFalse())
		})
	})
})

var _ = Describe("QuotaExceededError", func() {
	It("should format error message correctly", func() {
		err := &QuotaExceededError{
			Namespace:    "production",
			Resource:     TotalMaxTFlopsRequestResource,
			Requested:    *resource.NewQuantity(200, resource.DecimalSI),
			Limit:        *resource.NewQuantity(100, resource.DecimalSI),
			Unresolvable: false,
		}

		msg := err.Error()
		Expect(msg).To(ContainSubstring("production"))
		Expect(msg).To(ContainSubstring(TotalMaxTFlopsRequestResource))
		Expect(msg).To(ContainSubstring("200"))
		Expect(msg).To(ContainSubstring("100"))
	})

	It("should implement error interface", func() {
		var err error = &QuotaExceededError{
			Namespace: "test",
			Resource:  "test-resource",
		}
		Expect(err.Error()).NotTo(BeEmpty())
	})
})
