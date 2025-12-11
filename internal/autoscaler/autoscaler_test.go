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

package autoscaler

import (
	"context"
	"fmt"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/recommender"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/aws/smithy-go/ptr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var _ = Describe("Autoscaler", func() {
	Context("when creating an autoscaler", func() {
		It("should return an error if there is no client", func() {
			as, err := NewAutoscaler(nil, nil, nil)
			Expect(as).To(BeNil())
			Expect(err.Error()).To(ContainSubstring("must specify client"))
		})

		It("should return an error if there is no allocator", func() {
			as, err := NewAutoscaler(k8sClient, nil, nil)
			Expect(as).To(BeNil())
			Expect(err.Error()).To(ContainSubstring("must specify allocator"))
		})

		It("should return an error if there is no metrics provider", func() {
			as, err := NewAutoscaler(k8sClient, allocator, nil)
			Expect(as).To(BeNil())
			Expect(err.Error()).To(ContainSubstring("must specify metricsProvider"))
		})
	})

	Context("when loading history metrics", func() {
		It("should create the state of workloads and workers based on historical metrics", func() {
			scaler, _ := NewAutoscaler(k8sClient, allocator, &FakeMetricsProvider{})
			// History metrics are now loaded per-workload in goroutines
			// This test is kept for compatibility but the behavior has changed
			// The metrics loader will handle history loading after InitialDelayPeriod
			Expect(scaler).ToNot(BeNil())
		})
	})

	Context("when loading workloads", func() {
		It("should keep the state of workloads", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(3).
				Build()
			defer tfEnv.Cleanup()

			scaler, _ := NewAutoscaler(k8sClient, allocator, &FakeMetricsProvider{})
			scaler.loadWorkloads(ctx)
			Expect(scaler.workloads).To(BeEmpty())

			// create two workloads
			pool := tfEnv.GetGPUPool(0)
			// Use unique IDs to avoid conflicts
			// with two replicas
			workload0 := createWorkload(pool, 200, 2)
			workload0Workers := getWorkers(workload0)
			key0 := WorkloadID{workload0.Namespace, workload0.Name}
			// with one replica
			workload1 := createWorkload(pool, 201, 1)
			workload1Workers := getWorkers(workload1)
			key1 := WorkloadID{workload1.Namespace, workload1.Name}

			// Wait for workloads to have WorkerCount > 0 (set by controller)
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(workload0), workload0)).Should(Succeed())
				g.Expect(workload0.Status.WorkerCount).To(BeNumerically(">", 0))
			}).Should(Succeed())
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(workload1), workload1)).Should(Succeed())
				g.Expect(workload1.Status.WorkerCount).To(BeNumerically(">", 0))
			}).Should(Succeed())

			scaler.loadWorkloads(ctx)
			Expect(scaler.workloads).To(HaveLen(2))
			Expect(scaler.workloads).To(HaveKey(key0))
			Expect(scaler.workloads).To(HaveKey(key1))
			workers := scaler.workloads[key0].WorkerUsageSamplers
			Expect(workers).To(HaveLen(2))
			Expect(workers).To(HaveKey(workload0Workers[0].Name))
			Expect(workers).To(HaveKey(workload0Workers[1].Name))
			Expect(scaler.workloads[key1].WorkerUsageSamplers).To(HaveKey(workload1Workers[0].Name))

			updateWorkloadReplicas(workload0, 1)
			scaler.loadWorkloads(ctx)
			Expect(scaler.workloads[key0].WorkerUsageSamplers).To(HaveLen(1))

			deleteWorkload(workload0)
			deleteWorkload(workload1)
			scaler.loadWorkloads(ctx)
			Expect(scaler.workloads).NotTo(HaveKey(key0))
			Expect(scaler.workloads).NotTo(HaveKey(key1))
		})
	})

	Context("when loading real time metrics", func() {
		It("should correctly update the stat of the workload", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(1).
				Build()
			defer tfEnv.Cleanup()
			pool := tfEnv.GetGPUPool(0)
			// Use unique ID to avoid conflicts
			workload := createWorkload(pool, 202, 1)
			worker := getWorkers(workload)[0]
			key := WorkloadID{workload.Namespace, workload.Name}
			defer deleteWorkload(workload)

			// Wait for workload to have WorkerCount > 0
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(workload), workload)).Should(Succeed())
				g.Expect(workload.Status.WorkerCount).To(BeNumerically(">", 0))
			}).Should(Succeed())

			scaler, _ := NewAutoscaler(k8sClient, allocator, &FakeMetricsProvider{})
			scaler.loadWorkloads(ctx)
			ws, exists := scaler.workloads[key]
			Expect(exists).To(BeTrue())
			Expect(ws).ToNot(BeNil())
			now := time.Now()
			usage := &metrics.WorkerUsage{
				Namespace:    workload.Namespace,
				WorkloadName: workload.Name,
				WorkerName:   worker.Name,
				TflopsUsage:  12.0,
				VramUsage:    9000,
				Timestamp:    now,
			}

			scaler.metricsProvider = &FakeMetricsProvider{[]*metrics.WorkerUsage{usage}}
			// Realtime metrics are now loaded per-workload in goroutines
			// Manually add sample for testing
			ws.AddSample(usage)

			scalerWorkers := ws.WorkerUsageSamplers
			Expect(scalerWorkers[worker.Name].LastTflopsSampleTime).To(Equal(usage.Timestamp))
			Expect(ws.WorkerUsageAggregator.TflopsHistogram.IsEmpty()).To(BeFalse())
			Expect(scalerWorkers[worker.Name].VramPeak).To(Equal(usage.VramUsage))
			Expect(scalerWorkers[worker.Name].LastVramSampleTime).To(Equal(usage.Timestamp))
			Expect(ws.WorkerUsageAggregator.VramHistogram.IsEmpty()).To(BeFalse())
			usage = &metrics.WorkerUsage{
				Namespace:    workload.Namespace,
				WorkloadName: workload.Name,
				WorkerName:   worker.Name,
				TflopsUsage:  13.0,
				VramUsage:    10000,
				Timestamp:    now.Add(time.Minute),
			}
			scaler.metricsProvider = &FakeMetricsProvider{[]*metrics.WorkerUsage{usage}}
			// Realtime metrics are now loaded per-workload in goroutines
			// Manually add sample for testing
			ws.AddSample(usage)
			Expect(scalerWorkers[worker.Name].LastTflopsSampleTime).To(Equal(usage.Timestamp))
			Expect(scalerWorkers[worker.Name].VramPeak).To(Equal(usage.VramUsage))
			Expect(scalerWorkers[worker.Name].LastVramSampleTime).To(Equal(usage.Timestamp))
			Expect(ws.WorkerUsageAggregator.TotalSamplesCount).To(Equal(2))
		})
	})

	Context("when processing workloads", func() {
		var tfEnv *TensorFusionEnv
		var workload *tfv1.TensorFusionWorkload
		var key WorkloadID
		var scaler *Autoscaler
		var targetRes tfv1.Resources
		var workloadIDCounter = 100 // Start from 100 to avoid conflicts with other tests
		BeforeEach(func() {
			// Clean up any existing workload with the same ID first
			cleanupWorkload(client.ObjectKey{Namespace: "default", Name: getWorkloadName(workloadIDCounter)})
			tfEnv = NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(1).
				Build()
			go mockSchedulerLoop(ctx, cfg)
			workload = createWorkload(tfEnv.GetGPUPool(0), workloadIDCounter, 1)
			workloadIDCounter++
			key = WorkloadID{workload.Namespace, workload.Name}
			verifyGpuStatus(tfEnv)

			scaler, _ = NewAutoscaler(k8sClient, allocator, &FakeMetricsProvider{})
			scaler.loadWorkloads(ctx)
			targetRes = tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("110"),
					Vram:   resource.MustParse("110Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("110"),
					Vram:   resource.MustParse("110Gi"),
				},
			}
		})

		AfterEach(func() {
			deleteWorkload(workload)
			tfEnv.Cleanup()
		})

		It("should scale up if the recommended resources exceed the current allocation", func() {
			// Ensure workload is loaded
			scaler.loadWorkloads(ctx)
			workloadState, exists := scaler.workloads[key]
			Expect(exists).To(BeTrue())
			Expect(workloadState).ToNot(BeNil())
			scaler.recommenders = append(scaler.recommenders, &FakeRecommender{Resources: &targetRes})
			scaler.processSingleWorkload(ctx, workloadState)
			verifyRecommendationStatus(workload, &targetRes)

			// Upon reprocessing the workload, it should skip resource updates
			scaler.processSingleWorkload(ctx, workloadState)
			verifyRecommendationStatusConsistently(workload, &targetRes)
		})

		It("should update resources based on auto scaling config", func() {
			// Ensure workload is loaded
			scaler.loadWorkloads(ctx)
			workloadState, exists := scaler.workloads[key]
			Expect(exists).To(BeTrue())
			Expect(workloadState).ToNot(BeNil())
			scaler.recommenders = append(scaler.recommenders, &FakeRecommender{Resources: &targetRes})
			oldRes := workloadState.Spec.Resources

			// verify IsAutoScalingEnabled
			workloadState.Spec.AutoScalingConfig.AutoSetResources = &tfv1.AutoSetResources{
				Enable: false,
			}
			scaler.processSingleWorkload(ctx, workloadState)
			verifyWorkerResources(workload, &oldRes)

			// verify IsTargetResource
			workloadState.Spec.AutoScalingConfig.AutoSetResources = &tfv1.AutoSetResources{
				Enable:         true,
				TargetResource: tfv1.ScalingTargetResourceCompute,
			}
			scaler.processSingleWorkload(ctx, workloadState)
			expect := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("110"),
					Vram:   resource.MustParse("8Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("110"),
					Vram:   resource.MustParse("16Gi"),
				},
			}
			verifyWorkerResources(workload, &expect)
		})

		It("should not apply recommended resources if the worker has a dedicated GPU", func() {
			// Ensure workload is loaded
			scaler.loadWorkloads(ctx)
			workloadState, exists := scaler.workloads[key]
			Expect(exists).To(BeTrue())
			Expect(workloadState).ToNot(BeNil())
			scaler.recommenders = append(scaler.recommenders, &FakeRecommender{Resources: &targetRes})
			// set the worker in dedicated mode
			worker := getWorkers(workload)[0]
			workloadState.CurrentActiveWorkers[worker.Name].Annotations[constants.DedicatedGPUAnnotation] = constants.TrueStringValue
			oldRes := workloadState.Spec.Resources
			scaler.processSingleWorkload(ctx, workloadState)
			// verify the worker's resources have not been altered
			verifyWorkerResources(workload, &oldRes)
		})

		It("should not update resources if recommended resources exceeded quota", func() {
			excessiveRes := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("9999"),
					Vram:   resource.MustParse("9999Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("9999"),
					Vram:   resource.MustParse("9999Gi"),
				},
			}

			scaler.recommenders = append(scaler.recommenders, &FakeRecommender{Resources: &excessiveRes})

			// Ensure workload is loaded
			scaler.loadWorkloads(ctx)
			workloadState, exists := scaler.workloads[key]
			Expect(exists).To(BeTrue())
			Expect(workloadState).ToNot(BeNil())
			oldRes := workloadState.Spec.Resources
			scaler.processSingleWorkload(ctx, workloadState)
			verifyWorkerResources(workload, &oldRes)
		})

		It("should update resources based on cron scaling rule", func() {
			// Ensure workload is loaded
			scaler.loadWorkloads(ctx)
			workloadState, exists := scaler.workloads[key]
			Expect(exists).To(BeTrue())
			Expect(workloadState).ToNot(BeNil())
			resourcesInRule := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("120"),
					Vram:   resource.MustParse("120Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("120"),
					Vram:   resource.MustParse("120Gi"),
				},
			}

			workloadState.Spec.AutoScalingConfig.CronScalingRules = []tfv1.CronScalingRule{
				{
					Enable:           true,
					Name:             "test",
					Start:            "0 0 * * *",
					End:              "59 23 * * *",
					DesiredResources: resourcesInRule,
				},
			}
			scaler.processSingleWorkload(ctx, workloadState)
			verifyRecommendationStatus(workload, &resourcesInRule)

			// invalidate the rule by updating start and end fields
			workloadState.Spec.AutoScalingConfig.CronScalingRules = []tfv1.CronScalingRule{
				{
					Enable:           true,
					Name:             "test",
					Start:            "",
					End:              "",
					DesiredResources: resourcesInRule,
				},
			}

			scaler.processSingleWorkload(ctx, workloadState)
			originalResources := workloadState.Spec.Resources
			verifyRecommendationStatus(workload, &originalResources)

			// should not change after cron scaling rule inactive
			scaler.processSingleWorkload(ctx, workloadState)
			verifyRecommendationStatus(workload, &originalResources)
		})

		It("should not scale down when merging recommendations during active cron scaling progress", func() {
			// Ensure workload is loaded
			scaler.loadWorkloads(ctx)
			workloadState, exists := scaler.workloads[key]
			Expect(exists).To(BeTrue())
			Expect(workloadState).ToNot(BeNil())
			resourcesInRule := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("110"),
					Vram:   resource.MustParse("110Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("110"),
					Vram:   resource.MustParse("110Gi"),
				},
			}
			workloadState.Spec.AutoScalingConfig.CronScalingRules = []tfv1.CronScalingRule{
				{
					Enable:           true,
					Name:             "test",
					Start:            "0 0 * * *",
					End:              "59 23 * * *",
					DesiredResources: resourcesInRule,
				},
			}

			scaler.processSingleWorkload(ctx, workloadState)
			verifyRecommendationStatus(workload, &resourcesInRule)

			fakeRes := tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("1"),
					Vram:   resource.MustParse("1Gi"),
				},
				Limits: tfv1.Resource{
					Tflops: resource.MustParse("1"),
					Vram:   resource.MustParse("1Gi"),
				},
			}

			scaler.recommenders = append(scaler.recommenders, &FakeRecommender{Resources: &fakeRes})

			scaler.processSingleWorkload(ctx, workloadState)
			verifyRecommendationStatusConsistently(workload, &resourcesInRule)
		})

		It("should return max allowed resources spec per worker based on current worker count", func() {
			// Ensure workload is loaded
			scaler.loadWorkloads(ctx)
			workloadState, exists := scaler.workloads[key]
			Expect(exists).To(BeTrue())
			Expect(workloadState).ToNot(BeNil())
			workloadHandler := scaler.workloadHandler
			gpuList := tfEnv.GetPoolGpuList(0)
			capacity := gpuList.Items[0].Status.Capacity
			allTflops := int64(capacity.Tflops.AsApproximateFloat64())
			allVram := capacity.Vram.Value()

			// Wait for workers to have GPUs allocated by mockSchedulerLoop
			Eventually(func(g Gomega) {
				workers := getWorkers(workload)
				g.Expect(workers).To(HaveLen(1))
				// Check that worker has GPU allocated
				g.Expect(workers[0].Annotations).To(HaveKey(constants.GPUDeviceIDsAnnotation))
			}).Should(Succeed())

			// Reload workload state to get updated worker info
			scaler.loadWorkloads(ctx)
			workloadState = scaler.workloads[key]

			got, err := workloadHandler.GetMaxAllowedResourcesSpec(workloadState)
			Expect(err).To(Succeed())
			Expect(got.Tflops.Value()).To(Equal(allTflops))
			Expect(got.Vram.Value()).To(Equal(allVram))

			updateWorkloadReplicas(workload, 2)
			// Wait for new workers to have GPUs allocated, with longer timeout
			Eventually(func(g Gomega) {
				workers := getWorkers(workload)
				g.Expect(workers).To(HaveLen(2))
				for _, worker := range workers {
					g.Expect(worker.Annotations).To(HaveKey(constants.GPUDeviceIDsAnnotation))
				}
			}, 30*time.Second).Should(Succeed())
			scaler.loadWorkloads(ctx)
			workloadState, exists = scaler.workloads[key]
			Expect(exists).To(BeTrue())
			Expect(workloadState).ToNot(BeNil())
			got, err = workloadHandler.GetMaxAllowedResourcesSpec(workloadState)
			Expect(err).To(Succeed())
			Expect(got.Tflops.Value()).To(Equal(allTflops / 2))
			Expect(got.Vram.Value()).To(Equal(allVram / 2))

			updateWorkloadReplicas(workload, 0)
			// Wait for workload status to update
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(workload), workload)).Should(Succeed())
				g.Expect(workload.Status.WorkerCount).To(Equal(int32(0)))
			}).Should(Succeed())
			scaler.loadWorkloads(ctx)
			// After setting replicas to 0, workload should be removed from scaler.workloads
			// because WorkerCount == 0, so GetMaxAllowedResourcesSpec should return nil
			workloadState = scaler.workloads[key]
			if workloadState != nil {
				got, err = workloadHandler.GetMaxAllowedResourcesSpec(workloadState)
				// If workload still exists but has no workers, it should return nil
				if err == nil {
					Expect(got).To(BeNil())
				}
			} else {
				// Workload was removed from scaler.workloads, which is expected when WorkerCount == 0
				Expect(workloadState).To(BeNil())
			}
		})
	})
})

func createWorkload(pool *tfv1.GPUPool, id int, replicas int) *tfv1.TensorFusionWorkload {
	GinkgoHelper()
	tflopsRequests := resource.MustParse("10")
	vramRequests := resource.MustParse("8Gi")
	tflopsLimits := resource.MustParse("20")
	vramLimits := resource.MustParse("16Gi")

	poolName := pool.Name
	key := client.ObjectKey{Namespace: "default", Name: getWorkloadName(id)}
	workload := &tfv1.TensorFusionWorkload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      key.Name,
			Namespace: key.Namespace,
			Labels: map[string]string{
				constants.GpuPoolKey: poolName,
			},
		},
		Spec: tfv1.WorkloadProfileSpec{
			Replicas: ptr.Int32(int32(replicas)),
			PoolName: poolName,
			Resources: tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: tflopsRequests,
					Vram:   vramRequests,
				},
				Limits: tfv1.Resource{
					Tflops: tflopsLimits,
					Vram:   vramLimits,
				},
			},
			Qos: constants.QoSLevelMedium,
			AutoScalingConfig: tfv1.AutoScalingConfig{
				AutoSetResources: &tfv1.AutoSetResources{
					Enable:         true,
					TargetResource: tfv1.ScalingTargetResourceAll,
				},
			},
		},
	}

	Expect(k8sClient.Create(ctx, workload)).To(Succeed())

	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
	}).Should(Succeed())

	checkWorkerPodCount(workload)
	return workload
}

func verifyGpuStatus(tfEnv *TensorFusionEnv) {
	Eventually(func(g Gomega) bool {
		gpuList := tfEnv.GetPoolGpuList(0)
		ok := false
		_, ok = lo.Find(gpuList.Items, func(gpu tfv1.GPU) bool {
			return gpu.Status.Available.Tflops.Equal(resource.MustParse("1990")) && gpu.Status.Available.Vram.Equal(resource.MustParse("1992Gi"))
		})
		return ok
	}).Should(BeTrue())
}

func checkWorkerPodCount(workload *tfv1.TensorFusionWorkload) {
	GinkgoHelper()
	podList := &corev1.PodList{}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.List(ctx, podList,
			client.InNamespace(workload.Namespace),
			client.MatchingLabels{constants.WorkloadKey: workload.Name})).Should(Succeed())
		g.Expect(podList.Items).Should(HaveLen(int(*workload.Spec.Replicas)))
	}).Should(Succeed())
}

func getWorkloadName(index int) string {
	return fmt.Sprintf("workload-%d", index)
}

func getWorkers(workload *tfv1.TensorFusionWorkload) []*corev1.Pod {
	GinkgoHelper()
	podList := &corev1.PodList{}
	Expect(k8sClient.List(ctx, podList,
		client.InNamespace(workload.Namespace),
		client.MatchingLabels{constants.WorkloadKey: workload.Name})).Should(Succeed())
	return lo.Map(podList.Items, func(pod corev1.Pod, _ int) *corev1.Pod {
		return &pod
	})
}

type FakeMetricsProvider struct {
	Metrics []*metrics.WorkerUsage
}

func (f *FakeMetricsProvider) GetWorkersMetrics(ctx context.Context) ([]*metrics.WorkerUsage, error) {
	return f.Metrics, nil
}

func (f *FakeMetricsProvider) GetWorkloadHistoryMetrics(ctx context.Context, namespace, workloadName string, startTime, endTime time.Time) ([]*metrics.WorkerUsage, error) {
	// Filter metrics by namespace, workloadName, and time range
	result := []*metrics.WorkerUsage{}
	for _, m := range f.Metrics {
		if m.Namespace == namespace && m.WorkloadName == workloadName &&
			m.Timestamp.After(startTime) && m.Timestamp.Before(endTime) {
			result = append(result, m)
		}
	}
	return result, nil
}

func (f *FakeMetricsProvider) GetWorkloadRealtimeMetrics(ctx context.Context, namespace, workloadName string, startTime, endTime time.Time) ([]*metrics.WorkerUsage, error) {
	// Filter metrics by namespace, workloadName, and time range
	result := []*metrics.WorkerUsage{}
	for _, m := range f.Metrics {
		if m.Namespace == namespace && m.WorkloadName == workloadName &&
			m.Timestamp.After(startTime) && m.Timestamp.Before(endTime) {
			result = append(result, m)
		}
	}
	return result, nil
}

func (f *FakeMetricsProvider) LoadHistoryMetrics(ctx context.Context, processMetricsFunc func(*metrics.WorkerUsage)) error {
	startTime := time.Now().Add(-7 * 24 * time.Hour)
	for day := range 7 {
		for hour := range 24 {
			for minute := range 60 {
				// idx := day*24 + hour
				sample := &metrics.WorkerUsage{
					Namespace:    "default",
					WorkloadName: "workload-0",
					WorkerName:   fmt.Sprintf("worker-%d", 1),
					TflopsUsage:  100.0,
					VramUsage:    1 * 1000 * 1000 * 1000,
					Timestamp:    startTime.Add(time.Duration(day*24+hour)*time.Hour + time.Duration(minute)*time.Minute),
				}
				processMetricsFunc(sample)
			}
		}
	}

	return nil
}

func (f *FakeMetricsProvider) GetHistoryMetrics(ctx context.Context) ([]*metrics.WorkerUsage, error) {
	sample := []*metrics.WorkerUsage{}
	startTime := time.Now().Add(-8 * 24 * time.Hour)
	for day := 0; day < 8; day++ {
		for hour := 0; hour < 1; hour++ {
			for minute := 0; minute < 60; minute++ {
				// idx := day*24 + hour
				sample = append(sample, &metrics.WorkerUsage{
					Namespace:    "default",
					WorkloadName: "workload-0",
					WorkerName:   fmt.Sprintf("worker-%d", 1),
					TflopsUsage:  100.0,
					VramUsage:    1 * 1000 * 1000 * 1000,
					Timestamp:    startTime.Add(time.Duration(day*24+hour)*time.Hour + time.Duration(minute)*time.Minute),
				})
			}
		}
	}

	return sample, nil
}

type FakeRecommender struct {
	*tfv1.Resources
}

func (f *FakeRecommender) Name() string {
	return "fake"
}

func (f *FakeRecommender) Recommend(ctx context.Context, workload *workload.State) (*recommender.RecResult, error) {
	meta.SetStatusCondition(&workload.Status.Conditions, metav1.Condition{
		Type:               constants.ConditionStatusTypeRecommendationProvided,
		Status:             metav1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Reason:             "FakeReason",
		Message:            "Fake message",
	})
	return &recommender.RecResult{
		Resources: *f.Resources,
	}, nil
}

func verifyWorkerResources(workload *tfv1.TensorFusionWorkload, expectedRes *tfv1.Resources) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		res, _ := utils.GPUResourcesFromAnnotations(getWorkers(workload)[0].Annotations)
		g.Expect(res.Equal(expectedRes)).To(BeTrue())
	}).Should(Succeed())
}

func verifyRecommendationStatus(workload *tfv1.TensorFusionWorkload, expectedRes *tfv1.Resources) {
	GinkgoHelper()
	key := client.ObjectKeyFromObject(workload)
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
		g.Expect(workload.Status.Recommendation.Equal(expectedRes)).To(BeTrue())
		g.Expect(workload.Status.AppliedRecommendedReplicas).To(Equal(*workload.Spec.Replicas))
		// Check for migrated condition type (ConditionStatusTypeResourceUpdate)
		// The handler migrates ConditionStatusTypeRecommendationProvided to ConditionStatusTypeResourceUpdate
		condition := meta.FindStatusCondition(workload.Status.Conditions, constants.ConditionStatusTypeResourceUpdate)
		g.Expect(condition).ToNot(BeNil())
		if condition != nil {
			switch condition.Reason {
			case "RuleActive":
				g.Expect(workload.Status.ActiveCronScalingRule).ToNot(BeNil())
			case "RuleInactive":
				g.Expect(workload.Status.ActiveCronScalingRule).To(BeNil())
			}
		}
		res, _ := utils.GPUResourcesFromAnnotations(getWorkers(workload)[0].Annotations)
		g.Expect(res.Equal(expectedRes)).To(BeTrue())
	}).Should(Succeed())
}

func verifyRecommendationStatusConsistently(workload *tfv1.TensorFusionWorkload, expectedRes *tfv1.Resources) {
	GinkgoHelper()
	key := client.ObjectKeyFromObject(workload)
	Consistently(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
		g.Expect(workload.Status.Recommendation.Equal(expectedRes)).To(BeTrue())
		res, _ := utils.GPUResourcesFromAnnotations(getWorkers(workload)[0].Annotations)
		g.Expect(res.Equal(expectedRes)).To(BeTrue())
	}).Should(Succeed())
}

func updateWorkloadReplicas(workload *tfv1.TensorFusionWorkload, replicas int) {
	GinkgoHelper()
	key := client.ObjectKeyFromObject(workload)
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
		workload.Spec.Replicas = ptr.Int32(int32(replicas))
		g.Expect(k8sClient.Update(ctx, workload)).To(Succeed())
	}).Should(Succeed())

	checkWorkerPodCount(workload)
}

func deleteWorkload(workload *tfv1.TensorFusionWorkload) {
	cleanupWorkload(client.ObjectKeyFromObject(workload))
}

func cleanupWorkload(key client.ObjectKey) {
	GinkgoHelper()
	workload := &tfv1.TensorFusionWorkload{}

	if err := k8sClient.Get(ctx, key, workload); err != nil {
		if errors.IsNotFound(err) {
			return
		}
		// If there's an error other than NotFound, try to continue cleanup
		// Don't fail the test if workload doesn't exist
		return
	}

	// Set replicas to 0
	Eventually(func(g Gomega) {
		err := k8sClient.Get(ctx, key, workload)
		if errors.IsNotFound(err) {
			return
		}
		g.Expect(err).Should(Succeed())
		workload.Spec.Replicas = ptr.Int32(0)
		g.Expect(k8sClient.Update(ctx, workload)).To(Succeed())
	}).Should(Succeed())

	// Wait for pods to be deleted, but with a longer timeout and more lenient check
	Eventually(func(g Gomega) {
		podList := &corev1.PodList{}
		err := k8sClient.List(ctx, podList,
			client.InNamespace(key.Namespace),
			client.MatchingLabels{constants.WorkloadKey: key.Name})
		if err != nil {
			return
		}
		// Filter out pods that are being deleted
		activePods := []corev1.Pod{}
		for _, pod := range podList.Items {
			if pod.DeletionTimestamp.IsZero() {
				activePods = append(activePods, pod)
			}
		}
		g.Expect(activePods).Should(BeEmpty())
	}, 30*time.Second).Should(Succeed())

	// Try to delete, but don't fail if already deleted
	if err := k8sClient.Get(ctx, key, workload); err == nil {
		_ = k8sClient.Delete(ctx, workload)
		Eventually(func(g Gomega) {
			err := k8sClient.Get(ctx, key, workload)
			g.Expect(errors.IsNotFound(err)).To(BeTrue())
		}).Should(Succeed())
	}
}
func mockSchedulerLoop(ctx context.Context, cfg *rest.Config) {
	ticker := time.NewTicker(50 * time.Millisecond)
	clientset, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		Expect(err).To(Succeed())
	}
	for range ticker.C {
		select {
		case <-ctx.Done():
			return
		default:
			podList := &corev1.PodList{}
			_ = k8sClient.List(ctx, podList)
			for _, pod := range podList.Items {
				if pod.Spec.NodeName != "" {
					continue
				}
				go scheduleAndStartPod(&pod, clientset)
			}
		}
	}
}

func scheduleAndStartPod(pod *corev1.Pod, clientset *kubernetes.Clientset) {
	// simulate scheduling cycle Filter and Reserve
	allocRequest, _, err := allocator.ComposeAllocationRequest(pod)
	Expect(err).To(Succeed())
	gpus, err := allocator.Alloc(allocRequest)
	if err != nil {
		// some test cases are expected to fail, just continue
		return
	}
	Expect(gpus).To(HaveLen(int(allocRequest.Count)))
	allocator.SyncGPUsToK8s()

	// update pod annotation
	Eventually(func(g Gomega) {
		latestPod := &corev1.Pod{}
		err := k8sClient.Get(ctx, types.NamespacedName{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		}, latestPod)
		if errors.IsNotFound(err) {
			return
		}
		g.Expect(err).To(Succeed())

		if latestPod.Annotations == nil {
			latestPod.Annotations = map[string]string{}
		}
		latestPod.Annotations[constants.GPUDeviceIDsAnnotation] = strings.Join(
			lo.Map(gpus, func(gpu *tfv1.GPU, _ int) string {
				return gpu.Name
			}), ",")
		err = k8sClient.Status().Update(ctx, latestPod)
		if errors.IsNotFound(err) {
			return
		}
		g.Expect(err).To(Succeed())

		// update pod node name
		latestPod.Spec.NodeName = gpus[0].Status.NodeSelector[constants.KubernetesHostNameLabel]

		// simulate k8s scheduler binding cycle Bind function
		binding := &corev1.Binding{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pod.Name,
				Namespace: pod.Namespace,
			},
			Target: corev1.ObjectReference{
				Kind: "Node",
				Name: latestPod.Spec.NodeName,
			},
		}

		err = clientset.CoreV1().Pods(latestPod.Namespace).Bind(ctx, binding, metav1.CreateOptions{})
		if errors.IsNotFound(err) {
			return
		}
		g.Expect(err).To(Succeed())
	}).Should(Succeed())

	// simulate kubelet start the pod successfully
	patchPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
	}
	patchPod.Status.Phase = corev1.PodRunning
	patchPod.Status.Conditions = append(patchPod.Status.Conditions, corev1.PodCondition{
		Type:   corev1.PodReady,
		Status: corev1.ConditionTrue,
	})
	err = k8sClient.Status().Patch(ctx, patchPod, client.MergeFrom(&corev1.Pod{}))
	if errors.IsNotFound(err) {
		return
	}
	Expect(err).To(Succeed())
}
