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
	"fmt"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/aws/smithy-go/ptr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var _ = Describe("Autoscaler", func() {
	Context("when creating an autoscaler", func() {
		It("should return an error if there is no client", func() {
			as, err := NewAutoscaler(nil)
			Expect(as).To(BeNil())
			Expect(err.Error()).To(ContainSubstring("must specify client"))
		})
	})

	Context("when loading history metrics", func() {
		It("should create the state of workloads and workers based on historical metrics", func() {
			scaler, _ := NewAutoscaler(k8sClient)
			scaler.MetricsProvider = &FakeMetricsProvider{}
			scaler.LoadHistoryMetrics(ctx)
			metrics := scaler.MetricsProvider.GetHistoryMetrics()
			for _, m := range metrics {
				Expect(scaler.WorkloadStates).To(HaveKey(m.Workload))
				Expect(scaler.WorkerStates).To(HaveKey(m.Worker))
			}
		})
	})

	Context("when loading workloads", func() {
		It("should keep the state of workloads and workers with auto-scaling enabled", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(3).
				Build()
			defer tfEnv.Cleanup()

			scaler, _ := NewAutoscaler(k8sClient)
			scaler.LoadWorkloads(ctx)
			Expect(scaler.WorkloadStates).To(HaveLen(0))
			Expect(scaler.WorkerStates).To(HaveLen(0))

			// create two workloads
			pool := tfEnv.GetGPUPool(0)
			// with two replias
			workload0 := createWorkload(pool, 0, 2)
			workload0Workers := getWorkers(workload0)
			// with one replia
			workload1 := createWorkload(pool, 1, 1)
			workload1Workers := getWorkers(workload1)

			scaler.LoadWorkloads(ctx)
			Expect(scaler.WorkloadStates).To(HaveLen(2))
			Expect(scaler.WorkloadStates).To(HaveKey(workload0.Name))
			Expect(scaler.WorkloadStates).To(HaveKey(workload1.Name))
			Expect(scaler.WorkerStates).To(HaveLen(3))
			Expect(scaler.WorkerStates).To(HaveKey(workload0Workers[0].Name))
			Expect(scaler.WorkerStates).To(HaveKey(workload0Workers[1].Name))
			Expect(scaler.WorkerStates).To(HaveKey(workload1Workers[0].Name))

			updateWorkloadReplicas(workload0, 1)
			scaler.LoadWorkloads(ctx)
			Expect(scaler.WorkerStates).To(HaveLen(2))

			deleteWorkload(workload0)
			deleteWorkload(workload1)
			scaler.LoadWorkloads(ctx)
			Expect(scaler.WorkloadStates).NotTo(HaveKey(workload0.Name))
			Expect(scaler.WorkerStates).NotTo(HaveKey(workload0Workers[0].Name))
			Expect(scaler.WorkerStates).NotTo(HaveKey(workload0Workers[1].Name))
			Expect(scaler.WorkloadStates).NotTo(HaveKey(workload1.Name))
			Expect(scaler.WorkerStates).NotTo(HaveKey(workload1Workers[0].Name))
		})
	})

	Context("when loading real time metrics", func() {
		It("should update the state of workloads and workers", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(1).
				Build()
			defer tfEnv.Cleanup()
			pool := tfEnv.GetGPUPool(0)
			workload := createWorkload(pool, 0, 1)
			workers := getWorkers(workload)
			defer deleteWorkload(workload)

			worker := workers[0].Name

			scaler, _ := NewAutoscaler(k8sClient)
			scaler.LoadWorkloads(ctx)
			ws := scaler.WorkloadStates[workload.Name]
			metrics := &WorkerMetrics{
				Workload:    workload.Name,
				Worker:      worker,
				TflopsUsage: ResourceAmount(12.0),
				VramUsage:   9000,
				Timestamp:   time.Now(),
			}

			scaler.MetricsProvider = &FakeMetricsProvider{[]*WorkerMetrics{metrics}}
			scaler.LoadRealTimeMetrics(ctx)

			Expect(scaler.WorkerStates[worker].TflopsPeak).To(Equal(metrics.TflopsUsage))
			Expect(scaler.WorkerStates[worker].LastTflopsSampleTime).To(Equal(metrics.Timestamp))
			Expect(ws.TflopsHistogram.IsEmpty()).To(BeFalse())
			Expect(scaler.WorkerStates[worker].VramPeak).To(Equal(metrics.VramUsage))
			Expect(scaler.WorkerStates[worker].LastVramSampleTime).To(Equal(metrics.Timestamp))
			Expect(ws.VramHistogram.IsEmpty()).To(BeFalse())
		})
	})

	Context("when processing workloads", func() {
		It("should update worker annotations if resource out of bounds", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(1).
				Build()
			defer tfEnv.Cleanup()
			workload := createWorkload(tfEnv.GetGPUPool(0), 0, 1)
			defer deleteWorkload(workload)

			scaler, _ := NewAutoscaler(k8sClient)
			scaler.LoadWorkloads(ctx)

			recommender := &FakeRecommender{
				RecommendedResources: RecommendedResources{
					TargetTflops:     110,
					LowerBoundTflops: 100,
					UpperBoundTflops: 120,
					TargetVram:       110 * 1000 * 1000 * 1000,
					LowerBoundVram:   100 * 1000 * 1000 * 1000,
					UpperBoundVram:   120 * 1000 * 1000 * 1000,
				},
			}

			scaler.Recommender = recommender
			rr := recommender.GetRecommendedResources(nil)

			scaler.ProcessWorkloads(ctx)

			Eventually(func(g Gomega) {
				workers := getWorkers(workload)
				annotations := workers[0].GetAnnotations()

				tflopsRequest := resource.MustParse(annotations[constants.TFLOPSRequestAnnotation])
				g.Expect(tflopsRequest.Value()).To(Equal(int64(rr.TargetTflops)))

				tflopsLimit := resource.MustParse(annotations[constants.TFLOPSLimitAnnotation])
				g.Expect(tflopsLimit.Value()).To(Equal(int64(rr.TargetTflops * 2)))

				vramRequest := resource.MustParse(annotations[constants.VRAMRequestAnnotation])
				g.Expect(vramRequest.Value()).To(Equal(int64(rr.TargetVram)))

				vramLimit := resource.MustParse(annotations[constants.VRAMLimitAnnotation])
				g.Expect(vramLimit.Value()).To(Equal(int64(rr.TargetVram * 2)))

			}).Should(Succeed())
		})

		It("should not udpate worker annotations if resources in bounds", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(1).
				Build()
			defer tfEnv.Cleanup()
			workload := createWorkload(tfEnv.GetGPUPool(0), 0, 1)
			defer deleteWorkload(workload)

			scaler, _ := NewAutoscaler(k8sClient)
			scaler.LoadWorkloads(ctx)

			recommender := &FakeRecommender{
				RecommendedResources: RecommendedResources{
					TargetTflops:     110,
					LowerBoundTflops: 10,
					UpperBoundTflops: 120,
					TargetVram:       110 * 1000 * 1000 * 1000,
					LowerBoundVram:   5 * 1000 * 1000 * 1000,
					UpperBoundVram:   120 * 1000 * 1000 * 1000,
				},
			}

			scaler.Recommender = recommender

			scaler.ProcessWorkloads(ctx)

			Consistently(func(g Gomega) {
				workers := getWorkers(workload)
				annotations := workers[0].GetAnnotations()

				tflopsRequest := resource.MustParse(annotations[constants.TFLOPSRequestAnnotation])
				g.Expect(tflopsRequest.Equal(workload.Spec.Resources.Requests.Tflops)).To(BeTrue())

				tflopsLimit := resource.MustParse(annotations[constants.TFLOPSLimitAnnotation])
				g.Expect(tflopsLimit.Equal(workload.Spec.Resources.Limits.Tflops)).To(BeTrue())

				vramRequest := resource.MustParse(annotations[constants.VRAMRequestAnnotation])
				g.Expect(vramRequest.Equal(workload.Spec.Resources.Requests.Vram)).To(BeTrue())

				vramLimit := resource.MustParse(annotations[constants.VRAMLimitAnnotation])
				g.Expect(vramLimit.Equal(workload.Spec.Resources.Limits.Vram)).To(BeTrue())

			}).Should(Succeed())
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
				AutoSetLimits: tfv1.AutoSetLimits{
					Enable:         true,
					TargetResource: "",
				},
				AutoSetRequests: tfv1.AutoSetRequests{
					Enable:         true,
					TargetResource: "",
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
		client.InNamespace("default"),
		client.MatchingLabels{constants.WorkloadKey: workload.Name})).Should(Succeed())
	return lo.Map(podList.Items, func(pod corev1.Pod, _ int) *corev1.Pod {
		return &pod
	})
}

type FakeMetricsProvider struct {
	Metrics []*WorkerMetrics
}

func (f *FakeMetricsProvider) GetWorkersMetrics() []*WorkerMetrics {
	return f.Metrics
}

func (f *FakeMetricsProvider) GetHistoryMetrics() []*WorkerMetrics {
	metrics := []*WorkerMetrics{}
	startTime := time.Now().Add(-7 * 24 * time.Hour)
	for day := 0; day < 7; day++ {
		for hour := 0; hour < 24; hour++ {
			idx := day*24 + hour
			metrics = append(metrics, &WorkerMetrics{
				Workload:    "workload-0",
				Worker:      fmt.Sprintf("worker-%d", idx),
				TflopsUsage: ResourceAmount(10.0 + float64(idx%10)),
				VramUsage:   1 * 1024 * 1024 * 1024,
				Timestamp:   startTime.Add(time.Duration(day*24+hour) * time.Hour),
			})
		}
	}

	return metrics
}

type FakeRecommender struct {
	RecommendedResources
}

func (f *FakeRecommender) GetRecommendedResources(_ *WorkloadState) *RecommendedResources {
	return &f.RecommendedResources
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
		Expect(err).To(HaveOccurred())
	}

	// Set replicas to 0
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
		workload.Spec.Replicas = ptr.Int32(0)
		g.Expect(k8sClient.Update(ctx, workload)).To(Succeed())
	}).Should(Succeed())

	Eventually(func(g Gomega) {
		podList := &corev1.PodList{}
		g.Expect(k8sClient.List(ctx, podList,
			client.InNamespace(key.Namespace),
			client.MatchingLabels{constants.WorkloadKey: key.Name})).To(Succeed())
		g.Expect(podList.Items).Should(BeEmpty())
	}).Should(Succeed())

	Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
	Expect(k8sClient.Delete(ctx, workload)).To(Succeed())
	Eventually(func(g Gomega) {
		err := k8sClient.Get(ctx, key, workload)
		g.Expect(err).Should(HaveOccurred())
	}).Should(Succeed())
}
