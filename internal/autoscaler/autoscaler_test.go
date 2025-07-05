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
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/aws/smithy-go/ptr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// [x] tflops add all samples, like cpu in vpa
// [x] Reallocate resources before update annotation
// Add AutoSetResources, make it more configurable
// Log key events
// Add recommendation to workload status
// Write some documents
// cron scheduler stragegy,  parallisam ?
// Refactor main, setup database may not put in leader election runnable group
// Resolve conversation on github, thanks for reviews

var _ = Describe("Autoscaler", func() {
	Context("when creating an autoscaler", func() {
		It("should return an error if there is no client", func() {
			as, err := NewAutoscaler(nil, nil)
			Expect(as).To(BeNil())
			Expect(err.Error()).To(ContainSubstring("must specify client"))
		})

		It("should return an error if there is no allocator", func() {
			as, err := NewAutoscaler(k8sClient, nil)
			Expect(as).To(BeNil())
			Expect(err.Error()).To(ContainSubstring("must specify allocator"))
		})
	})

	Context("when loading history metrics", func() {
		It("should create the state of workloads and workers based on historical metrics", func() {
			scaler, _ := NewAutoscaler(k8sClient, allocator)
			scaler.MetricsProvider = &FakeMetricsProvider{}
			scaler.LoadHistoryMetrics(ctx)
			metrics, _ := scaler.GetHistoryMetrics()
			for _, m := range metrics {
				Expect(scaler.WorkloadStates).To(HaveKey(m.WorkloadName))
				Expect(scaler.WorkerStates).To(HaveKey(m.WorkerName))
			}
		})
	})

	Context("when loading workloads", func() {
		It("should keep the state of workloads and workers with auto-scaling enabled", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(3).
				Build()
			defer tfEnv.Cleanup()

			scaler, _ := NewAutoscaler(k8sClient, allocator)
			scaler.LoadWorkloads(ctx)
			Expect(scaler.WorkloadStates).To(BeEmpty())
			Expect(scaler.WorkerStates).To(BeEmpty())

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

			scaler, _ := NewAutoscaler(k8sClient, allocator)
			scaler.LoadWorkloads(ctx)
			ws := scaler.WorkloadStates[workload.Name]
			now := time.Now()
			metrics := &WorkerMetrics{
				WorkloadName: workload.Name,
				WorkerName:   worker,
				TflopsUsage:  ResourceAmount(12.0),
				VramUsage:    9000,
				Timestamp:    now,
			}

			scaler.MetricsProvider = &FakeMetricsProvider{[]*WorkerMetrics{metrics}}
			scaler.LoadRealTimeMetrics(ctx)

			Expect(scaler.WorkerStates[worker].LastTflopsSampleTime).To(Equal(metrics.Timestamp))
			Expect(ws.TflopsHistogram.IsEmpty()).To(BeFalse())
			Expect(scaler.WorkerStates[worker].VramPeak).To(Equal(metrics.VramUsage))
			Expect(scaler.WorkerStates[worker].LastVramSampleTime).To(Equal(metrics.Timestamp))
			Expect(ws.VramHistogram.IsEmpty()).To(BeFalse())
		})
	})

	Context("when processing workloads", func() {
		It("should update only those resources exceeding the recommended resource boundaries", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(1).
				Build()
			defer tfEnv.Cleanup()
			go mockSchedulerLoop(ctx, cfg)
			workload := createWorkload(tfEnv.GetGPUPool(0), 0, 1)
			defer deleteWorkload(workload)

			scaler, _ := NewAutoscaler(k8sClient, allocator)
			scaler.LoadWorkloads(ctx)

			scaler.ResourceRecommender = &FakeUpScalingRecommender{}
			rr := scaler.GetRecommendedResources(nil)

			scaler.ProcessWorkloads(ctx)
			Eventually(func(g Gomega) {
				assertWorkerAnnotations(getWorkers(workload)[0], rr)
			}).Should(Succeed())

			// Upon reprocessing the workload, it should skip resource updates since they are already within the recommended resource boundaries
			scaler.ProcessWorkloads(ctx)
			Consistently(func(g Gomega) {
				assertWorkerAnnotations(getWorkers(workload)[0], rr)
			}).Should(Succeed())
		})

		It("should update resources based on auto scaling config", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(1).
				Build()
			defer tfEnv.Cleanup()
			go mockSchedulerLoop(ctx, cfg)
			workload := createWorkload(tfEnv.GetGPUPool(0), 0, 1)
			defer deleteWorkload(workload)

			scaler, _ := NewAutoscaler(k8sClient, allocator)
			scaler.LoadWorkloads(ctx)

			scaler.ResourceRecommender = &FakeUpScalingRecommender{}
			rr := scaler.GetRecommendedResources(nil)

			workloadState := scaler.WorkloadStates[workload.Name]
			oldRes := workloadState.Resources

			// verify IsAutoScalingEnabled
			workloadState.AutoScalingConfig.AutoSetResources.Enable = false
			scaler.ProcessWorkloads(ctx)
			Eventually(func(g Gomega) {
				tflopsRequest, tflopsLimit, vramRequest, vramLimit := parseResourceAnnotations(getWorkers(workload)[0])
				Expect(tflopsRequest.Equal(oldRes.Requests.Tflops)).To(BeTrue())
				Expect(tflopsLimit.Equal(oldRes.Limits.Tflops)).To(BeTrue())
				Expect(vramRequest.Equal(oldRes.Requests.Vram)).To(BeTrue())
				Expect(vramLimit.Equal(oldRes.Limits.Vram)).To(BeTrue())
			}).Should(Succeed())

			// verify IsTargetResource
			workloadState.AutoScalingConfig.AutoSetResources.Enable = true
			workloadState.AutoScalingConfig.AutoSetResources.TargetResource = "tflops"
			scaler.ProcessWorkloads(ctx)
			Eventually(func(g Gomega) {
				tflopsRequest, tflopsLimit, vramRequest, vramLimit := parseResourceAnnotations(getWorkers(workload)[0])
				Expect(tflopsRequest.Value()).To(Equal(int64(rr.TargetTflops)))
				Expect(tflopsLimit.Value()).To(Equal(int64(rr.TargetTflops * 2)))
				Expect(vramRequest.Equal(oldRes.Requests.Vram)).To(BeTrue())
				Expect(vramLimit.Equal(oldRes.Limits.Vram)).To(BeTrue())
			}).Should(Succeed())
		})

		It("should return an error if recommended resources exceeded quota", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(1).
				Build()
			defer tfEnv.Cleanup()
			go mockSchedulerLoop(ctx, cfg)
			workload := createWorkload(tfEnv.GetGPUPool(0), 0, 1)
			defer deleteWorkload(workload)

			scaler, _ := NewAutoscaler(k8sClient, allocator)
			scaler.LoadWorkloads(ctx)
			scaler.ResourceRecommender = &FakeQuotaExceededRecommender{}
			rr := scaler.GetRecommendedResources(nil)
			err := scaler.updateWorkerResourcesIfNeeded(ctx, scaler.WorkloadStates[workload.Name], getWorkers(workload)[0], rr)
			Expect(err.Error()).To(ContainSubstring("failed to adjust allocation: scaling quota exceeded"))
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
				AutoSetResources: tfv1.AutoSetResources{
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

type FakeAllocator struct{}

type FakeMetricsProvider struct {
	Metrics []*WorkerMetrics
}

func (f *FakeMetricsProvider) GetWorkersMetrics() ([]*WorkerMetrics, error) {
	return f.Metrics, nil
}

func (f *FakeMetricsProvider) GetHistoryMetrics() ([]*WorkerMetrics, error) {
	metrics := []*WorkerMetrics{}
	startTime := time.Now().Add(-8 * 24 * time.Hour)
	for day := 0; day < 8; day++ {
		for hour := 0; hour < 1; hour++ {
			for minute := 0; minute < 60; minute++ {
				// idx := day*24 + hour
				metrics = append(metrics, &WorkerMetrics{
					WorkloadName: "workload-0",
					WorkerName:   fmt.Sprintf("worker-%d", 1),
					TflopsUsage:  ResourceAmount(100.0),
					VramUsage:    1 * 1000 * 1000 * 1000,
					Timestamp:    startTime.Add(time.Duration(day*24+hour)*time.Hour + time.Duration(minute)*time.Minute),
				})
			}
		}
	}

	return metrics, nil
}

type FakeUpScalingRecommender struct{}

func (f *FakeUpScalingRecommender) GetRecommendedResources(_ *WorkloadState) *RecommendedResources {
	return &RecommendedResources{
		TargetTflops:     110,
		LowerBoundTflops: 100,
		UpperBoundTflops: 120,
		TargetVram:       110 * 1000 * 1000 * 1000,
		LowerBoundVram:   100 * 1000 * 1000 * 1000,
		UpperBoundVram:   120 * 1000 * 1000 * 1000,
	}
}

type FakeQuotaExceededRecommender struct{}

func (f *FakeQuotaExceededRecommender) GetRecommendedResources(_ *WorkloadState) *RecommendedResources {
	return &RecommendedResources{
		TargetTflops:     9999,
		LowerBoundTflops: 9999,
		UpperBoundTflops: 9999,
		TargetVram:       9999 * 1000 * 1000 * 1000,
		LowerBoundVram:   9999 * 1000 * 1000 * 1000,
		UpperBoundVram:   9999 * 1000 * 1000 * 1000,
	}
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

func assertWorkerAnnotations(worker *corev1.Pod, rr *RecommendedResources) {
	GinkgoHelper()
	tflopsRequest, tflopsLimit, vramRequest, vramLimit := parseResourceAnnotations(worker)
	Expect(tflopsRequest.Value()).To(Equal(int64(rr.TargetTflops)))
	Expect(tflopsLimit.Value()).To(Equal(int64(rr.TargetTflops * 2)))
	Expect(vramRequest.Value()).To(Equal(int64(rr.TargetVram)))
	Expect(vramLimit.Value()).To(Equal(int64(rr.TargetVram * 2)))
}

func parseResourceAnnotations(worker *corev1.Pod) (tflopsRequest, tflopsLimit, vramRequest, vramLimit resource.Quantity) {
	annotations := worker.GetAnnotations()
	keys := []struct {
		key string
		dst *resource.Quantity
	}{
		{constants.TFLOPSRequestAnnotation, &tflopsRequest},
		{constants.TFLOPSLimitAnnotation, &tflopsLimit},
		{constants.VRAMRequestAnnotation, &vramRequest},
		{constants.VRAMLimitAnnotation, &vramLimit},
	}
	for _, k := range keys {
		*k.dst = resource.MustParse(annotations[k.key])
	}
	return
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
	if errors.IsNotFound(err) {
		return
	}
	Expect(err).To(Succeed())
	gpus, err := allocator.Alloc(&allocRequest)
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
		latestPod.Annotations[constants.GpuKey] = strings.Join(
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
