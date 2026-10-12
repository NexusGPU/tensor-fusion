//go:build e2e

package suite

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Cases that need no GPU node at all: control plane, admission, API surface.
// They are the gate for every other tier and the only ones a quota-less AWS
// account can run for real.

// noProvision keeps a pending TensorFusion pod from making Karpenter launch a
// node for it: with the chart's catch-all NodeOverlay every instance type looks
// like it offers tensor-fusion.ai/index_*, so Karpenter would otherwise start a
// (CPU!) node for a pod that only wants to be inspected. KRP-A08 is the case
// that studies that behaviour on purpose.
func noProvision() PodOpt {
	return withNodeSelector(map[string]string{"e2e.tensor-fusion.ai/no-such-node": "true"})
}

var tfCRDs = []string{
	"tensorfusionclusters", "gpupools", "gpunodes", "gpus", "gpunodeclaims", "gpunodeclasses",
	"tensorfusionworkloads", "tensorfusionconnections", "workloadprofiles",
	"schedulingconfigtemplates", "gpuresourcequotas", "providerconfigs",
}

var _ = Describe("install", func() {
	Register(Case{
		ID: "INS-01", Title: "Helm 安装后控制面就绪", Tiers: []string{TierSmoke}, Needs: []string{NeedTF}, NoNS: true,
		Run: func(ctx SpecContext, c *C) {
			By("CRDs are established")
			for _, plural := range tfCRDs {
				crd, err := env.get(ctx, gvkCRD, "", plural+"."+tfDomain)
				Expect(err).NotTo(HaveOccurred(), "CRD %s", plural)
				Expect(condition(crd, "Established")).To(Equal("True"), "CRD %s", plural)
			}

			By("controller deployment is available")
			dep, err := env.get(ctx, gvkDeployment, env.TFNamespace, env.TFRelease+"-controller")
			Expect(err).NotTo(HaveOccurred())
			Expect(condition(dep, "Available")).To(Equal("True"), "controller not Available: %s", toJSON(mapAt(dep, "status")))
			c.Note("operator image %s", firstContainerImage(dep, "controller"))

			By("controller pods are Ready and have not restarted")
			pods, err := env.list(ctx, gvkPod, env.TFNamespace, client.MatchingLabels{lblComponent: "operator"})
			Expect(err).NotTo(HaveOccurred())
			Expect(pods).NotTo(BeEmpty(), "no pod with %s=operator", lblComponent)
			restarts := 0.0
			for i := range pods {
				Expect(podReady(&pods[i])).To(BeTrue(), podWhy(&pods[i]))
				for _, cs := range sliceAt(&pods[i], "status", "containerStatuses") {
					m, _ := cs.(map[string]any)
					if n, ok := m["restartCount"].(int64); ok {
						restarts += float64(n)
					}
				}
			}
			c.Measure("controller_restarts", restarts, "")

			By("mutating webhook has a CA bundle")
			wh, err := env.get(ctx, gvk("admissionregistration.k8s.io", "v1", "MutatingWebhookConfiguration"), "", env.TFRelease+"-mutating-webhook")
			Expect(err).NotTo(HaveOccurred())
			hooks := sliceAt(wh, "webhooks")
			Expect(hooks).NotTo(BeEmpty())
			ca, _, _ := unstructured.NestedString(hooks[0].(map[string]any), "clientConfig", "caBundle")
			Expect(ca).NotTo(BeEmpty(), "webhook caBundle not injected")

			By("leader lease is held")
			lease, err := env.get(ctx, gvk("coordination.k8s.io", "v1", "Lease"), env.TFNamespace, "85104305.tensor-fusion.ai")
			Expect(err).NotTo(HaveOccurred())
			Expect(str(lease, "spec", "holderIdentity")).NotTo(BeEmpty())

			By("GreptimeDB and alertmanager are Ready")
			for _, target := range [][2]string{{"greptimedb", "greptimedb-standalone-0"}, {env.TFNamespace, "alert-manager-0"}} {
				name := env.TFRelease + "-" + target[1]
				p, err := env.get(ctx, gvkPod, target[0], name)
				Expect(err).NotTo(HaveOccurred(), "pod %s/%s", target[0], name)
				Expect(podReady(p)).To(BeTrue(), podWhy(p))
			}
		},
	})

	Register(Case{
		ID: "INS-02", Title: "默认集群与 Pool 被调谐（零节点）", Tiers: []string{TierSmoke}, Needs: []string{NeedTF}, NoNS: true,
		Run: func(ctx SpecContext, c *C) {
			tfc, err := env.get(ctx, gvkTFCluster, "", "tensor-fusion")
			Expect(err).NotTo(HaveOccurred())
			pool, err := env.get(ctx, gvkGPUPool, "", env.RealPool)
			Expect(err).NotTo(HaveOccurred(), "GPUPool %s (created by the TensorFusionCluster controller)", env.RealPool)
			_, err = env.get(ctx, gvkSchedTmpl, "", "tf-scheduling-config")
			Expect(err).NotTo(HaveOccurred())

			// "Reconciles" = the controllers wrote status. The phase itself depends on
			// capacity: with zero GPU nodes and the chart's minResources (20 TFLOPS /
			// 10Gi) the pool is held in Updating, which the docs do not mention
			//. With nodes it must be Running.
			waitFor(ctx, "GPUPool status.phase set", 2*time.Minute, func() (string, bool) {
				pool, _ = env.get(ctx, gvkGPUPool, "", env.RealPool)
				return "phase empty", str(pool, "status", "phase") != ""
			})
			c.Note("TensorFusionCluster phase=%s, GPUPool %s phase=%s totalNodes=%s totalGPUs=%s",
				str(tfc, "status", "phase"), env.RealPool, str(pool, "status", "phase"), str(pool, "status", "totalNodes"), str(pool, "status", "totalGPUs"))
			Expect(str(pool, "spec", "nodeManagerConfig", "provisioningMode")).To(Equal("AutoSelect"))
			if len(env.gpusOf(ctx, env.RealPool)) > 0 {
				Expect(str(pool, "status", "phase")).To(Equal("Running"))
			} else {
				Expect(str(pool, "status", "phase")).To(BeElementOf("Running", "Updating", "Pending"))
			}
			Expect(pool.GetAnnotations()[tfDomain+"/is-default-pool"]).To(Equal("true"), "chart pool must be the default pool")
		},
	})

	Register(Case{
		ID: "INS-04", Title: "非 GPU 节点不被纳管", Tiers: []string{TierSmoke, TierRehearsal}, Needs: []string{NeedTF}, NoNS: true,
		Run: func(ctx SpecContext, c *C) {
			gpuNodes, err := env.list(ctx, gvkGPUNode, "")
			Expect(err).NotTo(HaveOccurred())
			managed := map[string]bool{}
			for i := range gpuNodes {
				managed[gpuNodes[i].GetName()] = true
			}
			for _, role := range []string{"system", "cpu"} {
				for _, n := range env.nodes(ctx, map[string]string{env.RoleLabel: role}) {
					Expect(managed[n.GetName()]).To(BeFalse(), "node %s (role %s) has a GPUNode", n.GetName(), role)
					for k := range n.GetLabels() {
						Expect(strings.HasPrefix(k, tfDomain+"/pool-")).To(BeFalse(), "node %s carries pool label %s", n.GetName(), k)
					}
				}
			}
			c.Measure("gpunodes", float64(len(gpuNodes)), "")
		},
	})
})

var _ = Describe("webhook", func() {
	Register(Case{
		ID: "WHK-01", Title: "只加 label + annotation 即被改写", Tiers: []string{TierSmoke, TierRehearsal}, Needs: []string{NeedTF},
		Run: func(ctx SpecContext, c *C) {
			pod := tfPod(c.NS, "frac", ann("isolation", "soft", "compute-percent-request", "20", "compute-percent-limit", "40", "vram-request", "1Gi", "vram-limit", "2Gi"), noProvision())
			Expect(env.create(ctx, pod)).To(Succeed())
			got, err := env.get(ctx, gvkPod, c.NS, "frac")
			Expect(err).NotTo(HaveOccurred())

			Expect(str(got, "spec", "schedulerName")).To(Equal(schedulerName))
			Expect(got.GetLabels()).To(HaveKeyWithValue(lblComponent, "worker"))
			Expect(got.GetLabels()).To(HaveKeyWithValue(lblPool, env.Pool))
			a := got.GetAnnotations()
			Expect(a).To(HaveKeyWithValue(tfDomain+"/isolation", "soft"))
			Expect(a).To(HaveKeyWithValue(tfDomain+"/is-local-gpu", "true"))
			Expect(a).To(HaveKeyWithValue(tfDomain+"/gpu-count", "1"))
			Expect(a).To(HaveKey(tfDomain + "/qos"))

			By("the injected container asks for one tensor-fusion.ai/index_* slot")
			limits := toStringMap(sliceAt(got, "spec", "containers")[0].(map[string]any)["resources"].(map[string]any)["limits"])
			var index []string
			for k := range limits {
				if strings.HasPrefix(k, tfDomain+"/index_") {
					index = append(index, k)
				}
			}
			Expect(index).To(HaveLen(1), "limits: %v", limits)

			By("soft isolation injects the limiter init containers and env")
			var inits []string
			for _, ic := range sliceAt(got, "spec", "initContainers") {
				inits = append(inits, fmt.Sprint(ic.(map[string]any)["name"]))
			}
			Expect(inits).To(ContainElement("init-soft-limiter"), "init containers: %v", inits)
			envs := containerEnv(got, 0)
			Expect(envs).To(HaveKeyWithValue("TF_ISOLATION_MODE", "soft"))
			Expect(envs).To(HaveKey("TF_SHM_PATH"))
			c.Note("init containers %v, index resource %s", inits, index[0])

			By("a TensorFusionWorkload is created for the pod")
			waitFor(ctx, "TensorFusionWorkload", time.Minute, func() (string, bool) {
				wl, _ := env.list(ctx, gvkWorkload, c.NS)
				return fmt.Sprintf("%d workloads", len(wl)), len(wl) == 1
			})

			if env.Mode == ModeNoGPU {
				By("with no GPU in the pool the pod stays Pending with a diagnosable reason (OBS-04)")
				waitFor(ctx, "scheduler event naming the missing GPU capacity", 2*time.Minute, func() (string, bool) {
					evs := env.events(ctx, c.NS, "Pod", "frac")
					return strings.Join(evs, " | "), containsEvent(evs, "no gpu devices in pool") || containsEvent(evs, "GPUQuotaOrCapacityNotEnough")
				})
				Expect(podPhase(got)).To(Equal("Pending"))
			}
		},
	})

	Register(Case{
		ID: "WHK-02", Title: "引用 WorkloadProfile，annotation 可覆盖", Tiers: []string{TierSmoke}, Needs: []string{NeedTF},
		Run: func(ctx SpecContext, c *C) {
			profile := parseYAML(fmt.Sprintf(`
apiVersion: tensor-fusion.ai/v1
kind: WorkloadProfile
metadata: {name: small, namespace: %s}
spec:
  poolName: %s
  isLocalGPU: true
  isolation: soft
  qos: low
  resources:
    requests: {tflops: "5", vram: 1Gi}
    limits: {tflops: "10", vram: 2Gi}
`, c.NS, env.Pool))
			Expect(env.create(ctx, profile...)).To(Succeed())

			Expect(env.create(ctx, tfPod(c.NS, "from-profile", ann("workload-profile", "small"), noProvision()))).To(Succeed())
			p, _ := env.get(ctx, gvkPod, c.NS, "from-profile")
			a := p.GetAnnotations()
			Expect(a).To(HaveKeyWithValue(tfDomain+"/qos", "low"))
			Expect(a).To(HaveKeyWithValue(tfDomain+"/vram-limit", "2Gi"))
			Expect(a).To(HaveKeyWithValue(tfDomain+"/tflops-request", "5"))

			Expect(env.create(ctx, tfPod(c.NS, "override", ann("workload-profile", "small", "vram-limit", "3Gi", "qos", "medium"), noProvision()))).To(Succeed())
			p, _ = env.get(ctx, gvkPod, c.NS, "override")
			Expect(p.GetAnnotations()).To(HaveKeyWithValue(tfDomain+"/vram-limit", "3Gi"))
			Expect(p.GetAnnotations()).To(HaveKeyWithValue(tfDomain+"/qos", "medium"))
		},
	})

	Register(Case{
		ID: "WHK-03", Title: "多容器 Pod 只注入指定容器", Tiers: []string{TierSmoke}, Needs: []string{NeedTF},
		Run: func(ctx SpecContext, c *C) {
			By("two containers without inject-container are rejected")
			err := env.create(ctx, tfPod(c.NS, "ambiguous", ann("compute-percent-request", "10", "vram-request", "1Gi"),
				withExtraContainer("sidecar", env.Images.Busybox, "sleep", "86400"), noProvision()))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("inject container has to be specified"))

			By("with inject-container only that container gets the index resource")
			Expect(env.create(ctx, tfPod(c.NS, "two", ann("inject-container", "main", "compute-percent-request", "10", "vram-request", "1Gi"),
				withExtraContainer("sidecar", env.Images.Busybox, "sleep", "86400"), noProvision()))).To(Succeed())
			p, _ := env.get(ctx, gvkPod, c.NS, "two")
			for _, raw := range sliceAt(p, "spec", "containers") {
				ct := raw.(map[string]any)
				limits, _, _ := unstructured.NestedMap(ct, "resources", "limits")
				hasIndex := false
				for k := range limits {
					hasIndex = hasIndex || strings.HasPrefix(k, tfDomain+"/index_")
				}
				Expect(hasIndex).To(Equal(ct["name"] == "main"), "container %v limits %v", ct["name"], limits)
			}
		},
	})

	Register(Case{
		ID: "WHK-05", Title: "enabled-replicas 只接入部分副本", Tiers: []string{TierSmoke}, Needs: []string{NeedTF},
		KnownIssue: "intermittent: the webhook's per-workload pod counter is not atomic across the concurrent pod creations of one ReplicaSet; with 4 replicas and enabled-replicas=2, 3 pods were handed to TensorFusion in 1 of 3 runs (2.17.23)",
		Run: func(ctx SpecContext, c *C) {
			tmpl := tfPod(c.NS, "gray", ann("enabled-replicas", "2", "compute-percent-request", "10", "vram-request", "1Gi"))
			Expect(env.create(ctx, deployment(c.NS, "gray", 4, tmpl))).To(Succeed())
			var injected, plain int
			waitFor(ctx, "4 pods created, 2 of them mutated", 3*time.Minute, func() (string, bool) {
				injected, plain = 0, 0
				for _, p := range env.podsOf(ctx, c.NS, "gray") {
					if str(&p, "spec", "schedulerName") == schedulerName {
						injected++
					} else {
						plain++
					}
				}
				return fmt.Sprintf("injected=%d plain=%d", injected, plain), injected+plain == 4
			})
			Expect(injected).To(Equal(2), "exactly enabled-replicas pods are handed to TensorFusion")
			Expect(plain).To(Equal(2))
		},
	})

	Register(Case{
		ID: "WHK-06", Title: "未启用的 Pod 完全不被改动", Tiers: []string{TierSmoke}, Needs: []string{NeedTF},
		Run: func(ctx SpecContext, c *C) {
			// Annotations alone must not opt a pod in: the label is the gate.
			pod := plainPod(c.NS, "plain", noProvision())
			pod.SetAnnotations(ann("vram-request", "1Gi", "compute-percent-request", "10"))
			Expect(env.create(ctx, pod)).To(Succeed())
			got, _ := env.get(ctx, gvkPod, c.NS, "plain")
			Expect(str(got, "spec", "schedulerName")).To(Equal("default-scheduler"))
			Expect(got.GetLabels()).NotTo(HaveKey(lblComponent))
			Expect(sliceAt(got, "spec", "initContainers")).To(BeEmpty())
			wl, _ := env.list(ctx, gvkWorkload, c.NS)
			Expect(wl).To(BeEmpty())
		},
	})

	Register(Case{
		ID: "WHK-08", Title: "server-side dry-run 不产生副作用", Tiers: []string{TierSmoke}, Needs: []string{NeedTF},
		KnownIssue: "webhook declares sideEffects: None but creates the TensorFusionWorkload even for dry-run requests (<= 2.17.23)",
		Run: func(ctx SpecContext, c *C) {
			Expect(env.dryRun(ctx, tfPod(c.NS, "dry", ann("compute-percent-request", "10", "vram-request", "1Gi"), noProvision()))).To(Succeed())
			never(ctx, "a TensorFusionWorkload appears after a dry-run pod create", 20*time.Second, func() (string, bool) {
				wl, _ := env.list(ctx, gvkWorkload, c.NS)
				return fmt.Sprintf("%d workload(s): %v", len(wl), names(wl)), len(wl) > 0
			})
		},
	})

	Register(Case{
		ID: "WHK-07", Title: "删除负载后 TensorFusionWorkload 与 GPU 资源归还", Tiers: []string{TierSmoke, TierCore, TierRehearsal}, Needs: []string{NeedTF},
		Run: func(ctx SpecContext, c *C) {
			opts := []PodOpt{}
			if env.Mode == ModeNoGPU {
				opts = append(opts, noProvision())
			}
			tmpl := tfPod(c.NS, "leak", ann("isolation", "shared", "compute-percent-request", "10", "vram-request", "1Gi"), opts...)
			Expect(env.create(ctx, deployment(c.NS, "leak", 2, tmpl))).To(Succeed())
			waitFor(ctx, "TensorFusionWorkload for the deployment", time.Minute, func() (string, bool) {
				wl, _ := env.list(ctx, gvkWorkload, c.NS)
				return fmt.Sprintf("%d", len(wl)), len(wl) == 1
			})
			if env.Mode != ModeNoGPU {
				waitFor(ctx, "both replicas Running", 5*time.Minute, func() (string, bool) {
					ready := 0
					pods := env.podsOf(ctx, c.NS, "leak")
					for i := range pods {
						if podReady(&pods[i]) {
							ready++
						}
					}
					return fmt.Sprintf("%d/2 ready", ready), ready == 2
				})
			}
			env.delete(ctx, newObj(gvkDeployment, c.NS, "leak"))
			d := waitFor(ctx, "TensorFusionWorkload removed with its owner", 3*time.Minute, func() (string, bool) {
				wl, _ := env.list(ctx, gvkWorkload, c.NS)
				return fmt.Sprintf("%d left: %v", len(wl), names(wl)), len(wl) == 0
			})
			c.Measure("workload_gc_seconds", d.Seconds(), "s")
			if env.Mode != ModeNoGPU {
				c.Measure("gpu_return_seconds", env.waitResourcesReturned(ctx, 3*time.Minute).Seconds(), "s")
			}
		},
	})
})

var _ = Describe("negative", func() {
	rejected := func(id, title, want string, annotations map[string]string, opts ...PodOpt) {
		Register(Case{
			ID: id, Title: title, Tiers: []string{TierSmoke}, Needs: []string{NeedTF},
			Run: func(ctx SpecContext, c *C) {
				err := env.create(ctx, tfPod(c.NS, "bad", annotations, append(opts, noProvision())...))
				Expect(err).To(HaveOccurred(), "the webhook admitted an invalid pod")
				Expect(err.Error()).To(ContainSubstring(want))
				c.Note("rejected with: %s", firstLine(err.Error()))
			},
		})
	}
	rejected("NEG-02", "指定不存在的 pool 在准入被拒", "gpu pool", ann("gpupool", "does-not-exist", "vram-request", "1Gi", "compute-percent-request", "10"))
	rejected("NEG-05", "整卡请求缺 gpu-model 被拒", "dedicated GPU requires gpu-model", ann("dedicated-gpu", "true"))
	rejected("NEG-06", "gang-min-members=1 被拒", "gang", ann("gang-enabled", "true", "gang-min-members", "1", "vram-request", "1Gi", "compute-percent-request", "10"))
	rejected("NEG-12", "soft 模式未写任何资源请求被拒", "request is not set", ann("isolation", "soft"))

	Register(Case{
		ID: "NEG-03", Title: "本地 hard 不写 sidecar-worker：实际被接受并自动加 sidecar", Tiers: []string{TierSmoke}, Needs: []string{NeedTF},
		Run: func(ctx SpecContext, c *C) {
			// e2e/README (earlier draft) and tf_parser.go:222 say this combination is
			// rejected. The check is unreachable: local hard always gets a sidecar
			// worker. The case pins the real behaviour.
			Expect(env.create(ctx, tfPod(c.NS, "hard", ann("isolation", "hard", "is-local-gpu", "true", "compute-percent-limit", "50", "compute-percent-request", "50", "vram-request", "1Gi", "vram-limit", "1Gi"), noProvision()))).To(Succeed())
			p, _ := env.get(ctx, gvkPod, c.NS, "hard")
			var containers []string
			for _, raw := range sliceAt(p, "spec", "containers") {
				containers = append(containers, fmt.Sprint(raw.(map[string]any)["name"]))
			}
			Expect(containers).To(ContainElement("tensorfusion-worker"), "containers: %v", containers)
			Expect(p.GetAnnotations()).To(HaveKeyWithValue(tfDomain+"/sidecar-worker", "true"))
		},
	})

	Register(Case{
		ID: "NEG-11", Title: "tflops 与 compute-percent 同时指定：实际被接受，tflops 生效", Tiers: []string{TierSmoke}, Needs: []string{NeedTF},
		Run: func(ctx SpecContext, c *C) {
			// tf_parser.go has a "mutually exclusive" rejection, but the annotation
			// parser drops compute-percent-* as soon as a tflops-* annotation is
			// present, so the check never fires. Pin what users actually get
			//.
			Expect(env.create(ctx, tfPod(c.NS, "both", ann("tflops-request", "5", "tflops-limit", "5", "compute-percent-request", "10", "compute-percent-limit", "10", "vram-request", "1Gi", "vram-limit", "1Gi"), noProvision()))).To(Succeed())
			wl, err := env.list(ctx, gvkWorkload, c.NS)
			Expect(err).NotTo(HaveOccurred())
			Expect(wl).To(HaveLen(1))
			c.Note("workload resources: %s", toJSON(mapAt(&wl[0], "spec", "resources")))
			Expect(str(&wl[0], "spec", "resources", "requests", "tflops")).To(Equal("5"))
			Expect(str(&wl[0], "spec", "resources", "requests", "computePercent")).To(BeElementOf("", "0"))
		},
	})

	Register(Case{
		ID: "NEG-04", Title: "partitioned 不带 partition-id 被拒", Tiers: []string{TierSmoke}, Needs: []string{NeedTF},
		Run: func(ctx SpecContext, c *C) {
			err := env.create(ctx, tfPod(c.NS, "mig", ann("isolation", "partitioned"), noProvision()))
			Expect(err).To(HaveOccurred(), "a partitioned pod without a partition template was admitted")
			c.Note("rejected with: %s", firstLine(err.Error()))
		},
	})

	Register(Case{
		ID: "NEG-13", Title: "非法 isolation / qos 值的报错位置", Tiers: []string{TierSmoke}, Needs: []string{NeedTF},
		Run: func(ctx SpecContext, c *C) {
			// The webhook does not validate these enums itself; the pod is refused only
			// because creating the TensorFusionWorkload fails CRD validation
			//. Either way an invalid value must not be admitted.
			for _, kv := range [][2]string{{"isolation", "bogus"}, {"qos", "urgent"}} {
				err := env.create(ctx, tfPod(c.NS, "enum-"+kv[0], ann(kv[0], kv[1], "vram-request", "1Gi", "compute-percent-request", "10"), noProvision()))
				Expect(err).To(HaveOccurred(), "%s=%s was admitted", kv[0], kv[1])
				c.Note("%s=%s -> %s", kv[0], kv[1], firstLine(err.Error()))
			}
		},
	})
})

var _ = Describe("scheduling without GPUs", func() {
	Register(Case{
		ID: "SCH-10", Title: "命名空间配额拦截超限请求", Tiers: []string{TierSmoke, TierCore, TierRehearsal}, Needs: []string{NeedTF}, Timeout: 6 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			quota := parseYAML(fmt.Sprintf(`
apiVersion: tensor-fusion.ai/v1
kind: GPUResourceQuota
metadata: {name: quota, namespace: %s}
spec:
  total:
    requests: {tflops: "30", vram: 8Gi}
    limits: {tflops: "60", vram: 16Gi}
    maxWorkers: 10
  single:
    maxLimits: {tflops: "40", vram: 8Gi}
`, c.NS))
			Expect(env.create(ctx, quota...)).To(Succeed())
			// The quota store is fed by a controller; give it a moment to load.
			time.Sleep(10 * time.Second)

			opts := []PodOpt{}
			if env.Mode == ModeNoGPU {
				opts = append(opts, noProvision())
			}
			By("a request above the namespace total is refused by the scheduler with a quota reason")
			Expect(env.create(ctx, tfPod(c.NS, "too-big", ann("isolation", "shared", "tflops-request", "35", "tflops-limit", "35", "vram-request", "1Gi", "vram-limit", "1Gi"), opts...))).To(Succeed())
			waitFor(ctx, "quota event on the pod", 3*time.Minute, func() (string, bool) {
				evs := env.events(ctx, c.NS, "Pod", "too-big")
				return strings.Join(evs, " | "), containsEvent(evs, "quota")
			})
			p, _ := env.get(ctx, gvkPod, c.NS, "too-big")
			Expect(podPhase(p)).To(Equal("Pending"))
			Expect(str(p, "spec", "nodeName")).To(BeEmpty())

			q, _ := env.get(ctx, gvkQuota, c.NS, "quota")
			c.Note("quota status: %s", toJSON(mapAt(q, "status")))

			if env.Mode != ModeNoGPU {
				By("a request inside the quota is scheduled and accounted")
				Expect(env.create(ctx, tfPod(c.NS, "fits", ann("isolation", "shared", "tflops-request", "10", "tflops-limit", "10", "vram-request", "1Gi", "vram-limit", "1Gi")))).To(Succeed())
				env.waitPodReady(ctx, c.NS, "fits", 5*time.Minute)
				waitFor(ctx, "quota status.used reflects the running pod", 2*time.Minute, func() (string, bool) {
					q, _ = env.get(ctx, gvkQuota, c.NS, "quota")
					return toJSON(mapAt(q, "status")), qty(q, "status", "used", "requests", "tflops") >= 10
				})
			}
		},
	})

	Register(Case{
		ID: "SCH-12", Title: "分配查询与扩容器状态 API 可用", Tiers: []string{TierSmoke}, Needs: []string{NeedTF}, NoNS: true,
		Run: func(ctx SpecContext, c *C) {
			for _, path := range []string{"api/allocation", "api/node-scaler"} {
				body, err := env.controllerAPI(ctx, path, nil)
				Expect(err).NotTo(HaveOccurred(), "GET /%s", path)
				c.Measure(strings.ReplaceAll(path, "/", "_")+"_bytes", float64(len(body)), "B")
			}
		},
	})

	Register(Case{
		ID: "SCH-14", Title: "自动分配 HostPort", Tiers: []string{TierSmoke}, Needs: []string{NeedTF},
		Run: func(ctx SpecContext, c *C) {
			pod := tfPod(c.NS, "port", ann("compute-percent-request", "10", "vram-request", "1Gi"),
				withLabels(map[string]string{tfDomain + "/host-port": "auto", tfDomain + "/port-name": "http"}), noProvision(),
				func(pod map[string]any) {
					ct := pod["spec"].(map[string]any)["containers"].([]any)[0].(map[string]any)
					ct["ports"] = []any{map[string]any{"name": "http", "containerPort": int64(8080)}}
				})
			Expect(env.create(ctx, pod)).To(Succeed())
			p, _ := env.get(ctx, gvkPod, c.NS, "port")
			port := p.GetAnnotations()[tfDomain+"/port-number"]
			Expect(port).NotTo(BeEmpty(), "annotations: %v", p.GetAnnotations())
			var n int
			_, err := fmt.Sscan(port, &n)
			Expect(err).NotTo(HaveOccurred())
			Expect(n).To(BeNumerically(">=", 40000))
			Expect(n).To(BeNumerically("<=", 42000))
			c.Measure("host_port", float64(n), "")
		},
	})
})

var _ = Describe("observability without GPUs", func() {
	Register(Case{
		ID: "OBS-01", Title: "Operator metrics 端点可抓取", Tiers: []string{TierSmoke}, Needs: []string{NeedTF}, NoNS: true,
		Run: func(ctx SpecContext, c *C) {
			pods, err := env.list(ctx, gvkPod, env.TFNamespace, client.MatchingLabels{lblComponent: "operator"})
			Expect(err).NotTo(HaveOccurred())
			Expect(pods).NotTo(BeEmpty())
			// Through the API server's pod proxy: no ServiceMonitor / token dance needed
			// to prove the endpoint serves controller-runtime metrics.
			body, err := env.CS.CoreV1().Pods(env.TFNamespace).ProxyGet("http", pods[0].GetName(), "9000", "metrics", nil).DoRaw(ctx)
			if err != nil && (apierrors.IsServiceUnavailable(err) || apierrors.IsInternalError(err) || apierrors.IsForbidden(err)) {
				body, err = env.CS.CoreV1().Pods(env.TFNamespace).ProxyGet("https", pods[0].GetName(), "9000", "metrics", nil).DoRaw(ctx)
			}
			Expect(err).NotTo(HaveOccurred())
			Expect(string(body)).To(ContainSubstring("controller_runtime_reconcile_total"))
			c.Measure("metrics_bytes", float64(len(body)), "B")
		},
	})

	Register(Case{
		ID: "OBS-06", Title: "GreptimeDB 可通过 HTTP SQL 查询", Tiers: []string{TierSmoke}, Needs: []string{NeedTF}, NoNS: true,
		Run: func(ctx SpecContext, c *C) {
			rows, err := env.greptimeSQL(ctx, "SHOW TABLES")
			Expect(err).NotTo(HaveOccurred(), "GreptimeDB HTTP SQL API")
			var tables []string
			for _, r := range rows {
				if len(r) > 0 && strings.HasPrefix(fmt.Sprint(r[0]), "tf_") {
					tables = append(tables, fmt.Sprint(r[0]))
				}
			}
			// Which tables exist without any GPU node is recorded, not asserted: the
			// row-level assertions live in OBS-02 (needs a workload on a GPU).
			c.Note("tf_* tables present: %v", tables)
			c.Measure("tf_tables", float64(len(tables)), "")
		},
	})
})

func containerEnv(p *unstructured.Unstructured, idx int) map[string]string {
	out := map[string]string{}
	cs := sliceAt(p, "spec", "containers")
	if idx >= len(cs) {
		return out
	}
	envs, _, _ := unstructured.NestedSlice(cs[idx].(map[string]any), "env")
	for _, e := range envs {
		m := e.(map[string]any)
		out[fmt.Sprint(m["name"])] = fmt.Sprint(m["value"])
	}
	return out
}

func names(items []unstructured.Unstructured) []string {
	var out []string
	for i := range items {
		out = append(out, items[i].GetName())
	}
	return out
}
