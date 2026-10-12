//go:build e2e

package suite

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Cases that consume GPU capacity. They run unchanged on real GPUs (tier core /
// karpenter-finops) and on mock GPUs (tier rehearsal); everything that depends
// on a CUDA device is guarded by env.Mode == ModeRealGPU and reported per case.

// tfNodeSelector is what a worker pod needs so that Karpenter grows the right
// NodePool for it. Mock mode: the mock pool. Real mode: nothing - the product
// promise is "label + annotations only".
func tfNodeSelector() PodOpt {
	if env.Mode == ModeMockGPU {
		return withNodeSelector(map[string]string{"e2e.tensor-fusion.ai/mock-gpu": "true"})
	}
	return func(map[string]any) {}
}

// fraction is the standard fractional request of the suite, expressed against
// the GPU that is actually there: pct% of its compute, vramGi of its memory.
func fraction(g GPU, isolation string, pct float64, vramGi int) map[string]string {
	t := fmt.Sprintf("%.0f", g.CapTflops*pct/100)
	v := fmt.Sprintf("%dGi", vramGi)
	return ann("isolation", isolation, "tflops-request", t, "tflops-limit", t, "vram-request", v, "vram-limit", v)
}

// wholeGPU asks for one entire device (isolation shared takes the full card).
func wholeGPU() map[string]string {
	return ann("isolation", "shared", "gpu-count", "1")
}

// holdNodes keeps Karpenter and TensorFusion from removing the current GPU
// nodes while a case needs them empty-but-present; undone on cleanup.
func (e *Env) holdNodes(ctx context.Context) {
	for _, g := range e.poolGPUs(ctx) {
		node := newObj(gvkNode, "", g.Node)
		if err := e.patchMerge(ctx, node, `{"metadata":{"annotations":{"karpenter.sh/do-not-disrupt":"true"}}}`); err != nil {
			GinkgoWriter.Printf("hold node %s: %v\n", g.Node, err)
		}
		name := g.Node
		DeferCleanup(func() {
			_ = e.patchMerge(context.Background(), newObj(gvkNode, "", name), `{"metadata":{"annotations":{"karpenter.sh/do-not-disrupt":null}}}`)
		})
	}
}

// ensureGPUs makes sure the pool has at least n GPUs, growing it the way a user
// would: by creating workloads and letting the autoscaling path provide nodes.
// Returns the GPUs and how long the scale-up took (0 when nothing was needed).
func (e *Env) ensureGPUs(ctx context.Context, c *C, n int) ([]GPU, time.Duration) {
	GinkgoHelper()
	if gpus := e.readyGPUs(ctx); len(gpus) >= n {
		e.holdNodes(ctx)
		return gpus, 0
	}
	start := time.Now()
	for i := 0; i < n; i++ {
		Expect(e.create(ctx, tfPod(c.NS, fmt.Sprintf("warm-%d", i), wholeGPU(), tfNodeSelector()))).To(Succeed())
	}
	waitFor(ctx, fmt.Sprintf("%d GPUs Running in pool %s", n, e.Pool), 12*time.Minute, func() (string, bool) {
		gpus := e.readyGPUs(ctx)
		var claims []string
		for _, nc := range e.nodeClaims(ctx, "") {
			claims = append(claims, nodeClaimWhy(&nc))
		}
		return fmt.Sprintf("%d GPUs; nodeclaims: %s", len(gpus), strings.Join(claims, " | ")), len(gpus) >= n
	})
	for i := 0; i < n; i++ {
		e.waitPodReady(ctx, c.NS, fmt.Sprintf("warm-%d", i), 5*time.Minute)
	}
	d := time.Since(start)
	e.holdNodes(ctx)
	for i := 0; i < n; i++ {
		e.delete(ctx, newObj(gvkPod, c.NS, fmt.Sprintf("warm-%d", i)))
	}
	e.waitResourcesReturned(ctx, 3*time.Minute)
	return e.readyGPUs(ctx), d
}

func (e *Env) readyGPUs(ctx context.Context) []GPU {
	var out []GPU
	for _, g := range e.poolGPUs(ctx) {
		if g.Phase == "Running" {
			out = append(out, g)
		}
	}
	return out
}

func (e *Env) waitAllReady(ctx context.Context, ns string, names []string, timeout time.Duration) map[string]*unstructured.Unstructured {
	GinkgoHelper()
	out := map[string]*unstructured.Unstructured{}
	waitFor(ctx, fmt.Sprintf("pods %v Ready", names), timeout, func() (string, bool) {
		var notReady []string
		for _, n := range names {
			p, err := e.get(ctx, gvkPod, ns, n)
			if err != nil {
				notReady = append(notReady, n+": "+err.Error())
				continue
			}
			out[n] = p
			if !podReady(p) {
				notReady = append(notReady, n+": "+podWhy(p))
			}
		}
		return strings.Join(notReady, " | "), len(notReady) == 0
	})
	return out
}

var _ = Describe("discovery", func() {
	Register(Case{
		ID: "INS-03", Title: "GPU 节点与 GPU 自动发现（含 KRP-A01 纳管 Karpenter 节点）", Tiers: []string{TierCore, TierRehearsal, TierKarpenter, TierKWOK},
		Needs: []string{NeedTF}, Timeout: 20 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			if env.Mode == ModeNoGPU {
				Skip("no GPU capacity and no mock pool: run `make mock-install` (rehearsal) or wait for GPU quota")
			}
			gpus, scale := env.ensureGPUs(ctx, c, 1)
			if scale > 0 {
				c.Measure("cold_start_to_gpu_ready_seconds", scale.Seconds(), "s")
			}
			g := gpus[0]
			c.Note("GPU %s model=%q vendor=%q capacity=%.0f TFLOPS / %.1f GiB on node %s", g.Name, g.Model, g.Vendor, g.CapTflops, g.CapVRAM, g.Node)
			Expect(g.CapTflops).To(BeNumerically(">", 0))
			Expect(g.CapVRAM).To(BeNumerically(">", 1))
			if env.Mode == ModeRealGPU {
				Expect(g.Vendor).To(Equal("NVIDIA"))
			}

			gn, err := env.get(ctx, gvkGPUNode, "", g.Node)
			Expect(err).NotTo(HaveOccurred())
			Expect(str(gn, "status", "phase")).To(Equal("Running"))

			By("hypervisor pod is Ready and owned by both the GPUNode and the Node (so Karpenter may delete the node under it)")
			hv, err := env.get(ctx, gvkPod, env.TFNamespace, "hypervisor-"+g.Node)
			Expect(err).NotTo(HaveOccurred())
			Expect(podReady(hv)).To(BeTrue(), podWhy(hv))
			kinds := map[string]bool{}
			for _, o := range hv.GetOwnerReferences() {
				kinds[o.Kind] = true
			}
			Expect(kinds).To(HaveKey("GPUNode"))
			Expect(kinds).To(HaveKey("Node"))

			By("the node advertises the 16 tensor-fusion.ai/index_* resources")
			node, err := env.get(ctx, gvkNode, "", g.Node)
			Expect(err).NotTo(HaveOccurred())
			idx := 0
			for k := range mapAt(node, "status", "allocatable") {
				if strings.HasPrefix(k, tfDomain+"/index_") {
					idx++
				}
			}
			Expect(idx).To(Equal(16))
			if env.HasKarpenter {
				Expect(node.GetLabels()).To(HaveKey("karpenter.sh/nodepool"), "GPU node is not owned by a Karpenter NodePool: the expander cannot clone it")
			}
			_, hasPoolLabel := node.GetLabels()[tfDomain+"/pool-"+env.Pool]
			c.Note("node carries pool label %s/pool-%s: %v (the hypervisor pod and the GPUNode do)", tfDomain, env.Pool, hasPoolLabel)

			By("pool status aggregates the GPUs")
			waitFor(ctx, "GPUPool totals", 2*time.Minute, func() (string, bool) {
				p, _ := env.get(ctx, gvkGPUPool, "", env.Pool)
				return toJSON(mapAt(p, "status")), str(p, "status", "phase") == "Running" && qty(p, "status", "totalTFlops") >= g.CapTflops
			})
		},
	})
})

var _ = Describe("sharing and bin-packing", func() {
	Register(Case{
		ID: "ISO-01", Title: "soft 分片：一张卡上运行多个 Pod", Tiers: []string{TierCore, TierRehearsal}, Needs: []string{NeedTF, NeedGPU}, Timeout: 20 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			names := []string{"s-0", "s-1", "s-2"}
			for _, n := range names {
				Expect(env.create(ctx, tfPod(c.NS, n, fraction(g, "soft", 20, 2), tfNodeSelector()))).To(Succeed())
			}
			pods := env.waitAllReady(ctx, c.NS, names, 8*time.Minute)

			ids := map[string]bool{}
			for _, p := range pods {
				for _, id := range podGPUIDs(p) {
					ids[id] = true
				}
			}
			Expect(ids).To(HaveLen(1), "three 20%% pods must share one GPU, got %v", ids)

			var gid string
			for id := range ids {
				gid = id
			}
			// The allocator syncs its in-memory state to the GPU CR asynchronously.
			var got GPU
			wantT := 3 * float64(int(g.CapTflops*0.2+0.5))
			d := waitFor(ctx, "GPU CR accounting to show three allocations", 2*time.Minute, func() (string, bool) {
				u, err := env.get(ctx, gvkGPU, "", gid)
				if err != nil {
					return err.Error(), false
				}
				got = toGPU(u)
				usedT, usedV := got.CapTflops-got.AvailTflops, got.CapVRAM-got.AvailVRAM
				return fmt.Sprintf("used %.1f TFLOPS (want %.1f), %.1f GiB (want 6), mode %q", usedT, wantT, usedV, got.Mode),
					usedT > wantT-1.5 && usedT < wantT+1.5 && usedV > 5.9 && usedV < 6.1
			})
			c.Measure("gpu_cr_accounting_lag_seconds", d.Seconds(), "s")
			Expect(got.Mode).To(Equal("soft"), "GPU.status.activeIsolationMode")
			c.Measure("pods_on_one_gpu", 3, "")

			if env.Mode == ModeRealGPU {
				By("every container sees exactly its GPU with its VRAM quota")
				for _, n := range names {
					out, err := env.exec(ctx, c.NS, n, "main", "nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits")
					Expect(err).NotTo(HaveOccurred())
					c.Note("%s nvidia-smi memory.total=%s MiB", n, strings.TrimSpace(out))
				}
			}

			for _, n := range names {
				env.delete(ctx, newObj(gvkPod, c.NS, n))
			}
			c.Measure("gpu_return_seconds", env.waitResourcesReturned(ctx, 3*time.Minute).Seconds(), "s")
		},
	})

	Register(Case{
		ID: "ISO-05", Title: "shared 整卡：独占后其它 Pod 不会落到这张卡", Tiers: []string{TierCore, TierRehearsal}, Needs: []string{NeedTF, NeedGPU}, Timeout: 20 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			env.ensureGPUs(ctx, c, 1)
			Expect(env.create(ctx, tfPod(c.NS, "whole", wholeGPU(), tfNodeSelector()))).To(Succeed())
			p, _ := env.waitPodReady(ctx, c.NS, "whole", 8*time.Minute)
			ids := podGPUIDs(p)
			Expect(ids).To(HaveLen(1), "annotations: %v", p.GetAnnotations())
			var got GPU
			waitFor(ctx, "GPU CR to show the whole card allocated", 2*time.Minute, func() (string, bool) {
				u, err := env.get(ctx, gvkGPU, "", ids[0])
				if err != nil {
					return err.Error(), false
				}
				got = toGPU(u)
				return fmt.Sprintf("available %.1f TFLOPS / %.1f GiB", got.AvailTflops, got.AvailVRAM), got.AvailTflops < 0.01 && got.AvailVRAM < 0.01
			})
			c.Note("GPU %s fully allocated to the shared pod (activeIsolationMode=%s)", got.Name, got.Mode)

			By("a fractional pod cannot be placed on the card that is taken as a whole")
			Expect(env.create(ctx, tfPod(c.NS, "late", fraction(got, "soft", 10, 1), tfNodeSelector()))).To(Succeed())
			deadline := time.Now().Add(40 * time.Second)
			for time.Now().Before(deadline) {
				late, _ := env.get(ctx, gvkPod, c.NS, "late")
				Expect(podGPUIDs(late)).NotTo(ContainElement(ids[0]), "fractional pod was placed on a whole-GPU card")
				time.Sleep(poll)
			}
			env.delete(ctx, newObj(gvkPod, c.NS, "late"))
		},
	})

	Register(Case{
		ID: "ISO-14", Title: "soft 模式对业务镜像的要求：无 libstdc++ 的镜像进程无法启动", Tiers: []string{TierRehearsal, TierCore}, Needs: []string{NeedTF, NeedGPU}, Timeout: 15 * time.Minute,
		KnownIssue: "the soft limiter is injected through /etc/ld.so.preload and needs glibc + libstdc++.so.6 in the workload image; on images without it every dynamically linked process exits 127 (<= 2.17.23, vgpu-provider 1.3.9)",
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			img := getenv("E2E_IMAGE_NO_LIBSTDCXX", "public.ecr.aws/docker/library/busybox:1.37")
			Expect(env.create(ctx, tfPod(c.NS, "minimal", fraction(gpus[0], "soft", 10, 1), tfNodeSelector(), withImage(img, "sleep", "86400")))).To(Succeed())
			// Expected (correct) behaviour: the container runs. A missing dependency
			// of the preloaded limiter must not take the workload down.
			waitFor(ctx, "container of a libstdc++-less image to stay up under soft isolation", 4*time.Minute, func() (string, bool) {
				p, err := env.get(ctx, gvkPod, c.NS, "minimal")
				if err != nil {
					return err.Error(), false
				}
				if podReady(p) {
					return "", true
				}
				why := podWhy(p)
				if strings.Contains(why, "exited 127") || strings.Contains(why, "CrashLoopBackOff") {
					logs := env.podLogs(ctx, c.NS, "minimal", "main")
					Fail("workload container cannot start under soft isolation: " + firstLine(logs) + " (" + why + ")")
				}
				return why, false
			})
		},
	})

	Register(Case{
		ID: "SCH-01", Title: "默认装箱：小负载集中到一个节点", Tiers: []string{TierCore, TierRehearsal, TierKarpenter}, Needs: []string{NeedTF, NeedGPU}, Timeout: 25 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 2)
			nodes := map[string]bool{}
			for _, g := range gpus {
				nodes[g.Node] = true
			}
			if len(nodes) < 2 {
				Skip("needs GPUs on two different nodes; all GPUs are on one node")
			}
			names := []string{"p-0", "p-1", "p-2", "p-3"}
			for _, n := range names {
				Expect(env.create(ctx, tfPod(c.NS, n, fraction(gpus[0], "soft", 20, 2), tfNodeSelector()))).To(Succeed())
				env.waitPodReady(ctx, c.NS, n, 6*time.Minute) // one at a time: placement must compact, not race
			}
			used := map[string]int{}
			for _, n := range names {
				p, _ := env.get(ctx, gvkPod, c.NS, n)
				used[str(p, "spec", "nodeName")]++
			}
			c.Note("pods per node: %v (placement mode NodeCompactGPULowLoad)", used)
			Expect(used).To(HaveLen(1), "4 x 20%% pods fit one GPU and must not be spread over nodes")
			c.Measure("nodes_used", float64(len(used)), "")
		},
	})
})

var _ = Describe("recovery", func() {
	Register(Case{
		ID: "NLC-01", Title: "Hypervisor 重启：业务不中断、GPU 先 Pending 后恢复、不丢账", Tiers: []string{TierCore, TierRehearsal}, Needs: []string{NeedTF, NeedGPU}, Timeout: 25 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			Expect(env.create(ctx, tfPod(c.NS, "app", fraction(gpus[0], "soft", 20, 2), tfNodeSelector()))).To(Succeed())
			p, _ := env.waitPodReady(ctx, c.NS, "app", 8*time.Minute)
			node := str(p, "spec", "nodeName")
			gid := podGPUIDs(p)[0]
			before, _ := env.get(ctx, gvkGPU, "", gid)
			availBefore := toGPU(before).AvailTflops

			hvName := "hypervisor-" + node
			old, err := env.get(ctx, gvkPod, env.TFNamespace, hvName)
			Expect(err).NotTo(HaveOccurred())
			oldUID := old.GetUID()
			start := time.Now()
			env.delete(ctx, old)

			sawPending := false
			waitFor(ctx, "replacement hypervisor Ready and GPU back to Running", 8*time.Minute, func() (string, bool) {
				g, _ := env.get(ctx, gvkGPU, "", gid)
				phase := str(g, "status", "phase")
				if phase == "Pending" {
					sawPending = true
				}
				hv, err := env.get(ctx, gvkPod, env.TFNamespace, hvName)
				if err != nil {
					return "hypervisor pod: " + err.Error(), false
				}
				return fmt.Sprintf("gpu phase=%s hypervisor: %s", phase, podWhy(hv)), hv.GetUID() != oldUID && podReady(hv) && phase == "Running"
			})
			c.Measure("hypervisor_recovery_seconds", time.Since(start).Seconds(), "s")
			c.Note("GPU went through phase Pending during the replacement: %v (poll interval %s)", sawPending, poll)

			app, _ := env.get(ctx, gvkPod, c.NS, "app")
			Expect(podReady(app)).To(BeTrue(), "workload pod must survive a hypervisor restart: %s", podWhy(app))
			Expect(restarts(app)).To(Equal(0), "workload container restarted")
			after, _ := env.get(ctx, gvkGPU, "", gid)
			Expect(toGPU(after).AvailTflops).To(BeNumerically("~", availBefore, 0.01), "allocation must be unchanged after the restart")

			env.delete(ctx, app)
			env.waitResourcesReturned(ctx, 3*time.Minute)
		},
	})

	Register(Case{
		ID: "NLC-02", Title: "Operator 重启：分配状态重建、无重复分配", Tiers: []string{TierCore, TierRehearsal}, Needs: []string{NeedTF, NeedGPU}, Timeout: 25 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			Expect(env.create(ctx, tfPod(c.NS, "a", fraction(g, "soft", 60, 4), tfNodeSelector()))).To(Succeed())
			env.waitPodReady(ctx, c.NS, "a", 8*time.Minute)
			before := allocationKeys(ctx)
			Expect(before).NotTo(BeEmpty())

			pods, _ := env.list(ctx, gvkPod, env.TFNamespace, client.MatchingLabels{lblComponent: "operator"})
			Expect(pods).NotTo(BeEmpty())
			start := time.Now()
			for i := range pods {
				env.delete(ctx, &pods[i])
			}
			waitFor(ctx, "controller back and allocation API answering", 6*time.Minute, func() (string, bool) {
				dep, err := env.get(ctx, gvkDeployment, env.TFNamespace, env.TFRelease+"-controller")
				if err != nil || condition(dep, "Available") != "True" || str(dep, "status", "unavailableReplicas") != "" {
					return "deployment not available", false
				}
				after := allocationKeys(ctx)
				return fmt.Sprintf("allocations before=%v after=%v", before, after), len(after) == len(before)
			})
			c.Measure("operator_recovery_seconds", time.Since(start).Seconds(), "s")
			Expect(allocationKeys(ctx)).To(Equal(before), "allocator state after restart differs")

			By("a pod that no longer fits the same GPU must not be double-allocated onto it")
			aPod, _ := env.get(ctx, gvkPod, c.NS, "a")
			gid := podGPUIDs(aPod)[0]
			Expect(env.create(ctx, tfPod(c.NS, "b", fraction(g, "soft", 60, 4), tfNodeSelector()))).To(Succeed())
			deadline := time.Now().Add(60 * time.Second)
			for time.Now().Before(deadline) {
				bPod, _ := env.get(ctx, gvkPod, c.NS, "b")
				Expect(podGPUIDs(bPod)).NotTo(ContainElement(gid), "pod b was allocated the GPU that pod a already uses 60%% of")
				time.Sleep(poll)
			}
			u, _ := env.get(ctx, gvkGPU, "", gid)
			Expect(toGPU(u).AvailTflops).To(BeNumerically("~", g.CapTflops*0.4, 1.5), "GPU %s available TFLOPS after restart", gid)
			env.delete(ctx, newObj(gvkPod, c.NS, "b"))
		},
	})
})

func restarts(p *unstructured.Unstructured) int {
	n := 0
	for _, cs := range sliceAt(p, "status", "containerStatuses") {
		m, _ := cs.(map[string]any)
		switch v := m["restartCount"].(type) {
		case int64:
			n += int(v)
		case float64:
			n += int(v)
		}
	}
	return n
}

// allocationKeys returns the pod UIDs the allocator currently accounts for.
func allocationKeys(ctx context.Context) []string {
	raw, err := env.controllerAPI(ctx, "api/allocation", nil)
	if err != nil {
		return nil
	}
	var resp struct {
		Allocation map[string]json.RawMessage `json:"allocation"`
	}
	if json.Unmarshal(raw, &resp) != nil {
		return nil
	}
	var keys []string
	for k := range resp.Allocation {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
