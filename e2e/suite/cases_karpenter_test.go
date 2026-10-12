//go:build e2e

package suite

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// Karpenter compatibility, path A (GPUPool AutoSelect + NodePool) and FinOps
// accounting. Path B (provisioningMode: Karpenter) is in cases_karpenter_b_test.go.

func targetNodePool() string {
	if env.Mode == ModeMockGPU {
		return env.MockNodePool
	}
	return env.GPUNodePool
}

// finopsPriceAs is the instance type whose price the FinOps numbers use. On
// real GPUs: the instance that actually ran. In the rehearsal the nodes are CPU
// instances standing in for single-GPU ones, so the GPU price is substituted
// and the case is reported as mock-gpu.
func finopsPriceAs() string {
	if env.Mode == ModeMockGPU {
		return getenv("E2E_FINOPS_GPU_TYPE", "g4dn.xlarge")
	}
	return ""
}

// expansionBindLag is measured by KRP-A02 and judged by KRP-A16.
var expansionBindLag time.Duration

// gpuReadyBindLag is the sharper form of the same number: from the moment the
// new node's GPU CR turned Running (sampled twice a second) to the pod being
// bound. Before the re-queue fix of 2.17.24 this was the scheduler's 5-minute
// unschedulable flush. Measured by KRP-A02, judged by KRP-A16.
var (
	gpuReadyBindLag         time.Duration
	gpuReadyBindLagMeasured bool
)

// gpuWatch records when each GPU of the pool was first seen Running.
type gpuWatch struct {
	mu     sync.Mutex
	seen   map[string]time.Time
	cancel context.CancelFunc
}

func watchGPUsRunning(parent context.Context) *gpuWatch {
	ctx, cancel := context.WithCancel(parent)
	w := &gpuWatch{seen: map[string]time.Time{}, cancel: cancel}
	for _, g := range env.readyGPUs(ctx) {
		w.seen[g.Name] = time.Time{} // already Running before the case: not a measurement
	}
	go func() {
		t := time.NewTicker(500 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
			}
			now := time.Now()
			for _, g := range env.readyGPUs(ctx) {
				w.mu.Lock()
				if _, ok := w.seen[g.Name]; !ok {
					w.seen[g.Name] = now
				}
				w.mu.Unlock()
			}
		}
	}()
	return w
}

func (w *gpuWatch) runningAt(gpu string) time.Time {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.seen[gpu]
}

func claimEvents(ctx context.Context, name string) []string {
	return env.events(ctx, "default", "NodeClaim", name)
}

var _ = Describe("karpenter path A", func() {
	Register(Case{
		ID: "KRP-A08", Title: "从 0 个 GPU 节点冷启动：Pending 的 TF Pod → Karpenter NodeClaim → 节点 → GPU 注册 → Pod 运行",
		Tiers: []string{TierKarpenter, TierRehearsal, TierDryRun, TierKWOK}, Needs: []string{NeedTF, NeedKarpenter}, Timeout: 45 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			pool := targetNodePool()
			// Nodes of earlier cases leave on the NodePool's consolidateAfter (10m by
			// default in this stack) plus drain and termination.
			waitFor(ctx, "NodePool "+pool+" empty (cold start precondition)", 20*time.Minute, func() (string, bool) {
				ncs := env.nodeClaims(ctx, pool)
				for i := range ncs {
					_ = env.patchMerge(ctx, newObj(gvkNode, "", str(&ncs[i], "status", "nodeName")), `{"metadata":{"annotations":{"karpenter.sh/do-not-disrupt":null}}}`)
				}
				return fmt.Sprintf("NodeClaims: %v", nodeClaimNames(ncs)), len(ncs) == 0
			})
			led := newNodeLedger("")
			start := time.Now()
			// No nodeSelector in real mode: exactly what the docs tell users to write.
			Expect(env.create(ctx, tfPod(c.NS, "cold", ann("isolation", "soft", "compute-percent-request", "20", "compute-percent-limit", "20", "vram-request", "2Gi", "vram-limit", "2Gi"), tfNodeSelector()))).To(Succeed())

			var nominated string
			waitFor(ctx, "Karpenter nominates a NodeClaim for the pod", 3*time.Minute, func() (string, bool) {
				led.observe(ctx)
				evs := env.events(ctx, c.NS, "Pod", "cold")
				for _, ev := range evs {
					if strings.HasPrefix(ev, "Nominated:") {
						nominated = ev[strings.LastIndex(ev, "/")+1:]
						return "", true
					}
				}
				return strings.Join(evs, " | "), false
			})
			c.Measure("pending_to_nodeclaim_seconds", time.Since(start).Seconds(), "s")
			Expect(nominated).To(HavePrefix(pool+"-"), "Karpenter picked NodePool of %q; a TensorFusion pod must only ever grow the TensorFusion NodePool %q", nominated, pool)

			// Without GPU quota the launch fails at EC2. That is the one expected
			// failure of the dry run: capture the error text as evidence and stop.
			var quotaErr string
			var gpu GPU
			waitFor(ctx, "node joins and its GPU registers (or EC2 refuses the launch)", 12*time.Minute, func() (string, bool) {
				led.observe(ctx)
				for _, ev := range append(claimEvents(ctx, nominated), allClaimEvents(ctx, pool)...) {
					if strings.Contains(ev, "InsufficientCapacityError") && (strings.Contains(ev, "VcpuLimitExceeded") || strings.Contains(ev, "MaxSpotInstanceCountExceeded")) {
						quotaErr = ev
					}
				}
				if gpus := env.readyGPUs(ctx); len(gpus) > 0 {
					gpu = gpus[0]
					return "", true
				}
				if quotaErr != "" && len(env.nodeClaims(ctx, pool)) == 0 && env.Mode != ModeMockGPU {
					return "", true
				}
				var why []string
				for _, nc := range env.nodeClaims(ctx, pool) {
					why = append(why, nodeClaimWhy(&nc))
				}
				return strings.Join(why, " | "), false
			})
			if gpu.Name == "" {
				c.Note("EC2 refused the GPU launch (quota), everything before it worked: %s", firstLine(quotaErr))
				for _, ev := range allClaimEvents(ctx, pool) {
					if strings.Contains(ev, "InsufficientCapacityError") {
						c.Note("NodeClaim event: %s", firstLine(ev))
					}
				}
				c.SetMode(ModeDryRun)
				Skip("dry-run result: NodeClaim for NodePool " + pool + " was created and EC2 answered with a quota error; no GPU capacity to continue")
			}
			c.Measure("pending_to_gpu_registered_seconds", time.Since(start).Seconds(), "s")
			_, d := env.waitPodReady(ctx, c.NS, "cold", 8*time.Minute)
			_ = d
			c.Measure("pending_to_pod_running_seconds", time.Since(start).Seconds(), "s")
			c.Note("first GPU %s (%s) on %s", gpu.Name, gpu.Model, gpu.Node)
			led.observe(ctx)
			led.report(c, finopsPriceAs())
		},
	})

	Register(Case{
		ID: "KRP-A15", Title: "chart 默认 NodeOverlay 会让 Karpenter 为 TF Pod 拉起非 TF 节点", Tiers: []string{TierKarpenter, TierDryRun, TierKWOK},
		Needs: []string{NeedTF, NeedKarpenter}, Timeout: 12 * time.Minute, Mode: ModeNoGPU,
		KnownIssue: "NodeOverlay/tensor-fusion-overlay has `requirements: []` (chart 1.8.8): every NodePool advertises tensor-fusion.ai/index_*, so Karpenter launches a non-GPU node for a pending TF pod; the NodeClaim stays Initialized=Unknown (ResourceNotRegistered) and is never consolidated",
		Run: func(ctx SpecContext, c *C) {
			if ov, err := env.get(ctx, gvkNodeOverlay, "", "tensor-fusion-overlay"); err == nil {
				c.Note("chart overlay requirements: %s", toJSON(sliceAt(ov, "spec", "requirements")))
			}
			DeferCleanup(func() {
				// Whatever the bug launched must not outlive the case.
				bg := context.Background()
				for _, nc := range env.nodeClaims(bg, env.CPUNodePool) {
					env.delete(bg, &nc)
				}
			})
			guard, err := env.get(ctx, gvkNodeOverlay, "", "tf-e2e-no-index")
			if err != nil {
				// No stack guard (KWOK tier): the cluster runs with the chart overlay
				// only, which is exactly the configuration the case is about.
				c.Note("no guard overlay tf-e2e-no-index on this cluster: asserting against the chart overlay alone")
			} else {
				// Take the stack's guard away for the duration of the case.
				saved := guard.DeepCopy()
				saved.SetResourceVersion("")
				saved.SetUID("")
				unstructured.RemoveNestedField(saved.Object, "status")
				unstructured.RemoveNestedField(saved.Object, "metadata", "managedFields")
				unstructured.RemoveNestedField(saved.Object, "metadata", "creationTimestamp")
				unstructured.RemoveNestedField(saved.Object, "metadata", "generation")
				env.delete(ctx, guard)
				DeferCleanup(func() { _ = env.C.Create(context.Background(), saved) })
				time.Sleep(20 * time.Second) // Karpenter re-evaluates overlays on its own cadence
			}

			Expect(env.create(ctx, tfPod(c.NS, "stray", ann("isolation", "soft", "compute-percent-request", "20", "vram-request", "1Gi", lblPool[len(tfDomain)+1:], env.RealPool)))).To(Succeed())
			// Correct behaviour: no NodeClaim outside the GPU NodePool, ever.
			never(ctx, "Karpenter launched a non-GPU node for a TensorFusion pod", 90*time.Second, func() (string, bool) {
				for _, pool := range []string{env.CPUNodePool, env.MockNodePool} {
					if pool == targetNodePool() {
						continue
					}
					for _, nc := range env.nodeClaims(ctx, pool) {
						return nodeClaimWhy(&nc), true
					}
				}
				return "", false
			})
		},
	})

	Register(Case{
		ID: "KRP-A02", Title: "GPU 不足 → 扩容器创建 NodeClaim → 新节点 → Pod 运行（核心闭环）", Tiers: []string{TierKarpenter, TierRehearsal, TierKWOK},
		Needs: []string{NeedTF, NeedKarpenter, NeedGPU}, Timeout: 30 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			env.ensureGPUs(ctx, c, 1)
			By("occupy every GPU that exists")
			var fillers []string
			for i := range env.readyGPUs(ctx) {
				n := fmt.Sprintf("fill-%d", i)
				fillers = append(fillers, n)
				Expect(env.create(ctx, tfPod(c.NS, n, wholeGPU(), tfNodeSelector()))).To(Succeed())
			}
			env.waitAllReady(ctx, c.NS, fillers, 8*time.Minute)
			before := nodeClaimNames(env.nodeClaims(ctx, ""))
			gpusBefore := len(env.readyGPUs(ctx))
			led := newNodeLedger("")
			gw := watchGPUsRunning(ctx)
			defer gw.cancel()

			start := time.Now()
			g := env.readyGPUs(ctx)[0]
			Expect(env.create(ctx, tfPod(c.NS, "extra", fraction(g, "soft", 20, 2), tfNodeSelector()))).To(Succeed())

			var created []unstructured.Unstructured
			waitFor(ctx, "a new NodeClaim", 4*time.Minute, func() (string, bool) {
				led.observe(ctx)
				created = created[:0]
				for _, nc := range env.nodeClaims(ctx, "") {
					if !contains(before, nc.GetName()) {
						created = append(created, nc)
					}
				}
				return "pod events: " + strings.Join(env.events(ctx, c.NS, "Pod", "extra"), " | "), len(created) > 0
			})
			c.Measure("pending_to_nodeclaim_seconds", time.Since(start).Seconds(), "s")

			extra, _ := env.waitPodReady(ctx, c.NS, "extra", 12*time.Minute)
			c.Measure("pending_to_pod_running_seconds", time.Since(start).Seconds(), "s")
			led.observe(ctx)
			// How long the new node sat "empty" before its pod was bound: the lower
			// bound for the NodePool's consolidateAfter (Karpenter removes a node the
			// expander added if it stays empty longer than that).
			for _, nc := range env.nodeClaims(ctx, "") {
				if str(&nc, "status", "nodeName") != str(extra, "spec", "nodeName") {
					continue
				}
				if ready, bound := conditionTime(&nc, "Initialized"), conditionTime(extra, "PodScheduled"); !ready.IsZero() && !bound.IsZero() {
					expansionBindLag = bound.Sub(ready)
					c.Measure("node_initialized_to_pod_bound_seconds", expansionBindLag.Seconds(), "s")
				}
			}

			// The same lag from the GPU's side: GPU CR Running (500 ms sampling) to
			// PodScheduled (1 s resolution of the condition timestamp).
			if ids := podGPUIDs(extra); len(ids) > 0 {
				if ready, bound := gw.runningAt(ids[0]), conditionTime(extra, "PodScheduled"); !ready.IsZero() && !bound.IsZero() {
					gpuReadyBindLag, gpuReadyBindLagMeasured = bound.Sub(ready), true
					if gpuReadyBindLag < 0 {
						gpuReadyBindLag = 0
					}
					c.Measure("gpu_running_to_pod_bound_seconds", gpuReadyBindLag.Seconds(), "s")
				}
			}

			By("exactly one node was added, by the TensorFusion expander, in the pool's NodePool")
			created = created[:0]
			for _, nc := range env.nodeClaims(ctx, "") {
				if !contains(before, nc.GetName()) {
					created = append(created, nc)
				}
			}
			var desc []string
			byExpander := 0
			for i := range created {
				nc := &created[i]
				src := nc.GetLabels()[tfDomain+"/expansion-source"]
				if src != "" {
					byExpander++
				}
				desc = append(desc, fmt.Sprintf("%s(pool=%s expansion-source=%q do-not-disrupt=%q owner=%v)", nc.GetName(),
					nc.GetLabels()["karpenter.sh/nodepool"], src, nc.GetAnnotations()["karpenter.sh/do-not-disrupt"], ownerKinds(nc)))
				Expect(nc.GetLabels()["karpenter.sh/nodepool"]).To(Equal(targetNodePool()))
				Expect(nc.GetAnnotations()).NotTo(HaveKeyWithValue("karpenter.sh/do-not-disrupt", "true"), "path A NodeClaims must stay disruptable")
			}
			c.Note("NodeClaims created for one pending pod: %s", strings.Join(desc, ", "))
			evs := env.events(ctx, c.NS, "Pod", "extra")
			c.Note("pod events: %s", strings.Join(compactEvents(evs), " | "))
			c.Measure("nodeclaims_created", float64(len(created)), "")
			c.Measure("nodeclaims_by_tf_expander", float64(byExpander), "")
			Expect(created).To(HaveLen(1), "one pending pod must cost one node; got %s", strings.Join(desc, ", "))
			Expect(len(env.readyGPUs(ctx))).To(BeNumerically(">", gpusBefore))
			led.report(c, finopsPriceAs())
		},
	})

	Register(Case{
		ID: "KRP-A16", Title: "扩容节点的 GPU 就绪后，Pending Pod 应及时绑定（不等调度器 5 分钟兜底重试）", Tiers: []string{TierKarpenter, TierRehearsal, TierKWOK},
		Needs: []string{NeedTF, NeedKarpenter, NeedGPU}, NoNS: true,
		KnownIssue: "a pod that failed scheduling while the new node's GPU was not registered yet is not re-queued when the GPU CR becomes Running; it is retried only by the scheduler's 5-minute unschedulable flush. Until then the new node is empty for Karpenter, so with consolidateAfter <= 5m the node is removed before the pod lands (observed twice in the rehearsal, operator 2.17.23)",
		Run: func(ctx SpecContext, c *C) {
			if expansionBindLag == 0 {
				Skip("KRP-A02 did not measure the lag in this run (run them together: FOCUS='KRP-A02|KRP-A16')")
			}
			c.Measure("node_initialized_to_pod_bound_seconds", expansionBindLag.Seconds(), "s")
			// Hypervisor start + GPU registration took 15-30 s on mock nodes; a real
			// GPU node needs image pulls on top. 120 s is generous for "promptly".
			Expect(expansionBindLag.Seconds()).To(BeNumerically("<", 120), "the pod waited %.0f s on a node that was ready for it", expansionBindLag.Seconds())
			if gpuReadyBindLagMeasured {
				c.Measure("gpu_running_to_pod_bound_seconds", gpuReadyBindLag.Seconds(), "s")
				// A re-queue on the GPU event binds within a scheduling cycle; 30 s
				// leaves room for a busy control plane and is far from the 300 s flush.
				Expect(gpuReadyBindLag.Seconds()).To(BeNumerically("<", 30), "the GPU was Running for %.0f s before the pod was bound", gpuReadyBindLag.Seconds())
			}
		},
	})

	Register(Case{
		ID: "KRP-A03", Title: "一次扩容承接多个 Pending Pod（不是一个 Pod 一台机器）", Tiers: []string{TierKarpenter, TierRehearsal, TierKWOK},
		Needs: []string{NeedTF, NeedKarpenter, NeedGPU}, Timeout: 30 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			env.ensureGPUs(ctx, c, 1)
			var fillers []string
			for i := range env.readyGPUs(ctx) {
				n := fmt.Sprintf("fill-%d", i)
				fillers = append(fillers, n)
				Expect(env.create(ctx, tfPod(c.NS, n, wholeGPU(), tfNodeSelector()))).To(Succeed())
			}
			env.waitAllReady(ctx, c.NS, fillers, 8*time.Minute)
			before := nodeClaimNames(env.nodeClaims(ctx, ""))
			led := newNodeLedger("")
			g := env.readyGPUs(ctx)[0]

			start := time.Now()
			names := []string{"burst-0", "burst-1", "burst-2"}
			for _, n := range names {
				Expect(env.create(ctx, tfPod(c.NS, n, fraction(g, "soft", 20, 2), tfNodeSelector()))).To(Succeed())
			}
			pods := env.waitAllReady(ctx, c.NS, names, 12*time.Minute)
			c.Measure("burst_to_all_running_seconds", time.Since(start).Seconds(), "s")
			led.observe(ctx)

			added := 0
			for _, nc := range env.nodeClaims(ctx, "") {
				if !contains(before, nc.GetName()) {
					added++
				}
			}
			nodes := map[string]bool{}
			gpuIDs := map[string]bool{}
			for _, p := range pods {
				nodes[str(p, "spec", "nodeName")] = true
				for _, id := range podGPUIDs(p) {
					gpuIDs[id] = true
				}
			}
			scaler, _ := env.controllerAPI(ctx, "api/node-scaler", nil)
			c.Note("node-scaler state after the burst: %s", firstLine(string(scaler)))
			c.Measure("nodeclaims_created", float64(added), "")
			c.Measure("nodes_hosting_burst", float64(len(nodes)), "")
			c.Measure("gpus_hosting_burst", float64(len(gpuIDs)), "")
			Expect(added).To(Equal(1), "3 x 20%% pods fit one GPU: exactly one new node expected")
			Expect(nodes).To(HaveLen(1))
			Expect(gpuIDs).To(HaveLen(1))
			led.report(c, finopsPriceAs())
		},
	})

	Register(Case{
		ID: "KRP-A17", Title: "N 个整卡 Pod 同时 Pending：恰好扩 N 台（不多扩、不少扩）", Tiers: []string{TierKarpenter, TierKWOK},
		Needs: []string{NeedTF, NeedKarpenter, NeedGPU}, Timeout: 30 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			n, _ := strconv.Atoi(getenv("E2E_BURST", "4"))
			env.ensureGPUs(ctx, c, 1)
			var fillers []string
			for i := range env.readyGPUs(ctx) {
				name := fmt.Sprintf("fill-%d", i)
				fillers = append(fillers, name)
				Expect(env.create(ctx, tfPod(c.NS, name, wholeGPU(), tfNodeSelector()))).To(Succeed())
			}
			env.waitAllReady(ctx, c.NS, fillers, 8*time.Minute)
			before := nodeClaimNames(env.nodeClaims(ctx, ""))
			led := newNodeLedger("")

			start := time.Now()
			var names []string
			for i := 0; i < n; i++ {
				names = append(names, fmt.Sprintf("burst-%d", i))
				Expect(env.create(ctx, tfPod(c.NS, names[i], wholeGPU(), tfNodeSelector()))).To(Succeed())
			}
			// Every NodeClaim that ever existed during the burst counts, including
			// ones Karpenter or the expander created and removed again.
			everSeen := map[string]string{}
			pods := map[string]*unstructured.Unstructured{}
			waitFor(ctx, fmt.Sprintf("%d whole-GPU pods Running", n), 20*time.Minute, func() (string, bool) {
				led.observe(ctx)
				for _, nc := range env.nodeClaims(ctx, "") {
					if !contains(before, nc.GetName()) {
						everSeen[nc.GetName()] = nc.GetLabels()["karpenter.sh/nodepool"] + "/" + nc.GetLabels()[tfDomain+"/expansion-source"]
					}
				}
				var notReady []string
				for _, name := range names {
					p, err := env.get(ctx, gvkPod, c.NS, name)
					if err != nil || !podReady(p) {
						notReady = append(notReady, name)
						continue
					}
					pods[name] = p
				}
				return fmt.Sprintf("not ready: %v; nodeclaims seen: %d", notReady, len(everSeen)), len(notReady) == 0
			})
			c.Measure("burst_to_all_running_seconds", time.Since(start).Seconds(), "s")
			// Late over-provisioning shows up after the pods are bound.
			never(ctx, "an extra NodeClaim after the burst was satisfied", 60*time.Second, func() (string, bool) {
				for _, nc := range env.nodeClaims(ctx, "") {
					if !contains(before, nc.GetName()) {
						everSeen[nc.GetName()] = nc.GetLabels()["karpenter.sh/nodepool"] + "/" + nc.GetLabels()[tfDomain+"/expansion-source"]
					}
				}
				return fmt.Sprintf("%d NodeClaims for %d pods: %v", len(everSeen), n, everSeen), len(everSeen) > n
			})
			nodes := map[string]bool{}
			for _, p := range pods {
				nodes[str(p, "spec", "nodeName")] = true
			}
			scaler, _ := env.controllerAPI(ctx, "api/node-scaler", nil)
			c.Note("node-scaler state after the burst: %s", firstLine(string(scaler)))
			c.Note("NodeClaims (name: nodepool/expansion-source): %v", everSeen)
			c.Measure("pods", float64(n), "")
			c.Measure("nodeclaims_created", float64(len(everSeen)), "")
			c.Measure("nodes_hosting_burst", float64(len(nodes)), "")
			for name, pool := range everSeen {
				Expect(pool).To(HavePrefix(targetNodePool()+"/"), "NodeClaim %s is not from the TensorFusion NodePool", name)
			}
			Expect(everSeen).To(HaveLen(n), "%d whole-GPU pods on single-GPU instance types need exactly %d new nodes", n, n)
			Expect(nodes).To(HaveLen(n))
			led.report(c, finopsPriceAs())
		},
	})

	Register(Case{
		ID: "KRP-A09", Title: "负载清空后节点被回收（should-delete 标记 + Karpenter consolidation）", Tiers: []string{TierKarpenter, TierRehearsal, TierKWOK},
		Needs: []string{NeedTF, NeedKarpenter, NeedGPU}, Timeout: 30 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			// No holdNodes here: the point is that nothing keeps an empty node alive.
			gpus := env.readyGPUs(ctx)
			if len(gpus) == 0 {
				Expect(env.create(ctx, tfPod(c.NS, "seed", wholeGPU(), tfNodeSelector()))).To(Succeed())
				env.waitPodReady(ctx, c.NS, "seed", 12*time.Minute)
				env.delete(ctx, newObj(gvkPod, c.NS, "seed"))
			}
			env.waitResourcesReturned(ctx, 3*time.Minute)
			nodes := map[string]bool{}
			for _, g := range env.readyGPUs(ctx) {
				nodes[g.Node] = true
				// A previous case may have pinned the node.
				_ = env.patchMerge(ctx, newObj(gvkNode, "", g.Node), `{"metadata":{"annotations":{"karpenter.sh/do-not-disrupt":null}}}`)
			}
			Expect(nodes).NotTo(BeEmpty())
			start := time.Now()
			marked := map[string]time.Duration{}
			waitFor(ctx, "every empty GPU node removed (NodeClaim, Node, GPUNode, GPU CRs)", 20*time.Minute, func() (string, bool) {
				left := 0
				var state []string
				for n := range nodes {
					node, err := env.get(ctx, gvkNode, "", n)
					if err != nil {
						continue
					}
					left++
					if node.GetLabels()[tfDomain+"/should-delete"] == "true" {
						if _, ok := marked[n]; !ok {
							marked[n] = time.Since(start)
						}
					}
					state = append(state, fmt.Sprintf("%s should-delete=%q deleting=%v", n, node.GetLabels()[tfDomain+"/should-delete"], node.GetDeletionTimestamp() != nil))
				}
				gn, _ := env.list(ctx, gvkGPUNode, "")
				return fmt.Sprintf("%d node(s) left: %s; %d GPUNode(s), %d GPU CR(s)", left, strings.Join(state, " | "), len(gn), len(env.poolGPUs(ctx))),
					left == 0 && len(env.poolGPUs(ctx)) == 0
			})
			c.Measure("empty_to_node_gone_seconds", time.Since(start).Seconds(), "s")
			c.Measure("nodes_removed", float64(len(nodes)), "")
			c.Measure("nodes_marked_should_delete_first", float64(len(marked)), "")
			for n, d := range marked {
				c.Note("node %s was marked should-delete after %.0fs", n, d.Seconds())
			}
			if len(marked) == 0 {
				c.Note("no node carried tensor-fusion.ai/should-delete before Karpenter removed it: the NodePool's own consolidateAfter fired first (TF compaction protects new nodes for 5 min)")
			}
		},
	})

	Register(Case{
		ID: "FIN-01", Title: "FinOps：N 个分片负载 vs 整卡基线的节点小时与成本", Tiers: []string{TierKarpenter, TierRehearsal, TierKWOK},
		Needs: []string{NeedTF, NeedKarpenter}, Timeout: 45 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			if env.Mode == ModeNoGPU {
				Skip("needs GPU capacity (real or mock)")
			}
			n := 3
			hold := 3 * time.Minute
			if v, err := time.ParseDuration(getenv("E2E_FINOPS_HOLD", "")); err == nil && v > 0 {
				hold = v // fake nodes: the window only has to be long enough to be sampled
			}
			run := func(label string, annotations func(g GPU) map[string]string) (nodes int, hours, usd float64, ready time.Duration) {
				By("scenario " + label + ": start from an empty pool")
				waitFor(ctx, "pool "+targetNodePool()+" empty", 20*time.Minute, func() (string, bool) {
					ncs := env.nodeClaims(ctx, targetNodePool())
					for i := range ncs {
						_ = env.patchMerge(ctx, newObj(gvkNode, "", str(&ncs[i], "status", "nodeName")), `{"metadata":{"annotations":{"karpenter.sh/do-not-disrupt":null}}}`)
					}
					return fmt.Sprintf("NodeClaims: %v", nodeClaimNames(ncs)), len(ncs) == 0
				})
				led := newNodeLedger(targetNodePool())
				start := time.Now()
				// The first pod brings up the first node; its GPU defines the fractions.
				ref := GPU{CapTflops: 100}
				var names []string
				for i := 0; i < n; i++ {
					name := fmt.Sprintf("%s-%d", label, i)
					names = append(names, name)
					a := annotations(ref)
					Expect(env.create(ctx, tfPod(c.NS, name, a, tfNodeSelector()))).To(Succeed())
				}
				waitFor(ctx, fmt.Sprintf("%d %s workloads Running", n, label), 20*time.Minute, func() (string, bool) {
					led.observe(ctx)
					var notReady []string
					for _, name := range names {
						p, err := env.get(ctx, gvkPod, c.NS, name)
						if err != nil || !podReady(p) {
							notReady = append(notReady, name)
						}
					}
					return fmt.Sprintf("not ready: %v", notReady), len(notReady) == 0
				})
				ready = time.Since(start)
				// Steady state: the workloads "serve" for a fixed window.
				end := time.Now().Add(hold)
				for time.Now().Before(end) {
					led.observe(ctx)
					time.Sleep(15 * time.Second)
				}
				used := map[string]bool{}
				for _, name := range names {
					p, _ := env.get(ctx, gvkPod, c.NS, name)
					used[str(p, "spec", "nodeName")] = true
				}
				led.observe(ctx)
				for _, name := range names {
					env.delete(ctx, newObj(gvkPod, c.NS, name))
				}
				nodes = len(used)
				hours, usd = 0, 0
				for _, ln := range led.nodes {
					if ln.InstanceType == "" {
						continue
					}
					h := ln.Last.Sub(ln.First).Hours()
					priced := ln.InstanceType
					if p := finopsPriceAs(); p != "" {
						priced = p
					}
					hours += h
					usd += h * env.hourlyPrice(priced)
				}
				return nodes, hours, usd, ready
			}

			// 3 x (20% compute, 2 GiB) fit one single-GPU node; 3 whole GPUs need three.
			fn, fh, fu, fr := run("frac", func(GPU) map[string]string {
				return ann("isolation", "soft", "compute-percent-request", "20", "compute-percent-limit", "20", "vram-request", "2Gi", "vram-limit", "2Gi")
			})
			wn, wh, wu, wr := run("whole", func(GPU) map[string]string { return wholeGPU() })

			c.Measure("workloads", float64(n), "")
			c.Measure("fractional_nodes", float64(fn), "")
			c.Measure("fractional_node_hours", fh, "h")
			c.Measure("fractional_cost_usd", fu, "$")
			c.Measure("fractional_time_to_all_running_seconds", fr.Seconds(), "s")
			c.Measure("wholegpu_nodes", float64(wn), "")
			c.Measure("wholegpu_node_hours", wh, "h")
			c.Measure("wholegpu_cost_usd", wu, "$")
			c.Measure("wholegpu_time_to_all_running_seconds", wr.Seconds(), "s")
			if wu > 0 {
				c.Measure("cost_saving_percent", (1-fu/wu)*100, "%")
			}
			price := finopsPriceAs()
			if price == "" {
				price = "the launched instance types"
			}
			c.Note("steady-state window %s per scenario; prices: on-demand %s from e2e/pricing; a node-hour is NodeClaim creation to last sighting", hold, price)
			c.Attach("finops", map[string]any{
				"workloads": n, "hold_seconds": hold.Seconds(), "priced_as": price, "mode": env.Mode,
				"fractional": map[string]any{"nodes": fn, "node_hours": fh, "usd": fu, "seconds_to_running": fr.Seconds()},
				"whole_gpu":  map[string]any{"nodes": wn, "node_hours": wh, "usd": wu, "seconds_to_running": wr.Seconds()},
			})
			Expect(fn).To(Equal(1), "fractional workloads must share one GPU node")
			Expect(wn).To(BeNumerically(">", fn), "whole-GPU baseline must need more nodes")
			Expect(fh).To(BeNumerically("<", wh))
		},
	})

	Register(Case{
		ID: "NLC-09", Title: "节点被删除后状态收敛，Deployment 的 Pod 在新节点重建", Tiers: []string{TierKarpenter, TierRehearsal, TierKWOK},
		Needs: []string{NeedTF, NeedKarpenter, NeedGPU}, Timeout: 30 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			tmpl := tfPod(c.NS, "svc", fraction(gpus[0], "soft", 20, 2), tfNodeSelector())
			Expect(env.create(ctx, deployment(c.NS, "svc", 1, tmpl))).To(Succeed())
			var first *unstructured.Unstructured
			waitFor(ctx, "deployment pod Running", 10*time.Minute, func() (string, bool) {
				pods := env.podsOf(ctx, c.NS, "svc")
				if len(pods) == 0 {
					return "no pod", false
				}
				first = pods[0].DeepCopy()
				return podWhy(first), podReady(first)
			})
			node := str(first, "spec", "nodeName")
			var claim string
			for _, nc := range env.nodeClaims(ctx, "") {
				if str(&nc, "status", "nodeName") == node {
					claim = nc.GetName()
				}
			}
			Expect(claim).NotTo(BeEmpty(), "node %s has no NodeClaim", node)
			_ = env.patchMerge(ctx, newObj(gvkNode, "", node), `{"metadata":{"annotations":{"karpenter.sh/do-not-disrupt":null}}}`)

			start := time.Now()
			env.delete(ctx, newObj(gvkNodeClaim, "", claim)) // Karpenter drains and terminates the instance
			waitFor(ctx, "GPUNode and GPU CRs of the deleted node removed", 10*time.Minute, func() (string, bool) {
				_, err := env.get(ctx, gvkGPUNode, "", node)
				left := 0
				for _, g := range env.poolGPUs(ctx) {
					if g.Node == node {
						left++
					}
				}
				return fmt.Sprintf("GPUNode present=%v, %d GPU CR(s) left", err == nil, left), err != nil && left == 0
			})
			c.Measure("node_delete_to_cr_cleanup_seconds", time.Since(start).Seconds(), "s")
			waitFor(ctx, "replacement pod Running on another node", 15*time.Minute, func() (string, bool) {
				for _, p := range env.podsOf(ctx, c.NS, "svc") {
					if podReady(&p) && str(&p, "spec", "nodeName") != node {
						return "", true
					}
				}
				return podsWhy(ctx, c.NS, "svc"), false
			})
			c.Measure("node_delete_to_pod_running_seconds", time.Since(start).Seconds(), "s")
		},
	})
})

func conditionTime(u *unstructured.Unstructured, typ string) time.Time {
	for _, c := range sliceAt(u, "status", "conditions") {
		m, _ := c.(map[string]any)
		if m["type"] == typ && m["status"] == "True" {
			t, _ := time.Parse(time.RFC3339, fmt.Sprint(m["lastTransitionTime"]))
			return t
		}
	}
	return time.Time{}
}

func ownerKinds(u *unstructured.Unstructured) []string {
	var out []string
	for _, o := range u.GetOwnerReferences() {
		out = append(out, o.Kind+"/"+o.Name)
	}
	return out
}

func allClaimEvents(ctx context.Context, pool string) []string {
	evs, _ := env.list(ctx, gvkEvent, "default")
	var out []string
	for i := range evs {
		if str(&evs[i], "involvedObject", "kind") == "NodeClaim" && strings.HasPrefix(str(&evs[i], "involvedObject", "name"), pool+"-") {
			out = append(out, str(&evs[i], "reason")+": "+str(&evs[i], "message"))
		}
	}
	return out
}

// compactEvents keeps reason + the first words of each event, deduplicated.
func compactEvents(evs []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, e := range evs {
		if len(e) > 110 {
			e = e[:110] + "…"
		}
		if !seen[e] {
			seen[e] = true
			out = append(out, e)
		}
	}
	return out
}
