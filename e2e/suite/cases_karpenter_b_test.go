//go:build e2e

package suite

import (
	"context"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Karpenter compatibility, path B: GPUPool provisioningMode=Karpenter. The
// operator creates GPUNodeClaim -> NodeClaim itself, against an EC2NodeClass,
// with no NodePool involved.
//
// Not executable on the released images (<= 2.17.23): the pool never reconciles
// because the static pricing provider is not initialised. The cases require
// E2E_FIXED_IMAGES=1 and are otherwise reported as skipped with that reason.
// Path B cannot be rehearsed on mock GPUs either: instance types come from the
// operator's built-in GPU price table and the NodeClaim requests nvidia.com/gpu.

const (
	pathBPool           = "karp"
	pathBPoolName       = "tensor-fusion-" + pathBPool
	pathBCleanupTimeout = 10 * time.Minute
	// A label value holds at most 63 bytes; a node name up to 253.
	maxLabelValueLen = 63
)

// pathBCleanupSeconds is how long the last removal of the path B pool took
// until nothing of it was left; 0 until a removal has completed.
var pathBCleanupSeconds float64

// pathBLeftovers lists what still exists of the path B pool: the pool, its
// GPUNodeClaims, their NodeClaims and nodes, its GPUNodes and GPUs.
func pathBLeftovers(ctx context.Context) string {
	var left []string
	if _, err := env.get(ctx, gvkGPUPool, "", pathBPoolName); err == nil {
		left = append(left, "GPUPool/"+pathBPoolName)
	}
	owned := client.MatchingLabels{tfDomain + "/managed-by": pathBPoolName}
	claims, _ := env.list(ctx, gvkNodeClaimT, "", owned)
	for _, o := range claims {
		left = append(left, "GPUNodeClaim/"+o.GetName())
	}
	gpuNodes, _ := env.list(ctx, gvkGPUNode, "", owned)
	for _, o := range gpuNodes {
		left = append(left, "GPUNode/"+o.GetName())
	}
	for _, g := range env.gpusOf(ctx, pathBPoolName) {
		left = append(left, "GPU/"+g.Name)
	}
	for _, gvk := range []schema.GroupVersionKind{gvkNodeClaim, gvkNode} {
		items, _ := env.list(ctx, gvk, "", client.HasLabels{tfDomain + "/node-provisioner"})
		for _, o := range items {
			if strings.HasPrefix(o.GetLabels()[tfDomain+"/node-provisioner"], pathBPoolName+"-") {
				left = append(left, gvk.Kind+"/"+o.GetName())
			}
		}
	}
	return strings.Join(left, ", ")
}

// addPathBPool adds GPUPool tensor-fusion-karp to the chart's TensorFusionCluster
// and registers its removal.
func addPathBPool(ctx context.Context, instanceTypes []string, capacityType string, minTflops string) string {
	GinkgoHelper()
	tfc, err := env.get(ctx, gvkTFCluster, "", "tensor-fusion")
	Expect(err).NotTo(HaveOccurred())
	pools := sliceAt(tfc, "spec", "gpuPools")
	var def map[string]any
	var kept []any
	for _, p := range pools {
		m := p.(map[string]any)
		if m["isDefault"] == true {
			def = m
		}
		if m["name"] != pathBPool {
			kept = append(kept, p)
		}
	}
	Expect(def).NotTo(BeNil(), "no default pool to copy the component config from")
	pool := (&unstructured.Unstructured{Object: def}).DeepCopy().Object
	pool["name"], pool["isDefault"] = pathBPool, false
	tmpl := pool["specTemplate"].(map[string]any)
	tmpl["capacityConfig"] = map[string]any{
		"minResources":  map[string]any{"cpu": "0", "memory": "0", "tflops": minTflops, "vram": "1Gi"},
		"warmResources": map[string]any{"cpu": "0", "memory": "0", "tflops": "0", "vram": "0"},
		"maxResources":  map[string]any{"cpu": "0", "memory": "0", "tflops": "130", "vram": "48Gi"},
	}
	nm := tmpl["nodeManagerConfig"].(map[string]any)
	delete(nm, "nodeSelector")
	nm["provisioningMode"] = "Karpenter"
	nm["nodeCompaction"] = map[string]any{"period": "1m"}
	nm["nodeProvisioner"] = map[string]any{
		"karpenterNodeClassRef": pathBNodeClassRef(),
		"gpuRequirements": []any{
			map[string]any{"key": "node.kubernetes.io/instance-type", "operator": "In", "values": toAnySlice(instanceTypes)},
			map[string]any{"key": "karpenter.sh/capacity-type", "operator": "In", "values": []any{capacityType}},
		},
		// Matched by the chart's initialGpuNodeLabelSelector, not by the default
		// pool (which selects on the stack's shared-pool label).
		"gpuNodeLabels": map[string]any{"nvidia.com/gpu.present": "true"},
	}
	patch := map[string]any{"spec": map[string]any{
		"computingVendor": map[string]any{"name": "karpenter", "type": "karpenter", "authType": "serviceAccountRole", "enable": true,
			"params": map[string]any{"defaultRegion": getenv("E2E_PRICING_REGION", "us-west-2")}},
		"gpuPools": append(kept, pool),
	}}
	Expect(env.patchMerge(ctx, tfc, toJSON(patch))).To(Succeed())
	DeferCleanup(func() {
		bg := context.Background()
		cur, err := env.get(bg, gvkTFCluster, "", "tensor-fusion")
		if err != nil {
			return
		}
		var rest []any
		for _, p := range sliceAt(cur, "spec", "gpuPools") {
			if p.(map[string]any)["name"] != pathBPool {
				rest = append(rest, p)
			}
		}
		_ = env.patchMerge(bg, cur, toJSON(map[string]any{"spec": map[string]any{"gpuPools": rest}}))
		// KRP-B09: deleting the pool must take every GPUNodeClaim, NodeClaim,
		// GPUNode and GPU of the pool with it. An object left behind fails the case.
		removed := time.Now()
		left := ""
		deadline := removed.Add(pathBCleanupTimeout)
		for time.Now().Before(deadline) {
			if left = pathBLeftovers(bg); left == "" {
				pathBCleanupSeconds = time.Since(removed).Seconds()
				GinkgoWriter.Printf("path B pool removed: nothing left after %.0fs\n", pathBCleanupSeconds)
				return
			}
			time.Sleep(5 * time.Second)
		}
		Fail(fmt.Sprintf("%s after removing pool %s: still present: %s", pathBCleanupTimeout, pathBPool, left))
	})
	return pathBPoolName
}

// pathBNodeClassRef names the NodeClass path B NodeClaims use: the stack's
// EC2NodeClass on AWS, the ECSNodeClass on Alibaba Cloud, the KWOKNodeClass on
// a KWOK cluster.
func pathBNodeClassRef() map[string]any {
	if env.HasAlibaba && !env.HasAWS {
		return map[string]any{"group": "karpenter.alibabacloud.com", "kind": "ECSNodeClass", "version": "v1alpha1", "name": getenv("E2E_GPU_NODECLASS", "tf-gpu")}
	}
	if env.HasKWOK && !env.HasAWS {
		return map[string]any{"group": "karpenter.kwok.sh", "kind": "KWOKNodeClass", "version": "v1alpha1", "name": getenv("E2E_GPU_NODECLASS", "tf-kwok")}
	}
	return map[string]any{"group": "karpenter.k8s.aws", "kind": "EC2NodeClass", "version": "v1", "name": getenv("E2E_GPU_NODECLASS", "tf-gpu")}
}

var _ = Describe("karpenter path B", func() {
	Register(Case{
		ID: "KRP-B02", Title: "路径 B：容量水位 → GPUNodeClaim → 同名 NodeClaim → 节点 → Bound（含 B01 连接检查、B09 删除 Pool 清理）",
		Tiers: []string{TierKarpenter, TierDryRun, TierKWOK}, Needs: []string{NeedTF, NeedKarpenter, NeedNodeClass, NeedFixedImg}, Timeout: 40 * time.Minute,
		KnownIssue: "<= 2.17.23: provisioningMode=Karpenter pools never reconcile (pricing provider not initialised). 2.17.24 - 2.17.27: the GPUNodeClaim and (given a chart whose ClusterRole lets the operator read the NodeClass, >= 1.8.9) the NodeClaim and the node appear, but the node controller matches nodes to pools by node selector only, so the node of a pool without one gets no GPUNode, no hypervisor, and the claim never becomes Bound; and a deleted pool leaves its GPUNodeClaim behind",
		Run: func(ctx SpecContext, c *C) {
			// g4dn.2xlarge: the built-in price table cannot resolve g4dn.xlarge's GPU
			// model ("NVIDIA T4 Tensor Core" has no TFLOPS entry), so it is skipped
			// as zero capacity.
			types := strings.Split(getenv("E2E_PATHB_INSTANCE_TYPES", "g4dn.2xlarge"), ",")
			pool := addPathBPool(ctx, types, getenv("E2E_PATHB_CAPACITY", "on-demand"), "1")
			start := time.Now()

			By("KRP-B01: cloud vendor connection check")
			waitFor(ctx, "TensorFusionCluster condition CloudVendorConnectionReady", 3*time.Minute, func() (string, bool) {
				tfc, _ := env.get(ctx, gvkTFCluster, "", "tensor-fusion")
				return toJSON(sliceAt(tfc, "status", "conditions")), condition(tfc, "CloudVendorConnectionReady") == "True"
			})

			By("a GPUNodeClaim and the NodeClaim of the same name are created")
			var claim *unstructured.Unstructured
			waitFor(ctx, "GPUNodeClaim for pool "+pool, 5*time.Minute, func() (string, bool) {
				claims, _ := env.list(ctx, gvkNodeClaimT, "")
				p, _ := env.get(ctx, gvkGPUPool, "", pool)
				if len(claims) == 0 {
					return "no GPUNodeClaim; pool status " + toJSON(mapAt(p, "status")), false
				}
				claim = claims[0].DeepCopy()
				return "", true
			})
			c.Measure("pool_to_gpunodeclaim_seconds", time.Since(start).Seconds(), "s")
			c.Note("GPUNodeClaim %s: instance type %s, hourly-pricing=%s", claim.GetName(), str(claim, "spec", "instanceType"), claim.GetAnnotations()[tfDomain+"/hourly-pricing"])

			var nc *unstructured.Unstructured
			waitFor(ctx, "Karpenter NodeClaim "+claim.GetName(), 3*time.Minute, func() (string, bool) {
				n, err := env.get(ctx, gvkNodeClaim, "", claim.GetName())
				if err != nil {
					return err.Error(), false
				}
				nc = n
				return "", true
			})
			Expect(ownerKinds(nc)).To(ContainElement("GPUNodeClaim/" + claim.GetName()))
			Expect(nc.GetAnnotations()).To(HaveKeyWithValue("karpenter.sh/do-not-disrupt", "true"), "path B NodeClaims are TensorFusion's to remove, not Karpenter's")
			Expect(nc.GetLabels()).To(HaveKey(tfDomain + "/node-provisioner"))
			Expect(str(nc, "spec", "resources", "requests", "nvidia.com/gpu")).To(Equal("1"))

			By("the node joins and the GPUNodeClaim becomes Bound (or EC2 refuses the launch for quota)")
			quotaErr := ""
			bound := false
			waitFor(ctx, "GPUNodeClaim Bound", 15*time.Minute, func() (string, bool) {
				for _, ev := range claimEvents(ctx, claim.GetName()) {
					if strings.Contains(ev, "InsufficientCapacityError") {
						quotaErr = ev
						return "", true
					}
				}
				cur, err := env.get(ctx, gvkNodeClaimT, "", claim.GetName())
				if err != nil {
					return err.Error(), false
				}
				bound = str(cur, "status", "phase") == "Bound"
				return fmt.Sprintf("phase=%s; nodeclaim %s", str(cur, "status", "phase"), nodeClaimWhy(nc)), bound
			})
			if !bound {
				c.Note("EC2 refused the launch: %s", firstLine(quotaErr))
				c.SetMode(ModeDryRun)
				Skip("dry-run result: path B created GPUNodeClaim and NodeClaim; EC2 answered with a capacity / quota error")
			}
			c.Measure("pool_to_bound_seconds", time.Since(start).Seconds(), "s")
			if cur, err := env.get(ctx, gvkNodeClaim, "", claim.GetName()); err == nil {
				nc = cur
			}
			if node, err := env.get(ctx, gvkNode, "", str(nc, "status", "nodeName")); err == nil {
				c.Note("node %s: instance type %s, capacity type %s", node.GetName(), node.GetLabels()["node.kubernetes.io/instance-type"], node.GetLabels()["karpenter.sh/capacity-type"])
			}
			var gpu GPU
			waitFor(ctx, "a Running GPU CR in pool "+pool, 5*time.Minute, func() (string, bool) {
				for _, g := range env.gpusOf(ctx, pool) {
					if g.Phase == "Running" {
						gpu = g
						return "", true
					}
				}
				return fmt.Sprintf("%d GPU CRs, none Running", len(env.gpusOf(ctx, pool))), false
			})
			c.Measure("pool_to_gpu_running_seconds", time.Since(start).Seconds(), "s")
			Expect(gpu.Node).To(Equal(str(nc, "status", "nodeName")), "the GPU belongs to the node of the NodeClaim")
			if gn, err := env.get(ctx, gvkGPUNode, "", gpu.Node); Expect(err).NotTo(HaveOccurred()) {
				Expect(gn.GetLabels()).To(HaveKeyWithValue(tfDomain+"/node-provisioner", claim.GetName()))
				Expect(gn.GetLabels()).To(HaveKeyWithValue(tfDomain+"/managed-by", pool))
			}

			By("a workload that asks for this pool is scheduled onto the provisioned node")
			podStart := time.Now()
			Expect(env.create(ctx, tfPod(c.NS, "on-path-b", merge(fraction(gpu, "soft", 20, 2), map[string]string{lblPool: pool})))).To(Succeed())
			pod, _ := env.waitPodReady(ctx, c.NS, "on-path-b", 5*time.Minute)
			c.Measure("pod_create_to_running_seconds", time.Since(podStart).Seconds(), "s")
			Expect(str(pod, "spec", "nodeName")).To(Equal(gpu.Node))
			Expect(podGPUIDs(pod)).To(ConsistOf(gpu.Name))
			Consistently(func() int { return len(mustList(ctx, gvkNodeClaimT)) }, 30*time.Second, 5*time.Second).
				Should(Equal(1), "a pool at its minimum does not provision a second node")

			By("KRP-B09: removing the pool removes the GPUNodeClaim, NodeClaim, node, GPUNode and GPU")
			// The removal itself is the cleanup registered by addPathBPool; it fails
			// this case when anything is left, and KRP-B10 reports how long it took.
			env.delete(ctx, pod)
		},
	})

	Register(Case{
		ID: "KRP-B10", Title: "路径 B：删除 Pool 后 GPUNodeClaim / NodeClaim / 节点 / GPUNode / GPU 全部回收（接在 KRP-B02 之后）",
		Tiers: []string{TierKarpenter, TierKWOK}, Needs: []string{NeedTF, NeedKarpenter, NeedNodeClass, NeedFixedImg}, Timeout: 15 * time.Minute, NoNS: true,
		Run: func(ctx SpecContext, c *C) {
			if pathBCleanupSeconds == 0 {
				Skip("KRP-B02 did not get as far as removing its pool in this run")
			}
			Expect(pathBLeftovers(ctx)).To(BeEmpty())
			c.Measure("pool_delete_to_clean_seconds", pathBCleanupSeconds, "s")
		},
	})
})

func mustList(ctx context.Context, gvk schema.GroupVersionKind) []unstructured.Unstructured {
	items, err := env.list(ctx, gvk, "")
	Expect(err).NotTo(HaveOccurred())
	return items
}
