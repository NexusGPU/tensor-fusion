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

const pathBPool = "karp"

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
		// KRP-B09: deleting the pool must take every GPUNodeClaim / NodeClaim with it.
		deadline := time.Now().Add(10 * time.Minute)
		for time.Now().Before(deadline) {
			claims, _ := env.list(bg, gvkNodeClaimT, "")
			if len(claims) == 0 {
				return
			}
			time.Sleep(15 * time.Second)
		}
		GinkgoWriter.Printf("WARNING: GPUNodeClaims still present 10 min after removing pool %s\n", pathBPool)
	})
	return "tensor-fusion-" + pathBPool
}

// pathBNodeClassRef names the NodeClass path B NodeClaims use: the stack's
// EC2NodeClass on AWS, the KWOKNodeClass on a KWOK cluster.
func pathBNodeClassRef() map[string]any {
	if env.HasKWOK && !env.HasAWS {
		return map[string]any{"group": "karpenter.kwok.sh", "kind": "KWOKNodeClass", "version": "v1alpha1", "name": getenv("E2E_GPU_NODECLASS", "tf-kwok")}
	}
	return map[string]any{"group": "karpenter.k8s.aws", "kind": "EC2NodeClass", "version": "v1", "name": getenv("E2E_GPU_NODECLASS", "tf-gpu")}
}

var _ = Describe("karpenter path B", func() {
	Register(Case{
		ID: "KRP-B02", Title: "路径 B：容量水位 → GPUNodeClaim → 同名 NodeClaim → 节点 → Bound（含 B01 连接检查、B09 删除 Pool 清理）",
		Tiers: []string{TierKarpenter, TierDryRun, TierKWOK}, Needs: []string{NeedTF, NeedKarpenter, NeedNodeClass, NeedFixedImg}, Timeout: 40 * time.Minute, NoNS: true,
		KnownIssue: "<= 2.17.23: provisioningMode=Karpenter pools never reconcile (pricing provider not initialised). 2.17.24 / 2.17.25: the pool reconciles and creates the GPUNodeClaim, but the chart's ClusterRole lets the operator read no Karpenter NodeClass kind, so the GPUNodeClaim controller fails with `failed to query NodeClass ... is forbidden` and no NodeClaim is created",
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
			waitFor(ctx, "GPU CR in pool "+pool, 5*time.Minute, func() (string, bool) {
				n := len(env.gpusOf(ctx, pool))
				return fmt.Sprintf("%d GPU CRs", n), n > 0
			})
		},
	})
})
