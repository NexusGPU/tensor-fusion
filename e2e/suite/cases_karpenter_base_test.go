//go:build e2e

package suite

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	authv1 "k8s.io/api/authorization/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// Karpenter cases that need no GPU: the integration surface (NodeOverlay, RBAC),
// Karpenter's own health, and a real launch + consolidation of a CPU node.

var _ = Describe("karpenter base", func() {
	Register(Case{
		ID: "KRP-00", Title: "Karpenter is ready: NodePool / EC2NodeClass Ready, nodeOverlay feature gate on", Tiers: []string{TierSmoke, TierKWOK}, Needs: []string{NeedKarpenter}, NoNS: true,
		Run: func(ctx SpecContext, c *C) {
			pools, err := env.list(ctx, gvkNodePool, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(pools).NotTo(BeEmpty())
			for i := range pools {
				Expect(condition(&pools[i], "Ready")).To(Equal("True"), "NodePool %s: %s", pools[i].GetName(), toJSON(sliceAt(&pools[i], "status", "conditions")))
			}
			c.Note("NodePools: %v", names(pools))
			if env.HasKWOK {
				ncs, _ := env.list(ctx, gvkKWOKNC, "")
				Expect(ncs).NotTo(BeEmpty(), "no KWOKNodeClass")
				for i := range ncs {
					Expect(condition(&ncs[i], "Ready")).To(Equal("True"), "KWOKNodeClass %s: %s", ncs[i].GetName(), toJSON(sliceAt(&ncs[i], "status", "conditions")))
				}
				c.Note("KWOKNodeClasses: %v (fake nodes, nodeRegistrationDelay %s)", names(ncs), str(&ncs[0], "spec", "nodeRegistrationDelay"))
			}
			if env.HasAWS {
				ncs, _ := env.list(ctx, gvkEC2NC, "")
				for i := range ncs {
					Expect(condition(&ncs[i], "Ready")).To(Equal("True"), "EC2NodeClass %s: %s", ncs[i].GetName(), toJSON(sliceAt(&ncs[i], "status", "conditions")))
					var amis []string
					for _, a := range sliceAt(&ncs[i], "status", "amis") {
						amis = append(amis, fmt.Sprint(a.(map[string]any)["name"]))
					}
					c.Note("EC2NodeClass %s resolved AMIs: %v", ncs[i].GetName(), amis)
				}
			}
			if env.HasAlibaba {
				ncs, _ := env.list(ctx, gvkECSNC, "")
				Expect(ncs).NotTo(BeEmpty(), "no ECSNodeClass")
				for i := range ncs {
					Expect(condition(&ncs[i], "Ready")).To(Equal("True"), "ECSNodeClass %s: %s", ncs[i].GetName(), toJSON(sliceAt(&ncs[i], "status", "conditions")))
					var images []string
					for _, a := range sliceAt(&ncs[i], "status", "images") {
						images = append(images, fmt.Sprint(a.(map[string]any)["id"]))
					}
					c.Note("ECSNodeClass %s resolved images: %v", ncs[i].GetName(), images)
				}
			}
			dep, err := env.get(ctx, gvkDeployment, "kube-system", "karpenter")
			if err == nil {
				envs := map[string]string{}
				for _, raw := range sliceAt(dep, "spec", "template", "spec", "containers") {
					es, _, _ := unstructured.NestedSlice(raw.(map[string]any), "env")
					for _, e := range es {
						m := e.(map[string]any)
						envs[fmt.Sprint(m["name"])] = fmt.Sprint(m["value"])
					}
				}
				Expect(envs["FEATURE_GATES"]).To(ContainSubstring("NodeOverlay=true"), "Karpenter FEATURE_GATES")
				c.Note("Karpenter image %s", firstContainerImage(dep, "controller"))
			}
		},
	})

	Register(Case{
		ID: "KRP-01", Title: "NodeOverlay and the operator's NodeClaim RBAC are in place", Tiers: []string{TierSmoke, TierKWOK}, Needs: []string{NeedTF, NeedKarpenter}, NoNS: true,
		Run: func(ctx SpecContext, c *C) {
			ov, err := env.get(ctx, gvkNodeOverlay, "", "tensor-fusion-overlay")
			Expect(err).NotTo(HaveOccurred(), "NodeOverlay missing: the chart only renders it when the Karpenter CRD existed at install time")
			capacity := toStringMap(mapAt(ov, "spec", "capacity"))
			for _, suffix := range strings.Split("0123456789abcdef", "") {
				Expect(capacity).To(HaveKeyWithValue(tfDomain+"/index_"+suffix, "36"))
			}
			Expect(capacity).To(HaveKeyWithValue(tfDomain+"/index", "512"))
			Expect(condition(ov, "Ready")).To(Equal("True"), "NodeOverlay not accepted by Karpenter: %s", toJSON(sliceAt(ov, "status", "conditions")))

			dep, err := env.get(ctx, gvkDeployment, env.TFNamespace, env.TFRelease+"-controller")
			Expect(err).NotTo(HaveOccurred())
			sa := str(dep, "spec", "template", "spec", "serviceAccountName")
			for _, verb := range []string{"create", "delete", "list", "watch"} {
				sar, err := env.CS.AuthorizationV1().SubjectAccessReviews().Create(ctx, &authv1.SubjectAccessReview{
					Spec: authv1.SubjectAccessReviewSpec{
						User:               fmt.Sprintf("system:serviceaccount:%s:%s", env.TFNamespace, sa),
						ResourceAttributes: &authv1.ResourceAttributes{Group: "karpenter.sh", Resource: "nodeclaims", Verb: verb},
					},
				}, metav1.CreateOptions{})
				Expect(err).NotTo(HaveOccurred())
				Expect(sar.Status.Allowed).To(BeTrue(), "operator service account %s may not %s nodeclaims", sa, verb)
			}
		},
	})

	Register(Case{
		ID: "KRP-C01", Title: "Karpenter launches and reclaims a non-GPU node (general-purpose NodePool)", Tiers: []string{TierSmoke, TierKWOK}, Needs: []string{NeedKarpenter}, Timeout: 15 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			// A node left in the pool would simply absorb the test pod. An idle pool
			// is the precondition; Karpenter gets a consolidation cycle to reach it.
			waitFor(ctx, "NodePool "+env.CPUNodePool+" empty before the test", 4*time.Minute, func() (string, bool) {
				ncs := env.nodeClaims(ctx, env.CPUNodePool)
				return fmt.Sprintf("NodeClaims present: %v", nodeClaimNames(ncs)), len(ncs) == 0
			})
			before := nodeClaimNames(env.nodeClaims(ctx, env.CPUNodePool))
			// GPU nodes of earlier cases may still be around (KWOK tier); what
			// matters is that this pod adds none.
			gpuBefore := nodeClaimNames(env.nodeClaims(ctx, env.GPUNodePool))
			led := newNodeLedger(env.CPUNodePool)

			start := time.Now()
			tmpl := plainPod(c.NS, "inflate", withNodeSelector(map[string]string{env.RoleLabel: "cpu"}), withResources(map[string]string{"cpu": "1", "memory": "512Mi"}))
			Expect(env.create(ctx, deployment(c.NS, "inflate", 1, tmpl))).To(Succeed())

			var claim *unstructured.Unstructured
			waitFor(ctx, "a new NodeClaim in NodePool "+env.CPUNodePool, 3*time.Minute, func() (string, bool) {
				led.observe(ctx)
				for _, nc := range env.nodeClaims(ctx, env.CPUNodePool) {
					if !contains(before, nc.GetName()) {
						claim = nc.DeepCopy()
						return "", true
					}
				}
				return "no new NodeClaim; pod: " + podsWhy(ctx, c.NS, "inflate"), false
			})
			c.Measure("pending_to_nodeclaim_seconds", time.Since(start).Seconds(), "s")

			waitFor(ctx, "NodeClaim launched, registered and initialized", 6*time.Minute, func() (string, bool) {
				led.observe(ctx)
				nc, err := env.get(ctx, gvkNodeClaim, "", claim.GetName())
				if err != nil {
					return err.Error(), false
				}
				claim = nc
				return nodeClaimWhy(nc), condition(nc, "Initialized") == "True"
			})
			c.Measure("pending_to_node_ready_seconds", time.Since(start).Seconds(), "s")

			waitFor(ctx, "pod Running on the new node", 3*time.Minute, func() (string, bool) {
				pods := env.podsOf(ctx, c.NS, "inflate")
				if len(pods) == 0 {
					return "no pod", false
				}
				return podWhy(&pods[0]), podReady(&pods[0]) && str(&pods[0], "spec", "nodeName") == str(claim, "status", "nodeName")
			})
			c.Measure("pending_to_pod_running_seconds", time.Since(start).Seconds(), "s")
			c.Note("launched %s (%s) in %s", claim.GetLabels()["node.kubernetes.io/instance-type"], claim.GetLabels()["karpenter.sh/capacity-type"], claim.GetLabels()["topology.kubernetes.io/zone"])

			By("the system node group is untouched and no GPU or mock node was started")
			for _, name := range nodeClaimNames(env.nodeClaims(ctx, env.GPUNodePool)) {
				Expect(gpuBefore).To(ContainElement(name), "a GPU NodeClaim appeared for a pod that asks for no GPU")
			}

			By("deleting the workload: Karpenter consolidates the now empty node")
			del := time.Now()
			env.delete(ctx, newObj(gvkDeployment, c.NS, "inflate"))
			waitFor(ctx, "NodeClaim and node removed", 8*time.Minute, func() (string, bool) {
				led.observe(ctx)
				_, err := env.get(ctx, gvkNodeClaim, "", claim.GetName())
				if err == nil {
					return "NodeClaim still present", false
				}
				_, err = env.get(ctx, gvkNode, "", str(claim, "status", "nodeName"))
				return "node still present", err != nil
			})
			c.Measure("empty_to_node_gone_seconds", time.Since(del).Seconds(), "s")
			led.report(c, "")
		},
	})
})

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

func podsWhy(ctx context.Context, ns, app string) string {
	var out []string
	for _, p := range env.podsOf(ctx, ns, app) {
		out = append(out, p.GetName()+": "+podWhy(&p))
		for _, ev := range env.events(ctx, ns, "Pod", p.GetName()) {
			out = append(out, "  "+ev)
		}
	}
	return strings.Join(out, "; ")
}

// ---------------------------------------------------------------------------
// nodeLedger: measured node-hours and cost of the nodes a case caused
// ---------------------------------------------------------------------------

type ledgerNode struct {
	Claim, InstanceType, CapacityType, Zone string
	First, Last                             time.Time
	Registered                              time.Time
}

// nodeLedger watches the NodeClaims of one NodePool ("" = all) by polling and
// turns their lifetimes into node-hours and dollars. Lifetime is counted from
// the NodeClaim's creation to the last time it was seen, which is what EC2
// bills for within the poll interval.
type nodeLedger struct {
	pool  string
	known map[string]bool // NodeClaims that existed before the case
	nodes map[string]*ledgerNode
}

func newNodeLedger(pool string) *nodeLedger {
	l := &nodeLedger{pool: pool, known: map[string]bool{}, nodes: map[string]*ledgerNode{}}
	for _, nc := range env.nodeClaims(context.Background(), pool) {
		l.known[nc.GetName()] = true
	}
	return l
}

func (l *nodeLedger) observe(ctx context.Context) {
	now := time.Now()
	for _, nc := range env.nodeClaims(ctx, l.pool) {
		if l.known[nc.GetName()] {
			continue
		}
		n := l.nodes[nc.GetName()]
		if n == nil {
			n = &ledgerNode{Claim: nc.GetName(), First: nc.GetCreationTimestamp().Time}
			l.nodes[nc.GetName()] = n
		}
		labels := nc.GetLabels()
		if v := labels["node.kubernetes.io/instance-type"]; v != "" {
			n.InstanceType, n.CapacityType, n.Zone = v, labels["karpenter.sh/capacity-type"], labels["topology.kubernetes.io/zone"]
		}
		if n.Registered.IsZero() && condition(&nc, "Registered") == "True" {
			n.Registered = now
		}
		n.Last = now
	}
}

// report records node count, node-hours and on-demand cost. priceAs substitutes
// the price of another instance type (rehearsal: mock GPUs run on CPU instances
// but the FinOps comparison is about GPU prices); "" uses the real type.
func (l *nodeLedger) report(c *C, priceAs string) (nodeHours, usd float64) {
	var rows []map[string]any
	var keys []string
	for k := range l.nodes {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		n := l.nodes[k]
		if n.InstanceType == "" {
			continue // never launched (e.g. insufficient capacity): not billed
		}
		hours := n.Last.Sub(n.First).Hours()
		priced := n.InstanceType
		if priceAs != "" {
			priced = priceAs
		}
		cost := hours * env.hourlyPrice(priced)
		nodeHours += hours
		usd += cost
		rows = append(rows, map[string]any{
			"nodeclaim": n.Claim, "instance_type": n.InstanceType, "capacity_type": n.CapacityType, "zone": n.Zone,
			"priced_as": priced, "hours": hours, "usd_on_demand": cost,
		})
	}
	c.Measure("nodes_launched", float64(len(rows)), "")
	c.Measure("node_hours", nodeHours, "h")
	c.Measure("cost_on_demand_usd", usd, "$")
	if len(rows) > 0 {
		c.Attach("nodes", rows)
	}
	return nodeHours, usd
}
