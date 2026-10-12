//go:build e2e

package suite

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/util/validation"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

// hypervisorPodName is the name the operator gives the hypervisor pod of a
// node: "hypervisor-<node>", shortened with a hash when that exceeds 63.
func hypervisorPodName(node string) string {
	return utils.BuildHypervisorPodName(node)
}

var _ = Describe("long node names", func() {
	Register(Case{
		ID: "NLC-10", Title: "节点名超过 63 个字符：GPU 仍能注册、被选中、被调度",
		Tiers: []string{TierKWOK}, Needs: []string{NeedTF, NeedKarpenter, NeedGPU}, Timeout: 30 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			// The KWOK provider names a node kwok-<nodeclaim>-<two words>-<n>-
			// <random>-<uint32>. For a NodeClaim of the TensorFusion expander
			// (tensor-fusion-<10 characters>) that is 65 characters or more, longer
			// than a label value may be. So: fill what is there, and let one more pod
			// make the expander add a node.
			env.ensureGPUs(ctx, c, 1)
			By("occupy every GPU that exists, then ask for one more")
			var fillers []string
			for i := range env.readyGPUs(ctx) {
				n := fmt.Sprintf("fill-%d", i)
				fillers = append(fillers, n)
				Expect(env.create(ctx, tfPod(c.NS, n, wholeGPU(), tfNodeSelector()))).To(Succeed())
			}
			env.waitAllReady(ctx, c.NS, fillers, 8*time.Minute)
			start := time.Now()
			like := env.readyGPUs(ctx)[0]
			Expect(env.create(ctx, tfPod(c.NS, "on-long-node", fraction(like, "soft", 20, 2), tfNodeSelector()))).To(Succeed())
			pod, _ := env.waitPodReady(ctx, c.NS, "on-long-node", 12*time.Minute)
			c.Measure("pending_to_pod_running_seconds", time.Since(start).Seconds(), "s")

			node := str(pod, "spec", "nodeName")
			if len(node) <= maxLabelValueLen {
				Skip(fmt.Sprintf("the node the expander added (%s) has a name of %d characters: nothing to show here", node, len(node)))
			}
			ids := podGPUIDs(pod)
			Expect(ids).To(HaveLen(1))
			c.Note("node %s (%d characters), GPU %s", node, len(node), ids[0])
			c.Measure("node_name_length", float64(len(node)), "chars")

			By("the GPU's owner label is a valid label value that the selectors compute from the node name")
			obj, err := env.get(ctx, gvkGPU, "", ids[0])
			Expect(err).NotTo(HaveOccurred())
			owner := obj.GetLabels()[tfDomain+"/managed-by"]
			Expect(validation.IsValidLabelValue(owner)).To(BeEmpty())
			Expect(owner).To(Equal(utils.LabelSafeValue(node)))
			Expect(owner).NotTo(Equal(node))
			Expect(ownerKinds(obj)).To(ContainElement("GPUNode/"+node), "the full node name is in the owner reference")
			Expect(str(obj, "status", "nodeSelector", "kubernetes.io/hostname")).To(Equal(node))
			Expect(str(obj, "status", "phase")).To(Equal("Running"))

			By("the operator found the GPU through that label: the GPUNode is Running and accounts for the pod")
			gn, err := env.get(ctx, gvkGPUNode, "", node)
			Expect(err).NotTo(HaveOccurred())
			Expect(str(gn, "status", "phase")).To(Equal("Running"))
			Expect(qty(gn, "status", "totalTFlops")).To(BeNumerically(">", 0))
			Eventually(func() float64 {
				cur, _ := env.get(ctx, gvkGPUNode, "", node)
				return qty(cur, "status", "totalTFlops") - qty(cur, "status", "availableTFlops")
			}, 2*time.Minute, 5*time.Second).Should(BeNumerically(">", 0),
				"the node's allocation is summed over the GPUs the owner selector returns")
			hv, err := env.get(ctx, gvkPod, env.TFNamespace, hypervisorPodName(node))
			Expect(err).NotTo(HaveOccurred())
			Expect(str(hv, "spec", "nodeName")).To(Equal(node))
		},
	})
})
