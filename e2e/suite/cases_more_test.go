//go:build e2e

package suite

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var _ = Describe("scheduling", func() {
	Register(Case{
		ID: "SCH-04", Title: "Scheduling by GPU model: a matching model is schedulable, a model that does not exist stays Pending", Tiers: []string{TierCore, TierRehearsal},
		Needs: []string{NeedTF, NeedGPU}, Timeout: 20 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			ok := merge(fraction(g, "soft", 10, 1), ann("gpu-model", g.Model))
			Expect(env.create(ctx, tfPod(c.NS, "match", ok, tfNodeSelector()))).To(Succeed())
			p, _ := env.waitPodReady(ctx, c.NS, "match", 8*time.Minute)
			u, err := env.get(ctx, gvkGPU, "", podGPUIDs(p)[0])
			Expect(err).NotTo(HaveOccurred())
			Expect(toGPU(u).Model).To(Equal(g.Model))

			bad := merge(fraction(g, "soft", 10, 1), ann("gpu-model", "No-Such-GPU-9000"))
			// noProvision: a model that exists nowhere must not buy a node either;
			// whether the expander tries is recorded from the events.
			Expect(env.create(ctx, tfPod(c.NS, "nomatch", bad, tfNodeSelector()))).To(Succeed())
			before := nodeClaimNames(env.nodeClaims(ctx, ""))
			never(ctx, "pod asking for a non-existent GPU model was scheduled", 60*time.Second, func() (string, bool) {
				q, _ := env.get(ctx, gvkPod, c.NS, "nomatch")
				return "node " + str(q, "spec", "nodeName"), str(q, "spec", "nodeName") != ""
			})
			after := nodeClaimNames(env.nodeClaims(ctx, ""))
			c.Measure("nodeclaims_created_for_unsatisfiable_model", float64(len(after)-len(before)), "")
			c.Note("events of the unsatisfiable pod: %s", strings.Join(compactEvents(env.events(ctx, c.NS, "Pod", "nomatch")), " | "))
			env.delete(ctx, newObj(gvkPod, c.NS, "nomatch"))
			Expect(len(after)).To(Equal(len(before)), "a node was requested for a GPU model that no node can provide")
		},
	})

	Register(Case{
		ID: "SCH-11", Title: "Gang scheduling (positive): members run together once all are present; annotations and status are visible", Tiers: []string{TierCore, TierRehearsal},
		Needs: []string{NeedTF, NeedGPU}, Timeout: 20 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			tmpl := tfPod(c.NS, "gang", merge(fraction(gpus[0], "soft", 15, 1), ann("gang-enabled", "true", "gang-min-members", "2", "gang-timeout", "3m")), tfNodeSelector())
			Expect(env.create(ctx, deployment(c.NS, "gang", 2, tmpl))).To(Succeed())
			start := time.Now()
			waitFor(ctx, "both gang members Running", 8*time.Minute, func() (string, bool) {
				ready := 0
				pods := env.podsOf(ctx, c.NS, "gang")
				for i := range pods {
					if podReady(&pods[i]) {
						ready++
					}
				}
				return fmt.Sprintf("%d/2 ready; %s", ready, podsWhy(ctx, c.NS, "gang")), ready == 2
			})
			c.Measure("gang_all_running_seconds", time.Since(start).Seconds(), "s")
			pods := env.podsOf(ctx, c.NS, "gang")
			keys := map[string]bool{}
			for i := range pods {
				a := pods[i].GetAnnotations()
				Expect(a).To(HaveKeyWithValue(tfDomain+"/gang-required-members", "2"))
				keys[a[tfDomain+"/gang-group-key"]] = true
			}
			Expect(keys).To(HaveLen(1), "gang members must share one group key: %v", keys)
			wl, _ := env.list(ctx, gvkWorkload, c.NS)
			Expect(wl).To(HaveLen(1))
			c.Note("workload status.gang: %s", toJSON(mapAt(&wl[0], "status", "gang")))
		},
	})

	Register(Case{
		ID: "ISO-06", Title: "remote mode: the application Pod is on a CPU node and uses a remote GPU through a TensorFusionConnection", Tiers: []string{TierCore},
		Needs: []string{NeedTF, NeedRealGPU}, Timeout: 30 * time.Minute,
		KnownIssue: "the remote worker of the default ProviderConfig (tensorfusion/tensor-fusion-worker v2.25.0, community edition) quits at start with \"VerifyLicense: License is not set\" and the pod goes into CrashLoopBackOff with exit code 0; neither the chart nor the operator has a setting that hands it a license, so the client never gets a GPU (seen on a real T4, 2026-10-12)",
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			env.shipWorkloads(ctx, c.NS)
			a := merge(shareAnnotations(g, "soft", 0.3, 4), ann("is-local-gpu", "false"))
			pod := tfPod(c.NS, "remote", a, withShell(env.Images.Torch, "python3 /e2e/gpu_probe.py info && sleep 600"), withScripts(), loadNodeSelector("system"))
			Expect(env.create(ctx, pod)).To(Succeed())
			p, _ := env.waitPodReady(ctx, c.NS, "remote", 15*time.Minute)
			Expect(p.GetLabels()).To(HaveKeyWithValue(lblComponent, "client"))
			// Any GPU of the pool may serve the connection, not only the first one.
			var gpuNodes []string
			for _, x := range gpus {
				gpuNodes = append(gpuNodes, x.Node)
			}
			Expect(gpuNodes).NotTo(ContainElement(str(p, "spec", "nodeName")), "the client must not run on a GPU node")

			conns, err := env.list(ctx, gvkConnection, c.NS)
			Expect(err).NotTo(HaveOccurred())
			Expect(conns).NotTo(BeEmpty(), "no TensorFusionConnection for the remote client")
			c.Note("connection status: %s", toJSON(mapAt(&conns[0], "status")))
			workers, _ := env.list(ctx, gvkPod, c.NS, client.MatchingLabels{lblComponent: "worker"})
			Expect(workers).NotTo(BeEmpty(), "no worker pod")
			Expect(gpuNodes).To(ContainElement(str(&workers[0], "spec", "nodeName")), "the worker must run on a GPU node of the pool")

			var info map[string]any
			waitFor(ctx, "probe output from the client", 5*time.Minute, func() (string, bool) {
				res := results(env.podLogs(ctx, c.NS, "remote", "main"))
				if len(res) == 0 {
					return "no RESULT line yet", false
				}
				info = res[0]
				return "", true
			})
			Expect(info["cuda_available"]).To(Equal(true), "CUDA not usable through the remote GPU: %v", info)
			Expect(num(info, "device_count")).To(Equal(1.0))
		},
	})
})

var _ = Describe("cloud console", func() {
	Register(Case{
		ID: "CLD-01", Title: "cluster-agent sidecar is running (cluster registered with the console)", Tiers: []string{TierSmoke}, Needs: []string{NeedTF}, NoNS: true,
		Run: func(ctx SpecContext, c *C) {
			dep, err := env.get(ctx, gvkDeployment, env.TFNamespace, env.TFRelease+"-controller")
			Expect(err).NotTo(HaveOccurred())
			has := false
			for _, raw := range sliceAt(dep, "spec", "template", "spec", "containers") {
				has = has || raw.(map[string]any)["name"] == "cluster-agent"
			}
			if !has {
				Skip("cloud console agent not configured (install with TF_AGENT_ID / TF_ENROLL_TOKEN in the environment)")
			}
			pods, _ := env.list(ctx, gvkPod, env.TFNamespace, client.MatchingLabels{lblComponent: "operator"})
			Expect(pods).NotTo(BeEmpty())
			waitFor(ctx, "cluster-agent container Ready", 3*time.Minute, func() (string, bool) {
				p, _ := env.get(ctx, gvkPod, env.TFNamespace, pods[0].GetName())
				for _, cs := range sliceAt(p, "status", "containerStatuses") {
					m := cs.(map[string]any)
					if m["name"] == "cluster-agent" {
						return podWhy(p), m["ready"] == true
					}
				}
				return "container status missing", false
			})
			logs := env.podLogs(ctx, env.TFNamespace, pods[0].GetName(), "cluster-agent")
			// The enrol token must never be echoed by the agent.
			if tok := os.Getenv("TF_ENROLL_TOKEN"); tok != "" {
				Expect(logs).NotTo(ContainSubstring(tok), "cluster-agent logs contain the enrol token")
			}
			c.Note("agent log tail: %s", lastLines(logs, 2))
			c.Note("whether the cluster shows up in the console UI is a manual check")
		},
	})
})

var _ = Describe("uninstall", func() {
	Register(Case{
		ID: "UPG-03", Title: "The uninstall script cleans up completely (CRs, CRDs, webhook, NodeOverlay, node labels and index capacity)", Tiers: []string{TierDisrupt},
		Needs: []string{NeedTF}, NoNS: true, Timeout: 25 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			script := filepath.Join(env.RepoRoot, "scripts", "uninstall.sh")
			start := time.Now()
			cmd := exec.CommandContext(ctx, "bash", script)
			cmd.Env = append(os.Environ(), "NAMESPACE="+env.TFNamespace, "HELM_RELEASE="+env.TFRelease)
			out, err := cmd.CombinedOutput()
			Expect(err).NotTo(HaveOccurred(), "uninstall.sh failed: %s", lastLines(string(out), 12))
			c.Measure("uninstall_seconds", time.Since(start).Seconds(), "s")

			for _, plural := range tfCRDs {
				_, err := env.get(ctx, gvkCRD, "", plural+"."+tfDomain)
				Expect(apierrors.IsNotFound(err)).To(BeTrue(), "CRD %s still present", plural)
			}
			_, err = env.get(ctx, gvk("admissionregistration.k8s.io", "v1", "MutatingWebhookConfiguration"), "", env.TFRelease+"-mutating-webhook")
			Expect(apierrors.IsNotFound(err)).To(BeTrue(), "mutating webhook still present")
			_, err = env.get(ctx, gvkNamespace, "", env.TFNamespace)
			Expect(apierrors.IsNotFound(err)).To(BeTrue(), "namespace %s still present", env.TFNamespace)
			if env.HasKarpenter {
				_, err = env.get(ctx, gvkNodeOverlay, "", "tensor-fusion-overlay")
				Expect(apierrors.IsNotFound(err)).To(BeTrue(), "NodeOverlay still present")
			}
			for _, n := range env.nodes(ctx, nil) {
				for k := range n.GetLabels() {
					Expect(strings.HasPrefix(k, tfDomain+"/")).To(BeFalse(), "node %s still has label %s", n.GetName(), k)
				}
				for k := range mapAt(&n, "status", "capacity") {
					Expect(strings.HasPrefix(k, tfDomain+"/")).To(BeFalse(), "node %s still advertises %s", n.GetName(), k)
				}
			}
			// A plain pod must be creatable again without the webhook.
			ns := env.newNamespace(ctx, "upg-03")
			DeferCleanup(func() { env.deleteNamespace(ns) })
			Expect(env.dryRun(ctx, plainPod(ns, "after-uninstall"))).To(Succeed())
		},
	})
})
