//go:build e2e

package suite

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Annotation / label names. Kept as literals (not imported from pkg/constants):
// they are the public contract the docs promise, so a rename in the code must
// fail these tests.
const (
	lblEnabled   = tfDomain + "/enabled"
	lblPool      = tfDomain + "/gpupool"
	lblComponent = tfDomain + "/component"
	lblWorkload  = tfDomain + "/workload"

	schedulerName = "tensor-fusion-scheduler"
)

// ann builds a TensorFusion annotation map from short keys:
// ann("isolation", "soft", "vram-limit", "4Gi").
func ann(kv ...string) map[string]string {
	if len(kv)%2 != 0 {
		panic("ann needs key/value pairs")
	}
	m := map[string]string{}
	for i := 0; i < len(kv); i += 2 {
		m[tfDomain+"/"+kv[i]] = kv[i+1]
	}
	return m
}

func merge(maps ...map[string]string) map[string]string {
	out := map[string]string{}
	for _, m := range maps {
		for k, v := range m {
			out[k] = v
		}
	}
	return out
}

// PodOpt tweaks a pod built by tfPod.
type PodOpt func(pod map[string]any)

func withImage(image string, command ...string) PodOpt {
	return func(pod map[string]any) {
		c := pod["spec"].(map[string]any)["containers"].([]any)[0].(map[string]any)
		c["image"] = image
		if len(command) > 0 {
			c["command"] = toAnySlice(command)
		}
	}
}

func withNodeSelector(sel map[string]string) PodOpt {
	return func(pod map[string]any) {
		pod["spec"].(map[string]any)["nodeSelector"] = toAnyMap(sel)
	}
}

func withLabels(l map[string]string) PodOpt {
	return func(pod map[string]any) {
		md := pod["metadata"].(map[string]any)
		md["labels"] = toAnyMap(merge(toStringMap(md["labels"]), l))
	}
}

func withExtraContainer(name, image string, command ...string) PodOpt {
	return func(pod map[string]any) {
		spec := pod["spec"].(map[string]any)
		spec["containers"] = append(spec["containers"].([]any), map[string]any{
			"name": name, "image": image, "command": toAnySlice(command),
		})
	}
}

func withEnv(kv ...string) PodOpt {
	return func(pod map[string]any) {
		c := pod["spec"].(map[string]any)["containers"].([]any)[0].(map[string]any)
		var envs []any
		for i := 0; i < len(kv); i += 2 {
			envs = append(envs, map[string]any{"name": kv[i], "value": kv[i+1]})
		}
		c["env"] = envs
	}
}

func withGrace(seconds int64) PodOpt {
	return func(pod map[string]any) {
		pod["spec"].(map[string]any)["terminationGracePeriodSeconds"] = seconds
	}
}

func withResources(requests map[string]string) PodOpt {
	return func(pod map[string]any) {
		c := pod["spec"].(map[string]any)["containers"].([]any)[0].(map[string]any)
		c["resources"] = map[string]any{"requests": toAnyMap(requests)}
	}
}

func toAnySlice(s []string) []any {
	out := make([]any, len(s))
	for i := range s {
		out[i] = s[i]
	}
	return out
}

func toAnyMap(m map[string]string) map[string]any {
	out := map[string]any{}
	for k, v := range m {
		out[k] = v
	}
	return out
}

func toStringMap(v any) map[string]string {
	out := map[string]string{}
	if m, ok := v.(map[string]any); ok {
		for k, val := range m {
			out[k] = fmt.Sprint(val)
		}
	}
	return out
}

// tfPod builds a pod the TensorFusion webhook will pick up: the enabled label
// plus the given annotations. The container sleeps; cases that need a real
// workload replace image and command.
func tfPod(ns, name string, annotations map[string]string, opts ...PodOpt) *unstructured.Unstructured {
	pod := plainPod(ns, name, opts...)
	labels := pod.GetLabels()
	labels[lblEnabled] = "true"
	pod.SetLabels(labels)
	if _, ok := annotations[lblPool]; !ok && env.Pool != "" && env.Pool != env.RealPool {
		// The chart's pool is the default pool; any other pool (the mock pool of
		// the rehearsal) has to be named explicitly.
		annotations = merge(annotations, map[string]string{lblPool: env.Pool})
	}
	pod.SetAnnotations(annotations)
	return pod
}

// plainPod is a pod TensorFusion must leave alone.
func plainPod(ns, name string, opts ...PodOpt) *unstructured.Unstructured {
	obj := map[string]any{
		"apiVersion": "v1",
		"kind":       "Pod",
		"metadata": map[string]any{
			"name": name, "namespace": ns,
			"labels": map[string]any{"app": name, "tf-e2e/run": env.RunID},
		},
		"spec": map[string]any{
			"restartPolicy":                 "Never",
			"terminationGracePeriodSeconds": int64(1),
			"containers": []any{map[string]any{
				"name":    "main",
				"image":   env.Images.Busybox,
				"command": []any{"sh", "-c", "sleep 86400"},
			}},
		},
	}
	for _, o := range opts {
		o(obj)
	}
	return &unstructured.Unstructured{Object: obj}
}

// deployment wraps a pod template built by tfPod / plainPod.
func deployment(ns, name string, replicas int, tmpl *unstructured.Unstructured) *unstructured.Unstructured {
	labels := merge(tmpl.GetLabels(), map[string]string{"app": name})
	spec := tmpl.Object["spec"].(map[string]any)
	spec["restartPolicy"] = "Always"
	return &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "apps/v1",
		"kind":       "Deployment",
		"metadata":   map[string]any{"name": name, "namespace": ns, "labels": map[string]any{"tf-e2e/run": env.RunID}},
		"spec": map[string]any{
			"replicas": int64(replicas),
			"selector": map[string]any{"matchLabels": map[string]any{"app": name}},
			"template": map[string]any{
				"metadata": map[string]any{"labels": toAnyMap(labels), "annotations": toAnyMap(tmpl.GetAnnotations())},
				"spec":     spec,
			},
		},
	}}
}

func (e *Env) podsOf(ctx context.Context, ns, app string) []unstructured.Unstructured {
	pods, err := e.list(ctx, gvkPod, ns, client.MatchingLabels{"app": app})
	Expect(err).NotTo(HaveOccurred())
	sort.Slice(pods, func(i, j int) bool { return pods[i].GetName() < pods[j].GetName() })
	return pods
}

// ---------------------------------------------------------------------------
// GPU inventory
// ---------------------------------------------------------------------------

// GPU is the slice of a GPU CR the cases reason about.
type GPU struct {
	Name, Node, Model, Vendor, Phase, UUID     string
	CapTflops, CapVRAM, AvailTflops, AvailVRAM float64 // TFLOPS, GiB
	Mode                                       string
}

func qty(u *unstructured.Unstructured, path ...string) float64 {
	s := str(u, path...)
	if s == "" {
		return 0
	}
	q, err := resource.ParseQuantity(s)
	if err != nil {
		return 0
	}
	return q.AsApproximateFloat64()
}

const gib = 1024 * 1024 * 1024

func toGPU(u *unstructured.Unstructured) GPU {
	node := str(u, "status", "nodeSelector", "kubernetes.io/hostname")
	if node == "" {
		node = u.GetLabels()["kubernetes.io/hostname"]
	}
	return GPU{
		Name: u.GetName(), Node: node,
		Model: str(u, "status", "gpuModel"), Vendor: str(u, "status", "vendor"),
		Phase: str(u, "status", "phase"), UUID: str(u, "status", "uuid"),
		CapTflops: qty(u, "status", "capacity", "tflops"), CapVRAM: qty(u, "status", "capacity", "vram") / gib,
		AvailTflops: qty(u, "status", "available", "tflops"), AvailVRAM: qty(u, "status", "available", "vram") / gib,
		Mode: str(u, "status", "activeIsolationMode"),
	}
}

// poolGPUs lists the GPU CRs of the pool the run targets.
func (e *Env) poolGPUs(ctx context.Context) []GPU {
	return e.gpusOf(ctx, e.Pool)
}

func (e *Env) gpusOf(ctx context.Context, pool string) []GPU {
	items, err := e.list(ctx, gvkGPU, "", client.MatchingLabels{lblPool: pool})
	if err != nil {
		return nil
	}
	out := make([]GPU, 0, len(items))
	for i := range items {
		out = append(out, toGPU(&items[i]))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// waitResourcesReturned is the suite's single definition of "no leak": every
// GPU of the pool is back to available == capacity.
func (e *Env) waitResourcesReturned(ctx context.Context, timeout time.Duration) time.Duration {
	GinkgoHelper()
	return waitFor(ctx, "GPU resources returned (available == capacity on every GPU of pool "+e.Pool+")", timeout, func() (string, bool) {
		var busy []string
		for _, g := range e.poolGPUs(ctx) {
			if g.CapTflops-g.AvailTflops > 0.01 || g.CapVRAM-g.AvailVRAM > 0.01 {
				busy = append(busy, fmt.Sprintf("%s tflops %.1f/%.1f vram %.1f/%.1fGi", g.Name, g.AvailTflops, g.CapTflops, g.AvailVRAM, g.CapVRAM))
			}
		}
		return strings.Join(busy, ", "), len(busy) == 0
	})
}

// podGPUIDs returns the GPU CR names the scheduler bound the pod to.
func podGPUIDs(p *unstructured.Unstructured) []string {
	v := p.GetAnnotations()[tfDomain+"/gpu-ids"]
	if v == "" {
		return nil
	}
	return strings.Split(v, ",")
}

// ---------------------------------------------------------------------------
// controller HTTP API (through the API server's service proxy)
// ---------------------------------------------------------------------------

func (e *Env) controllerAPI(ctx context.Context, path string, params map[string]string) ([]byte, error) {
	return e.svcGet(ctx, e.TFNamespace, e.TFRelease, "8080", path, params)
}

// greptimeSQL runs a query through GreptimeDB's HTTP API and returns the rows.
func (e *Env) greptimeSQL(ctx context.Context, sql string) ([][]any, error) {
	raw, err := e.svcGet(ctx, "greptimedb", "greptimedb-standalone", "4000", "v1/sql", map[string]string{"sql": sql, "db": "public"})
	if err != nil {
		return nil, err
	}
	var resp struct {
		Output []struct {
			Records struct {
				Rows [][]any `json:"rows"`
			} `json:"records"`
		} `json:"output"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("greptime response: %w: %s", err, string(raw))
	}
	if resp.Error != "" {
		return nil, fmt.Errorf("greptime: %s", resp.Error)
	}
	if len(resp.Output) == 0 {
		return nil, nil
	}
	return resp.Output[0].Records.Rows, nil
}

// tableRows counts rows written to a metrics table in the last `window`.
func (e *Env) tableRows(ctx context.Context, table string, window time.Duration) (int, error) {
	// greptime_timestamp is the time index of the influx-line tables the
	// operator and hypervisor write through vector.
	sql := fmt.Sprintf("SELECT count(*) FROM %s WHERE greptime_timestamp > now() - INTERVAL '%d seconds'", table, int(window.Seconds()))
	rows, err := e.greptimeSQL(ctx, sql)
	if err != nil {
		return 0, err
	}
	if len(rows) == 0 || len(rows[0]) == 0 {
		return 0, nil
	}
	n, _ := strconv.Atoi(fmt.Sprint(rows[0][0]))
	if f, ok := rows[0][0].(float64); ok {
		n = int(f)
	}
	return n, nil
}

// ---------------------------------------------------------------------------
// Karpenter
// ---------------------------------------------------------------------------

// nodeClaims lists NodeClaims, optionally of one NodePool.
func (e *Env) nodeClaims(ctx context.Context, nodePool string) []unstructured.Unstructured {
	var opts []client.ListOption
	if nodePool != "" {
		opts = append(opts, client.MatchingLabels{"karpenter.sh/nodepool": nodePool})
	}
	items, _ := e.list(ctx, gvkNodeClaim, "", opts...)
	return items
}

func nodeClaimNames(items []unstructured.Unstructured) []string {
	var out []string
	for i := range items {
		out = append(out, items[i].GetName())
	}
	sort.Strings(out)
	return out
}

func nodeClaimWhy(nc *unstructured.Unstructured) string {
	var parts []string
	for _, c := range sliceAt(nc, "status", "conditions") {
		m, _ := c.(map[string]any)
		parts = append(parts, fmt.Sprintf("%v=%v(%v)", m["type"], m["status"], m["reason"]))
	}
	return fmt.Sprintf("%s type=%s node=%s %s", nc.GetName(), nc.GetLabels()["node.kubernetes.io/instance-type"], str(nc, "status", "nodeName"), strings.Join(parts, " "))
}

// nodes returns the nodes matching the labels.
func (e *Env) nodes(ctx context.Context, sel map[string]string) []unstructured.Unstructured {
	items, err := e.list(ctx, gvkNode, "", client.MatchingLabels(sel))
	Expect(err).NotTo(HaveOccurred())
	return items
}

func nodeReady(n *unstructured.Unstructured) bool { return condition(n, "Ready") == "True" }

// hourlyPrice returns the on-demand $/h of an instance type from the committed
// price table (0 when unknown).
func (e *Env) hourlyPrice(instanceType string) float64 {
	return e.Prices.OnDemand[instanceType]
}
