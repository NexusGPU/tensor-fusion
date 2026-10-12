//go:build e2e

package suite

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// Plumbing shared by the isolation and perf tiers: ship the workload scripts
// into the cluster, run them as pods, turn their RESULT lines into PerfPoints.

//go:embed workloads/*.py
var workloadFS embed.FS

const workloadsCM = "tf-e2e-workloads"

// shipWorkloads puts every embedded script into a ConfigMap of the namespace.
func (e *Env) shipWorkloads(ctx context.Context, ns string) {
	GinkgoHelper()
	if _, err := e.get(ctx, gvk("", "v1", "ConfigMap"), ns, workloadsCM); err == nil {
		return
	}
	entries, err := workloadFS.ReadDir("workloads")
	Expect(err).NotTo(HaveOccurred())
	data := map[string]any{}
	for _, f := range entries {
		b, err := workloadFS.ReadFile("workloads/" + f.Name())
		Expect(err).NotTo(HaveOccurred())
		data[f.Name()] = string(b)
	}
	cm := &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "v1", "kind": "ConfigMap",
		"metadata": map[string]any{"name": workloadsCM, "namespace": ns},
		"data":     data,
	}}
	Expect(e.create(ctx, cm)).To(Succeed())
}

// withScripts mounts the workload scripts at /e2e and a per-node cache at
// /cache (pip wheels, Hugging Face models: downloaded once per node, not once
// per measurement).
func withScripts() PodOpt {
	return func(pod map[string]any) {
		spec := pod["spec"].(map[string]any)
		vols, _ := spec["volumes"].([]any)
		spec["volumes"] = append(vols,
			map[string]any{"name": "e2e-scripts", "configMap": map[string]any{"name": workloadsCM}},
			map[string]any{"name": "e2e-cache", "hostPath": map[string]any{"path": "/var/lib/tf-e2e/cache", "type": "DirectoryOrCreate"}},
		)
		ct := spec["containers"].([]any)[0].(map[string]any)
		mounts, _ := ct["volumeMounts"].([]any)
		ct["volumeMounts"] = append(mounts,
			map[string]any{"name": "e2e-scripts", "mountPath": "/e2e"},
			map[string]any{"name": "e2e-cache", "mountPath": "/cache"},
		)
		envs, _ := ct["env"].([]any)
		ct["env"] = append(envs,
			map[string]any{"name": "HF_HOME", "value": "/cache/hf"},
			map[string]any{"name": "PIP_CACHE_DIR", "value": "/cache/pip"},
			map[string]any{"name": "PYTHONUNBUFFERED", "value": "1"},
			map[string]any{"name": "PIP_DISABLE_PIP_VERSION_CHECK", "value": "1"},
		)
	}
}

func withShell(image, script string) PodOpt {
	return withImage(image, "sh", "-c", script)
}

func withPort(port int64) PodOpt {
	return func(pod map[string]any) {
		ct := pod["spec"].(map[string]any)["containers"].([]any)[0].(map[string]any)
		ct["ports"] = []any{map[string]any{"containerPort": port}}
	}
}

// loadNodeSelector places load generators and CPU stand-ins away from GPU
// nodes when the cluster has the stack's role labels; on any other cluster
// (bare metal) the scheduler decides.
func loadNodeSelector(role string) PodOpt {
	return func(pod map[string]any) {
		if len(env.nodes(context.Background(), map[string]string{env.RoleLabel: "system"})) == 0 {
			return
		}
		pod["spec"].(map[string]any)["nodeSelector"] = map[string]any{env.RoleLabel: role}
	}
}

// native makes a plain pod see the GPU without TensorFusion: the baseline all
// overhead numbers are measured against. Works where the node's default
// container runtime is the NVIDIA one (EKS AL2023 NVIDIA AMI); elsewhere set
// E2E_NATIVE_RUNTIME_CLASS.
func nativeGPU(node string) PodOpt {
	return func(pod map[string]any) {
		spec := pod["spec"].(map[string]any)
		spec["nodeName"] = node
		if rc := getenv("E2E_NATIVE_RUNTIME_CLASS", ""); rc != "" {
			spec["runtimeClassName"] = rc
		}
		ct := spec["containers"].([]any)[0].(map[string]any)
		envs, _ := ct["env"].([]any)
		ct["env"] = append(envs, map[string]any{"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
			map[string]any{"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"})
	}
}

// waitDone waits for a run-to-completion pod and returns its logs.
func (e *Env) waitDone(ctx context.Context, ns, name string, timeout time.Duration) (string, bool) {
	GinkgoHelper()
	ok := false
	waitFor(ctx, fmt.Sprintf("pod %s/%s to finish", ns, name), timeout, func() (string, bool) {
		p, err := e.get(ctx, gvkPod, ns, name)
		if err != nil {
			return err.Error(), false
		}
		switch podPhase(p) {
		case "Succeeded":
			ok = true
			return "", true
		case "Failed":
			return "", true
		}
		return podWhy(p), false
	})
	return e.podLogs(ctx, ns, name, "main"), ok
}

// results extracts the `RESULT {json}` lines a workload script printed.
func results(logs string) []map[string]any {
	var out []map[string]any
	for _, line := range strings.Split(logs, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "RESULT ") {
			continue
		}
		var m map[string]any
		if json.Unmarshal([]byte(line[len("RESULT "):]), &m) == nil {
			out = append(out, m)
		}
	}
	return out
}

func num(m map[string]any, key string) float64 {
	switch v := m[key].(type) {
	case float64:
		return v
	case int:
		return float64(v)
	}
	return 0
}

func dist(m map[string]any, key string) Dist {
	d, _ := m[key].(map[string]any)
	if d == nil {
		return Dist{}
	}
	return Dist{N: int(num(d, "n")), P50: num(d, "p50"), P95: num(d, "p95"), P99: num(d, "p99"), Max: num(d, "max"), Mean: num(d, "mean")}
}

func text(m map[string]any, key string) string {
	if s, ok := m[key].(string); ok {
		return s
	}
	return ""
}

// llmPoint converts an llm_client.py result.
func llmPoint(res map[string]any, base PerfPoint) PerfPoint {
	p := base
	p.Throughput, p.ThroughputUnit, p.TPM = num(res, "tokens_per_s"), "tokens/s", num(res, "tpm")
	p.TTFT, p.TPOT, p.ITL, p.Latency = dist(res, "ttft_s"), dist(res, "tpot_s"), dist(res, "itl_s"), dist(res, "latency_s")
	p.Requests, p.Samples, p.WindowS = int(num(res, "requests")), int(num(res, "output_tokens")), num(res, "window_s")
	p.ErrorRate, p.Timeouts = num(res, "error_rate"), int(num(res, "timeouts"))
	p.Gaps400, p.Gaps900 = int(num(res, "gaps_ge_400ms")), int(num(res, "gaps_ge_900ms"))
	p.GapsNear500, p.GapsNear1s = int(num(res, "gaps_near_500ms")), int(num(res, "gaps_near_1s"))
	if b, ok := res["itl_bimodality"].(float64); ok {
		p.ITLBimodality = &b
	}
	if p.Requests == 0 {
		p.Error = "no request completed: " + text(res, "first_error")
	}
	return p
}

// benchPoint converts a result of ort_bench.py / torch_mini.py / mlperf_mini.py / gpu_probe.py burn.
func benchPoint(res map[string]any, base PerfPoint) PerfPoint {
	p := base
	if m := text(res, "model"); m != "" {
		p.Model = m
	}
	if t := text(res, "test"); t != "" {
		p.Test = t
	}
	if s := text(res, "scenario"); s != "" {
		p.Test = s
	}
	p.Error, p.Skipped = text(res, "error"), text(res, "skipped")
	switch {
	case res["qps"] != nil:
		p.Throughput, p.ThroughputUnit, p.Samples = num(res, "qps"), "samples/s", int(num(res, "calls"))
	case res["samples_per_second"] != nil || res["completed_samples_per_second"] != nil:
		p.Throughput, p.ThroughputUnit = num(res, "samples_per_second"), "samples/s"
		if p.Throughput == 0 {
			p.Throughput = num(res, "completed_samples_per_second")
		}
		p.Latency = Dist{P50: num(res, "p50_latency_ns") / 1e9, P99: num(res, "p99_latency_ns") / 1e9, Max: num(res, "max_latency_ns") / 1e9}
		p.Samples = int(p.Throughput * num(res, "min_duration_s"))
	default:
		p.Throughput, p.ThroughputUnit, p.Samples = num(res, "iters_per_s"), "iters/s", int(num(res, "iters"))
	}
	if res["latency_s"] != nil {
		p.Latency = dist(res, "latency_s")
	} else if res["iter_s"] != nil {
		p.Latency = dist(res, "iter_s")
	}
	if p.Latency.N == 0 {
		p.Latency.N = p.Samples
	}
	p.WindowS = num(res, "window_s")
	p.Gaps400, p.Gaps900 = int(num(res, "stalls_ge_400ms")), int(num(res, "stalls_ge_900ms"))
	p.MaxMemoryMiB = num(res, "max_memory_mib")
	return p
}

// shareAnnotations expresses "share of one GPU's compute + a VRAM limit" for a
// TensorFusion pod. vramGi 0 = the whole card's memory.
func shareAnnotations(g GPU, mode string, share float64, vramGi float64) map[string]string {
	if mode == "shared" {
		return wholeGPU()
	}
	pct := fmt.Sprintf("%.0f", share*100)
	if vramGi <= 0 {
		vramGi = float64(int(g.CapVRAM))
	}
	v := fmt.Sprintf("%.0fGi", vramGi)
	return ann("isolation", mode, "compute-percent-request", pct, "compute-percent-limit", pct, "vram-request", v, "vram-limit", v)
}

// profile selects the measurement budget. quick: a few minutes per runtime,
// fine for every run and for spot nodes. full: the complete design.
type perfProfile struct {
	Name           string
	Warmup, Window float64 // seconds
	Shares         []float64
	Concurrency    int
	MaxTokens      int
}

func profile() perfProfile {
	if getenv("E2E_PERF_PROFILE", "quick") == "full" {
		// 180 s at 8 streams x ~64 tokens: ~600 requests (TTFT p99 from 6 samples
		// above it - reported, TTFT p95 is the gate) and > 30k tokens (ITL p99
		// from > 300 samples above it).
		return perfProfile{Name: "full", Warmup: 30, Window: 180, Shares: []float64{1, 0.75, 0.5, 0.25}, Concurrency: 8, MaxTokens: 64}
	}
	// 60 s: ~200 requests, ~10k tokens. Enough for ITL / TPOT p99, TTFT p95.
	return perfProfile{Name: "quick", Warmup: 15, Window: 60, Shares: []float64{1, 0.5, 0.25}, Concurrency: 8, MaxTokens: 64}
}

// llmModel picks an ungated model that fits the VRAM the pod is given.
// Qwen2.5 Instruct, Apache-2.0, no Hugging Face token needed.
func llmModel(vramGi float64) string {
	if m := getenv("E2E_LLM_MODEL", ""); m != "" {
		return m
	}
	if vramGi >= 10 {
		return "Qwen/Qwen2.5-1.5B-Instruct" // ~3.1 GiB fp16 weights + KV cache
	}
	return "Qwen/Qwen2.5-0.5B-Instruct" // ~1 GiB fp16 weights: fits a 4 GiB slice of a T4
}

func (e *Env) podIP(ctx context.Context, ns, name string) string {
	p, err := e.get(ctx, gvkPod, ns, name)
	if err != nil {
		return ""
	}
	return str(p, "status", "podIP")
}

// runLLMClient runs llm_client.py against url and returns its result.
func (e *Env) runLLMClient(ctx context.Context, c *C, name, url, model, mode string, prof perfProfile, rate float64) map[string]any {
	GinkgoHelper()
	args := fmt.Sprintf("python3 /e2e/llm_client.py --url %s --model %s --warmup %.0f --duration %.0f --max-tokens %d --label %s",
		url, model, prof.Warmup, prof.Window, prof.MaxTokens, name)
	if mode == "open" {
		args += fmt.Sprintf(" --mode open --rate %.3f", rate)
	} else {
		args += fmt.Sprintf(" --mode closed --concurrency %d", prof.Concurrency)
	}
	pod := plainPod(c.NS, name, withShell(env.Images.Curl, args), withScripts(), loadNodeSelector("system"))
	Expect(e.create(ctx, pod)).To(Succeed())
	logs, ok := e.waitDone(ctx, c.NS, name, time.Duration(prof.Warmup+prof.Window+240)*time.Second)
	res := results(logs)
	if !ok || len(res) == 0 {
		Fail(fmt.Sprintf("load generator %s produced no result: %s", name, lastLines(logs, 5)))
	}
	return res[0]
}

func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, " / ")
}

// recordPerf evaluates the points, attaches them for the report generator and
// puts the headline numbers into the summary. Returns the failed criteria.
func recordPerf(c *C, points []PerfPoint) []string {
	for i := range points {
		points[i].Case = c.ID
		if points[i].ExecMode == "" {
			points[i].ExecMode = env.Mode
		}
	}
	evaluate(points)
	c.Attach("perf", points)
	for _, p := range points {
		if p.Skipped != "" || p.Error != "" {
			continue
		}
		key := strings.NewReplacer("/", "_", " ", "_").Replace(fmt.Sprintf("%s_%s_%s_%s_%03.0f", p.Workload, p.Model, p.Mode, p.Load, p.Share*100))
		if p.Test != "" {
			key += "_" + p.Test
		}
		c.Measure(key+"_throughput", p.Throughput, p.ThroughputUnit)
		if p.TPOT.N > 0 {
			c.Measure(key+"_tpot_p99", p.TPOT.P99*1000, "ms")
			c.Measure(key+"_ttft_p95", p.TTFT.P95*1000, "ms")
			c.Measure(key+"_itl_max", p.ITL.Max*1000, "ms")
		} else if p.Latency.P99 > 0 {
			c.Measure(key+"_latency_p99", p.Latency.P99*1000, "ms")
		}
	}
	return failedVerdicts(points)
}
