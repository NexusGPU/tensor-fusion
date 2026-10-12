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

// Tiers isolation and perf: everything here needs a CUDA device. None of it
// has been executed yet (no GPU quota when it was written); the tier `dryrun`
// proves the manifests, the scripts and the evaluation. Read failures of a
// first real run with that in mind.

// ---------------------------------------------------------------------------
// pod builders (also used by DRY-ISO-01)
// ---------------------------------------------------------------------------

const pipGPU = "pip install -q onnx onnxruntime-gpu transformers"

func ortCommand(models string, warmup, duration, vramGi float64) string {
	return fmt.Sprintf("%s && python3 /e2e/ort_bench.py --provider cuda --models %s --warmup %.0f --duration %.0f --vram-limit-mib %.0f", pipGPU, models, warmup, duration, vramGi*1024)
}

func torchCommand(models, tests string, warmup, duration, vramGi float64) string {
	return fmt.Sprintf("pip install -q transformers && python3 /e2e/torch_mini.py --models %s --tests %s --warmup %.0f --duration %.0f --vram-limit-mib %.0f", models, tests, warmup, duration, vramGi*1024)
}

func mlperfCommand(models, scenarios string, duration float64) string {
	return fmt.Sprintf("%s mlcommons-loadgen && python3 /e2e/mlperf_mini.py --provider cuda --models %s --scenarios %s --duration %.0f", pipGPU, models, scenarios, duration)
}

func burnCommand(seconds, warmup float64) string {
	return fmt.Sprintf("python3 /e2e/gpu_probe.py burn --seconds %.0f --warmup %.0f --size 2048 --fp16", seconds, warmup)
}

// benchPod runs a script under the given isolation mode. mode native = no
// TensorFusion at all, pinned to node.
func benchPod(ns, name string, g GPU, mode string, share, vramGi float64, node, command string) *unstructured.Unstructured {
	if mode == "native" {
		return plainPod(ns, name, withShell(env.Images.Torch, command), withScripts(), nativeGPU(node))
	}
	return tfPod(ns, name, shareAnnotations(g, mode, share, vramGi), withShell(env.Images.Torch, command), withScripts(), tfNodeSelector())
}

// vllmServer is an OpenAI-compatible vLLM server; Ready means the model is loaded.
func vllmServer(ns, name string, g GPU, mode string, share, vramGi float64, extra string) *unstructured.Unstructured {
	model := llmModel(vramGi)
	// Under a VRAM limit the container sees its quota as the card's total, so
	// the utilisation fraction applies to the quota.
	cmd := fmt.Sprintf("python3 -m vllm.entrypoints.openai.api_server --model %s --served-model-name model --port 8000 --dtype half --max-model-len 2048 --gpu-memory-utilization 0.85 %s", model, extra)
	probe := func(pod map[string]any) {
		ct := pod["spec"].(map[string]any)["containers"].([]any)[0].(map[string]any)
		ct["ports"] = []any{map[string]any{"containerPort": int64(8000)}}
		ct["readinessProbe"] = map[string]any{
			"httpGet":          map[string]any{"path": "/health", "port": int64(8000)},
			"periodSeconds":    int64(5),
			"failureThreshold": int64(120),
		}
	}
	if mode == "native" {
		return plainPod(ns, name, withShell(env.Images.VLLM, cmd), withScripts(), nativeGPU(g.Node), probe)
	}
	return tfPod(ns, name, shareAnnotations(g, mode, share, vramGi), withShell(env.Images.VLLM, cmd), withScripts(), tfNodeSelector(), probe)
}

// runBench runs one bench pod to completion and returns its points.
func (e *Env) runBench(ctx context.Context, c *C, name string, g GPU, base PerfPoint, command string, timeout time.Duration) []PerfPoint {
	GinkgoHelper()
	start := time.Now()
	Expect(e.create(ctx, benchPod(c.NS, name, g, base.Mode, base.Share, base.VRAMGiB, g.Node, command))).To(Succeed())
	logs, ok := e.waitDone(ctx, c.NS, name, timeout)
	var points []PerfPoint
	for _, r := range results(logs) {
		p := benchPoint(r, base)
		p.SetupSeconds = time.Since(start).Seconds() - p.WindowS
		p.GPUModel = g.Model
		points = append(points, p)
	}
	if !ok || len(points) == 0 {
		p := base
		p.Error = "pod failed: " + lastLines(logs, 4)
		points = append(points, p)
	}
	e.delete(ctx, newObj(gvkPod, c.NS, name))
	e.waitGone(ctx, gvkPod, c.NS, name, 3*time.Minute)
	if base.Mode != "native" {
		e.waitResourcesReturned(ctx, 3*time.Minute)
	}
	return points
}

// noisyNeighbour starts a pod that burns its whole share on the same GPU for
// as long as the measurement lasts.
func (e *Env) noisyNeighbour(ctx context.Context, c *C, g GPU, mode string, share float64) func() {
	GinkgoHelper()
	name := fmt.Sprintf("noisy-%d", time.Now().UnixNano()%100000)
	Expect(e.create(ctx, benchPod(c.NS, name, g, mode, share, 2, g.Node, burnCommand(3600, 0)))).To(Succeed())
	e.waitPodReady(ctx, c.NS, name, 8*time.Minute)
	return func() {
		e.delete(ctx, newObj(gvkPod, c.NS, name))
		e.waitGone(ctx, gvkPod, c.NS, name, 3*time.Minute)
	}
}

// llmSweep serves one model under `mode` at each share and measures it at
// saturation, plus one fixed-rate run. The server is restarted per share:
// TensorFusion has no in-place limit API besides the autoscaler (ISO-16), and
// the model comes from the node's cache after the first load.
func (e *Env) llmSweep(ctx context.Context, c *C, g GPU, mode string, shares []float64, vramGi float64, prof perfProfile, extra string, fixedRateAt float64) []PerfPoint {
	GinkgoHelper()
	var points []PerfPoint
	saturationRPS := 0.0
	for i, s := range shares {
		name := fmt.Sprintf("vllm-%s-%03.0f", mode, s*100)
		start := time.Now()
		Expect(e.create(ctx, vllmServer(c.NS, name, g, mode, s, vramGi, extra))).To(Succeed())
		e.waitPodReady(ctx, c.NS, name, 25*time.Minute)
		setup := time.Since(start).Seconds()
		url := "http://" + e.podIP(ctx, c.NS, name) + ":8000"
		base := PerfPoint{Workload: "vllm", Model: llmModel(vramGi), Mode: mode, Share: s, VRAMGiB: vramGi, Cotenancy: "alone", Load: "saturation", GPUModel: g.Model, SetupSeconds: setup}
		res := e.runLLMClient(ctx, c, fmt.Sprintf("load-%s-%03.0f", mode, s*100), url, "model", "closed", prof, 0)
		p := llmPoint(res, base)
		points = append(points, p)
		if i == 0 {
			saturationRPS = num(res, "requests_per_s")
		}
		if fixedRateAt > 0 && s == fixedRateAt && saturationRPS > 0 {
			// 30 % of the unrestricted capacity: below what a 50 % share can serve,
			// so a healthy limiter must not degrade it at all.
			rate := saturationRPS * 0.3
			fr := e.runLLMClient(ctx, c, fmt.Sprintf("rate-%s-%03.0f", mode, s*100), url, "model", "open", prof, rate)
			fb := base
			fb.Load = "fixed-rate"
			points = append(points, llmPoint(fr, fb))
		}
		e.delete(ctx, newObj(gvkPod, c.NS, name))
		e.waitGone(ctx, gvkPod, c.NS, name, 4*time.Minute)
		if mode != "native" {
			e.waitResourcesReturned(ctx, 3*time.Minute)
		}
	}
	return points
}

func gpuVRAMSlice(g GPU) float64 {
	// Roughly 40 % of the card, at least 4 GiB: 6 GiB on a T4, 9 GiB on A10G/L4/3090.
	v := float64(int(g.CapVRAM * 0.4))
	if v < 4 {
		v = 4
	}
	return v
}

// ---------------------------------------------------------------------------
// isolation tier
// ---------------------------------------------------------------------------

var _ = Describe("isolation", func() {
	Register(Case{
		ID: "ISO-02", Title: "soft 算力限额实测：吞吐随 share 成比例、无停顿", Tiers: []string{TierIsolation}, Needs: []string{NeedTF, NeedRealGPU}, Timeout: 40 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			env.shipWorkloads(ctx, c.NS)
			var points []PerfPoint
			for _, s := range []float64{1, 0.5, 0.25} {
				base := PerfPoint{Workload: "matmul", Model: "fp16-2048", Mode: "soft", Share: s, VRAMGiB: 4, Cotenancy: "alone", Load: "saturation"}
				points = append(points, env.runBench(ctx, c, fmt.Sprintf("burn-%03.0f", s*100), g, base, burnCommand(45, 10), 10*time.Minute)...)
			}
			failed := recordPerf(c, points)
			Expect(failed).To(BeEmpty(), "compute limit not honoured:\n%s", strings.Join(failed, "\n"))
		},
	})

	Register(Case{
		ID: "ISO-03", Title: "显存上限实测：容器内只见配额、超限分配失败、邻居不受影响", Tiers: []string{TierIsolation}, Needs: []string{NeedTF, NeedRealGPU}, Timeout: 30 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			env.shipWorkloads(ctx, c.NS)
			for _, mode := range []string{"soft", "hard"} {
				By(mode + ": a neighbour holds 2 GiB of its own 4 GiB limit")
				neighbour := "hold-" + mode
				Expect(env.create(ctx, benchPod(c.NS, neighbour, g, mode, 0.2, 4, "", "python3 /e2e/gpu_probe.py vram --target-mib 2048 --hold 600"))).To(Succeed())
				env.waitPodReady(ctx, c.NS, neighbour, 8*time.Minute)

				By(mode + ": a pod with a 4 GiB limit asks for 6 GiB")
				name := "greedy-" + mode
				Expect(env.create(ctx, benchPod(c.NS, name, g, mode, 0.2, 4, "", "python3 /e2e/gpu_probe.py vram --target-mib 6144"))).To(Succeed())
				logs, _ := env.waitDone(ctx, c.NS, name, 8*time.Minute)
				res := results(logs)
				Expect(res).NotTo(BeEmpty(), "probe produced no result: %s", lastLines(logs, 5))
				r := res[len(res)-1]
				c.Measure(mode+"_allocated_mib", num(r, "allocated_mib"), "MiB")
				c.Measure(mode+"_visible_total_mib", num(r, "visible_total_mib"), "MiB")
				Expect(r["hit_limit"]).To(Equal(true), "%s: 6 GiB were allocated under a 4 GiB limit", mode)
				Expect(num(r, "allocated_mib")).To(BeNumerically("<=", 4096+64), "%s: allocated more than the limit", mode)
				Expect(num(r, "allocated_mib")).To(BeNumerically(">=", 4096*0.85), "%s: could not use its own quota", mode)
				Expect(num(r, "visible_total_mib")).To(BeNumerically("~", 4096, 300), "%s: the container must see its quota as the card's memory", mode)

				By(mode + ": the neighbour is still running and still holds its memory")
				n, _ := env.get(ctx, gvkPod, c.NS, neighbour)
				Expect(podReady(n)).To(BeTrue(), podWhy(n))
				Expect(restarts(n)).To(Equal(0))
				for _, p := range []string{name, neighbour} {
					env.delete(ctx, newObj(gvkPod, c.NS, p))
					env.waitGone(ctx, gvkPod, c.NS, p, 3*time.Minute)
				}
				env.waitResourcesReturned(ctx, 3*time.Minute)
			}
		},
	})

	Register(Case{
		ID: "ISO-04", Title: "hard 算力限额实测（sidecar worker）", Tiers: []string{TierIsolation}, Needs: []string{NeedTF, NeedRealGPU}, Timeout: 40 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			env.shipWorkloads(ctx, c.NS)
			var points []PerfPoint
			for _, s := range []float64{1, 0.5, 0.25} {
				base := PerfPoint{Workload: "matmul", Model: "fp16-2048", Mode: "hard", Share: s, VRAMGiB: 4, Cotenancy: "alone", Load: "saturation"}
				points = append(points, env.runBench(ctx, c, fmt.Sprintf("hburn-%03.0f", s*100), g, base, burnCommand(45, 10), 12*time.Minute)...)
			}
			failed := recordPerf(c, points)
			Expect(failed).To(BeEmpty(), "hard compute limit not honoured:\n%s", strings.Join(failed, "\n"))
		},
	})

	Register(Case{
		ID: "ISO-15", Title: "邻居隔离：吵闹的 Pod 不会把安静的 Pod 压到份额以下", Tiers: []string{TierIsolation}, Needs: []string{NeedTF, NeedRealGPU}, Timeout: 45 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			env.shipWorkloads(ctx, c.NS)
			for _, mode := range []string{"soft", "hard"} {
				base := PerfPoint{Workload: "matmul", Model: "fp16-2048", Mode: mode, Share: 0.4, VRAMGiB: 3, Cotenancy: "alone", Load: "saturation"}
				alone := env.runBench(ctx, c, "quiet-alone-"+mode, g, base, burnCommand(45, 10), 12*time.Minute)
				Expect(alone[0].Error).To(BeEmpty())

				stop := env.noisyNeighbour(ctx, c, g, mode, 0.6)
				nb := base
				nb.Cotenancy = "noisy-neighbour"
				with := env.runBench(ctx, c, "quiet-noisy-"+mode, g, nb, burnCommand(45, 10), 12*time.Minute)
				stop()
				Expect(with[0].Error).To(BeEmpty())

				ratio := with[0].Throughput / alone[0].Throughput
				c.Measure(mode+"_quiet_alone_iters_per_s", alone[0].Throughput, "it/s")
				c.Measure(mode+"_quiet_with_noisy_iters_per_s", with[0].Throughput, "it/s")
				c.Measure(mode+"_quiet_retained_ratio", ratio, "")
				c.Measure(mode+"_quiet_stalls_ge_400ms", float64(with[0].Gaps400), "")
				// The quiet pod keeps its 40 %: at most 20 % below what it gets alone
				// at the same limit, and no limiter stalls.
				Expect(ratio).To(BeNumerically(">=", 0.8), "%s: a neighbour using its own 60%% pushed the quiet pod to %.0f%% of its share", mode, ratio*100)
				Expect(with[0].Gaps900).To(Equal(0), "%s: quiet pod stalled >= 0.9 s", mode)
			}
		},
	})

	Register(Case{
		ID: "ISO-16", Title: "运行期调整限额生效（cron 伸缩规则，即 ASC-02）", Tiers: []string{TierIsolation}, Needs: []string{NeedTF, NeedRealGPU}, Timeout: 30 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			env.shipWorkloads(ctx, c.NS)
			// The only supported in-place limit change is the autoscaler. A cron rule
			// makes it deterministic: full share now, 30 % inside the window.
			now := time.Now().UTC()
			start, end := now.Add(3*time.Minute), now.Add(7*time.Minute)
			low := fmt.Sprintf("%.0f", g.CapTflops*0.3)
			full := fmt.Sprintf("%.0f", g.CapTflops)
			profile := parseYAML(fmt.Sprintf(`
apiVersion: tensor-fusion.ai/v1
kind: WorkloadProfile
metadata: {name: cron, namespace: %s}
spec:
  poolName: %s
  isLocalGPU: true
  isolation: soft
  resources:
    requests: {tflops: "%s", vram: 4Gi}
    limits: {tflops: "%s", vram: 4Gi}
  autoScalingConfig:
    cronScalingRules:
      - enable: true
        name: shrink
        start: "%d %d * * *"
        end: "%d %d * * *"
        desiredResources:
          requests: {tflops: "%s", vram: 4Gi}
          limits: {tflops: "%s", vram: 4Gi}
`, c.NS, env.Pool, low, full, start.Minute(), start.Hour(), end.Minute(), end.Hour(), low, low))
			Expect(env.create(ctx, profile...)).To(Succeed())
			pod := tfPod(c.NS, "cron", ann("workload-profile", "cron", "autoscale", "true"), withShell(env.Images.Torch, burnCommand(540, 5)), withScripts(), tfNodeSelector())
			Expect(env.create(ctx, pod)).To(Succeed())
			logs, _ := env.waitDone(ctx, c.NS, "cron", 14*time.Minute)
			res := results(logs)
			Expect(res).NotTo(BeEmpty(), lastLines(logs, 5))
			var series []float64
			if raw, ok := res[0]["series_iters_per_s"].([]any); ok {
				for _, v := range raw {
					f, _ := v.(float64)
					series = append(series, f)
				}
			}
			Expect(len(series)).To(BeNumerically(">", 400), "burn series too short")
			mean := func(a, b int) float64 {
				s := 0.0
				for _, v := range series[a:b] {
					s += v
				}
				return s / float64(b-a)
			}
			before, during, after := mean(30, 150), mean(240, 380), mean(len(series)-60, len(series))
			c.Measure("iters_per_s_before", before, "it/s")
			c.Measure("iters_per_s_in_window", during, "it/s")
			c.Measure("iters_per_s_after", after, "it/s")
			c.Measure("window_ratio", during/before, "")
			c.Attach("series", series)
			p, _ := env.get(ctx, gvkPod, c.NS, "cron")
			c.Note("pod annotations after the run: tflops-limit=%s", p.GetAnnotations()[tfDomain+"/tflops-limit"])
			Expect(during/before).To(BeNumerically("<", 0.3*1.25+0.05), "limit change did not take effect inside the cron window")
			Expect(during/before).To(BeNumerically(">", 0.3*0.75), "limit change overshot")
			Expect(after/before).To(BeNumerically(">", 0.8), "limit was not restored after the window")
		},
	})

	Register(Case{
		ID: "ISO-13", Title: "disable-features: gpu-limiter 关闭限流后不再受限", Tiers: []string{TierIsolation}, Needs: []string{NeedTF, NeedRealGPU}, Timeout: 30 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			env.shipWorkloads(ctx, c.NS)
			base := PerfPoint{Workload: "matmul", Model: "fp16-2048", Mode: "soft", Share: 1, VRAMGiB: 4, Cotenancy: "alone", Load: "saturation"}
			full := env.runBench(ctx, c, "full", g, base, burnCommand(40, 10), 10*time.Minute)
			a := merge(shareAnnotations(g, "soft", 0.3, 4), ann("disable-features", "gpu-limiter"))
			Expect(env.create(ctx, tfPod(c.NS, "nolimit", a, withShell(env.Images.Torch, burnCommand(40, 10)), withScripts(), tfNodeSelector()))).To(Succeed())
			logs, _ := env.waitDone(ctx, c.NS, "nolimit", 10*time.Minute)
			res := results(logs)
			Expect(res).NotTo(BeEmpty(), lastLines(logs, 5))
			ratio := num(res[0], "iters_per_s") / full[0].Throughput
			c.Measure("unlimited_ratio_with_30pct_limit", ratio, "")
			Expect(ratio).To(BeNumerically(">", 0.85), "with the limiter disabled a 30%% limit must not throttle")
		},
	})
})

// ---------------------------------------------------------------------------
// observability with a workload
// ---------------------------------------------------------------------------

var _ = Describe("observability", func() {
	Register(Case{
		ID: "OBS-02", Title: "GPU / worker / 节点 / pool 指标在限定时间内写入 GreptimeDB", Tiers: []string{TierCore, TierRehearsal},
		Needs: []string{NeedTF, NeedGPU}, Timeout: 25 * time.Minute,
		KnownIssue: "released images (<= 2.17.23): hypervisor metrics never reach GreptimeDB in a default install (metrics file written outside the vector-tailed directory), so tf_gpu_usage / tf_worker_usage stay empty; the tables feed the usage alerts, the autoscaler and the cloud console charts. Needs the fixed operator / hypervisor images and chart",
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			Expect(env.create(ctx, tfPod(c.NS, "metric", fraction(gpus[0], "soft", 20, 2), tfNodeSelector()))).To(Succeed())
			env.waitPodReady(ctx, c.NS, "metric", 8*time.Minute)
			tables := []string{"tf_gpu_usage", "tf_worker_usage", "tf_node_metrics", "tf_pool_metrics", "tf_worker_resources"}
			start := time.Now()
			counts := map[string]int{}
			arrived := map[string]float64{}
			// Deliberately not Fail-on-timeout inside the loop: report every table.
			deadline := time.Now().Add(6 * time.Minute)
			for time.Now().Before(deadline) && len(arrived) < len(tables) {
				for _, t := range tables {
					if _, ok := arrived[t]; ok {
						continue
					}
					n, err := env.tableRows(ctx, t, 15*time.Minute)
					if err == nil && n > 0 {
						counts[t] = n
						arrived[t] = time.Since(start).Seconds()
					}
				}
				time.Sleep(15 * time.Second)
			}
			var missing []string
			for _, t := range tables {
				if s, ok := arrived[t]; ok {
					c.Measure(t+"_first_row_seconds", s, "s")
					c.Measure(t+"_rows", float64(counts[t]), "")
				} else {
					missing = append(missing, t)
				}
			}
			Expect(missing).To(BeEmpty(), "no rows within 6 min of a running workload in: %v", missing)
		},
	})
})

// ---------------------------------------------------------------------------
// perf tier
// ---------------------------------------------------------------------------

var _ = Describe("perf", func() {
	Register(Case{
		ID: "PERF-LLM-01", Title: "vLLM 流式推理 vs 算力份额（soft）：吞吐成比例、TTFT / TPOT 尾延迟无断崖", Tiers: []string{TierPerf, "perf-quick"},
		Needs: []string{NeedTF, NeedRealGPU}, Timeout: 90 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			env.shipWorkloads(ctx, c.NS)
			prof := profile()
			vram := gpuVRAMSlice(g)
			var points []PerfPoint
			points = append(points, env.llmSweep(ctx, c, g, "native", []float64{1}, vram, prof, "", 0)...)
			points = append(points, env.llmSweep(ctx, c, g, "soft", prof.Shares, vram, prof, "", 0.5)...)
			failed := recordPerf(c, points)
			c.Note("model %s, VRAM limit %.0f GiB, profile %s (%0.fs warm-up + %.0fs window, %d streams)", llmModel(vram), vram, prof.Name, prof.Warmup, prof.Window, prof.Concurrency)
			Expect(failed).To(BeEmpty(), "criteria failed:\n%s", strings.Join(failed, "\n"))
		},
	})

	Register(Case{
		ID: "PERF-LLM-02", Title: "vLLM 流式推理 vs 算力份额（hard）", Tiers: []string{TierPerf, "perf-quick"},
		Needs: []string{NeedTF, NeedRealGPU}, Timeout: 90 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			env.shipWorkloads(ctx, c.NS)
			prof := profile()
			if prof.Name == "quick" {
				prof.Shares = []float64{1, 0.5}
			}
			vram := gpuVRAMSlice(g)
			var points []PerfPoint
			points = append(points, env.llmSweep(ctx, c, g, "native", []float64{1}, vram, prof, "", 0)...)
			points = append(points, env.llmSweep(ctx, c, g, "hard", prof.Shares, vram, prof, "", 0.5)...)
			failed := recordPerf(c, points)
			Expect(failed).To(BeEmpty(), "criteria failed:\n%s", strings.Join(failed, "\n"))
		},
	})

	Register(Case{
		ID: "PERF-LLM-03", Title: "vLLM 变体：VRAM 限额松 / 紧、eager（无 CUDA graph）、整卡 shared", Tiers: []string{TierPerf},
		Needs: []string{NeedTF, NeedRealGPU}, Timeout: 90 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			env.shipWorkloads(ctx, c.NS)
			prof := profile()
			var points []PerfPoint
			tight, loose := 4.0, float64(int(g.CapVRAM*0.8))
			points = append(points, env.llmSweep(ctx, c, g, "shared", []float64{1}, 0, prof, "", 0)...)
			for _, v := range []float64{tight, loose} {
				points = append(points, env.llmSweep(ctx, c, g, "soft", []float64{1, 0.5}, v, prof, "", 0)...)
			}
			eager := env.llmSweep(ctx, c, g, "soft", []float64{1, 0.5}, tight, prof, "--enforce-eager", 0)
			for i := range eager {
				eager[i].Model += " (eager)"
			}
			points = append(points, eager...)
			failed := recordPerf(c, points)
			Expect(failed).To(BeEmpty(), "criteria failed:\n%s", strings.Join(failed, "\n"))
		},
	})

	benchCase := func(id, title string, quick bool, build func(prof perfProfile, vram float64) (workload string, command string, timeout time.Duration)) {
		tiers := []string{TierPerf}
		if quick {
			tiers = append(tiers, "perf-quick")
		}
		Register(Case{
			ID: id, Title: title, Tiers: tiers, Needs: []string{NeedTF, NeedRealGPU}, Timeout: 120 * time.Minute,
			Run: func(ctx SpecContext, c *C) {
				gpus, _ := env.ensureGPUs(ctx, c, 1)
				g := gpus[0]
				env.shipWorkloads(ctx, c.NS)
				prof := profile()
				vram := gpuVRAMSlice(g)
				workload, command, timeout := build(prof, vram)
				type cfg struct {
					mode  string
					share float64
					noisy bool
				}
				cfgs := []cfg{{"native", 1, false}, {"soft", 1, false}, {"soft", 0.5, false}}
				if prof.Name == "full" {
					cfgs = append(cfgs, cfg{"soft", 0.25, false}, cfg{"shared", 1, false}, cfg{"hard", 1, false}, cfg{"hard", 0.5, false}, cfg{"soft", 0.5, true})
				}
				var points []PerfPoint
				for i, k := range cfgs {
					base := PerfPoint{Workload: workload, Mode: k.mode, Share: k.share, VRAMGiB: vram, Cotenancy: "alone", Load: "saturation"}
					if k.mode == "native" || k.mode == "shared" {
						base.VRAMGiB = 0
					}
					stop := func() {}
					if k.noisy {
						base.Cotenancy = "noisy-neighbour"
						stop = env.noisyNeighbour(ctx, c, g, k.mode, 1-k.share)
					}
					points = append(points, env.runBench(ctx, c, fmt.Sprintf("%s-%d", strings.ToLower(id), i), g, base, command, timeout)...)
					stop()
				}
				failed := recordPerf(c, points)
				Expect(failed).To(BeEmpty(), "criteria failed:\n%s", strings.Join(failed, "\n"))
			},
		})
	}
	benchCase("PERF-ORT-01", "ONNX Runtime (CUDA EP)：CNN / 编码器 / 语音编码器的 QPS 与延迟分位 vs 份额", true,
		func(prof perfProfile, vram float64) (string, string, time.Duration) {
			models := "resnet50,minilm"
			if prof.Name == "full" {
				models = "resnet50,mobilenet_v2,bert_base,minilm,whisper_enc"
			}
			return "onnxruntime", ortCommand(models, prof.Warmup, prof.Window*0.75, vram), 40 * time.Minute
		})
	benchCase("PERF-TB-01", "TorchBench 代表性子集（conv / transformer / 小 kernel 密集）vs 份额", true,
		func(prof perfProfile, vram float64) (string, string, time.Duration) {
			models, tests := "resnet50,bert_base,tiny_mlp", "eval"
			if prof.Name == "full" {
				models, tests = "resnet50,mobilenet_v2,bert_base,gpt2,tiny_mlp", "eval,train"
			}
			return "torchbench-mini", torchCommand(models, tests, 5, prof.Window*0.5, vram), 60 * time.Minute
		})
	benchCase("PERF-MLP-01", "MLPerf Inference 风格（LoadGen Offline / Server，缩减时长，非合规提交）vs 份额", true,
		func(prof perfProfile, vram float64) (string, string, time.Duration) {
			models, scenarios := "resnet50", "Offline"
			if prof.Name == "full" {
				models, scenarios = "resnet50,bert", "Offline,Server"
			}
			return "mlperf", mlperfCommand(models, scenarios, prof.Window*0.5), 60 * time.Minute
		})

	Register(Case{
		ID: "PERF-PW-01", Title: "成对正交设计：模式 × 份额 × 负载 × 邻居 × VRAM 的主效应", Tiers: []string{TierPerf},
		Needs: []string{NeedTF, NeedRealGPU}, Timeout: 240 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			gpus, _ := env.ensureGPUs(ctx, c, 1)
			g := gpus[0]
			env.shipWorkloads(ctx, c.NS)
			runs := perfDesign()
			c.Measure("design_runs", float64(len(runs)), "")
			c.Attach("design", runs)
			prof := profile()
			tight, loose := 4.0, float64(int(g.CapVRAM*0.8))
			var points []PerfPoint
			for i, r := range runs {
				share := map[string]float64{"30": 0.3, "60": 0.6, "100": 1}[r["share"]]
				vram := tight
				if r["vram"] == "loose" {
					vram = loose
				}
				kind, model, _ := strings.Cut(r["workload"], ":")
				var command, workload string
				switch kind {
				case "ort":
					workload, command = "onnxruntime", ortCommand(model, 10, 40, vram)
				case "tb":
					workload, command = "torchbench-mini", torchCommand(model, "eval", 5, 30, vram)
				default:
					workload, command = "matmul", burnCommand(40, 10)
				}
				base := PerfPoint{Workload: workload, Mode: r["mode"], Share: share, VRAMGiB: vram, Cotenancy: "alone", Load: "saturation"}
				stop := func() {}
				if r["cotenancy"] == "noisy" && share < 1 {
					base.Cotenancy = "noisy-neighbour"
					stop = env.noisyNeighbour(ctx, c, g, r["mode"], 1-share)
				}
				points = append(points, env.runBench(ctx, c, fmt.Sprintf("pw-%02d", i), g, base, command, 20*time.Minute)...)
				stop()
			}
			_ = prof
			failed := recordPerf(c, points)
			Expect(failed).To(BeEmpty(), "criteria failed:\n%s", strings.Join(failed, "\n"))
		},
	})
})

// perfDesign is the pairwise design of PERF-PW-01. 3 x 3 x 5 x 2 x 2 = 180
// combinations; all-pairs needs about 17 runs.
func perfDesign() []Run {
	factors := []Factor{
		{"workload", []string{"ort:resnet50", "ort:minilm", "tb:tiny_mlp", "tb:gpt2", "burn:matmul"}},
		{"mode", []string{"soft", "hard", "shared"}},
		{"share", []string{"100", "60", "30"}},
		{"cotenancy", []string{"alone", "noisy"}},
		{"vram", []string{"tight", "loose"}},
	}
	return pairwise(factors, func(r Run) bool {
		if r["mode"] == "shared" {
			// A whole-GPU pod has no share, no neighbour and no VRAM limit.
			return (r["share"] == "" || r["share"] == "100") && (r["cotenancy"] == "" || r["cotenancy"] == "alone") && (r["vram"] == "" || r["vram"] == "loose")
		}
		if r["share"] == "100" && r["cotenancy"] == "noisy" {
			return false // nothing left for a neighbour
		}
		return true
	})
}
