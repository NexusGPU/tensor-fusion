//go:build e2e

package suite

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// Tier dryrun: everything that truly needs a GPU, proven as far as it can be
// without one. A pass here says "the only thing missing is the device", never
// "the feature works": every case is recorded with mode dry-run.

// standin starts the CPU stand-in LLM server and returns its URL.
func (e *Env) standin(ctx context.Context, c *C, name, limiter string, extra string) string {
	GinkgoHelper()
	e.shipWorkloads(ctx, c.NS)
	cmd := fmt.Sprintf("python3 /e2e/llm_standin.py --port 8000 --limiter %s --tokens-per-s 400 %s", limiter, extra)
	Expect(e.create(ctx, plainPod(c.NS, name, withShell(e.Images.Curl, cmd), withScripts(), withPort(8000), loadNodeSelector("system")))).To(Succeed())
	e.waitPodReady(ctx, c.NS, name, 4*time.Minute)
	waitFor(ctx, "stand-in server answering", 2*time.Minute, func() (string, bool) {
		_, err := e.CS.CoreV1().Pods(c.NS).ProxyGet("http", name, "8000", "health", nil).DoRaw(ctx)
		if err != nil {
			return err.Error(), false
		}
		return "", true
	})
	return "http://" + e.podIP(ctx, c.NS, name) + ":8000"
}

// setStandinShare changes the stand-in's share in place, the way a runtime
// limit change reaches a running server.
func (e *Env) setStandinShare(ctx context.Context, c *C, pod string, share float64) {
	GinkgoHelper()
	_, err := e.exec(ctx, c.NS, pod, "main", "python3", "-c",
		fmt.Sprintf("import urllib.request as u; r=u.Request('http://127.0.0.1:8000/share?value=%.0f', method='PUT'); print(u.urlopen(r).read())", share*100))
	Expect(err).NotTo(HaveOccurred())
}

var _ = Describe("dry run", func() {
	Register(Case{
		ID: "DRY-PERF-01", Title: "LLM measurement and judging pipeline (CPU stand-in): a proportional result passes; stalls / bimodality / no throttling are detected",
		Tiers: []string{TierDryRun}, Mode: ModeDryRun, Timeout: 25 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			prof := perfProfile{Name: "dry", Warmup: 4, Window: 16, Concurrency: 8, MaxTokens: 48}
			var points []PerfPoint
			measure := func(server, url, mode string, share float64) {
				env.setStandinShare(ctx, c, server, share)
				name := fmt.Sprintf("load-%s-%03.0f", strings.TrimPrefix(mode, "standin-"), share*100)
				res := env.runLLMClient(ctx, c, name, url, "standin", "closed", prof, 0)
				points = append(points, llmPoint(res, PerfPoint{
					Workload: "llm-standin", Model: "standin", Mode: mode, Share: share, Cotenancy: "alone", Load: "saturation", ExecMode: "cpu-standin",
				}))
			}

			// One server per behaviour; the share is changed in place between runs.
			smooth := env.standin(ctx, c, "srv-smooth", "smooth", "")
			for _, s := range []float64{1, 0.5, 0.25} {
				measure("srv-smooth", smooth, "standin-smooth", s)
			}
			bucket := env.standin(ctx, c, "srv-bucket", "bucket", "")
			for _, s := range []float64{1, 0.25} {
				measure("srv-bucket", bucket, "standin-bucket", s)
			}
			stall := env.standin(ctx, c, "srv-stall", "smooth", "--stall-every 150 --stall-ms 1000")
			measure("srv-stall", stall, "standin-stall", 1)
			none := env.standin(ctx, c, "srv-none", "none", "")
			for _, s := range []float64{1, 0.25} {
				measure("srv-none", none, "standin-none", s)
			}

			recordPerf(c, points)
			verdict := func(mode string, share float64, criterion string) (bool, bool) {
				for _, p := range points {
					if p.Mode == mode && p.Share == share {
						for _, v := range p.Verdicts {
							if v.Criterion == criterion {
								return v.Pass, true
							}
						}
					}
				}
				return false, false
			}
			expect := func(mode string, share float64, criterion string, want bool) {
				got, found := verdict(mode, share, criterion)
				Expect(found).To(BeTrue(), "criterion %s was not evaluated for %s @ %.0f%%", criterion, mode, share*100)
				Expect(got).To(Equal(want), "%s @ %.0f%%: criterion %s", mode, share*100, criterion)
			}
			By("a limiter that scales smoothly passes effectiveness and no-cliff at 50% and 25%")
			for _, s := range []float64{0.5, 0.25} {
				for _, p := range points {
					if p.Mode == "standin-smooth" && p.Share == s {
						Expect(p.Pass).NotTo(BeNil())
						Expect(*p.Pass).To(BeTrue(), "smooth @ %.0f%% should pass: %v", s*100, p.Verdicts)
					}
				}
			}
			By("a limiter that grants work in 500 ms steps keeps the average but is flagged as a cliff")
			expect("standin-bucket", 0.25, "throughput-proportional", true)
			expect("standin-bucket", 0.25, "itl-not-bimodal", false)
			expect("standin-bucket", 0.25, "itl-p99-no-cliff", false)
			By("1 s stalls are flagged by themselves, without any baseline")
			expect("standin-stall", 1, "no-stall-near-1s", false)
			By("a limiter that does not limit is flagged as ineffective")
			expect("standin-none", 0.25, "throughput-proportional", false)
			expect("standin-none", 0.25, "limit-is-effective", false)
			c.Note("evaluation pipeline verified against 4 synthetic limiter behaviours, %d measurement points", len(points))
		},
	})

	Register(Case{
		ID: "DRY-PERF-02", Title: "ONNX Runtime / TorchBench-mini / MLPerf-LoadGen scripts run end to end on CPU (export, session, measurement, result parsing)",
		Tiers: []string{TierDryRun}, Mode: ModeDryRun, Timeout: 30 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			env.shipWorkloads(ctx, c.NS)
			script := strings.Join([]string{
				"set -e",
				"pip install -q --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu",
				"pip install -q --no-cache-dir onnx onnxruntime transformers mlcommons-loadgen",
				"cd /e2e",
				"python3 ort_bench.py --provider cpu --models tiny_mlp,minilm --warmup 2 --duration 8 --label dry",
				"python3 torch_mini.py --device cpu --models tiny_mlp,mobilenet_v2 --tests eval,train --warmup 2 --duration 6 --label dry",
				"python3 mlperf_mini.py --provider cpu --models tiny_mlp --scenarios Offline,Server --duration 6 --label dry",
			}, " && ")
			pod := plainPod(c.NS, "bench", withShell(env.Images.Curl, script), withScripts(), loadNodeSelector("cpu"),
				withResources(map[string]string{"cpu": "1", "memory": "5Gi"}))
			Expect(env.create(ctx, pod)).To(Succeed())
			logs, ok := env.waitDone(ctx, c.NS, "bench", 22*time.Minute)
			Expect(ok).To(BeTrue(), "bench pod failed: %s", lastLines(logs, 8))
			var points []PerfPoint
			for _, r := range results(logs) {
				w := text(r, "runtime")
				points = append(points, benchPoint(r, PerfPoint{Workload: w, Mode: "native", Share: 1, Cotenancy: "alone", Load: "saturation", ExecMode: "cpu-standin"}))
			}
			recordPerf(c, points)
			Expect(points).To(HaveLen(2+4+2), "expected one result per model x test; got %d: %s", len(points), lastLines(logs, 12))
			for _, p := range points {
				Expect(p.Error).To(BeEmpty(), "%s %s %s", p.Workload, p.Model, p.Test)
				Expect(p.Throughput).To(BeNumerically(">", 0), "%s %s %s", p.Workload, p.Model, p.Test)
			}
		},
	})

	Register(Case{
		ID: "DRY-ISO-01", Title: "All manifests of the GPU tiers pass server-side dry-run (admission, CRD validation, image fields)",
		Tiers: []string{TierDryRun}, Mode: ModeDryRun, Needs: []string{NeedTF}, Timeout: 10 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			env.shipWorkloads(ctx, c.NS)
			// The dry run goes through the webhook against the pool that will be used
			// for real, so this proves the annotations are accepted by this operator.
			g := GPU{CapTflops: 65, CapVRAM: 15} // T4
			saved := env.Pool
			env.Pool = env.RealPool
			defer func() { env.Pool = saved }()
			manifests := map[string]*unstructured.Unstructured{
				"iso-soft-burn":   tfPod(c.NS, "iso-soft", shareAnnotations(g, "soft", 0.5, 4), withShell(env.Images.Torch, "python3 /e2e/gpu_probe.py burn --seconds 30"), withScripts()),
				"iso-hard-burn":   tfPod(c.NS, "iso-hard", shareAnnotations(g, "hard", 0.5, 4), withShell(env.Images.Torch, "python3 /e2e/gpu_probe.py burn --seconds 30"), withScripts()),
				"iso-shared":      tfPod(c.NS, "iso-shared", wholeGPU(), withShell(env.Images.Torch, "python3 /e2e/gpu_probe.py info"), withScripts()),
				"iso-vram":        tfPod(c.NS, "iso-vram", shareAnnotations(g, "soft", 0.3, 4), withShell(env.Images.Torch, "python3 /e2e/gpu_probe.py vram --target-mib 6144"), withScripts()),
				"perf-vllm-soft":  vllmServer(c.NS, "vllm-soft", g, "soft", 0.5, 6, ""),
				"perf-vllm-hard":  vllmServer(c.NS, "vllm-hard", g, "hard", 0.5, 6, ""),
				"perf-vllm-eager": vllmServer(c.NS, "vllm-eager", g, "soft", 1, 6, "--enforce-eager"),
				"perf-ort":        benchPod(c.NS, "ort", g, "soft", 0.5, 6, "", ortCommand("resnet50,minilm", 10, 45, 6)),
				"perf-torch":      benchPod(c.NS, "tb", g, "soft", 0.5, 6, "", torchCommand("resnet50,bert_base,tiny_mlp", "eval", 5, 30, 6)),
				"perf-mlperf":     benchPod(c.NS, "mlp", g, "soft", 0.5, 6, "", mlperfCommand("resnet50", "Offline,Server", 30)),
				"perf-native":     benchPod(c.NS, "native", g, "native", 1, 0, "ip-0-0-0-0", ortCommand("resnet50", 10, 45, 0)),
			}
			for name, m := range manifests {
				err := env.dryRun(ctx, m)
				Expect(err).NotTo(HaveOccurred(), "manifest %s rejected", name)
			}
			c.Measure("manifests_accepted", float64(len(manifests)), "")
			wl, _ := env.list(ctx, gvkWorkload, c.NS)
			c.Note("dry-run left %d TensorFusionWorkload object(s) behind (see WHK-08); removed with the namespace", len(wl))
		},
	})

	Register(Case{
		ID: "DRY-IMG-01", Title: "Images and models used by the GPU tiers resolve (manifest / HEAD only, nothing is pulled)",
		Tiers: []string{TierDryRun}, Mode: ModeDryRun, NoNS: true, Timeout: 10 * time.Minute,
		Run: func(ctx SpecContext, c *C) {
			images := []string{env.Images.Torch, env.Images.VLLM, env.Images.Curl, env.Images.Busybox}
			if env.OperatorImage != "" {
				images = append(images, env.OperatorImage)
			}
			if p, err := env.get(ctx, gvkGPUPool, "", env.RealPool); err == nil {
				for _, path := range [][]string{{"hypervisor", "image"}, {"worker", "image"}, {"client", "image"}} {
					if img := str(p, append([]string{"spec", "componentConfig"}, path...)...); img != "" {
						images = append(images, img)
					}
				}
			}
			if pcs, _ := env.list(ctx, tfGVK("ProviderConfig"), ""); len(pcs) > 0 {
				for i := range pcs {
					if str(&pcs[i], "spec", "vendor") == "NVIDIA" {
						images = append(images, str(&pcs[i], "spec", "images", "middleware"))
					}
				}
			}
			seen := map[string]bool{}
			var rows []map[string]any
			for _, img := range images {
				if img == "" || seen[img] {
					continue
				}
				seen[img] = true
				digest, sizeMiB, err := resolveImage(ctx, img)
				Expect(err).NotTo(HaveOccurred(), "image %s does not resolve", img)
				rows = append(rows, map[string]any{"image": img, "digest": digest, "compressed_mib": sizeMiB})
				c.Note("%s -> %s, %.0f MiB compressed (linux/amd64)", img, shortDigest(digest), sizeMiB)
			}
			c.Attach("images", rows)

			for _, model := range []string{llmModel(4), llmModel(16)} {
				req, _ := http.NewRequestWithContext(ctx, http.MethodHead, "https://huggingface.co/"+model+"/resolve/main/config.json", nil)
				resp, err := httpDo(req)
				Expect(err).NotTo(HaveOccurred(), "Hugging Face unreachable for %s", model)
				resp.Body.Close()
				Expect(resp.StatusCode).To(BeNumerically("<", 400), "model %s is not publicly downloadable (HTTP %d): gated or renamed", model, resp.StatusCode)
				c.Note("model %s: config.json reachable without a token (HTTP %d)", model, resp.StatusCode)
			}
		},
	})
})

func shortDigest(d string) string {
	if len(d) > 19 {
		return d[:19]
	}
	return d
}

// httpDo retries transient transport errors (proxies and registries drop the
// odd connection); HTTP status codes are returned to the caller as they are.
func httpDo(req *http.Request) (*http.Response, error) {
	var err error
	for attempt := 0; attempt < 4; attempt++ {
		var resp *http.Response
		if resp, err = http.DefaultClient.Do(req.Clone(req.Context())); err == nil {
			return resp, nil
		}
		time.Sleep(time.Duration(attempt+1) * 2 * time.Second)
	}
	return nil, err
}

var wwwAuthRe = regexp.MustCompile(`(\w+)="([^"]*)"`)

// resolveImage asks the registry for the image manifest: proves the tag exists
// and returns the digest plus the compressed size of the linux/amd64 image.
func resolveImage(ctx context.Context, ref string) (digest string, sizeMiB float64, err error) {
	host, repo, tag := "registry-1.docker.io", ref, "latest"
	if i := strings.Index(ref, "@"); i > 0 {
		ref = ref[:i]
		repo = ref
	}
	if first, rest, ok := strings.Cut(ref, "/"); ok && (strings.Contains(first, ".") || strings.Contains(first, ":")) {
		host, repo = first, rest
		if host == "docker.io" {
			host = "registry-1.docker.io"
		}
	}
	if i := strings.LastIndex(repo, ":"); i > 0 {
		repo, tag = repo[:i], repo[i+1:]
	}
	if host == "registry-1.docker.io" && !strings.Contains(repo, "/") {
		repo = "library/" + repo
	}
	token := ""
	fetch := func(reference string) (*http.Response, []byte, error) {
		for attempt := 0; attempt < 6; attempt++ {
			req, _ := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("https://%s/v2/%s/manifests/%s", host, repo, reference), nil)
			req.Header.Set("Accept", "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json")
			if token != "" {
				req.Header.Set("Authorization", "Bearer "+token)
			}
			resp, err := httpDo(req)
			if err != nil {
				return nil, nil, err
			}
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
			resp.Body.Close()
			if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
				// Anonymous pulls from public registries are rate limited per second.
				time.Sleep(time.Duration(attempt+1) * 3 * time.Second)
				continue
			}
			if resp.StatusCode == http.StatusUnauthorized && token == "" {
				params := map[string]string{}
				for _, m := range wwwAuthRe.FindAllStringSubmatch(resp.Header.Get("Www-Authenticate"), -1) {
					params[m[1]] = m[2]
				}
				if params["realm"] == "" {
					return resp, body, nil
				}
				scope := params["scope"]
				if scope == "" {
					scope = "repository:" + repo + ":pull"
				}
				treq, _ := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s?service=%s&scope=%s", params["realm"], params["service"], scope), nil)
				tresp, err := httpDo(treq)
				if err != nil {
					return nil, nil, err
				}
				var t struct {
					Token       string `json:"token"`
					AccessToken string `json:"access_token"`
				}
				_ = json.NewDecoder(tresp.Body).Decode(&t)
				tresp.Body.Close()
				token = t.Token
				if token == "" {
					token = t.AccessToken
				}
				continue
			}
			return resp, body, nil
		}
		return nil, nil, fmt.Errorf("registry kept answering 401 / 429 / 5xx for %s", ref)
	}
	resp, body, err := fetch(tag)
	if err != nil {
		return "", 0, err
	}
	if resp.StatusCode != http.StatusOK {
		return "", 0, fmt.Errorf("%s: registry answered HTTP %d", ref, resp.StatusCode)
	}
	digest = resp.Header.Get("Docker-Content-Digest")
	if digest == "" {
		digest = fmt.Sprintf("sha256:%x", sha256.Sum256(body))
	}
	var m struct {
		Manifests []struct {
			Digest   string `json:"digest"`
			Platform struct {
				Architecture string `json:"architecture"`
				OS           string `json:"os"`
			} `json:"platform"`
		} `json:"manifests"`
		Layers []struct {
			Size float64 `json:"size"`
		} `json:"layers"`
	}
	_ = json.Unmarshal(body, &m)
	if len(m.Manifests) > 0 {
		found := false
		for _, e := range m.Manifests {
			if e.Platform.Architecture == "amd64" && e.Platform.OS == "linux" {
				found = true
				_, b2, err := fetch(e.Digest)
				if err != nil {
					return digest, 0, err
				}
				m.Layers = nil
				_ = json.Unmarshal(b2, &m)
				break
			}
		}
		if !found {
			return digest, 0, fmt.Errorf("%s has no linux/amd64 manifest", ref)
		}
	}
	for _, l := range m.Layers {
		sizeMiB += l.Size / (1 << 20)
	}
	return digest, sizeMiB, nil
}
