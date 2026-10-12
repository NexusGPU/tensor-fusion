//go:build e2e

package suite

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// The suite talks to the cluster with unstructured objects only. It does not
// import the operator's Go types on purpose: the same test binary must be able
// to judge any released operator version (and a bare-metal cluster) without
// being recompiled against it, and it must not break while api/ is being edited.

const (
	tfDomain     = "tensor-fusion.ai"
	fieldManager = "tf-e2e"

	// Poll interval for Kubernetes state. In-process client, no fork per poll.
	poll = 5 * time.Second
)

// Execution modes. Every case result carries one, so a rehearsal pass on mock
// devices can never be mistaken for hardware validation.
const (
	ModeRealGPU = "real-gpu"
	ModeMockGPU = "mock-gpu"
	ModeNoGPU   = "no-gpu"
	ModeDryRun  = "dry-run"
)

var (
	gvkPod        = schema.GroupVersionKind{Version: "v1", Kind: "Pod"}
	gvkNode       = schema.GroupVersionKind{Version: "v1", Kind: "Node"}
	gvkNamespace  = schema.GroupVersionKind{Version: "v1", Kind: "Namespace"}
	gvkEvent      = schema.GroupVersionKind{Version: "v1", Kind: "Event"}
	gvkDeployment = schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}
	gvkCRD        = schema.GroupVersionKind{Group: "apiextensions.k8s.io", Version: "v1", Kind: "CustomResourceDefinition"}

	gvkTFCluster  = tfGVK("TensorFusionCluster")
	gvkGPUPool    = tfGVK("GPUPool")
	gvkGPUNode    = tfGVK("GPUNode")
	gvkGPU        = tfGVK("GPU")
	gvkWorkload   = tfGVK("TensorFusionWorkload")
	gvkConnection = tfGVK("TensorFusionConnection")
	gvkProfile    = tfGVK("WorkloadProfile")
	gvkQuota      = tfGVK("GPUResourceQuota")
	gvkNodeClaimT = tfGVK("GPUNodeClaim")
	gvkSchedTmpl  = tfGVK("SchedulingConfigTemplate")

	gvkNodeClaim   = schema.GroupVersionKind{Group: "karpenter.sh", Version: "v1", Kind: "NodeClaim"}
	gvkNodePool    = schema.GroupVersionKind{Group: "karpenter.sh", Version: "v1", Kind: "NodePool"}
	gvkNodeOverlay = schema.GroupVersionKind{Group: "karpenter.sh", Version: "v1alpha1", Kind: "NodeOverlay"}
	gvkEC2NC       = schema.GroupVersionKind{Group: "karpenter.k8s.aws", Version: "v1", Kind: "EC2NodeClass"}
)

func gvk(group, version, kind string) schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: group, Version: version, Kind: kind}
}

func tfGVK(kind string) schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: tfDomain, Version: "v1", Kind: kind}
}

// Env is everything the suite learned about the cluster under test.
type Env struct {
	C   client.Client
	CS  kubernetes.Interface
	Cfg *rest.Config

	RunID      string
	ResultsDir string
	RepoRoot   string

	TFNamespace string // namespace of the TensorFusion release
	TFRelease   string

	// Pool that GPU-consuming cases target, and the mode that pool implies.
	Pool string
	Mode string

	RealPool string // GPUPool with real NVIDIA devices (chart default)
	MockPool string // GPUPool backed by the mock accelerator (rehearsal)

	HasKarpenter bool
	HasAWS       bool // EC2NodeClass CRD present: Karpenter AWS provider
	GPUNodePool  string
	CPUNodePool  string
	MockNodePool string
	RoleLabel    string

	Images Images
	Prices Prices

	OperatorImage   string
	HypervisorImage string
	K8sVersion      string
}

// Images used by test workloads. Everything is overridable so the suite also
// runs behind a registry mirror.
type Images struct {
	Busybox string // small base image with a shell, glibc and libstdc++; mutation / scheduling cases
	Torch   string // CUDA runtime + PyTorch; isolation cases
	VLLM    string
	ORT     string
	Curl    string
}

var env *Env

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func newEnv() *Env {
	rules := clientcmd.NewDefaultClientConfigLoadingRules()
	cfg, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(rules, &clientcmd.ConfigOverrides{}).ClientConfig()
	Expect(err).NotTo(HaveOccurred(), "no usable kubeconfig (set KUBECONFIG)")
	cfg.QPS, cfg.Burst = 20, 40
	cfg.UserAgent = "tf-e2e"

	c, err := client.New(cfg, client.Options{})
	Expect(err).NotTo(HaveOccurred())
	cs, err := kubernetes.NewForConfig(cfg)
	Expect(err).NotTo(HaveOccurred())

	wd, _ := os.Getwd()
	repo := filepath.Clean(filepath.Join(wd, "..", ".."))
	runID := getenv("E2E_RUN_ID", time.Now().UTC().Format("20060102-150405"))
	results := getenv("E2E_RESULTS_DIR", filepath.Join(repo, "e2e", "results", runID))
	Expect(os.MkdirAll(results, 0o755)).To(Succeed())

	e := &Env{
		C: c, CS: cs, Cfg: cfg,
		RunID: runID, ResultsDir: results, RepoRoot: repo,
		TFNamespace:  getenv("E2E_TF_NAMESPACE", "tensor-fusion-sys"),
		TFRelease:    getenv("E2E_TF_RELEASE", "tensor-fusion-sys"),
		RealPool:     getenv("E2E_POOL", "tensor-fusion-shared"),
		MockPool:     getenv("E2E_MOCK_POOL", "tensor-fusion-mock"),
		GPUNodePool:  getenv("E2E_GPU_NODEPOOL", "tf-gpu"),
		CPUNodePool:  getenv("E2E_CPU_NODEPOOL", "tf-cpu"),
		MockNodePool: getenv("E2E_MOCK_NODEPOOL", "tf-mock"),
		RoleLabel:    getenv("E2E_ROLE_LABEL", "e2e.tensor-fusion.ai/role"),
		Images: Images{
			// Not busybox: the soft limiter is preloaded into every process and needs
			// glibc + libstdc++ in the image (case ISO-14 pins that limitation).
			Busybox: getenv("E2E_IMAGE_BASE", "public.ecr.aws/docker/library/debian:bookworm-slim"),
			Torch:   getenv("E2E_IMAGE_TORCH", "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"),
			VLLM:    getenv("E2E_IMAGE_VLLM", "vllm/vllm-openai:v0.11.0"),
			ORT:     getenv("E2E_IMAGE_ORT", "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"),
			Curl:    getenv("E2E_IMAGE_PYTHON", "public.ecr.aws/docker/library/python:3.12-slim"),
		},
	}
	e.Prices = loadPrices(filepath.Join(repo, "e2e", "pricing", getenv("E2E_PRICING_REGION", "us-west-2")+".json"))
	return e
}

// detect fills in what depends on the live cluster. Called once from BeforeSuite.
func (e *Env) detect(ctx context.Context) {
	if v, err := e.CS.Discovery().ServerVersion(); err == nil {
		e.K8sVersion = v.GitVersion
	}
	e.HasKarpenter = e.crdExists(ctx, "nodeclaims.karpenter.sh")
	e.HasAWS = e.crdExists(ctx, "ec2nodeclasses.karpenter.k8s.aws")

	if d, err := e.get(ctx, gvkDeployment, e.TFNamespace, e.TFRelease+"-controller"); err == nil {
		e.OperatorImage = firstContainerImage(d, "controller")
	} else if l, err := e.list(ctx, gvkDeployment, e.TFNamespace, client.MatchingLabels{tfDomain + "/component": "operator"}); err == nil && len(l) > 0 {
		e.OperatorImage = firstContainerImage(&l[0], "controller")
	}

	// Mode: explicit, or inferred from the GPU CRs that exist right now.
	e.Mode = getenv("E2E_GPU_MODE", "auto")
	if e.Mode == "auto" {
		e.Mode = ModeNoGPU
		gpus, _ := e.list(ctx, gvkGPU, "")
		for i := range gpus {
			pool := gpus[i].GetLabels()[tfDomain+"/gpupool"]
			if pool == e.MockPool && e.Mode == ModeNoGPU {
				e.Mode = ModeMockGPU
			}
			if pool == e.RealPool {
				e.Mode = ModeRealGPU
			}
		}
	}
	switch e.Mode {
	case "real", ModeRealGPU:
		e.Mode, e.Pool = ModeRealGPU, e.RealPool
	case "mock", ModeMockGPU:
		e.Mode, e.Pool = ModeMockGPU, e.MockPool
	default:
		e.Mode, e.Pool = ModeNoGPU, e.RealPool
	}
	if p, err := e.get(ctx, gvkGPUPool, "", e.Pool); err == nil {
		e.HypervisorImage, _, _ = unstructured.NestedString(p.Object, "spec", "componentConfig", "hypervisor", "image")
	}
}

func firstContainerImage(u *unstructured.Unstructured, name string) string {
	containers, _, _ := unstructured.NestedSlice(u.Object, "spec", "template", "spec", "containers")
	for _, c := range containers {
		m, _ := c.(map[string]any)
		if n, _ := m["name"].(string); n == name || name == "" {
			img, _ := m["image"].(string)
			return img
		}
	}
	if len(containers) > 0 {
		m, _ := containers[0].(map[string]any)
		img, _ := m["image"].(string)
		return img
	}
	return ""
}

// ---------------------------------------------------------------------------
// object helpers
// ---------------------------------------------------------------------------

func newObj(gvk schema.GroupVersionKind, ns, name string) *unstructured.Unstructured {
	u := &unstructured.Unstructured{}
	u.SetGroupVersionKind(gvk)
	u.SetNamespace(ns)
	u.SetName(name)
	return u
}

func (e *Env) get(ctx context.Context, gvk schema.GroupVersionKind, ns, name string) (*unstructured.Unstructured, error) {
	u := newObj(gvk, ns, name)
	err := e.C.Get(ctx, client.ObjectKey{Namespace: ns, Name: name}, u)
	return u, err
}

func (e *Env) list(ctx context.Context, gvk schema.GroupVersionKind, ns string, opts ...client.ListOption) ([]unstructured.Unstructured, error) {
	l := &unstructured.UnstructuredList{}
	l.SetGroupVersionKind(gvk)
	if ns != "" {
		opts = append(opts, client.InNamespace(ns))
	}
	if err := e.C.List(ctx, l, opts...); err != nil {
		if meta.IsNoMatchError(err) {
			return nil, nil
		}
		return nil, err
	}
	return l.Items, nil
}

func (e *Env) crdExists(ctx context.Context, name string) bool {
	_, err := e.get(ctx, gvkCRD, "", name)
	return err == nil
}

// parseYAML splits a multi-document manifest into objects.
func parseYAML(manifest string) []*unstructured.Unstructured {
	var out []*unstructured.Unstructured
	dec := utilyaml.NewYAMLOrJSONDecoder(strings.NewReader(manifest), 4096)
	for {
		u := &unstructured.Unstructured{}
		if err := dec.Decode(&u.Object); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			Fail("bad manifest: " + err.Error())
		}
		if len(u.Object) > 0 {
			out = append(out, u)
		}
	}
	return out
}

// create creates the objects; AlreadyExists is an error (cases own their names).
func (e *Env) create(ctx context.Context, objs ...*unstructured.Unstructured) error {
	for _, o := range objs {
		if err := e.C.Create(ctx, o); err != nil {
			return fmt.Errorf("create %s %s/%s: %w", o.GetKind(), o.GetNamespace(), o.GetName(), err)
		}
	}
	return nil
}

// dryRun sends the object through admission (webhooks, CRD validation, quota)
// without persisting it: the server-side equivalent of
// `kubectl apply --dry-run=server`.
func (e *Env) dryRun(ctx context.Context, o *unstructured.Unstructured) error {
	c := o.DeepCopy()
	return e.C.Create(ctx, c, client.DryRunAll)
}

func (e *Env) delete(ctx context.Context, o *unstructured.Unstructured, opts ...client.DeleteOption) {
	err := e.C.Delete(ctx, o, opts...)
	if err != nil && !apierrors.IsNotFound(err) && !meta.IsNoMatchError(err) {
		GinkgoWriter.Printf("delete %s %s/%s: %v\n", o.GetKind(), o.GetNamespace(), o.GetName(), err)
	}
}

func (e *Env) patchMerge(ctx context.Context, o *unstructured.Unstructured, patch string) error {
	return e.C.Patch(ctx, o, client.RawPatch("application/merge-patch+json", []byte(patch)))
}

// str / num / slice accessors that never panic on a missing path.
func str(u *unstructured.Unstructured, path ...string) string {
	if u == nil {
		return ""
	}
	v, _, _ := unstructured.NestedFieldNoCopy(u.Object, path...)
	switch t := v.(type) {
	case string:
		return t
	case nil:
		return ""
	default:
		return fmt.Sprint(t)
	}
}

func sliceAt(u *unstructured.Unstructured, path ...string) []any {
	v, _, _ := unstructured.NestedFieldNoCopy(u.Object, path...)
	s, _ := v.([]any)
	return s
}

func mapAt(u *unstructured.Unstructured, path ...string) map[string]any {
	v, _, _ := unstructured.NestedFieldNoCopy(u.Object, path...)
	m, _ := v.(map[string]any)
	return m
}

func condition(u *unstructured.Unstructured, typ string) string {
	for _, c := range sliceAt(u, "status", "conditions") {
		m, _ := c.(map[string]any)
		if m["type"] == typ {
			s, _ := m["status"].(string)
			return s
		}
	}
	return ""
}

func toJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// ---------------------------------------------------------------------------
// waiting
// ---------------------------------------------------------------------------

// waitFor polls fn until it returns ("", true) or the timeout expires. The last
// "why not yet" string ends up in the failure message, which is what makes a
// timeout diagnosable from the JUnit report alone.
func waitFor(ctx context.Context, what string, timeout time.Duration, fn func() (string, bool)) time.Duration {
	GinkgoHelper()
	start := time.Now()
	last := ""
	for {
		why, ok := fn()
		if ok {
			return time.Since(start)
		}
		last = why
		if time.Since(start) > timeout {
			Fail(fmt.Sprintf("timed out after %s waiting for %s; last state: %s", timeout, what, last))
		}
		select {
		case <-ctx.Done():
			Fail(fmt.Sprintf("spec deadline hit while waiting for %s; last state: %s", what, last))
		case <-time.After(poll):
		}
	}
}

// never asserts that fn stays false for the whole window (negative checks:
// "no NodeClaim is created").
func never(ctx context.Context, what string, window time.Duration, fn func() (string, bool)) {
	GinkgoHelper()
	deadline := time.Now().Add(window)
	for time.Now().Before(deadline) {
		if why, happened := fn(); happened {
			Fail(fmt.Sprintf("%s happened: %s", what, why))
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(poll):
		}
	}
}

func podPhase(p *unstructured.Unstructured) string { return str(p, "status", "phase") }

func podReady(p *unstructured.Unstructured) bool {
	return podPhase(p) == "Running" && condition(p, "Ready") == "True"
}

// podWhy is a compact human reason for a pod that is not running yet.
func podWhy(p *unstructured.Unstructured) string {
	var parts []string
	parts = append(parts, "phase="+podPhase(p))
	if n := str(p, "spec", "nodeName"); n != "" {
		parts = append(parts, "node="+n)
	}
	for _, c := range sliceAt(p, "status", "conditions") {
		m, _ := c.(map[string]any)
		if m["status"] == "False" && m["message"] != nil {
			parts = append(parts, fmt.Sprintf("%v: %v", m["type"], m["message"]))
		}
	}
	for _, key := range []string{"initContainerStatuses", "containerStatuses"} {
		for _, c := range sliceAt(p, "status", key) {
			m, _ := c.(map[string]any)
			if st, ok := m["state"].(map[string]any); ok {
				if w, ok := st["waiting"].(map[string]any); ok {
					parts = append(parts, fmt.Sprintf("%v waiting: %v %v", m["name"], w["reason"], w["message"]))
				}
				if t, ok := st["terminated"].(map[string]any); ok && fmt.Sprint(t["exitCode"]) != "0" {
					parts = append(parts, fmt.Sprintf("%v exited %v: %v", m["name"], t["exitCode"], t["reason"]))
				}
			}
		}
	}
	return strings.Join(parts, "; ")
}

func (e *Env) waitPodReady(ctx context.Context, ns, name string, timeout time.Duration) (*unstructured.Unstructured, time.Duration) {
	GinkgoHelper()
	var pod *unstructured.Unstructured
	d := waitFor(ctx, fmt.Sprintf("pod %s/%s Ready", ns, name), timeout, func() (string, bool) {
		p, err := e.get(ctx, gvkPod, ns, name)
		if err != nil {
			return err.Error(), false
		}
		pod = p
		return podWhy(p), podReady(p)
	})
	return pod, d
}

func (e *Env) waitGone(ctx context.Context, gvk schema.GroupVersionKind, ns, name string, timeout time.Duration) time.Duration {
	GinkgoHelper()
	return waitFor(ctx, fmt.Sprintf("%s %s/%s gone", gvk.Kind, ns, name), timeout, func() (string, bool) {
		u, err := e.get(ctx, gvk, ns, name)
		if apierrors.IsNotFound(err) || meta.IsNoMatchError(err) {
			return "", true
		}
		if err != nil {
			return err.Error(), false
		}
		return fmt.Sprintf("still present, deletionTimestamp=%v finalizers=%v", u.GetDeletionTimestamp(), u.GetFinalizers()), false
	})
}

// events returns "reason: message" lines for events about the named object.
func (e *Env) events(ctx context.Context, ns, kind, name string) []string {
	evs, err := e.list(ctx, gvkEvent, ns)
	if err != nil {
		return nil
	}
	var out []string
	for i := range evs {
		if str(&evs[i], "involvedObject", "name") == name && (kind == "" || str(&evs[i], "involvedObject", "kind") == kind) {
			out = append(out, str(&evs[i], "reason")+": "+str(&evs[i], "message"))
		}
	}
	return out
}

func containsEvent(evs []string, substr ...string) bool {
	for _, ev := range evs {
		ok := true
		for _, s := range substr {
			if !strings.Contains(ev, s) {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// talking to things inside the cluster
// ---------------------------------------------------------------------------

// svcGet does an HTTP GET against a Service through the API server proxy. No
// port-forward process, works with any kubeconfig.
func (e *Env) svcGet(ctx context.Context, ns, svc, port, path string, params map[string]string) ([]byte, error) {
	return e.CS.CoreV1().Services(ns).ProxyGet("http", svc, port, path, params).DoRaw(ctx)
}

func (e *Env) podLogs(ctx context.Context, ns, pod, container string) string {
	b, err := e.CS.CoreV1().Pods(ns).GetLogs(pod, &corev1.PodLogOptions{Container: container}).DoRaw(ctx)
	if err != nil {
		return "<logs unavailable: " + err.Error() + ">"
	}
	return string(b)
}

// kubectl is used only where client-go would need packages the repo does not
// vendor (exec streams). It inherits KUBECONFIG from the environment.
func kubectl(ctx context.Context, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "kubectl", args...)
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		return out.String(), fmt.Errorf("kubectl %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(errb.String()))
	}
	return out.String(), nil
}

func (e *Env) exec(ctx context.Context, ns, pod, container string, command ...string) (string, error) {
	args := []string{"-n", ns, "exec", pod}
	if container != "" {
		args = append(args, "-c", container)
	}
	args = append(args, "--")
	return kubectl(ctx, append(args, command...)...)
}

// ---------------------------------------------------------------------------
// namespaces: one per case, deleted afterwards
// ---------------------------------------------------------------------------

var nsCounter struct {
	sync.Mutex
	n int
}

func (e *Env) newNamespace(ctx context.Context, caseID string) string {
	GinkgoHelper()
	nsCounter.Lock()
	nsCounter.n++
	n := nsCounter.n
	nsCounter.Unlock()
	run := strings.ToLower(strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			return r
		}
		return -1
	}, e.RunID))
	if len(run) > 10 {
		run = run[len(run)-10:]
	}
	name := fmt.Sprintf("tfe2e-%s-%s-%d", strings.ToLower(strings.ReplaceAll(caseID, "_", "-")), run, n)
	ns := newObj(gvkNamespace, "", name)
	ns.SetLabels(map[string]string{"tf-e2e/run": e.RunID, "tf-e2e/case": strings.ToLower(caseID)})
	Expect(e.C.Create(ctx, ns)).To(Succeed())
	return name
}

// deleteNamespace removes a case namespace and waits until the TensorFusion
// objects in it are really gone: a leaked TensorFusionWorkload keeps GPU
// capacity allocated and would poison the next case.
func (e *Env) deleteNamespace(name string) {
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	e.delete(ctx, newObj(gvkNamespace, "", name), client.PropagationPolicy(metav1.DeletePropagationBackground))
	deadline := time.Now().Add(3 * time.Minute)
	for time.Now().Before(deadline) {
		if _, err := e.get(ctx, gvkNamespace, "", name); apierrors.IsNotFound(err) {
			return
		}
		time.Sleep(poll)
	}
	GinkgoWriter.Printf("WARNING: namespace %s still terminating after 3m\n", name)
}
