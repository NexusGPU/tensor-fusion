//go:build e2e

// Package suite is the cluster-level end-to-end suite for TensorFusion.
//
// It runs against whatever cluster KUBECONFIG points at (the EKS stack in
// e2e/terraform, or a bare-metal cluster), selects cases by tier label, and
// writes JUnit plus a JSON / Markdown summary with every measured number.
//
//	go test -tags e2e -p 2 ./e2e/suite -timeout 60m \
//	   -ginkgo.label-filter='smoke-nogpu' -ginkgo.junit-report=junit.xml
//
// See e2e/README.md for the case list and e2e/Makefile for the entry points.
package suite

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/types"
	. "github.com/onsi/gomega"
)

func TestE2E(t *testing.T) {
	RegisterFailHandler(Fail)
	suiteCfg, reporterCfg := GinkgoConfiguration()
	// Cases share GPUs and NodePools: they run one at a time, in file order, and
	// a failure does not stop the rest (each case cleans up after itself).
	suiteCfg.RandomizeAllSpecs = false
	reporterCfg.Verbose = true
	RunSpecs(t, "TensorFusion cluster e2e", suiteCfg, reporterCfg)
}

var _ = BeforeSuite(func(ctx SpecContext) {
	env = newEnv()
	env.detect(ctx)
	By(fmt.Sprintf("run %s: k8s=%s mode=%s pool=%s karpenter=%v operator=%s results=%s",
		env.RunID, env.K8sVersion, env.Mode, env.Pool, env.HasKarpenter, env.OperatorImage, env.ResultsDir))
}, NodeTimeout(2*time.Minute))

// ---------------------------------------------------------------------------
// Case registration
// ---------------------------------------------------------------------------

// Tiers. A case may belong to several.
const (
	TierSmoke     = "smoke-nogpu"      // needs no GPU node at all
	TierCore      = "core"             // documented core features, needs a GPU (real or mock)
	TierIsolation = "isolation"        // limit enforcement, needs a real GPU
	TierKarpenter = "karpenter-finops" // node scaling + cost accounting
	TierPerf      = "perf"             // serving / benchmark performance vs share
	TierRehearsal = "rehearsal"        // subset proven on mock GPUs (CPU instances only)
	TierDryRun    = "dryrun"           // GPU-only cases proven as far as possible without a GPU
	TierDisrupt   = "disruptive"       // breaks the control plane on purpose; run last
	TierKWOK      = "kwok"             // Karpenter paths on fake (KWOK) nodes + fake hypervisor: no cloud, no GPU, control plane only
)

// Requirement names for Case.Needs.
const (
	NeedGPU       = "gpu"       // at least one GPU CR in the target pool (real or mock)
	NeedRealGPU   = "real-gpu"  // CUDA device; never satisfied by the mock accelerator
	NeedKarpenter = "karpenter" // karpenter.sh CRDs
	NeedAWS       = "aws"       // Karpenter AWS provider (EC2NodeClass)
	NeedNodeClass = "nodeclass" // a Karpenter provider whose NodeClass the suite can reference: AWS, Alibaba Cloud or KWOK
	NeedTwoGPUs   = "two-gpus"  // two GPU CRs (any nodes)
	NeedFixedImg  = "fixed-img" // behaviour known broken on the released images, see KnownIssue
	NeedLicense   = "license"   // a vGPU worker license (hard isolation, remote mode), see workerLicense
	NeedTF        = "tensor-fusion"
)

// C is the per-case context handed to a case body.
type C struct {
	ID   string
	NS   string
	mode string
	meta map[string]string
}

// SetMode changes the execution mode recorded for the case, for cases whose
// outcome decides it (a GPU launch that ends at the quota error is a dry run).
func (c *C) SetMode(mode string) {
	c.mode = mode
	m := map[string]string{}
	for k, v := range c.meta {
		m[k] = v
	}
	m["mode"] = mode
	AddReportEntry("case", m, ReportEntryVisibilityNever)
}

// Case describes one test case. ID is the stable identifier from e2e/README.md.
type Case struct {
	ID      string
	Title   string
	Tiers   []string
	Timeout time.Duration
	Needs   []string
	// Mode overrides the execution mode recorded for this case (e.g. dry-run).
	Mode string
	// KnownIssue: the case is expected to fail on the images named here; a
	// failure is then reported as "known-fail" in the summary (and still as a
	// failure in JUnit, on purpose).
	KnownIssue string
	NoNS       bool
	Run        func(ctx SpecContext, c *C)
}

var registry = map[string]Case{}

// Register adds the case to Ginkgo's tree. Called from package-level vars.
func Register(tc Case) bool {
	if _, dup := registry[tc.ID]; dup {
		panic("duplicate case id " + tc.ID)
	}
	registry[tc.ID] = tc
	if tc.Timeout == 0 {
		tc.Timeout = 5 * time.Minute
	}
	labels := append([]string{}, tc.Tiers...)
	args := []any{Label(labels...), SpecTimeout(tc.Timeout), func(ctx SpecContext) {
		c := &C{ID: tc.ID, mode: tc.Mode}
		if c.mode == "" {
			c.mode = env.Mode
		}
		c.meta = map[string]string{"id": tc.ID, "mode": c.mode, "tiers": strings.Join(tc.Tiers, ","), "known_issue": tc.KnownIssue}
		AddReportEntry("case", c.meta, ReportEntryVisibilityNever)

		if why := unmet(ctx, tc.Needs); why != "" {
			Skip(why)
		}
		if !tc.NoNS {
			c.NS = env.newNamespace(ctx, tc.ID)
			DeferCleanup(func() { env.deleteNamespace(c.NS) })
		}
		tc.Run(ctx, c)
	}}
	return It(tc.ID+" "+tc.Title, args...)
}

func hasTier(tiers []string, t string) bool {
	for _, x := range tiers {
		if x == t {
			return true
		}
	}
	return false
}

// unmet returns why the cluster cannot run a case ("" = it can). A skipped
// case is always reported with its reason; nothing is silently green.
func unmet(ctx context.Context, reqs []string) string {
	for _, r := range reqs {
		switch r {
		case NeedTF:
			if !env.crdExists(ctx, "gpupools."+tfDomain) {
				return "TensorFusion is not installed (no GPUPool CRD)"
			}
		case NeedKarpenter:
			if !env.HasKarpenter {
				return "Karpenter CRDs not installed on this cluster"
			}
		case NeedAWS:
			if !env.HasAWS {
				return "not an AWS Karpenter cluster (no EC2NodeClass CRD)"
			}
		case NeedNodeClass:
			if !env.HasAWS && !env.HasAlibaba && !env.HasKWOK {
				return "no known Karpenter NodeClass kind (EC2NodeClass, ECSNodeClass or KWOKNodeClass)"
			}
		case NeedGPU:
			if env.Mode == ModeNoGPU {
				return "no GPU in pool " + env.Pool + " (neither real nor mock); run the rehearsal or the GPU tiers"
			}
		case NeedRealGPU:
			if env.Mode != ModeRealGPU {
				return "needs a real CUDA device; current mode is " + env.Mode
			}
		case NeedTwoGPUs:
			if gpus := env.poolGPUs(ctx); len(gpus) < 2 {
				return fmt.Sprintf("needs 2 GPUs in pool %s, found %d", env.Pool, len(gpus))
			}
		case NeedLicense:
			if license, _ := workerLicense(); license == "" {
				return "needs a vGPU worker license: set TF_LICENSE and TF_LICENSE_SIGN or write them to ~/.config/tensor-fusion/license.env (tools/tf-license.sh)"
			}
		case NeedFixedImg:
			if os.Getenv("E2E_FIXED_IMAGES") == "" {
				return "known broken on the released images (see KnownIssue in the case list); set E2E_FIXED_IMAGES=1 when testing images that contain the fix"
			}
		}
	}
	return ""
}

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------

// Measurement is one recorded number (or short string) of a case.
type Measurement struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
	Unit  string  `json:"unit,omitempty"`
	Text  string  `json:"text,omitempty"`
}

// Measure records a number in the run summary.
func (c *C) Measure(name string, value float64, unit string) {
	AddReportEntry("measure", Measurement{Name: name, Value: value, Unit: unit}, ReportEntryVisibilityNever)
	GinkgoWriter.Printf("  measure %s = %.3f %s\n", name, value, unit)
}

// Note records a short observation (actual behaviour where docs and code disagree, etc.).
func (c *C) Note(format string, a ...any) {
	s := fmt.Sprintf(format, a...)
	AddReportEntry("note", Measurement{Name: "note", Text: s}, ReportEntryVisibilityNever)
	GinkgoWriter.Printf("  note: %s\n", s)
}

// Attach stores an arbitrary JSON document (tables for the report generator).
func (c *C) Attach(name string, v any) {
	path := filepath.Join(env.ResultsDir, fmt.Sprintf("%s.%s.json", c.ID, name))
	b, _ := json.MarshalIndent(v, "", "  ")
	Expect(os.WriteFile(path, b, 0o644)).To(Succeed())
	AddReportEntry("attachment", Measurement{Name: name, Text: filepath.Base(path)}, ReportEntryVisibilityNever)
}

// ---------------------------------------------------------------------------
// Summary: results/<run>/summary.json + summary.md
// ---------------------------------------------------------------------------

type caseResult struct {
	ID           string        `json:"id"`
	Title        string        `json:"title"`
	Tiers        []string      `json:"tiers"`
	Mode         string        `json:"mode"`
	Status       string        `json:"status"` // passed | failed | known-fail | skipped | timedout
	Seconds      float64       `json:"seconds"`
	Reason       string        `json:"reason,omitempty"`
	KnownIssue   string        `json:"known_issue,omitempty"`
	Measurements []Measurement `json:"measurements,omitempty"`
	Notes        []string      `json:"notes,omitempty"`
	Attachments  []string      `json:"attachments,omitempty"`
}

type runSummary struct {
	RunID       string         `json:"run_id"`
	Started     time.Time      `json:"started"`
	Seconds     float64        `json:"seconds"`
	Cluster     map[string]any `json:"cluster"`
	LabelFilter string         `json:"label_filter"`
	Counts      map[string]int `json:"counts"`
	Cases       []caseResult   `json:"cases"`
}

var _ = ReportAfterSuite("summary", func(report Report) {
	if env == nil {
		return
	}
	sum := runSummary{
		RunID:   env.RunID,
		Started: report.StartTime,
		Seconds: report.RunTime.Seconds(),
		Cluster: map[string]any{
			"kubernetes": env.K8sVersion, "gpu_mode": env.Mode, "pool": env.Pool,
			"karpenter": env.HasKarpenter, "karpenter_kwok": env.HasKWOK, "karpenter_alibaba": env.HasAlibaba, "operator_image": env.OperatorImage, "hypervisor_image": env.HypervisorImage,
		},
		LabelFilter: report.SuiteConfig.LabelFilter,
		Counts:      map[string]int{},
	}
	for _, sr := range report.SpecReports {
		if sr.LeafNodeType != types.NodeTypeIt {
			continue
		}
		id, title, _ := strings.Cut(sr.LeafNodeText, " ")
		res := caseResult{ID: id, Title: title, Tiers: sr.Labels(), Seconds: sr.RunTime.Seconds(), Mode: env.Mode}
		for _, e := range sr.ReportEntries {
			raw := e.Value.AsJSON
			if v := e.Value.GetRawValue(); v != nil {
				if rb, err := json.Marshal(v); err == nil {
					raw = string(rb)
				}
			}
			switch e.Name {
			case "case":
				var m map[string]string
				if json.Unmarshal([]byte(raw), &m) == nil {
					res.Mode, res.KnownIssue = m["mode"], m["known_issue"]
				}
			case "measure", "note", "attachment":
				var m Measurement
				if json.Unmarshal([]byte(raw), &m) != nil {
					continue
				}
				switch e.Name {
				case "measure":
					res.Measurements = append(res.Measurements, m)
				case "note":
					res.Notes = append(res.Notes, m.Text)
				default:
					res.Attachments = append(res.Attachments, m.Text)
				}
			}
		}
		switch sr.State {
		case types.SpecStatePassed:
			res.Status = "passed"
		case types.SpecStateSkipped:
			res.Status = "skipped"
			res.Reason = sr.Failure.Message
		case types.SpecStatePending:
			continue
		case types.SpecStateTimedout:
			res.Status = "timedout"
			res.Reason = firstLine(sr.Failure.Message)
		default:
			res.Status = "failed"
			res.Reason = firstLine(sr.Failure.Message)
		}
		if res.Status == "skipped" && res.Reason == "" {
			continue // filtered out by the label filter, not part of this run
		}
		if (res.Status == "failed" || res.Status == "timedout") && res.KnownIssue != "" {
			res.Status = "known-fail"
		}
		sum.Counts[res.Status]++
		sum.Cases = append(sum.Cases, res)
	}
	sort.SliceStable(sum.Cases, func(i, j int) bool { return sum.Cases[i].ID < sum.Cases[j].ID })

	b, _ := json.MarshalIndent(sum, "", "  ")
	_ = os.WriteFile(filepath.Join(env.ResultsDir, "summary.json"), b, 0o644)
	_ = os.WriteFile(filepath.Join(env.ResultsDir, "summary.md"), []byte(renderMarkdown(sum)), 0o644)
	fmt.Fprintf(GinkgoWriter, "\nsummary: %s\n", filepath.Join(env.ResultsDir, "summary.md"))
})

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i > 0 {
		s = s[:i]
	}
	if len(s) > 400 {
		s = s[:400] + "…"
	}
	return s
}

func renderMarkdown(sum runSummary) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# TensorFusion e2e run %s\n\n", sum.RunID)
	fmt.Fprintf(&b, "- Kubernetes: %v\n- GPU mode: **%v** (pool `%v`)\n- Operator image: `%v`\n- Hypervisor image: `%v`\n- Label filter: `%s`\n- Duration: %.0f s\n",
		sum.Cluster["kubernetes"], sum.Cluster["gpu_mode"], sum.Cluster["pool"], sum.Cluster["operator_image"], sum.Cluster["hypervisor_image"], sum.LabelFilter, sum.Seconds)
	var keys []string
	for k := range sum.Counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	b.WriteString("- Result: ")
	for i, k := range keys {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%d %s", sum.Counts[k], k)
	}
	b.WriteString("\n\n| ID | Case | Mode | Status | Time (s) | Measurements / reason |\n|---|---|---|---|---|---|\n")
	for _, c := range sum.Cases {
		var detail []string
		for _, m := range c.Measurements {
			detail = append(detail, fmt.Sprintf("%s=%s%s", m.Name, trimFloat(m.Value), m.Unit))
		}
		if c.Reason != "" {
			detail = append(detail, c.Reason)
		}
		if c.Status == "known-fail" {
			detail = append(detail, "known issue: "+c.KnownIssue)
		}
		detail = append(detail, c.Notes...)
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %.0f | %s |\n", c.ID, c.Title, c.Mode, c.Status, c.Seconds,
			strings.ReplaceAll(strings.Join(detail, "; "), "|", "\\|"))
	}
	return b.String()
}

func trimFloat(f float64) string {
	s := fmt.Sprintf("%.3f", f)
	s = strings.TrimRight(strings.TrimRight(s, "0"), ".")
	return s
}

// ---------------------------------------------------------------------------
// Prices (e2e/pricing/<region>.json, shared with `make cost`)
// ---------------------------------------------------------------------------

type Prices struct {
	Region   string             `json:"region"`
	OnDemand map[string]float64 `json:"on_demand"`
	EKSHour  float64            `json:"eks_control_plane_per_hour"`
}

func loadPrices(path string) Prices {
	var p Prices
	b, err := os.ReadFile(path)
	if err != nil {
		return p
	}
	_ = json.Unmarshal(b, &p)
	return p
}
