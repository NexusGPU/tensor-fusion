//go:build e2e

package suite

import (
	"fmt"
	"math"
	"sort"
)

// Evaluation of performance under a compute share. Pure functions: the same
// code judges a real GPU run, a bare-metal run and the CPU stand-in of the dry
// run, and tools/report.py re-renders its output without re-deciding anything.

// Dist is a latency distribution summary in seconds.
type Dist struct {
	N    int     `json:"n"`
	P50  float64 `json:"p50"`
	P95  float64 `json:"p95"`
	P99  float64 `json:"p99"`
	Max  float64 `json:"max"`
	Mean float64 `json:"mean"`
}

// PerfPoint is one measurement: one workload under one configuration.
type PerfPoint struct {
	Case     string `json:"case"`
	Workload string `json:"workload"` // vllm | onnxruntime | torchbench-mini | mlperf | llm-standin | matmul
	Model    string `json:"model"`
	Test     string `json:"test,omitempty"` // eval | train | Offline | Server

	Mode      string  `json:"mode"`           // native | shared | soft | hard | standin-smooth | ...
	Share     float64 `json:"share"`          // 1.0 = 100 %
	VRAMGiB   float64 `json:"vram_limit_gib"` // 0 = no limit
	Cotenancy string  `json:"cotenancy"`      // alone | noisy-neighbour
	Load      string  `json:"load"`           // saturation | fixed-rate

	Throughput     float64 `json:"throughput"`
	ThroughputUnit string  `json:"throughput_unit"` // tokens/s | samples/s | iters/s
	TPM            float64 `json:"tpm,omitempty"`

	TTFT    Dist `json:"ttft_s"`
	TPOT    Dist `json:"tpot_s"`
	ITL     Dist `json:"itl_s"`
	Latency Dist `json:"latency_s"`

	Requests      int      `json:"requests"`
	Samples       int      `json:"samples"` // tokens, calls or iterations inside the window
	WindowS       float64  `json:"window_s"`
	ErrorRate     float64  `json:"error_rate"`
	Timeouts      int      `json:"timeouts"`
	Gaps400       int      `json:"gaps_ge_400ms"`
	Gaps900       int      `json:"gaps_ge_900ms"`
	GapsNear500   int      `json:"gaps_near_500ms"`
	GapsNear1s    int      `json:"gaps_near_1s"`
	ITLBimodality *float64 `json:"itl_bimodality,omitempty"`
	MaxMemoryMiB  float64  `json:"max_memory_mib,omitempty"`
	SetupSeconds  float64  `json:"setup_seconds,omitempty"`
	ExecMode      string   `json:"exec_mode"` // real-gpu | cpu-standin
	GPUModel      string   `json:"gpu_model,omitempty"`
	Error         string   `json:"error,omitempty"`
	Skipped       string   `json:"skipped,omitempty"`

	Verdicts []Verdict `json:"verdicts,omitempty"`
	Pass     *bool     `json:"pass,omitempty"`
}

// Verdict is one criterion applied to one point.
type Verdict struct {
	Criterion string  `json:"criterion"`
	Pass      bool    `json:"pass"`
	Value     float64 `json:"value"`
	Bound     string  `json:"bound"`
}

// Criteria are the pass bounds of one isolation mode. All of them are two-sided
// in spirit: (a) the limit is effective, (b) it does not wreck serving quality.
type Criteria struct {
	// (a) effectiveness: throughput(s) / throughput(100%) must lie in
	// [BandLow*s, BandHigh*s + BandAbs].
	BandLow, BandHigh, BandAbs float64
	// throughput(lowest share) / throughput(100%) above this = the limiter does
	// not limit (flat curve).
	FlatMax float64
	// overhead at a 100 % share against no TensorFusion at all.
	OverheadMax float64

	// (b) no cliff. With a share s the proportional expectation for a latency
	// is baseline/s; tails may exceed it by these factors.
	TailK   float64 // p99 of TPOT (LLM) or per-call latency (others)
	MedianK float64 // p50
	// TTFT p95 <= TTFTK * baseline_p95 / s + TTFTFixed. The fixed part is what
	// separates "proportionally slower" from "stalled once before the first token".
	TTFTK, TTFTFixed float64
	// Stall signature of the soft limiter (1 s token wait, 500 ms refill steps).
	MaxGapS       float64 // no inter-token gap / call may reach this
	GapRatePer1k  float64 // gaps >= 400 ms per 1000 tokens (or calls)
	TailRatioK    float64 // (ITL p99 / p50) may grow at most this much over the baseline's ratio: bimodal = cliff
	MaxErrorRate  float64
	MinSamplesP99 int // below this many samples a p99 is reported but not judged
}

// criteriaFor returns the bounds for an isolation mode.
//
// soft: an in-process token bucket fed by a PID loop on measured utilisation,
// refilled every 500 ms. It regulates around the target, so +-25 % of the
// proportional throughput is the honest band; tails get 1.5x.
// hard: the limit is applied by the worker that executes the calls, a direct
// mechanism: +-15 % and 1.3x. It pays for that with remoting overhead, hence
// the larger allowance at 100 %.
// native / shared have no share to honour; only the overhead rule applies.
func criteriaFor(mode string) Criteria {
	c := Criteria{
		BandLow: 0.75, BandHigh: 1.25, BandAbs: 0.05, FlatMax: 0.85, OverheadMax: 0.05,
		TailK: 1.5, MedianK: 1.3, TTFTK: 1.5, TTFTFixed: 0.25,
		MaxGapS: 0.9, GapRatePer1k: 5, TailRatioK: 3, MaxErrorRate: 0.01, MinSamplesP99: 300,
	}
	switch mode {
	case "hard":
		c.BandLow, c.BandHigh, c.OverheadMax, c.TailK = 0.85, 1.15, 0.15, 1.3
	case "shared":
		c.OverheadMax = 0.03
	}
	return c
}

func (p *PerfPoint) groupKey() string {
	return fmt.Sprintf("%s|%s|%s|%s|%.1f|%s|%s", p.Workload, p.Model, p.Test, p.Mode, p.VRAMGiB, p.Cotenancy, p.Load)
}

func (p *PerfPoint) nativeKey() string {
	return fmt.Sprintf("%s|%s|%s|%s", p.Workload, p.Model, p.Test, p.Load)
}

// tail returns the latency distribution that plays the "per unit of work" role.
func (p *PerfPoint) tail() Dist {
	if p.TPOT.N > 0 {
		return p.TPOT
	}
	return p.Latency
}

func (p *PerfPoint) add(name string, pass bool, value float64, bound string) {
	p.Verdicts = append(p.Verdicts, Verdict{Criterion: name, Pass: pass, Value: round3(value), Bound: bound})
}

func round3(f float64) float64 {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0
	}
	return math.Round(f*1000) / 1000
}

// evaluate judges every point against its 100 % baseline (same workload, mode,
// VRAM limit, co-tenancy and load) and against the native run when there is
// one. It fills Verdicts and Pass in place and returns the number of failures.
func evaluate(points []PerfPoint) int {
	baseline := map[string]*PerfPoint{}
	native := map[string]*PerfPoint{}
	lowest := map[string]*PerfPoint{}
	for i := range points {
		p := &points[i]
		if p.Error != "" || p.Skipped != "" {
			continue
		}
		if p.Mode == "native" {
			native[p.nativeKey()] = p
		}
		if p.Share >= 0.999 {
			baseline[p.groupKey()] = p
		}
		if l, ok := lowest[p.groupKey()]; !ok || p.Share < l.Share {
			lowest[p.groupKey()] = p
		}
	}
	failures := 0
	for i := range points {
		p := &points[i]
		p.Verdicts = nil
		if p.Skipped != "" {
			continue
		}
		if p.Error != "" {
			f := false
			p.Pass = &f
			failures++
			continue
		}
		c := criteriaFor(p.Mode)
		base := baseline[p.groupKey()]
		s := p.Share

		p.add("error-rate", p.ErrorRate <= c.MaxErrorRate, p.ErrorRate, fmt.Sprintf("<= %.2f", c.MaxErrorRate))

		// Stall signature, judged on every limited point by itself.
		if p.Mode != "native" {
			maxGap := math.Max(p.ITL.Max, 0)
			if p.ITL.N == 0 {
				maxGap = p.Latency.Max
			}
			if p.ITL.N > 0 || p.tail().P50 < 0.2 { // a workload whose normal call takes > 200 ms cannot be judged on absolute gaps
				p.add("no-stall-near-1s", p.Gaps900 == 0, maxGap, fmt.Sprintf("no gap >= %.1fs", c.MaxGapS))
				units := p.Samples
				if units > 0 {
					rate := float64(p.Gaps400) * 1000 / float64(units)
					p.add("stall-rate", rate <= c.GapRatePer1k, rate, fmt.Sprintf("<= %.0f gaps>=0.4s per 1k", c.GapRatePer1k))
				}
			}
		}

		// Overhead of the mode itself: 100 % share against native.
		if nat := native[p.nativeKey()]; nat != nil && p.Mode != "native" && s >= 0.999 && nat.Throughput > 0 {
			overhead := 1 - p.Throughput/nat.Throughput
			p.add("overhead-vs-native", overhead <= c.OverheadMax, overhead, fmt.Sprintf("<= %.0f%%", c.OverheadMax*100))
			if nat.tail().P99 > 0 && p.tail().N >= c.MinSamplesP99 {
				r := p.tail().P99 / nat.tail().P99
				p.add("p99-vs-native", r <= 1+2*c.OverheadMax+0.1, r, fmt.Sprintf("<= %.2fx", 1+2*c.OverheadMax+0.1))
			}
		}

		if base != nil && base != p && base.Throughput > 0 && s > 0 && s < 0.999 {
			ratio := p.Throughput / base.Throughput
			lo, hi := c.BandLow*s, c.BandHigh*s+c.BandAbs
			// A fixed-rate run below capacity is not expected to drop: only the
			// upper bound of the load itself applies there.
			if p.Load == "fixed-rate" {
				p.add("fixed-rate-served", ratio >= math.Min(1, lo/s)*0.95 || ratio >= 0.95, ratio, ">= 0.95 of offered load unless the share is below the load")
			} else {
				p.add("throughput-proportional", ratio >= lo && ratio <= hi, ratio, fmt.Sprintf("%.2f..%.2f (share %.0f%%)", lo, hi, s*100))
			}

			bt, pt := base.tail(), p.tail()
			if bt.P50 > 0 && pt.N > 0 && p.Load != "fixed-rate" {
				r := pt.P50 / (bt.P50 / s)
				p.add("p50-vs-proportional", r <= c.MedianK, r, fmt.Sprintf("<= %.1fx of baseline/share", c.MedianK))
			}
			if p.Load != "fixed-rate" {
				// Tail: p99 where there are enough samples for one, p95 otherwise.
				// For LLM serving the inter-token latency is judged as well: it has
				// one sample per token, and a per-request mean (TPOT) hides stalls
				// that a stream of tokens shows.
				if bt.P99 > 0 && pt.N >= c.MinSamplesP99 {
					r := pt.P99 / (bt.P99 / s)
					p.add("p99-no-cliff", r <= c.TailK, r, fmt.Sprintf("<= %.1fx of baseline/share", c.TailK))
				} else if bt.P95 > 0 && pt.N >= c.MinSamplesP99/5 {
					r := pt.P95 / (bt.P95 / s)
					p.add("p95-no-cliff", r <= c.TailK, r, fmt.Sprintf("<= %.1fx of baseline/share (too few samples for p99)", c.TailK))
				}
				if base.ITL.P99 > 0 && p.ITL.N >= c.MinSamplesP99 {
					r := p.ITL.P99 / (base.ITL.P99 / s)
					p.add("itl-p99-no-cliff", r <= c.TailK, r, fmt.Sprintf("<= %.1fx of baseline/share", c.TailK))
				}
			}
			if base.TTFT.P95 > 0 && p.TTFT.N > 0 {
				bound := c.TTFTK*base.TTFT.P95/s + c.TTFTFixed
				p.add("ttft-no-fixed-stall", p.TTFT.P95 <= bound, p.TTFT.P95, fmt.Sprintf("<= %.3fs", bound))
			}
			if base.ITL.P50 > 0 && base.ITL.P99 > 0 && p.ITL.P50 > 0 && p.ITL.N >= c.MinSamplesP99 {
				r := (p.ITL.P99 / p.ITL.P50) / (base.ITL.P99 / base.ITL.P50)
				p.add("itl-not-bimodal", r <= c.TailRatioK, r, fmt.Sprintf("p99/p50 <= %.0fx the baseline's", c.TailRatioK))
			}
		}

		// Flat curve: judged on the lowest share of a sweep.
		if base != nil && lowest[p.groupKey()] == p && base != p && base.Throughput > 0 && p.Load != "fixed-rate" && s <= 0.6 {
			ratio := p.Throughput / base.Throughput
			p.add("limit-is-effective", ratio <= c.FlatMax, ratio, fmt.Sprintf("<= %.2f of the 100%% throughput at share %.0f%%", c.FlatMax, s*100))
		}

		ok := true
		for _, v := range p.Verdicts {
			ok = ok && v.Pass
		}
		p.Pass = &ok
		if !ok {
			failures++
		}
	}
	return failures
}

func failedVerdicts(points []PerfPoint) []string {
	var out []string
	for i := range points {
		p := &points[i]
		if p.Error != "" {
			out = append(out, fmt.Sprintf("%s %s %s share=%.0f%%: %s", p.Workload, p.Model, p.Mode, p.Share*100, p.Error))
		}
		for _, v := range p.Verdicts {
			if !v.Pass {
				out = append(out, fmt.Sprintf("%s %s %s share=%.0f%% %s: %s=%.3f (bound %s)", p.Workload, p.Model, p.Mode, p.Share*100, p.Load, v.Criterion, v.Value, v.Bound))
			}
		}
	}
	sort.Strings(out)
	return out
}

// ---------------------------------------------------------------------------
// Experimental design
// ---------------------------------------------------------------------------

// Factor is one dimension of the perf design.
type Factor struct {
	Name   string
	Levels []string
}

// Run is one row of the design: factor name -> level.
type Run map[string]string

// pairwise returns a small set of runs in which every pair of levels of every
// two factors appears at least once (greedy all-pairs, deterministic). The
// main effect of each factor stays estimable because every level of a factor
// meets every level of every other factor; what is given up is three-way
// interactions. `valid` rejects combinations that cannot exist (a native run
// has no share).
func pairwise(factors []Factor, valid func(Run) bool) []Run {
	type pair struct {
		a, b   int
		la, lb string
	}
	need := map[pair]bool{}
	for i := 0; i < len(factors); i++ {
		for j := i + 1; j < len(factors); j++ {
			for _, la := range factors[i].Levels {
				for _, lb := range factors[j].Levels {
					r := Run{factors[i].Name: la, factors[j].Name: lb}
					if completable(factors, r, valid) {
						need[pair{i, j, la, lb}] = true
					}
				}
			}
		}
	}
	covers := func(r Run) int {
		n := 0
		for p := range need {
			if r[factors[p.a].Name] == p.la && r[factors[p.b].Name] == p.lb {
				n++
			}
		}
		return n
	}
	var all []Run
	var gen func(i int, cur Run)
	gen = func(i int, cur Run) {
		if i == len(factors) {
			if valid == nil || valid(cur) {
				cp := Run{}
				for k, v := range cur {
					cp[k] = v
				}
				all = append(all, cp)
			}
			return
		}
		for _, l := range factors[i].Levels {
			cur[factors[i].Name] = l
			gen(i+1, cur)
		}
		delete(cur, factors[i].Name)
	}
	gen(0, Run{})

	var out []Run
	for len(need) > 0 {
		best, bestN := -1, 0
		for i, r := range all {
			if n := covers(r); n > bestN {
				best, bestN = i, n
			}
		}
		if best < 0 {
			break
		}
		r := all[best]
		out = append(out, r)
		for p := range need {
			if r[factors[p.a].Name] == p.la && r[factors[p.b].Name] == p.lb {
				delete(need, p)
			}
		}
	}
	return out
}

func completable(factors []Factor, partial Run, valid func(Run) bool) bool {
	if valid == nil {
		return true
	}
	var rec func(i int, cur Run) bool
	rec = func(i int, cur Run) bool {
		if i == len(factors) {
			return valid(cur)
		}
		if _, fixed := partial[factors[i].Name]; fixed {
			return rec(i+1, cur)
		}
		for _, l := range factors[i].Levels {
			cur[factors[i].Name] = l
			if rec(i+1, cur) {
				delete(cur, factors[i].Name)
				return true
			}
		}
		delete(cur, factors[i].Name)
		return false
	}
	cur := Run{}
	for k, v := range partial {
		cur[k] = v
	}
	return rec(0, cur)
}
