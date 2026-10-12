#!/usr/bin/env python3
"""Turns one or more e2e result directories into a comparison report.

    python3 e2e/tools/report.py e2e/results/<run> [e2e/results/<other-run> ...] [-o out-dir]

Reads   <run>/summary.json and every <run>/*.perf.json / *.finops.json attachment.
Writes  report.md   case status side by side per run; perf tables (metric vs
                    share, per mode, per workload) with pass/fail against the
                    criteria and the native baseline; FinOps table
        perf.csv    one row per measurement point (spreadsheet / pandas)
        perf.json   charts-ready series: one object per (run, workload, model,
                    mode, load, metric) with [share, value] points

Nothing is re-judged here: verdicts come from the suite (perf_eval_test.go).
Runs on bare metal and on EKS can be passed together to compare hardware, and
runs of two releases to see regressions. Standard library only.
"""
import argparse
import csv
import glob
import json
import os
import sys

METRICS = [
    ("throughput", lambda p: p.get("throughput"), "{:.1f}"),
    ("ttft_p50_ms", lambda p: ms(p, "ttft_s", "p50"), "{:.0f}"),
    ("ttft_p95_ms", lambda p: ms(p, "ttft_s", "p95"), "{:.0f}"),
    ("ttft_p99_ms", lambda p: ms(p, "ttft_s", "p99"), "{:.0f}"),
    ("tpot_p50_ms", lambda p: ms(p, "tpot_s", "p50"), "{:.1f}"),
    ("tpot_p95_ms", lambda p: ms(p, "tpot_s", "p95"), "{:.1f}"),
    ("tpot_p99_ms", lambda p: ms(p, "tpot_s", "p99"), "{:.1f}"),
    ("itl_p99_ms", lambda p: ms(p, "itl_s", "p99"), "{:.1f}"),
    ("itl_max_ms", lambda p: ms(p, "itl_s", "max"), "{:.0f}"),
    ("latency_p50_ms", lambda p: ms(p, "latency_s", "p50"), "{:.1f}"),
    ("latency_p99_ms", lambda p: ms(p, "latency_s", "p99"), "{:.1f}"),
    ("tpm", lambda p: p.get("tpm"), "{:.0f}"),
    ("error_rate", lambda p: p.get("error_rate"), "{:.3f}"),
    ("gaps_ge_400ms", lambda p: p.get("gaps_ge_400ms"), "{:.0f}"),
    ("gaps_ge_900ms", lambda p: p.get("gaps_ge_900ms"), "{:.0f}"),
]


def ms(p, dist, q):
    d = p.get(dist) or {}
    v = d.get(q)
    return v * 1000 if v else None


def fmt(v, f="{:.1f}"):
    return "–" if v is None else f.format(v)


def load_run(path):
    with open(os.path.join(path, "summary.json")) as fh:
        summary = json.load(fh)
    points, finops = [], []
    for f in sorted(glob.glob(os.path.join(path, "*.perf.json"))):
        with open(f) as fh:
            points += json.load(fh) or []
    for f in sorted(glob.glob(os.path.join(path, "*.finops.json"))):
        with open(f) as fh:
            finops.append(json.load(fh))
    return {"name": summary.get("run_id") or os.path.basename(path.rstrip("/")), "summary": summary, "points": points, "finops": finops}


def cases_table(runs):
    ids = sorted({c["id"] for r in runs for c in r["summary"].get("cases", [])})
    by = [{c["id"]: c for c in r["summary"].get("cases", [])} for r in runs]
    out = ["| Case | Title | " + " | ".join(r["name"] for r in runs) + " |", "|---|---|" + "---|" * len(runs)]
    for cid in ids:
        title = next((b[cid]["title"] for b in by if cid in b), "")
        cells = []
        for b in by:
            c = b.get(cid)
            cells.append("–" if not c else "%s (%s, %.0fs)" % (c["status"], c.get("mode", "?"), c.get("seconds", 0)))
        out.append("| %s | %s | %s |" % (cid, title, " | ".join(cells)))
    return "\n".join(out)


def perf_tables(runs):
    out = []
    groups = {}
    for r in runs:
        for p in r["points"]:
            key = (p.get("workload"), p.get("model"), p.get("test", ""), p.get("load"), p.get("cotenancy"))
            groups.setdefault(key, []).append((r["name"], p))
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        workload, model, test, load, cot = key
        out.append("\n### %s · %s%s · %s · %s\n" % (workload, model, (" · " + test) if test else "", load, cot))
        out.append("| Run | GPU | Mode | VRAM GiB | Share | Throughput | vs 100% | vs native | TTFT p95 ms | TPOT/lat p50 ms | TPOT/lat p99 ms | Max gap ms | Gaps>=0.4s | Err | Verdict | Failed criteria |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows = sorted(groups[key], key=lambda x: (x[0], x[1].get("mode"), x[1].get("vram_limit_gib", 0), -x[1].get("share", 0)))
        for run, p in rows:
            same = [q for rn, q in rows if rn == run and q.get("mode") == p.get("mode") and q.get("vram_limit_gib") == p.get("vram_limit_gib") and q.get("share", 0) >= 0.999]
            native = [q for rn, q in rows if rn == run and q.get("mode") == "native"]
            base = same[0].get("throughput") if same else None
            nat = native[0].get("throughput") if native else None
            thr = p.get("throughput")
            tail = p.get("tpot_s") if (p.get("tpot_s") or {}).get("n") else p.get("latency_s") or {}
            gap = ms(p, "itl_s", "max") or ms(p, "latency_s", "max")
            failed = [v["criterion"] for v in p.get("verdicts") or [] if not v["pass"]]
            verdict = "skipped" if p.get("skipped") else ("ERROR" if p.get("error") else ("pass" if p.get("pass") else ("FAIL" if p.get("pass") is False else "–")))
            out.append("| %s | %s | %s | %s | %.0f%% | %s %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                run, p.get("gpu_model") or p.get("exec_mode", ""), p.get("mode"), fmt(p.get("vram_limit_gib"), "{:.0f}"), p.get("share", 0) * 100,
                fmt(thr), p.get("throughput_unit", ""),
                fmt(thr / base if thr and base else None, "{:.2f}"), fmt(thr / nat if thr and nat else None, "{:.2f}"),
                fmt(ms(p, "ttft_s", "p95"), "{:.0f}"), fmt((tail.get("p50") or 0) * 1000 or None, "{:.2f}"), fmt((tail.get("p99") or 0) * 1000 or None, "{:.2f}"),
                fmt(gap, "{:.0f}"), fmt(p.get("gaps_ge_400ms"), "{:.0f}"), fmt(p.get("error_rate"), "{:.3f}"), verdict,
                ", ".join(failed) or (p.get("error") or p.get("skipped") or "")[:80]))
    return "\n".join(out)


def finops_table(runs):
    rows = [(r["name"], f) for r in runs for f in r["finops"]]
    if not rows:
        return ""
    out = ["\n## FinOps: fractional workloads vs whole-GPU baseline\n",
           "| Run | Mode | Workloads | Priced as | Fractional nodes | node-h | $ | Whole-GPU nodes | node-h | $ | Saving |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, f in rows:
        a, b = f["fractional"], f["whole_gpu"]
        saving = (1 - a["usd"] / b["usd"]) * 100 if b["usd"] else None
        out.append("| %s | %s | %d | %s | %d | %.3f | %.4f | %d | %.3f | %.4f | %s |" % (
            run, f.get("mode"), f["workloads"], f.get("priced_as"), a["nodes"], a["node_hours"], a["usd"], b["nodes"], b["node_hours"], b["usd"], fmt(saving, "{:.0f}%")))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    runs = [load_run(r) for r in args.runs]
    out_dir = args.out or args.runs[0]
    os.makedirs(out_dir, exist_ok=True)

    md = ["# TensorFusion e2e report\n"]
    for r in runs:
        s = r["summary"]
        cl = s.get("cluster", {})
        md.append("- **%s**: mode `%s`, k8s %s, operator `%s`, filter `%s`, %s" % (
            r["name"], cl.get("gpu_mode"), cl.get("kubernetes"), cl.get("operator_image"), s.get("label_filter"),
            ", ".join("%d %s" % (v, k) for k, v in sorted(s.get("counts", {}).items()))))
    md.append("\nA case run on mock devices (`mock-gpu`) or as `dry-run` proves the control path only, never hardware behaviour.\n")
    md.append("## Cases\n")
    md.append(cases_table(runs))
    if any(r["points"] for r in runs):
        md.append("\n## Performance vs share\n")
        md.append("`vs 100%` is throughput relative to the same mode at a 100 % share; `vs native` is relative to the run without TensorFusion. "
                  "TPOT is used for LLM serving, per-call latency for the other runtimes.")
        md.append(perf_tables(runs))
    md.append(finops_table(runs))
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    cols = ["run", "case", "exec_mode", "gpu_model", "workload", "model", "test", "mode", "share", "vram_limit_gib", "cotenancy", "load",
            "throughput_unit", "samples", "requests", "window_s", "pass", "failed_criteria", "error"] + [m[0] for m in METRICS]
    series = {}
    with open(os.path.join(out_dir, "perf.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in runs:
            for p in r["points"]:
                row = {"run": r["name"], "failed_criteria": ";".join(v["criterion"] for v in p.get("verdicts") or [] if not v["pass"])}
                for c in cols:
                    if c in row:
                        continue
                    row[c] = p.get(c, "")
                for name, fn, _ in METRICS:
                    v = fn(p)
                    row[name] = "" if v is None else round(v, 4)
                    if v is not None:
                        key = (r["name"], p.get("workload"), p.get("model"), p.get("test", ""), p.get("mode"), p.get("load"), p.get("cotenancy"), p.get("vram_limit_gib"), name)
                        series.setdefault(key, []).append([p.get("share"), round(v, 4), p.get("pass")])
                w.writerow([row[c] for c in cols])
    charts = [{"run": k[0], "workload": k[1], "model": k[2], "test": k[3], "mode": k[4], "load": k[5], "cotenancy": k[6], "vram_limit_gib": k[7],
               "metric": k[8], "points": sorted(v, key=lambda x: x[0] or 0)} for k, v in sorted(series.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]
    with open(os.path.join(out_dir, "perf.json"), "w") as fh:
        json.dump(charts, fh, indent=1)
    print("wrote %s/report.md, perf.csv (%d rows), perf.json (%d series)" % (out_dir, sum(len(r["points"]) for r in runs), len(charts)))


if __name__ == "__main__":
    sys.exit(main())
