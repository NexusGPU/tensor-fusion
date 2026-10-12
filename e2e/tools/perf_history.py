#!/usr/bin/env python3
"""Keep the performance results of each release in the repository and feed the
local viewer (e2e/perf-history/index.html).

  perf_history.py record --version 2.17 --env ack-us-east-1-t4 [--label "..."] RESULTS_DIR...
      Merge the given e2e result directories (results/<run-id>/, git-ignored)
      into ONE history entry: perf-history/runs/<version>/<env>.json. Recording
      the same version and environment again replaces the entry; the previous
      content stays in git history.
  perf_history.py build
      Regenerate perf-history/data.js from perf-history/runs/**.json. The viewer
      is opened from the file system, where a page cannot fetch JSON, so the
      data is shipped as a script.

An entry holds every case of the runs (status, measurements) and every
performance point (<ID>.perf.json: throughput, TTFT / TPOT / ITL / latency
distributions, stall counts, verdicts, and the device readings of the
measurement window). Node names and GPU UUIDs are dropped: the repository is
public and they identify nothing a later reader needs.
"""

import argparse
import datetime
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.normpath(os.path.join(HERE, "..", "perf-history"))


def load(path):
    with open(path) as f:
        return json.load(f)


def image_tag(image):
    return image.rsplit(":", 1)[1] if ":" in image else ""


def record(args):
    cases, points, sources = {}, [], []
    cluster, started = {}, None
    for d in args.results:
        summary = load(os.path.join(d, "summary.json"))
        sources.append({"run_id": summary["run_id"], "label_filter": summary.get("label_filter", ""),
                        "started": summary.get("started", ""), "seconds": round(summary.get("seconds", 0))})
        # The entry is labelled with the release of the last run given: list the
        # runs oldest first.
        cluster = summary.get("cluster", {}) or cluster
        started = summary.get("started") or started
        for c in summary.get("cases", []):
            if c.get("status") == "skipped" and c["id"] in cases:
                continue
            # A later run of the same case replaces the earlier one (a rerun after a fix).
            cases[c["id"]] = {
                "id": c["id"], "title": c.get("title", ""), "status": c.get("status", ""), "mode": c.get("mode", ""),
                "seconds": round(c.get("seconds", 0)), "run_id": summary["run_id"],
                "measurements": c.get("measurements", []),
                "reason": scrub(c.get("reason") or c.get("failure") or ""),
                "known_issue": c.get("known_issue", ""),
            }
        for f in sorted(glob.glob(os.path.join(d, "*.perf.json"))):
            case_id = os.path.basename(f)[: -len(".perf.json")]
            # Same rule for points: the last run that measured a case wins.
            points[:] = [p for p in points if p.get("case") != case_id]
            for p in load(f) or []:
                m = p.get("machine")
                if m:
                    m.pop("gpu", None)
                    m.pop("node", None)
                p["run_id"] = summary["run_id"]
                points.append(p)

    operator, hypervisor = cluster.get("operator_image", ""), cluster.get("hypervisor_image", "")
    machines = [p["machine"] for p in points if p.get("machine")]
    entry = {
        "version": args.version,
        "env": args.env,
        "label": args.label or args.env,
        "recorded": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": (started or "")[:10],
        "release": {"operator": image_tag(operator), "hypervisor": image_tag(hypervisor),
                    "operator_image": operator, "hypervisor_image": hypervisor},
        "cluster": {
            "kubernetes": cluster.get("kubernetes", ""), "gpu_mode": cluster.get("gpu_mode", ""),
            "karpenter": "alibaba" if cluster.get("karpenter_alibaba") else ("kwok" if cluster.get("karpenter_kwok") else ("yes" if cluster.get("karpenter") else "no")),
            "instance_type": next((m.get("instance_type") for m in machines if m.get("instance_type")), ""),
            "gpu_model": next((p.get("gpu_model") for p in points if p.get("gpu_model")), ""),
        },
        "notes": args.note or [],
        "sources": sources,
        "cases": sorted(cases.values(), key=lambda c: c["id"]),
        "points": points,
    }
    out = os.path.join(HISTORY, "runs", args.version, args.env + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(entry, f, indent=1, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    print("recorded %s: %d cases, %d performance points from %d runs" % (os.path.relpath(out), len(entry["cases"]), len(points), len(sources)))
    build(args)


def scrub(text):
    """Drop node names and GPU UUIDs from free text."""
    text = re.sub(r"gpu-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", "gpu-<uuid>", text)
    return re.sub(r"\b[a-z]{2}-[a-z]+-\d\.\d+\.\d+\.\d+\.\d+\b|\b\d+\.\d+\.\d+\.\d+\b", "<node>", text)


def version_key(v):
    return [int(x) if x.isdigit() else x for x in re.split(r"[.\-]", v)]


def build(_args):
    runs = [load(f) for f in glob.glob(os.path.join(HISTORY, "runs", "*", "*.json"))]
    runs.sort(key=lambda r: (version_key(r["version"]), r["env"]))
    out = os.path.join(HISTORY, "data.js")
    with open(out, "w") as f:
        f.write("// Generated by e2e/tools/perf_history.py build. Do not edit.\n")
        f.write("window.TF_PERF_HISTORY = ")
        json.dump({"runs": runs}, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        f.write(";\n")
    print("built %s: %d entries (%s)" % (os.path.relpath(out), len(runs), ", ".join("%s/%s" % (r["version"], r["env"]) for r in runs)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--version", required=True, help="release line the entry belongs to, e.g. 2.17")
    r.add_argument("--env", required=True, help="environment key, e.g. ack-us-east-1-t4, bare-metal-3090")
    r.add_argument("--label", help="environment name shown in the viewer")
    r.add_argument("--note", action="append", help="free-text note shown with the entry (repeatable)")
    r.add_argument("results", nargs="+", help="result directories (results/<run-id>)")
    r.set_defaults(fn=record)
    b = sub.add_parser("build")
    b.set_defaults(fn=build)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
