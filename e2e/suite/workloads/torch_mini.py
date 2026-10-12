#!/usr/bin/env python3
"""TorchBench-style micro suite in PyTorch (eval, and train where cheap).

Two implementations behind one output format:

  --impl mini  (default)  the model classes TorchBench wraps (torchvision resnet50 /
                          mobilenet_v2, HF BERT / GPT-2, plus a launch-bound MLP in
                          the role of `drq`), random weights, no downloads. This is
                          a representative subset, NOT the pytorch/benchmark harness.
  --impl torchbench       the real harness: expects a pytorch/benchmark checkout at
                          --torchbench-dir with the models installed, and runs
                          `run.py <model> -d cuda -t <test>` per model (perf-full
                          on the bare-metal host; not verified in the rehearsal).

One `RESULT {json}` line per model x test with iterations/s and per-iteration
latency percentiles.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_models  # noqa: E402

# TorchBench names for the same architectures.
TORCHBENCH_NAMES = {"resnet50": "resnet50", "mobilenet_v2": "mobilenet_v2", "bert_base": "hf_Bert", "gpt2": "hf_GPT2", "tiny_mlp": "drq"}


def emit(obj):
    print("RESULT " + json.dumps(obj))
    sys.stdout.flush()


def run_mini(args, name, test, device):
    import torch
    spec = bench_models.specs()[name]
    res = {"runtime": "pytorch", "impl": "mini", "model": name, "family": spec["family"], "test": test, "label": args.label}
    if args.vram_limit_mib and spec["vram_mib"] * (2 if test == "train" else 1) > args.vram_limit_mib * 0.9:
        return dict(res, skipped="does not fit %d MiB" % args.vram_limit_mib)
    model, example, _ = bench_models.build(name)
    model = model.to(device)
    example = tuple(t.to(device) for t in example)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    if test == "train":
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=1e-4)

        def step():
            opt.zero_grad(set_to_none=True)
            out = model(*example)
            out.float().mean().backward()
            opt.step()
    else:
        def step():
            with torch.no_grad():
                model(*example)
    durations = []
    start = time.perf_counter()
    while True:
        t = time.perf_counter()
        step()
        sync()
        now = time.perf_counter()
        if now - start >= args.warmup:
            durations.append(now - t)
        if now - start >= args.warmup + args.duration:
            break
    res.update({
        "batch": int(example[0].shape[0]), "window_s": args.duration, "iters": len(durations),
        "iters_per_s": len(durations) / args.duration, "samples_per_s": len(durations) * int(example[0].shape[0]) / args.duration,
        "latency_s": bench_models.percentiles(durations),
    })
    res.update(bench_models.stall_stats(durations))
    if device.type == "cuda":
        res["max_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def run_torchbench(args, name, test):
    tb = TORCHBENCH_NAMES.get(name, name)
    res = {"runtime": "pytorch", "impl": "torchbench", "model": tb, "test": test, "label": args.label}
    cmd = [sys.executable, "run.py", tb, "-d", "cuda", "-t", test]
    try:
        out = subprocess.run(cmd, cwd=args.torchbench_dir, capture_output=True, text=True, timeout=args.duration * 20 + 600)
    except Exception as e:
        return dict(res, error=str(e)[:300])
    text = out.stdout + out.stderr
    m = re.search(r"GPU Time per batch:\s+([\d.]+)\s+milliseconds", text)
    w = re.search(r"CPU Wall Time per batch:\s+([\d.]+)\s+milliseconds", text)
    if out.returncode != 0 or not w:
        return dict(res, error="run.py rc=%d: %s" % (out.returncode, text[-300:]))
    wall = float(w.group(1)) / 1000.0
    return dict(res, iters_per_s=1.0 / wall, latency_s={"mean": wall}, gpu_time_per_batch_s=float(m.group(1)) / 1000.0 if m else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="resnet50,bert_base,tiny_mlp")
    ap.add_argument("--tests", default="eval")
    ap.add_argument("--impl", choices=["mini", "torchbench"], default="mini")
    ap.add_argument("--torchbench-dir", default="/workspace/benchmark")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=float, default=5)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--vram-limit-mib", type=float, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    device = None
    if args.impl == "mini":
        import torch
        device = torch.device(args.device)
    for name in args.models.split(","):
        for test in args.tests.split(","):
            try:
                emit(run_mini(args, name, test, device) if args.impl == "mini" else run_torchbench(args, name, test))
            except Exception as e:
                emit({"runtime": "pytorch", "impl": args.impl, "model": name, "test": test, "label": args.label, "error": "%s: %s" % (type(e).__name__, str(e)[:300])})


if __name__ == "__main__":
    main()
