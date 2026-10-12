#!/usr/bin/env python3
"""ONNX Runtime benchmark (CUDA EP; TensorRT EP when asked and available).

For each model: export with torch.onnx (random weights, see bench_models.py),
create an InferenceSession, warm up, then run a time-boxed closed loop of
`--threads` callers. Reports QPS (samples/s) and per-call latency percentiles,
plus stall counters (calls >= 0.4 s / 0.9 s) for the limiter's stall signature.

One `RESULT {json}` line per model.
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_models  # noqa: E402


def emit(obj):
    print("RESULT " + json.dumps(obj))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="resnet50,minilm")
    ap.add_argument("--provider", choices=["cuda", "tensorrt", "cpu"], default="cuda")
    ap.add_argument("--warmup", type=float, default=10)
    ap.add_argument("--duration", type=float, default=45)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--vram-limit-mib", type=float, default=0, help="skip models whose footprint does not fit")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import numpy as np
    # torch first: it loads the CUDA and cuDNN libraries of the image, which the
    # CUDA execution provider then finds by name.
    import torch
    import onnxruntime as ort

    providers = {"cuda": ["CUDAExecutionProvider"], "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider"], "cpu": ["CPUExecutionProvider"]}[args.provider]
    for name in args.models.split(","):
        spec = bench_models.specs()[name]
        res = {"runtime": "onnxruntime", "provider": args.provider, "model": name, "family": spec["family"], "label": args.label}
        if args.vram_limit_mib and spec["vram_mib"] > args.vram_limit_mib * 0.9:
            emit(dict(res, skipped="needs ~%d MiB, limit %d MiB" % (spec["vram_mib"], args.vram_limit_mib)))
            continue
        try:
            model, example, input_names = bench_models.build(name, args.batch or None)
            path = "/tmp/%s.onnx" % name
            t0 = time.time()
            with torch.no_grad():
                torch.onnx.export(model, example, path, input_names=input_names, output_names=["out"], opset_version=17, dynamo=False)
            res["export_s"] = time.time() - t0
            del model
            t0 = time.time()
            sess = ort.InferenceSession(path, providers=providers)
            res["session_s"] = time.time() - t0
            res["providers_in_use"] = sess.get_providers()
            if args.provider != "cpu" and "CUDAExecutionProvider" not in sess.get_providers():
                emit(dict(res, error="CUDA execution provider not active (fell back to %s)" % sess.get_providers()))
                continue
            feeds = {n: t.numpy() for n, t in zip(input_names, example)}
            batch = example[0].shape[0]

            durations, lock = [], threading.Lock()
            stop_at = time.perf_counter() + args.warmup + args.duration
            measure_from = time.perf_counter() + args.warmup

            def loop():
                local = []
                while True:
                    t = time.perf_counter()
                    if t >= stop_at:
                        break
                    sess.run(None, feeds)
                    d = time.perf_counter() - t
                    if t >= measure_from:
                        local.append(d)
                with lock:
                    durations.extend(local)

            threads = [threading.Thread(target=loop) for _ in range(args.threads)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            calls = len(durations)
            res.update({
                "batch": int(batch), "threads": args.threads, "window_s": args.duration, "calls": calls,
                "qps": calls * batch / args.duration, "calls_per_s": calls / args.duration,
                "latency_s": bench_models.percentiles(durations),
            })
            res.update(bench_models.stall_stats(durations))
            del sess
            emit(res)
        except Exception as e:  # one model failing must not hide the others
            emit(dict(res, error="%s: %s" % (type(e).__name__, str(e)[:300])))
    _ = np


if __name__ == "__main__":
    main()
