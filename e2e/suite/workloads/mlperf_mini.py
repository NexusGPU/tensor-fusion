#!/usr/bin/env python3
"""MLPerf-Inference-style run: the official LoadGen (pip: mlcommons-loadgen)
driving an ONNX Runtime system under test, Offline and Server scenarios.

Deliberately reduced and NOT a compliant submission:
  - models have random weights and the "dataset" is synthetic tensors (accuracy
    is meaningless; ImageNet / SQuAD are not downloaded);
  - min_duration is seconds, not the 10 minutes the rules require;
  - the Server target QPS is derived from a short Offline run, not tuned.
What it keeps from MLPerf is LoadGen's scenario semantics: Offline measures
throughput with all samples at once, Server issues queries with Poisson arrivals
at a target rate and judges the 99th-percentile latency.

One `RESULT {json}` line per model x scenario.
"""
import argparse
import array
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_models  # noqa: E402

MODELS = {"resnet50": "resnet50", "bert": "bert_base"}


def emit(obj):
    print("RESULT " + json.dumps(obj))
    sys.stdout.flush()


def parse_summary(path):
    out = {}
    try:
        text = open(path).read()
    except OSError:
        return out
    for key, pat in {
        "samples_per_second": r"Samples per second\s*:\s*([\d.e+]+)",
        "completed_samples_per_second": r"Completed samples per second\s*:\s*([\d.e+]+)",
        "scheduled_samples_per_second": r"Scheduled samples per second\s*:\s*([\d.e+]+)",
        "p50_latency_ns": r"50\.00 percentile latency \(ns\)\s*:\s*(\d+)",
        "p90_latency_ns": r"90\.00 percentile latency \(ns\)\s*:\s*(\d+)",
        "p99_latency_ns": r"99\.00 percentile latency \(ns\)\s*:\s*(\d+)",
        "max_latency_ns": r"Max latency \(ns\)\s*:\s*(\d+)",
    }.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"Result is\s*:\s*(\w+)", text)
    if m:
        out["loadgen_result"] = m.group(1)
    return out


def run_scenario(lg, sess, feeds, batch, scenario, args, target_qps, outdir):
    lock = threading.Lock()

    def issue(samples):
        # One sess.run per `batch` queries; synthetic data, so the index is ignored.
        for i in range(0, len(samples), batch):
            with lock:
                sess.run(None, feeds)
            responses = [lg.QuerySampleResponse(s.id, 0, 0) for s in samples[i:i + batch]]
            lg.QuerySamplesComplete(responses)

    def flush():
        pass

    settings = lg.TestSettings()
    settings.mode = lg.TestMode.PerformanceOnly
    settings.min_duration_ms = int(args.duration * 1000)
    settings.max_duration_ms = int(args.duration * 3000)
    if scenario == "Offline":
        settings.scenario = lg.TestScenario.Offline
        settings.offline_expected_qps = target_qps
        settings.min_query_count = 1
    else:
        settings.scenario = lg.TestScenario.Server
        settings.server_target_qps = target_qps
        settings.server_target_latency_ns = int(args.server_latency_ms * 1e6)
        settings.min_query_count = 10
    os.makedirs(outdir, exist_ok=True)
    log = lg.LogSettings()
    log.log_output.outdir = outdir
    log.log_output.copy_summary_to_stdout = False
    log.enable_trace = False
    sut = lg.ConstructSUT(issue, flush)
    qsl = lg.ConstructQSL(1024, 1024, lambda idx: None, lambda idx: None)
    t0 = time.time()
    lg.StartTestWithLogSettings(sut, qsl, settings, log)
    wall = time.time() - t0
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
    res = parse_summary(os.path.join(outdir, "mlperf_log_summary.txt"))
    res["wall_s"] = wall
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="resnet50")
    ap.add_argument("--scenarios", default="Offline,Server")
    ap.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--duration", type=float, default=30, help="LoadGen min_duration in seconds")
    ap.add_argument("--server-latency-ms", type=float, default=100, help="Server scenario p99 bound")
    ap.add_argument("--server-load", type=float, default=0.6, help="Server target QPS as a fraction of the measured Offline QPS")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import mlperf_loadgen as lg
    # torch first: it loads the CUDA and cuDNN libraries of the image, which the
    # CUDA execution provider then finds by name.
    import torch
    import onnxruntime as ort

    _ = array
    providers = ["CUDAExecutionProvider"] if args.provider == "cuda" else ["CPUExecutionProvider"]
    for key in args.models.split(","):
        name = MODELS.get(key, key)
        base = {"runtime": "mlperf-loadgen+onnxruntime", "model": key, "provider": args.provider, "label": args.label}
        try:
            model, example, input_names = bench_models.build(name, args.batch or None)
            path = "/tmp/mlperf_%s.onnx" % name
            with torch.no_grad():
                torch.onnx.export(model, example, path, input_names=input_names, output_names=["out"], opset_version=17, dynamo=False)
            del model
            sess = ort.InferenceSession(path, providers=providers)
            if args.provider == "cuda" and "CUDAExecutionProvider" not in sess.get_providers():
                emit(dict(base, error="CUDA execution provider not active"))
                continue
            feeds = {n: t.numpy() for n, t in zip(input_names, example)}
            batch = int(example[0].shape[0])
            for _ in range(3):
                sess.run(None, feeds)
            # Calibrate: LoadGen's Offline scenario issues expected_qps x min_duration
            # samples up front, so the estimate decides how long the run takes.
            t0 = time.perf_counter()
            for _ in range(5):
                sess.run(None, feeds)
            estimate = 5.0 * batch / (time.perf_counter() - t0)
            offline_qps = None
            for scenario in args.scenarios.split(","):
                target = estimate if scenario == "Offline" else max(1.0, (offline_qps or estimate) * args.server_load)
                res = run_scenario(lg, sess, feeds, batch, scenario, args, target, "/tmp/mlperf_%s_%s" % (name, scenario))
                if scenario == "Offline":
                    offline_qps = res.get("samples_per_second")
                emit(dict(base, scenario=scenario, batch=batch, target_qps=target, min_duration_s=args.duration, **res))
        except Exception as e:
            emit(dict(base, error="%s: %s" % (type(e).__name__, str(e)[:300])))


if __name__ == "__main__":
    main()
