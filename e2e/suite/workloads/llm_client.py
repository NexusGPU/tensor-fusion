#!/usr/bin/env python3
"""Load generator for an OpenAI-compatible streaming endpoint (vLLM, or the
stand-in server). Standard library only, so it runs in a bare python image.

Measures serving quality, not just throughput:
  TTFT, TPOT, inter-token latency (ITL) distribution and its largest gap,
  output tokens per second / per minute, request latency, error and timeout
  rates - in a closed loop (fixed concurrency = saturation) or an open loop
  (fixed request rate).

Stall signature: the TensorFusion soft limiter waits up to 1 s for tokens and
refills them in 500 ms steps, so a limiter that starves a workload shows up as
inter-token gaps near 0.5 s / 1 s. Those are counted explicitly.

Prints one line `RESULT {json}` at the end.
"""
import argparse
import http.client
import json
import math
import random
import sys
import threading
import time
from urllib.parse import urlparse

PROMPT_WORDS = ("explain how a gpu scheduler shares one accelerator between several inference "
                "services and why tail latency matters for interactive users ").split()


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summary(values):
    return {"n": len(values), "p50": pct(values, 50), "p95": pct(values, 95), "p99": pct(values, 99),
            "max": max(values) if values else None, "mean": (sum(values) / len(values)) if values else None}


def bimodality(values):
    """Sarle's bimodality coefficient; > 0.555 hints at a bimodal distribution."""
    n = len(values)
    if n < 20:
        return None
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    if m2 == 0:
        return 0.0
    m3 = sum((v - mean) ** 3 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    skew, kurt = m3 / m2 ** 1.5, m4 / m2 ** 2
    return (skew ** 2 + 1) / kurt


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = []

    def add(self, rec):
        with self.lock:
            self.requests.append(rec)


def one_request(args, stats, seq):
    url = urlparse(args.url)
    rng = random.Random(seq)
    words = [rng.choice(PROMPT_WORDS) for _ in range(args.prompt_words)]
    # A unique prefix defeats prefix caching, so every request does real work.
    prompt = "request %d-%d: %s" % (seq, time.time_ns(), " ".join(words))
    body = json.dumps({
        "model": args.model, "stream": True, "max_tokens": args.max_tokens, "temperature": 0.0,
        "ignore_eos": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    })
    rec = {"start": time.time(), "ok": False, "timeout": False, "tokens": 0, "ttft": None, "gaps": [], "error": None}
    t0 = time.perf_counter()
    conn = None
    try:
        conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=args.timeout)
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            rec["error"] = "http %d: %s" % (resp.status, resp.read(200).decode("utf-8", "replace"))
            return
        last = None
        usage_tokens = None
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if chunk.get("usage") and chunk["usage"].get("completion_tokens") is not None:
                usage_tokens = chunk["usage"]["completion_tokens"]
            choices = chunk.get("choices") or []
            if not choices or not (choices[0].get("delta") or {}).get("content"):
                continue
            now = time.perf_counter()
            if last is None:
                rec["ttft"] = now - t0
            else:
                rec["gaps"].append(now - last)
            last = now
            rec["tokens"] += 1
        rec["latency"] = time.perf_counter() - t0
        if usage_tokens:
            rec["usage_tokens"] = usage_tokens
        rec["ok"] = rec["tokens"] > 0
        if not rec["ok"]:
            rec["error"] = "no tokens"
    except (TimeoutError, OSError) as e:
        rec["timeout"] = "timed out" in str(e).lower()
        rec["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        rec["end"] = time.time()
        if conn:
            conn.close()
        stats.add(rec)


def run(args):
    stats = Stats()
    stop_at = time.time() + args.warmup + args.duration
    seq = [0]
    seq_lock = threading.Lock()

    def next_seq():
        with seq_lock:
            seq[0] += 1
            return seq[0]

    threads = []
    if args.mode == "closed":
        def worker():
            while time.time() < stop_at:
                one_request(args, stats, next_seq())
        for _ in range(args.concurrency):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)
    else:  # open loop: constant arrival rate, requests do not wait for each other
        interval = 1.0 / args.rate
        nxt = time.time()
        while time.time() < stop_at:
            t = threading.Thread(target=one_request, args=(args, stats, next_seq()), daemon=True)
            t.start()
            threads.append(t)
            nxt += interval
            time.sleep(max(0.0, nxt - time.time()))
    for t in threads:
        t.join(timeout=args.timeout + 5)

    w0 = stop_at - args.duration
    window = [r for r in stats.requests if r["start"] >= w0]
    ok = [r for r in window if r["ok"]]
    gaps = [g for r in ok for g in r["gaps"]]
    tokens = sum(r["tokens"] for r in ok)
    # Tokens are attributed to the window by request start; the window length is
    # extended to the last completion so in-flight requests do not inflate rates.
    span = (max([r["end"] for r in ok]) - w0) if ok else args.duration
    tpot = [(r["latency"] - r["ttft"]) / (r["tokens"] - 1) for r in ok if r["tokens"] > 1]
    out = {
        "mode": args.mode, "concurrency": args.concurrency, "rate": args.rate,
        "warmup_s": args.warmup, "window_s": args.duration, "span_s": span,
        "requests": len(window), "ok": len(ok),
        "errors": len([r for r in window if not r["ok"] and not r["timeout"]]),
        "timeouts": len([r for r in window if r["timeout"]]),
        "error_rate": (1 - len(ok) / len(window)) if window else 1.0,
        "first_error": next((r["error"] for r in window if r["error"]), None),
        "output_tokens": tokens,
        "tokens_per_s": tokens / span if span > 0 else 0.0,
        "tpm": tokens / span * 60 if span > 0 else 0.0,
        "requests_per_s": len(ok) / span if span > 0 else 0.0,
        "ttft_s": summary([r["ttft"] for r in ok if r["ttft"] is not None]),
        "tpot_s": summary(tpot),
        "itl_s": summary(gaps),
        "latency_s": summary([r["latency"] for r in ok]),
        "itl_bimodality": bimodality(gaps),
        "gaps_ge_400ms": len([g for g in gaps if g >= 0.4]),
        "gaps_ge_900ms": len([g for g in gaps if g >= 0.9]),
        "gaps_near_500ms": len([g for g in gaps if 0.42 <= g <= 0.58]),
        "gaps_near_1s": len([g for g in gaps if 0.9 <= g <= 1.1]),
        "gap_rate_per_1k_tokens": (len([g for g in gaps if g >= 0.4]) * 1000.0 / len(gaps)) if gaps else 0.0,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="model")
    ap.add_argument("--mode", choices=["closed", "open"], default="closed")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rate", type=float, default=2.0, help="requests per second (open loop)")
    ap.add_argument("--warmup", type=float, default=15)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt-words", type=int, default=96)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    res = run(args)
    res["label"] = args.label
    print("RESULT " + json.dumps(res))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
