#!/usr/bin/env python3
"""CPU stand-in for an LLM server: speaks enough of the OpenAI streaming chat
API for llm_client.py, and models a GPU shared through a compute limit.

It exists so the measurement / evaluation / reporting pipeline of the perf tier
can be exercised end to end without a GPU, including the failure modes the
evaluation is supposed to catch:

  --limiter smooth   token rate scales with the share (what a good limiter does)
  --limiter bucket   work is granted in 500 ms refill steps; when the bucket is
                     empty the stream stalls until the next refill (the soft
                     limiter's failure signature: periodic ~0.5 s gaps)
  --limiter none     ignores the share (a limiter that does not limit)
  --stall-every N    additionally inject a --stall-ms pause every N tokens

The share can be changed while running (PUT /share?value=50), like a runtime
limit change. One "GPU": concurrent streams split the token rate.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class GPU:
    def __init__(self, args):
        self.args = args
        self.share = args.share / 100.0
        self.lock = threading.Lock()
        self.active = 0
        self.bucket = 0.0
        self.last_refill = time.monotonic()
        self.count = 0

    def token_time(self):
        """Seconds one token costs for one stream right now."""
        base = 1.0 / self.args.tokens_per_s
        with self.lock:
            active = max(1, self.active)
            self.count += 1
            n = self.count
        share = 1.0 if self.args.limiter == "none" else self.share
        cost = base * active
        if self.args.limiter == "bucket" and share < 1.0:
            # Budget of GPU seconds, refilled every 500 ms; a stream whose token
            # does not fit waits for the next refill.
            while True:
                with self.lock:
                    now = time.monotonic()
                    steps = int((now - self.last_refill) / 0.5)
                    if steps > 0:
                        self.bucket = min(self.bucket + steps * 0.5 * share, 0.5 * share)
                        self.last_refill += steps * 0.5
                    if self.bucket >= base:
                        self.bucket -= base
                        break
                    wait = self.last_refill + 0.5 - now
                time.sleep(max(wait, 0.001))
        else:
            cost = cost / share
        if self.args.stall_every and n % self.args.stall_every == 0:
            cost += self.args.stall_ms / 1000.0
        return cost


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    gpu = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"status": "ok", "share": self.gpu.share * 100, "data": [{"id": "standin"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):
        q = parse_qs(urlparse(self.path).query)
        if "value" in q:
            self.gpu.share = max(0.01, min(1.0, float(q["value"][0]) / 100.0))
        self.do_GET()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        max_tokens = int(req.get("max_tokens", 64))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        gpu = self.gpu
        with gpu.lock:
            gpu.active += 1
        try:
            time.sleep(gpu.args.ttft_ms / 1000.0 / (1.0 if gpu.args.limiter == "none" else gpu.share))
            for i in range(max_tokens):
                time.sleep(gpu.token_time())
                chunk = {"choices": [{"index": 0, "delta": {"content": "tok%d " % i}}]}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
            usage = {"choices": [], "usage": {"completion_tokens": max_tokens}}
            self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\ndata: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with gpu.lock:
                gpu.active -= 1
        self.close_connection = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--share", type=float, default=100, help="compute share in percent")
    ap.add_argument("--tokens-per-s", type=float, default=400, help="unrestricted GPU token rate, all streams together")
    ap.add_argument("--ttft-ms", type=float, default=40)
    ap.add_argument("--limiter", choices=["smooth", "bucket", "none"], default="smooth")
    ap.add_argument("--stall-every", type=int, default=0)
    ap.add_argument("--stall-ms", type=float, default=1000)
    args = ap.parse_args()
    Handler.gpu = GPU(args)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.daemon_threads = True
    print("stand-in LLM server on :%d share=%s%% limiter=%s" % (args.port, args.share, args.limiter), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
