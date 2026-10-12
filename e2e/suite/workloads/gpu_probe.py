#!/usr/bin/env python3
"""GPU probes for the isolation tier. Needs PyTorch with CUDA.

  info                 what the container sees (device count, name, total VRAM, nvidia-smi)
  vram --step-mib N    allocate until it fails; reports how much was obtained
  burn --seconds S     matmul loop; reports work rate and stall statistics, and one
                       line per second so a runtime limit change is visible

Every subcommand ends with `RESULT {json}`.
"""
import argparse
import json
import subprocess
import sys
import time


def emit(obj):
    print("RESULT " + json.dumps(obj))
    sys.stdout.flush()


def smi():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return {"rc": out.returncode, "rows": [[c.strip() for c in l.split(",")] for l in out.stdout.strip().splitlines()],
                "stderr": out.stderr.strip()[:300]}
    except Exception as e:  # nvidia-smi missing or hanging is a finding, not a crash
        return {"rc": -1, "error": str(e)}


def cmd_info(_):
    import torch
    res = {"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(), "devices": [], "nvidia_smi": smi()}
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        res["devices"].append({"index": i, "name": torch.cuda.get_device_name(i), "total_mib": total / 2**20, "free_mib": free / 2**20})
    emit(res)


def cmd_vram(args):
    import torch
    dev = torch.device("cuda:%d" % args.device)
    step = args.step_mib * 2**20
    held, error = [], None
    target = args.target_mib * 2**20
    try:
        while sum(t.numel() for t in held) < target:
            held.append(torch.empty(step, dtype=torch.uint8, device=dev).fill_(1))
    except RuntimeError as e:  # torch.OutOfMemoryError is a RuntimeError
        error = str(e).splitlines()[0][:300]
    got = sum(t.numel() for t in held)
    free, total = torch.cuda.mem_get_info(args.device)
    res = {"allocated_mib": got / 2**20, "target_mib": args.target_mib, "hit_limit": error is not None, "error": error,
           "visible_total_mib": total / 2**20, "visible_free_mib": free / 2**20, "nvidia_smi": smi()}
    if args.hold > 0:
        emit(dict(res, holding=True))
        time.sleep(args.hold)
    emit(res)


def cmd_burn(args):
    import torch
    dev = torch.device("cuda:%d" % args.device)
    n = args.size
    dtype = torch.float16 if args.fp16 else torch.float32
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    flops_per_iter = 2.0 * n ** 3
    for _ in range(3):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    last = start
    sec_start, sec_iters = start, 0
    durations, series = [], []
    measure_from = start + args.warmup
    iters = 0
    while True:
        torch.matmul(a, b)
        torch.cuda.synchronize()
        now = time.perf_counter()
        if now >= measure_from:
            durations.append(now - last)
            iters += 1
        last = now
        sec_iters += 1
        if now - sec_start >= 1.0:
            rate = sec_iters / (now - sec_start)
            series.append(round(rate, 2))
            print("TICK t=%.0f iters_per_s=%.2f tflops=%.2f" % (now - start, rate, rate * flops_per_iter / 1e12))
            sys.stdout.flush()
            sec_start, sec_iters = now, 0
        if now - start >= args.seconds:
            break
    span = last - measure_from
    durations.sort()

    def pct(p):
        return durations[min(len(durations) - 1, int(len(durations) * p / 100.0))] if durations else None

    emit({
        "size": n, "fp16": args.fp16, "seconds": args.seconds, "warmup": args.warmup,
        "iters": iters, "iters_per_s": iters / span if span > 0 else 0.0,
        "tflops": iters / span * flops_per_iter / 1e12 if span > 0 else 0.0,
        "iter_s": {"p50": pct(50), "p95": pct(95), "p99": pct(99), "max": durations[-1] if durations else None},
        "stalls_ge_400ms": len([d for d in durations if d >= 0.4]),
        "stalls_ge_900ms": len([d for d in durations if d >= 0.9]),
        "series_iters_per_s": series,
    })


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    v = sub.add_parser("vram")
    v.add_argument("--target-mib", type=int, required=True)
    v.add_argument("--step-mib", type=int, default=256)
    v.add_argument("--device", type=int, default=0)
    v.add_argument("--hold", type=float, default=0, help="keep the memory for this many seconds (noisy neighbour)")
    b = sub.add_parser("burn")
    b.add_argument("--seconds", type=float, default=30)
    b.add_argument("--warmup", type=float, default=5)
    b.add_argument("--size", type=int, default=2048)
    b.add_argument("--fp16", action="store_true")
    b.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    {"info": cmd_info, "vram": cmd_vram, "burn": cmd_burn}[args.cmd](args)


if __name__ == "__main__":
    main()
