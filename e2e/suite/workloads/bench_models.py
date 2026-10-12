#!/usr/bin/env python3
"""Model zoo shared by ort_bench.py, torch_mini.py and mlperf_mini.py.

Every model is built from its architecture definition with random weights: the
perf tier measures throughput and latency, for which the weight values do not
matter, and it keeps the runs free of dataset / checkpoint downloads (ImageNet
is not freely downloadable; no Hugging Face token is needed).

The set is chosen for different GPU execution patterns, because a limiter that
intercepts kernel launches behaves differently per pattern:

  resnet50      CNN classifier            few large conv kernels per forward
  mobilenet_v2  CNN, depthwise            many small kernels per forward
  bert_base     transformer encoder       medium kernels, attention + GEMM
  minilm        small transformer encoder embedding-model shape, tiny kernels
  gpt2          transformer decoder       causal LM forward (no generation loop)
  whisper_enc   speech encoder            conv1d front-end + transformer
  tiny_mlp      launch-bound canary       ~40 tiny kernels, GPU time dominated by
                                          launch overhead (TorchBench `drq`-like)

`vram_mib` is a rough footprint at the default batch size, used to skip models
that do not fit the VRAM limit of the pod.
"""


def specs():
    return {
        "resnet50": {"family": "cnn-large-kernels", "batch": 16, "vram_mib": 2500},
        "mobilenet_v2": {"family": "cnn-many-small-kernels", "batch": 16, "vram_mib": 1500},
        "bert_base": {"family": "transformer-encoder", "batch": 8, "vram_mib": 2000},
        "minilm": {"family": "transformer-encoder-small", "batch": 16, "vram_mib": 800},
        "gpt2": {"family": "transformer-decoder", "batch": 4, "vram_mib": 2500},
        "whisper_enc": {"family": "speech-encoder", "batch": 2, "vram_mib": 1500},
        "tiny_mlp": {"family": "launch-bound", "batch": 1, "vram_mib": 300},
    }


def build(name, batch=None):
    """Returns (torch.nn.Module in eval mode, example input tuple, input names)."""
    import torch
    from torch import nn
    spec = specs()[name]
    b = batch or spec["batch"]
    if name in ("resnet50", "mobilenet_v2"):
        import torchvision
        model = getattr(torchvision.models, name)(weights=None)
        return model.eval(), (torch.randn(b, 3, 224, 224),), ["input"]
    if name in ("bert_base", "minilm"):
        from transformers import BertConfig, BertModel
        cfg = BertConfig() if name == "bert_base" else BertConfig(hidden_size=384, num_hidden_layers=6, num_attention_heads=12, intermediate_size=1536)
        model = BertModel(cfg, add_pooling_layer=False)
        ids = torch.randint(0, cfg.vocab_size, (b, 128))
        return _Wrap(model).eval(), (ids, torch.ones_like(ids)), ["input_ids", "attention_mask"]
    if name == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config()
        model = GPT2LMHeadModel(cfg)
        model.config.use_cache = False
        return _Wrap(model, key="logits").eval(), (torch.randint(0, cfg.vocab_size, (b, 256)),), ["input_ids"]
    if name == "whisper_enc":
        from transformers import WhisperConfig, WhisperModel
        cfg = WhisperConfig()  # whisper-tiny sized
        model = WhisperModel(cfg).get_encoder()
        return _Wrap(model).eval(), (torch.randn(b, cfg.num_mel_bins, 3000),), ["input_features"]
    if name == "tiny_mlp":
        layers = []
        for _ in range(20):
            layers += [nn.Linear(64, 64), nn.ReLU()]
        return nn.Sequential(*layers).eval(), (torch.randn(b, 64),), ["input"]
    raise SystemExit("unknown model " + name)


class _Wrap:
    """Makes a transformers model return a single tensor (ONNX export friendly)."""

    def __new__(cls, model, key="last_hidden_state"):
        import torch

        class W(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, *args):
                out = self.m(*args)
                return out[key] if isinstance(out, dict) or hasattr(out, "keys") else out[0]
        return W()


def percentiles(durations):
    s = sorted(durations)
    if not s:
        return {"n": 0}

    def p(q):
        return s[min(len(s) - 1, int(len(s) * q / 100.0))]
    return {"n": len(s), "p50": p(50), "p95": p(95), "p99": p(99), "max": s[-1], "mean": sum(s) / len(s)}


def stall_stats(durations):
    return {"stalls_ge_400ms": len([d for d in durations if d >= 0.4]), "stalls_ge_900ms": len([d for d in durations if d >= 0.9])}
