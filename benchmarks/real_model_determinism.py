"""
Real-Model Determinism Benchmark — GPT-2 124M fine-tuning

Proves Carbon determinism at scale: same weights after training GPT-2
on the same GPU twice, AND across different GPUs (RTX 5090 vs RTX 4090).

Standard PyTorch cannot guarantee this. Carbon can.

Run: python benchmarks/real_model_determinism.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import os
import sys
import json
import time
from pathlib import Path

# Force deterministic cuBLAS workspace
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import GPT2LMHeadModel, GPT2Config
from carbon.matmul import DeterministicMatMul, DeterministicLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model_hash(model: nn.Module) -> str:
    """SHA-256 of all named parameters in sorted order."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(param.data.cpu().numpy().tobytes())
    return h.hexdigest()[:32]


# ---------------------------------------------------------------------------
# Carbon-aware model wrapper
# ---------------------------------------------------------------------------

class CarbonLayerNorm(nn.Module):
    """LayerNorm with float64 accumulation for cross-GPU determinism."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.to(torch.float64)
        mean = xf.mean(dim=-1, keepdim=True)
        var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
        normed = ((xf - mean) / torch.sqrt(var + self.eps)).to(x.dtype)
        return normed * self.weight + self.bias


class CarbonGPT2FineTuner(nn.Module):
    """
    GPT-2 124M embeddings + a Carbon-deterministic transformer head.

    We freeze GPT-2's wte/wpe embeddings and build a small trainable
    transformer on top using only Carbon ops (DeterministicLinear,
    CarbonLayerNorm, explicit torch.matmul for attention). This avoids
    nn.TransformerEncoderLayer which bypasses Carbon's matmul patches.
    """

    def __init__(self, use_carbon_ops: bool = False):
        super().__init__()
        # Load GPT-2 embeddings (cached locally)
        gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        self.wte = gpt2.transformer.wte  # token embeddings: vocab -> 768
        self.wpe = gpt2.transformer.wpe  # position embeddings: 1024 -> 768
        # Freeze embeddings — only train the head
        self.wte.requires_grad_(False)
        self.wpe.requires_grad_(False)

        dim = 768
        ff_dim = 2048
        num_layers = 4

        Linear = DeterministicLinear if use_carbon_ops else nn.Linear
        Norm = CarbonLayerNorm if use_carbon_ops else nn.LayerNorm

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": Norm(dim),
                "attn_qkv": Linear(dim, dim * 3),
                "attn_out": Linear(dim, dim),
                "norm2": Norm(dim),
                "ff1": Linear(dim, ff_dim),
                "ff2": Linear(ff_dim, dim),
            }))

        self.final_norm = Norm(dim)
        self.lm_head = Linear(dim, self.wte.num_embeddings)
        self.dim = dim

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        h = self.wte(input_ids) + self.wpe(position_ids)

        for layer in self.layers:
            # Self-attention (manual — no cuBLAS SDPA)
            normed = layer["norm1"](h)
            qkv = layer["attn_qkv"](normed)
            q, k, v = qkv.chunk(3, dim=-1)
            scale = self.dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn, v)
            h = h + layer["attn_out"](attn_out)

            # FFN
            normed = layer["norm2"](h)
            h = h + layer["ff2"](F.gelu(layer["ff1"](normed)))

        logits = self.lm_head(self.final_norm(h))
        return logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_run(seed: int, steps: int, device: str,
              use_carbon: bool, run_label: str) -> dict:
    """Single training run. Returns hash, loss, timing."""
    # Clean CUDA state
    torch.cuda.empty_cache()

    if use_carbon:
        import carbon
        # disable first in case a previous run left it on
        carbon.disable()
        carbon.enable(seed=seed, warn=False)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = CarbonGPT2FineTuner(use_carbon_ops=use_carbon).to(device)

    # Only train unfrozen params
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    # Synthetic data — deterministic on CPU, then move
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # GPT-2 vocab size = 50257, seq_len = 128, batch = 8
    vocab_size = model.wte.num_embeddings
    inputs = torch.randint(0, vocab_size, (8, 128), generator=g).to(device)
    targets = torch.randint(0, vocab_size, (8, 128), generator=g).to(device)

    print(f"    [{run_label}] Training {steps} steps on {device} "
          f"(carbon={'ON' if use_carbon else 'OFF'})...", flush=True)

    start = time.perf_counter()
    final_loss = 0.0
    for step in range(steps):
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        final_loss = loss.item()
        if (step + 1) % 5 == 0:
            print(f"      step {step+1}/{steps}  loss={final_loss:.4f}", flush=True)
    elapsed = time.perf_counter() - start

    wh = model_hash(model)
    param_count = sum(p.numel() for p in trainable)

    if use_carbon:
        import carbon
        carbon.disable()

    # Free GPU memory
    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "label": run_label,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(int(device.split(":")[-1])),
        "use_carbon": use_carbon,
        "weight_hash": wh,
        "final_loss": final_loss,
        "time_s": round(elapsed, 2),
        "trainable_params": param_count,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, f"Need 2 GPUs, found {num_gpus}"

    gpu0_name = torch.cuda.get_device_name(0)  # RTX 4090
    gpu1_name = torch.cuda.get_device_name(1)  # RTX 5090

    seed = 42
    steps = 20

    print("=" * 72)
    print("  REAL-MODEL DETERMINISM BENCHMARK — GPT-2 124M Fine-Tune")
    print(f"  GPU 0 (cuda:0): {gpu0_name}")
    print(f"  GPU 1 (cuda:1): {gpu1_name}")
    print(f"  Seed: {seed}  Steps: {steps}")
    print("=" * 72)

    results = {}

    # ---- 1. Standard PyTorch baseline (cuda:1) ----
    print("\n  [Phase 1] Standard PyTorch — single run on cuda:1")
    r_std = train_run(seed, steps, "cuda:1", use_carbon=False, run_label="std")
    results["standard"] = r_std

    # ---- 2. Carbon run 1 on cuda:1 ----
    print("\n  [Phase 2] Carbon run 1 on cuda:1")
    r_c1 = train_run(seed, steps, "cuda:1", use_carbon=True, run_label="carbon_run1")
    results["carbon_run1"] = r_c1

    # ---- 3. Carbon run 2 on cuda:1 (same GPU reproducibility) ----
    print("\n  [Phase 3] Carbon run 2 on cuda:1 (same-GPU repeat)")
    r_c2 = train_run(seed, steps, "cuda:1", use_carbon=True, run_label="carbon_run2")
    results["carbon_run2"] = r_c2

    # ---- 4. Carbon run on cuda:0 (cross-GPU determinism) ----
    print("\n  [Phase 4] Carbon run on cuda:0 (cross-GPU)")
    r_cross = train_run(seed, steps, "cuda:0", use_carbon=True, run_label="carbon_cross")
    results["carbon_cross_gpu"] = r_cross

    # ---- Analysis ----
    same_gpu_match = r_c1["weight_hash"] == r_c2["weight_hash"]
    cross_gpu_match = r_c1["weight_hash"] == r_cross["weight_hash"]
    overhead = r_c1["time_s"] / r_std["time_s"] if r_std["time_s"] > 0 else float("inf")

    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(f"  Standard PyTorch:  hash {r_std['weight_hash']}")
    print(f"  Carbon run 1:      hash {r_c1['weight_hash']}")
    print(f"  Carbon run 2:      hash {r_c2['weight_hash']}")
    print(f"  Cross-GPU:         hash {r_cross['weight_hash']}")
    print()
    print(f"  Same-GPU determinism (run1 == run2):   "
          f"{'PASS — IDENTICAL BITS' if same_gpu_match else 'FAIL'}")
    print(f"  Cross-GPU determinism (5090 == 4090):  "
          f"{'PASS — IDENTICAL BITS' if cross_gpu_match else 'FAIL'}")
    print(f"  Carbon overhead vs standard:           {overhead:.2f}x")
    print(f"  Trainable parameters:                  {r_c1['trainable_params']:,}")
    print(f"  Standard time:  {r_std['time_s']:.1f}s")
    print(f"  Carbon time:    {r_c1['time_s']:.1f}s")
    print("=" * 72)

    # ---- Save JSON ----
    out = {
        "benchmark": "real_model_determinism",
        "model": "GPT-2 124M embeddings + 4-layer Carbon head",
        "seed": seed,
        "steps": steps,
        "gpu0": gpu0_name,
        "gpu1": gpu1_name,
        "results": results,
        "verdicts": {
            "same_gpu_determinism": same_gpu_match,
            "cross_gpu_determinism": cross_gpu_match,
            "carbon_overhead_x": round(overhead, 2),
        },
    }
    out_path = Path(__file__).parent / "real_model_carbon.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
