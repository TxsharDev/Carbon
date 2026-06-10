"""
Carbon Determinism Proof

The definitive test: train the same model twice. Compare weights.

Without Carbon: different weights every time (even same GPU, same seed).
With Carbon: bit-exact identical weights. Every time. Any GPU.

This is the paper's main result. If these numbers aren't clean, nothing else matters.

Run: CARBON_GPU=0 python benchmarks/prove_determinism.py
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
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RunResult:
    method: str
    run_id: int
    seed: int
    steps: int
    final_loss: float
    weight_hash: str
    grad_hash: str
    time_s: float
    device: str


def model_hash(model: nn.Module) -> str:
    """SHA-256 of all model parameters — the bit-exact fingerprint."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(param.data.cpu().numpy().tobytes())
    return h.hexdigest()[:32]


def grad_hash(model: nn.Module) -> str:
    """SHA-256 of all gradients after a backward pass."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if param.grad is not None:
            h.update(name.encode())
            h.update(param.grad.cpu().numpy().tobytes())
    return h.hexdigest()[:32]


class TinyTransformer(nn.Module):
    """Small transformer for determinism testing. Big enough to exercise
    all non-deterministic ops, small enough to train in seconds."""

    def __init__(self, vocab_size=1000, dim=256, heads=4, layers=4, ff_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=ff_dim,
                dropout=0.0, batch_first=True,
            )
            for _ in range(layers)
        ])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


def make_data(batch_size: int, seq_len: int, vocab_size: int,
              seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic training data."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
    return inputs.to(device), targets.to(device)


def init_model(seed: int, device: str) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Create model + optimizer with deterministic initialization."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = TinyTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def train_standard(seed: int, steps: int, device: str, run_id: int) -> RunResult:
    """Standard PyTorch training — no determinism guarantees."""
    model, optimizer = init_model(seed, device)
    inputs, targets = make_data(32, 64, 1000, seed, device)

    start = time.perf_counter()
    for step in range(steps):
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    elapsed = time.perf_counter() - start

    return RunResult(
        method="standard",
        run_id=run_id,
        seed=seed,
        steps=steps,
        final_loss=loss.item(),
        weight_hash=model_hash(model),
        grad_hash=grad_hash(model),
        time_s=elapsed,
        device=torch.cuda.get_device_name(),
    )


def train_carbon(seed: int, steps: int, device: str, run_id: int) -> RunResult:
    """Carbon deterministic training."""
    import carbon
    carbon.enable(seed=seed, warn=False)

    model, optimizer = init_model(seed, device)
    inputs, targets = make_data(32, 64, 1000, seed, device)

    start = time.perf_counter()
    for step in range(steps):
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    elapsed = time.perf_counter() - start

    carbon.disable()

    return RunResult(
        method="carbon",
        run_id=run_id,
        seed=seed,
        steps=steps,
        final_loss=loss.item(),
        weight_hash=model_hash(model),
        grad_hash=grad_hash(model),
        time_s=elapsed,
        device=torch.cuda.get_device_name(),
    )


def run_proof():
    gpu_id = int(os.environ.get("CARBON_GPU", "0"))
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    device_name = torch.cuda.get_device_name(gpu_id)

    seed = 42
    steps = 50
    num_runs = 5

    print(f"{'='*70}")
    print(f"  CARBON DETERMINISM PROOF")
    print(f"  Device: {device_name}")
    print(f"  Seed: {seed}  Steps: {steps}  Runs: {num_runs}")
    print(f"{'='*70}")

    # --- Standard PyTorch ---
    print(f"\n  Standard PyTorch (no determinism):")
    print(f"  {'run':>4s}  {'loss':>10s}  {'weight hash':>34s}  {'time':>6s}")
    print(f"  {'-'*62}")

    std_results = []
    for i in range(num_runs):
        r = train_standard(seed, steps, device, i)
        std_results.append(r)
        print(f"  {r.run_id:4d}  {r.final_loss:10.6f}  {r.weight_hash}  {r.time_s:5.2f}s")

    std_hashes = set(r.weight_hash for r in std_results)
    std_match = len(std_hashes) == 1

    print(f"\n  Unique weight hashes: {len(std_hashes)}/{num_runs}")
    if std_match:
        print(f"  Result: ALL MATCH (PyTorch was deterministic on this config)")
    else:
        print(f"  Result: DIFFERS — {len(std_hashes)} different weight states from same seed")

    # --- Carbon ---
    print(f"\n  Carbon deterministic training:")
    print(f"  {'run':>4s}  {'loss':>10s}  {'weight hash':>34s}  {'time':>6s}")
    print(f"  {'-'*62}")

    carbon_results = []
    for i in range(num_runs):
        r = train_carbon(seed, steps, device, i)
        carbon_results.append(r)
        print(f"  {r.run_id:4d}  {r.final_loss:10.6f}  {r.weight_hash}  {r.time_s:5.2f}s")

    carbon_hashes = set(r.weight_hash for r in carbon_results)
    carbon_match = len(carbon_hashes) == 1

    print(f"\n  Unique weight hashes: {len(carbon_hashes)}/{num_runs}")
    if carbon_match:
        print(f"  Result: ALL MATCH — bit-exact identical weights across {num_runs} runs")
    else:
        print(f"  Result: DIFFERS — Carbon failed to produce identical weights")

    # --- Overhead ---
    std_avg = sum(r.time_s for r in std_results) / num_runs
    carbon_avg = sum(r.time_s for r in carbon_results) / num_runs
    overhead = carbon_avg / std_avg if std_avg > 0 else 0

    print(f"\n  {'='*70}")
    print(f"  SUMMARY")
    print(f"  {'='*70}")
    print(f"  Standard PyTorch:  {len(std_hashes)} unique hashes from {num_runs} runs")
    print(f"  Carbon:            {len(carbon_hashes)} unique hash from {num_runs} runs")
    print(f"  Overhead:          {overhead:.2f}x ({carbon_avg:.2f}s vs {std_avg:.2f}s)")
    print(f"  Verdict:           {'DETERMINISM PROVEN' if carbon_match else 'FAILED'}")

    # Save
    all_results = [asdict(r) for r in std_results + carbon_results]
    out_path = Path(__file__).parent / "determinism_proof.json"
    with open(out_path, "w") as f:
        json.dump({
            "device": device_name,
            "seed": seed,
            "steps": steps,
            "num_runs": num_runs,
            "standard_unique_hashes": len(std_hashes),
            "carbon_unique_hashes": len(carbon_hashes),
            "carbon_deterministic": carbon_match,
            "overhead": overhead,
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_proof()
