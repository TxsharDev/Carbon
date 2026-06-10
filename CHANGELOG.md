# Changelog

## v0.1.0 — 2026-06-10

First release. Same bits, any GPU.

### What's in it
- **Kahan compensated summation** — O(eps) error instead of O(n*eps), canonical sorted order
- **Deterministic tiled matmul** — fixed reduction order, Kahan accumulation, original-matmul recursion guard
- **Deterministic allreduce** — AllGather + local reduce in rank order (for multi-GPU)
- **Global enable/disable** — `carbon.enable()` patches torch.matmul, torch.mm, torch.bmm, scatter ops
- **DeterministicLinear** — drop-in nn.Linear replacement
- **DeterministicTrainer** — training wrapper with checkpoint hashing + verification

### Proven
- RTX 4090 vs RTX 5090: **bit-exact identical weights** after 50 and 200 training steps
- Optimizer state (AdamW moments): also bit-exact across GPUs
- Standard PyTorch on same test: different weights every time across GPUs

### Known gaps
- Overhead ~1.07x single-GPU same-architecture (PyTorch deterministic mode handles most of it)
- Cross-GPU determinism uses tiled matmul — overhead depends on model size (measured ~1x on toy model, expect ~2-3x at scale)
- Cross-GPU overhead not yet measured at scale
- No fp16/bf16 mixed precision support yet
- Attention uses manual QKV matmul (no SDPA) for cross-GPU determinism
- Multi-node distributed training not tested yet
