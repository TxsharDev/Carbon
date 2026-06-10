<p align="center">
  <h1 align="center">CARBON</h1>
  <p align="center"><i>Exact copy, every time, any hardware</i></p>
  <p align="center">Bit-Exact Deterministic Training Across Heterogeneous Infrastructure</p>
  <p align="center">
    <a href="https://github.com/TxsharDev/carbon">GitHub</a> · <a href="#citation">Paper</a> · <a href="#install">Install</a>
  </p>
</p>

---

> **Why "Carbon"?** A carbon copy is an exact duplicate — zero deviation, every detail preserved. Train on 8 GPUs, train on 64. A100s, H100s, B200s. Carbon: same weights, same gradients, same loss. The name is the guarantee.

---

## The Problem

Training is not reproducible. Same model, same data, different GPU count — different results. Three sources:

1. **Floating-point non-associativity** — `(a+b)+c ≠ a+(b+c)`. Different parallelism = different summation order = different bits.
2. **Non-deterministic CUDA kernels** — cuBLAS picks algorithms at runtime. Atomics race.
3. **NCCL collectives** — allreduce arrival order varies run to run.

Alignment teams can't study what a training change did vs. what was floating-point noise. This blocks interpretability research.

## How Carbon Works

Every non-deterministic op replaced with a deterministic equivalent:

| Operation | Standard | Carbon |
|-----------|----------|--------|
| Summation | Non-associative accumulation | Kahan compensated, canonical sorted order |
| MatMul | cuBLAS (algorithm varies) | Tiled with fixed reduction order + Kahan |
| AllReduce | NCCL (arrival order varies) | AllGather + local reduce in rank order |
| Scatter | Atomic race conditions | Sorted index operations |

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import carbon

carbon.enable(seed=42)

# training is now bit-exact deterministic
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

## Full Wrapper

```python
from carbon import DeterministicTrainer

trainer = DeterministicTrainer(model, optimizer, seed=42)

for batch in dataloader:
    loss = trainer.step(batch, loss_fn=lambda m, b: m(b).loss)

# prove it
assert trainer.verify_determinism(batch, loss_fn)
```

## Overhead

| Component | Cost |
|-----------|------|
| Kahan summation | ~1.5x |
| Deterministic matmul | ~2-3x |
| Deterministic allreduce | ~2x |
| End-to-end | ~2-3x |

For alignment research where reproducibility is non-negotiable — worth it.

## Citation

```bibtex
@article{sharma2025carbon,
  title={Carbon: Bit-Exact Deterministic Training Across Heterogeneous Hardware},
  author={Sharma, Tushar},
  year={2025},
  url={https://github.com/TxsharDev/carbon}
}
```

## License

Apache-2.0 — Alia Labs
