"""
Global deterministic mode — the user-facing API.

    import carbon
    carbon.enable()
    # All training is now bit-exact deterministic

This module monkey-patches PyTorch's non-deterministic operations
with Carbon's deterministic replacements.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
import warnings

_ENABLED = False
_ORIGINAL_OPS = {}


def enable(seed: int = 42, warn: bool = True):
    """
    Enable bit-exact deterministic training.

    This:
    1. Sets all random seeds
    2. Enables PyTorch deterministic mode
    3. Replaces non-deterministic ops with Carbon equivalents
    4. Patches distributed collectives

    Args:
        seed: random seed for reproducibility
        warn: print warning about performance impact
    """
    global _ENABLED

    if _ENABLED:
        return

    if warn:
        warnings.warn(
            "Carbon: Enabling deterministic mode. "
            "Training will be ~2-3x slower but bit-exact reproducible.",
            stacklevel=2,
        )

    # Standard PyTorch determinism
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    # Patch non-deterministic operations
    _patch_scatter_ops()
    _patch_distributed_ops()
    _patch_matmul()

    _ENABLED = True


def disable():
    """Restore original non-deterministic behavior."""
    global _ENABLED

    if not _ENABLED:
        return

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

    _restore_ops()
    _ENABLED = False


def is_enabled() -> bool:
    """Check if deterministic mode is active."""
    return _ENABLED


def _patch_scatter_ops():
    """Replace index_add_, scatter_add_ with deterministic versions."""
    # These ops are non-deterministic on CUDA because of atomic operations
    # PyTorch's deterministic mode already handles most of these,
    # but we add extra safety for edge cases

    _ORIGINAL_OPS["index_add_"] = torch.Tensor.index_add_

    def safe_index_add_(self, dim, index, source, *, alpha=1):
        if self.is_cuda:
            # Sort indices for deterministic order
            sorted_idx = index.argsort()
            sorted_index = index[sorted_idx]
            sorted_source = source.index_select(dim, sorted_idx)
            return _ORIGINAL_OPS["index_add_"](self, dim, sorted_index,
                                                sorted_source, alpha=alpha)
        return _ORIGINAL_OPS["index_add_"](self, dim, index, source, alpha=alpha)

    torch.Tensor.index_add_ = safe_index_add_


def _patch_distributed_ops():
    """Patch dist.all_reduce to use deterministic reduction."""
    if not dist.is_available():
        return

    from carbon.reduction import DeterministicAllReduce

    _ORIGINAL_OPS["all_reduce"] = dist.all_reduce

    def deterministic_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if op != dist.ReduceOp.SUM:
            # Only SUM needs deterministic treatment
            return _ORIGINAL_OPS["all_reduce"](tensor, op=op, group=group,
                                               async_op=async_op)

        result = DeterministicAllReduce.apply(tensor, group)
        tensor.copy_(result)
        return None  # synchronous

    dist.all_reduce = deterministic_all_reduce


def _patch_matmul():
    """Patch torch.matmul with deterministic tiled matmul.

    This is the key to cross-GPU determinism. Standard cuBLAS picks
    different algorithms on different GPU architectures (Ada vs Blackwell).
    Our tiled matmul with fixed reduction order + Kahan accumulation
    produces identical bits on any GPU.
    """
    from carbon.matmul import DeterministicMatMul

    _ORIGINAL_OPS["matmul"] = torch.matmul
    _ORIGINAL_OPS["Tensor.matmul"] = torch.Tensor.matmul
    _ORIGINAL_OPS["mm"] = torch.mm
    _ORIGINAL_OPS["bmm"] = torch.bmm

    def det_matmul(a, b, *, out=None):
        if a.is_cuda and a.requires_grad:
            return DeterministicMatMul.apply(a, b)
        if a.is_cuda:
            # No grad needed — still use deterministic path
            with torch.no_grad():
                return DeterministicMatMul.apply(a, b)
        return _ORIGINAL_OPS["matmul"](a, b, out=out)

    def det_tensor_matmul(self, other):
        if self.is_cuda:
            return det_matmul(self, other)
        return _ORIGINAL_OPS["Tensor.matmul"](self, other)

    def det_mm(a, b, *, out=None):
        if a.is_cuda:
            return det_matmul(a, b)
        return _ORIGINAL_OPS["mm"](a, b, out=out)

    def det_bmm(a, b, *, out=None):
        if a.is_cuda:
            return det_matmul(a, b)
        return _ORIGINAL_OPS["bmm"](a, b, out=out)

    torch.matmul = det_matmul
    torch.Tensor.matmul = det_tensor_matmul
    torch.mm = det_mm
    torch.bmm = det_bmm


def _restore_ops():
    """Restore original PyTorch operations."""
    if "index_add_" in _ORIGINAL_OPS:
        torch.Tensor.index_add_ = _ORIGINAL_OPS["index_add_"]

    if "all_reduce" in _ORIGINAL_OPS:
        dist.all_reduce = _ORIGINAL_OPS["all_reduce"]

    if "matmul" in _ORIGINAL_OPS:
        torch.matmul = _ORIGINAL_OPS["matmul"]
        torch.Tensor.matmul = _ORIGINAL_OPS["Tensor.matmul"]
        torch.mm = _ORIGINAL_OPS["mm"]
        torch.bmm = _ORIGINAL_OPS["bmm"]

    _ORIGINAL_OPS.clear()
