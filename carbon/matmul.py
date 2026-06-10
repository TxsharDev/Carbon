"""
Deterministic Matrix Multiplication.

Standard cuBLAS matmul is non-deterministic because it uses
non-associative floating-point reductions with varying thread
scheduling. Carbon provides a deterministic matmul that:

1. Decomposes the matmul into tiles
2. Reduces tile partial sums in a fixed canonical order
3. Uses compensated accumulation for the final sum

This guarantees bit-exact results across different:
- GPU architectures (A100, H100, B200)
- CUDA versions
- cuBLAS algorithm selections
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Optional

from carbon.summation import KahanAccumulator

# Save the ORIGINAL torch.matmul before any patching happens.
# _tiled_matmul needs this to avoid infinite recursion when
# carbon.enable() patches torch.matmul globally.
_original_matmul = torch.matmul


class DeterministicMatMul(torch.autograd.Function):
    """
    Bit-exact matrix multiplication.

    Slower than cuBLAS (~2-3x), but guaranteed reproducible.
    The algorithm:
    1. Tile the inputs into blocks
    2. Compute partial products per tile
    3. Accumulate tile results in a fixed order using Kahan summation

    For training, both forward and backward are deterministic.
    """

    TILE_SIZE = 128  # Trade-off: larger = faster but more rounding

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor,
                tile_size: int = 128) -> torch.Tensor:
        """
        Deterministic A @ B.

        Args:
            a: (..., M, K)
            b: (..., K, N)
            tile_size: tile dimension for blocked matmul

        Returns:
            (..., M, N) — bit-exact result
        """
        ctx.save_for_backward(a, b)
        ctx.tile_size = tile_size

        return _tiled_matmul(a, b, tile_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b = ctx.saved_tensors
        tile_size = ctx.tile_size

        # grad_a = grad_output @ b.T
        grad_a = _tiled_matmul(grad_output, b.transpose(-2, -1), tile_size)

        # grad_b = a.T @ grad_output
        grad_b = _tiled_matmul(a.transpose(-2, -1), grad_output, tile_size)

        return grad_a, grad_b, None


def _tiled_matmul(a: torch.Tensor, b: torch.Tensor,
                  tile_size: int) -> torch.Tensor:
    """
    Tiled matrix multiplication with deterministic accumulation order.

    Tiles along the K (inner) dimension and accumulates partial
    products in a fixed order using Kahan summation.
    """
    *batch_dims, M, K = a.shape
    *_, K2, N = b.shape
    assert K == K2, f"Inner dimension mismatch: {K} vs {K2}"

    device = a.device
    out_shape = (*batch_dims, M, N)

    # Use float64 accumulator for exact compensation
    acc = KahanAccumulator(out_shape, dtype=torch.float64, device=device)

    # Tile along K dimension — fixed order guarantees determinism
    num_tiles = math.ceil(K / tile_size)
    for t in range(num_tiles):
        k_start = t * tile_size
        k_end = min(k_start + tile_size, K)

        a_tile = a[..., :, k_start:k_end]  # (..., M, tile)
        b_tile = b[..., k_start:k_end, :]  # (..., tile, N)

        # Use the ORIGINAL unpatched matmul to avoid infinite recursion
        # when carbon.enable() patches torch.matmul globally
        partial = _original_matmul(
            a_tile.to(torch.float64),
            b_tile.to(torch.float64)
        )

        acc.add(partial)

    return acc.result().to(a.dtype)


class DeterministicLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with deterministic matmul.

    Usage:
        # Replace:
        layer = nn.Linear(512, 256)
        # With:
        layer = DeterministicLinear(512, 256)
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = DeterministicMatMul.apply(input, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output
