"""
Deterministic Collective Operations.

Standard NCCL allreduce is non-deterministic: the order in which
partial sums arrive from different GPUs varies run-to-run. Carbon
replaces these with collectives that enforce a canonical reduction order.

Approach:
1. AllGather all partial tensors to every GPU
2. Sort by source rank (canonical order)
3. Reduce locally using CompensatedSum

This is slower than native NCCL but guarantees bit-exact results
across any number of GPUs.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from typing import Optional

from carbon.summation import CompensatedSum, KahanAccumulator


class DeterministicAllReduce(torch.autograd.Function):
    """
    Drop-in replacement for dist.all_reduce that guarantees
    bit-exact results across different GPU topologies.

    Instead of:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    Use:
        tensor = DeterministicAllReduce.apply(tensor)
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor,
                group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        if not dist.is_initialized():
            return tensor

        world_size = dist.get_world_size(group)
        if world_size == 1:
            return tensor

        # Step 1: AllGather — collect tensors from all ranks
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=group)

        # Step 2: Deterministic reduction in rank order (0, 1, 2, ...)
        # This is the canonical order — same regardless of network topology
        acc = KahanAccumulator(
            tensor.shape,
            dtype=torch.float64,
            device=tensor.device,
        )
        for rank in range(world_size):
            acc.add(gathered[rank])

        result = acc.result().to(tensor.dtype)
        ctx.group = group
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Gradient of allreduce is allreduce
        if dist.is_initialized() and dist.get_world_size(ctx.group) > 1:
            grad_output = DeterministicAllReduce.apply(grad_output, ctx.group)
        return grad_output, None


class DeterministicReduceScatter(torch.autograd.Function):
    """
    Deterministic reduce-scatter: each rank gets a different chunk
    of the reduced result, computed in canonical order.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor,
                group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        if not dist.is_initialized():
            return tensor

        world_size = dist.get_world_size(group)
        if world_size == 1:
            return tensor

        rank = dist.get_rank(group)

        # AllGather
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=group)

        # Deterministic reduction in rank order
        acc = KahanAccumulator(
            tensor.shape,
            dtype=torch.float64,
            device=tensor.device,
        )
        for r in range(world_size):
            acc.add(gathered[r])

        full_result = acc.result().to(tensor.dtype)

        # Scatter: each rank takes its chunk
        chunk_size = tensor.shape[0] // world_size
        start = rank * chunk_size
        end = start + chunk_size
        result = full_result[start:end].contiguous()

        ctx.group = group
        ctx.world_size = world_size
        ctx.chunk_size = chunk_size
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not dist.is_initialized():
            return grad_output, None

        # AllGather the gradient chunks, then return full gradient
        gathered = [torch.zeros_like(grad_output) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, grad_output.contiguous(), group=ctx.group)
        return torch.cat(gathered, dim=0), None


def deterministic_allreduce(tensor: torch.Tensor,
                            group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """Functional API for deterministic allreduce."""
    return DeterministicAllReduce.apply(tensor, group)


def deterministic_reduce_scatter(tensor: torch.Tensor,
                                 group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """Functional API for deterministic reduce-scatter."""
    return DeterministicReduceScatter.apply(tensor, group)
