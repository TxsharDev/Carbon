"""Tests for deterministic matrix multiplication."""

import torch
import pytest
from carbon.matmul import DeterministicMatMul, DeterministicLinear


class TestDeterministicMatMul:
    def test_correct_result(self):
        a = torch.randn(4, 8, dtype=torch.float32)
        b = torch.randn(8, 6, dtype=torch.float32)
        result = DeterministicMatMul.apply(a, b)
        expected = a.to(torch.float64) @ b.to(torch.float64)
        assert torch.allclose(result, expected.float(), atol=1e-5)

    def test_batched(self):
        a = torch.randn(2, 4, 8)
        b = torch.randn(2, 8, 6)
        result = DeterministicMatMul.apply(a, b)
        assert result.shape == (2, 4, 6)

    def test_reproducible(self):
        a = torch.randn(32, 64)
        b = torch.randn(64, 32)
        r1 = DeterministicMatMul.apply(a, b)
        r2 = DeterministicMatMul.apply(a, b)
        assert torch.equal(r1, r2)

    def test_gradient_flows(self):
        a = torch.randn(4, 8, requires_grad=True)
        b = torch.randn(8, 6, requires_grad=True)
        result = DeterministicMatMul.apply(a, b)
        loss = result.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_gradient_correctness(self):
        """Gradients should match standard matmul gradients."""
        a = torch.randn(4, 8, requires_grad=True, dtype=torch.float64)
        b = torch.randn(8, 6, requires_grad=True, dtype=torch.float64)

        # Deterministic
        r1 = DeterministicMatMul.apply(a, b)
        r1.sum().backward()
        grad_a_det = a.grad.clone()
        grad_b_det = b.grad.clone()

        a.grad = None
        b.grad = None

        # Standard
        r2 = a @ b
        r2.sum().backward()

        assert torch.allclose(grad_a_det, a.grad, atol=1e-10)
        assert torch.allclose(grad_b_det, b.grad, atol=1e-10)


class TestDeterministicLinear:
    def test_output_shape(self):
        layer = DeterministicLinear(64, 32)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_reproducible(self):
        layer = DeterministicLinear(64, 32)
        x = torch.randn(4, 64)
        r1 = layer(x)
        r2 = layer(x)
        assert torch.equal(r1, r2)

    def test_no_bias(self):
        layer = DeterministicLinear(64, 32, bias=False)
        assert layer.bias is None
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 32)
