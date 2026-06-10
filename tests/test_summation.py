"""Tests for deterministic summation."""

import torch
import pytest
from carbon.summation import KahanAccumulator, CompensatedSum, deterministic_sum


class TestKahanAccumulator:
    def test_basic_sum(self):
        acc = KahanAccumulator((1,), dtype=torch.float64)
        for v in [1.0, 2.0, 3.0]:
            acc.add(torch.tensor([v]))
        assert acc.result().item() == pytest.approx(6.0)

    def test_compensates_catastrophic_cancellation(self):
        """Kahan should handle the classic 1e16 + 1 - 1e16 case."""
        acc = KahanAccumulator((1,), dtype=torch.float64)
        acc.add(torch.tensor([1e16]))
        acc.add(torch.tensor([1.0]))
        acc.add(torch.tensor([-1e16]))
        # Naive float64 would lose the 1.0
        assert acc.result().item() == pytest.approx(1.0, abs=1e-6)

    def test_order_independent_with_sort(self):
        """Same elements in different order, sorted before summing."""
        vals_a = torch.randn(1000)
        vals_b = vals_a[torch.randperm(1000)]

        def sorted_kahan_sum(vals):
            sorted_vals, _ = vals.sort()
            acc = KahanAccumulator((1,), dtype=torch.float64)
            for v in sorted_vals:
                acc.add(v.unsqueeze(0).to(torch.float64))
            return acc.result().item()

        assert sorted_kahan_sum(vals_a) == sorted_kahan_sum(vals_b)

    def test_reset(self):
        acc = KahanAccumulator((3,))
        acc.add(torch.ones(3))
        acc.reset()
        assert (acc.result() == 0).all()


class TestCompensatedSum:
    def test_matches_exact_sum(self):
        summer = CompensatedSum(use_kahan=True)
        tensor = torch.randn(100)
        result = summer(tensor, dim=0)
        expected = tensor.to(torch.float64).sum().to(tensor.dtype)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_deterministic_across_orders(self):
        """Same values in different order should give same result."""
        summer = CompensatedSum(use_kahan=True)

        vals = torch.randn(500)
        result1 = summer(vals, dim=0)

        shuffled = vals[torch.randperm(500)]
        result2 = summer(shuffled, dim=0)

        # With sorted Kahan, order shouldn't matter
        assert torch.allclose(result1, result2, atol=1e-10)

    def test_pairwise_sum(self):
        summer = CompensatedSum(use_kahan=False)
        tensor = torch.randn(64)
        result = summer(tensor, dim=0)
        expected = tensor.to(torch.float64).sum().to(tensor.dtype)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_multidimensional(self):
        summer = CompensatedSum(use_kahan=True)
        tensor = torch.randn(8, 16, 32)
        result = summer(tensor, dim=-1)
        assert result.shape == (8, 16)

    def test_pairwise_power_of_two_padding(self):
        """Pairwise sum should handle non-power-of-2 sizes."""
        summer = CompensatedSum(use_kahan=False)
        tensor = torch.randn(37)  # not a power of 2
        result = summer(tensor, dim=0)
        expected = tensor.to(torch.float64).sum().to(tensor.dtype)
        assert torch.allclose(result, expected, atol=1e-4)


class TestFunctionalAPI:
    def test_deterministic_sum(self):
        tensor = torch.randn(10, 20)
        result = deterministic_sum(tensor, dim=-1)
        assert result.shape == (10,)
