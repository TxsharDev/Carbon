"""Tests for the WebGPU deterministic backend."""

import numpy as np
import pytest

try:
    from carbon.wgpu_backend import WgpuDeterministicEngine, HAS_WGPU
except ImportError:
    HAS_WGPU = False

pytestmark = pytest.mark.skipif(not HAS_WGPU, reason="wgpu not installed")


@pytest.fixture(scope="module")
def engine():
    return WgpuDeterministicEngine()


class TestWgpuMatMul:
    def test_correct_result(self, engine):
        """WebGPU matmul should match numpy f64 reference."""
        np.random.seed(42)
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(8, 6).astype(np.float32)

        result = engine.matmul(A, B)
        expected = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)

        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_reproducible(self, engine):
        """Same inputs must produce identical outputs every time."""
        np.random.seed(42)
        A = np.random.randn(32, 64).astype(np.float32)
        B = np.random.randn(64, 32).astype(np.float32)

        r1 = engine.matmul(A, B)
        r2 = engine.matmul(A, B)

        np.testing.assert_array_equal(r1, r2)

    def test_shape_validation(self, engine):
        """Inner dimension mismatch should raise."""
        A = np.zeros((4, 8), dtype=np.float32)
        B = np.zeros((7, 6), dtype=np.float32)

        with pytest.raises(ValueError, match="Inner dimension mismatch"):
            engine.matmul(A, B)

    def test_dimension_validation(self, engine):
        """Non-2D inputs should raise."""
        A = np.zeros((4,), dtype=np.float32)
        B = np.zeros((4, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected 2D"):
            engine.matmul(A, B)

    def test_large_matmul(self, engine):
        """Larger matmul to exercise tiling."""
        np.random.seed(123)
        A = np.random.randn(64, 256).astype(np.float32)
        B = np.random.randn(256, 64).astype(np.float32)

        result = engine.matmul(A, B)
        expected = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)

        np.testing.assert_allclose(result, expected, atol=1e-4)
        assert result.shape == (64, 64)

    def test_square_matmul(self, engine):
        """Square matrix multiply."""
        np.random.seed(99)
        A = np.random.randn(16, 16).astype(np.float32)
        B = np.random.randn(16, 16).astype(np.float32)

        result = engine.matmul(A, B)
        assert result.shape == (16, 16)

        # Verify reproducibility
        r2 = engine.matmul(A, B)
        np.testing.assert_array_equal(result, r2)

    def test_large_values_no_nan(self, engine):
        """Large values should not produce NaN (Dekker overflow guard)."""
        np.random.seed(77)
        A = np.array([[1e30, -1e30], [1e35, 1e35]], dtype=np.float32)
        B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        result = engine.matmul(A, B)
        assert not np.any(np.isnan(result)), f"NaN in result: {result}"
        assert result.shape == (2, 2)


class TestWgpuSum:
    def test_basic_sum(self, engine):
        """1D sum should match numpy."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = engine.sum(data)
        assert abs(result - 15.0) < 1e-6

    def test_reproducible(self, engine):
        """Same data must produce identical sum."""
        np.random.seed(42)
        data = np.random.randn(1000).astype(np.float32)

        r1 = engine.sum(data)
        r2 = engine.sum(data)

        assert r1 == r2

    def test_compensates_catastrophic_cancellation(self, engine):
        """Kahan should handle 1e8 + 1 - 1e8."""
        data = np.array([1e8, 1.0, -1e8], dtype=np.float32)
        result = engine.sum(data)
        assert abs(result - 1.0) < 1e-3

    def test_2d_sum_last_axis(self, engine):
        """Sum along last axis of 2D array."""
        np.random.seed(42)
        data = np.random.randn(8, 16).astype(np.float32)

        result = engine.sum(data, axis=-1)
        expected = data.astype(np.float64).sum(axis=-1).astype(np.float32)

        assert result.shape == (8,)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_3d_sum(self, engine):
        """Sum along last axis of 3D array."""
        np.random.seed(42)
        data = np.random.randn(4, 8, 16).astype(np.float32)

        result = engine.sum(data, axis=-1)
        expected = data.astype(np.float64).sum(axis=-1).astype(np.float32)

        assert result.shape == (4, 8)
        np.testing.assert_allclose(result, expected, atol=1e-4)


class TestWgpuReduce:
    def test_basic_reduce(self, engine):
        """Reduce two arrays element-wise."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        result = engine.reduce([a, b])
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)

        np.testing.assert_array_almost_equal(result, expected)

    def test_reproducible(self, engine):
        """Same partials must produce identical reduction."""
        np.random.seed(42)
        partials = [np.random.randn(100).astype(np.float32) for _ in range(4)]

        r1 = engine.reduce(partials)
        r2 = engine.reduce(partials)

        np.testing.assert_array_equal(r1, r2)

    def test_many_partials(self, engine):
        """Reduce many partials in fixed order."""
        np.random.seed(42)
        partials = [np.random.randn(50).astype(np.float32) for _ in range(8)]

        result = engine.reduce(partials)
        # Reference: sum in order using numpy f64
        expected = sum(
            p.astype(np.float64) for p in partials
        ).astype(np.float32)

        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_2d_reduce(self, engine):
        """Reduce 2D partial arrays."""
        np.random.seed(42)
        partials = [np.random.randn(4, 8).astype(np.float32) for _ in range(3)]

        result = engine.reduce(partials)
        assert result.shape == (4, 8)

    def test_shape_mismatch_raises(self, engine):
        """Mismatched shapes should raise."""
        a = np.zeros((4,), dtype=np.float32)
        b = np.zeros((5,), dtype=np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            engine.reduce([a, b])

    def test_empty_raises(self, engine):
        """Empty partial list should raise."""
        with pytest.raises(ValueError, match="at least one"):
            engine.reduce([])


class TestWgpuSumExtra:
    def test_sum_returns_ndarray(self, engine):
        """sum() should always return ndarray, even for 1D input."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = engine.sum(data)
        assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
        assert result.shape == ()  # 0-D array

    def test_sum_axis_0(self, engine):
        """Sum along axis 0."""
        np.random.seed(42)
        data = np.random.randn(8, 4).astype(np.float32)
        result = engine.sum(data, axis=0)
        expected = data.astype(np.float64).sum(axis=0).astype(np.float32)
        assert result.shape == (4,)
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestWgpuEngine:
    def test_adapter_info(self, engine):
        """Should return adapter info dict."""
        info = engine.adapter_info
        assert isinstance(info, dict)
        assert "vendor" in info
        assert "backend" in info

    def test_import_guard(self):
        """Importing without wgpu installed should set HAS_WGPU flag."""
        from carbon.wgpu_backend import HAS_WGPU as flag
        # If we got here, wgpu IS installed
        assert flag is True
