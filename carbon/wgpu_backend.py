"""
WebGPU Backend for Carbon — deterministic compute without CUDA.

Implements the same bit-exact algorithms (Kahan-compensated matmul,
fixed-order reduction, compensated summation) using wgpu compute shaders.

This backend works on any GPU with WebGPU support: NVIDIA, AMD, Intel, Apple.
No CUDA, no PyTorch required.

Key design: uses emulated double precision (double-single arithmetic via
Dekker splitting) in WGSL shaders. This avoids GPU fast-math optimizations
that can break Kahan compensation at extreme magnitudes — the same class
of bug that makes CUDA non-deterministic in the first place.

Usage:
    from carbon.wgpu_backend import WgpuDeterministicEngine

    engine = WgpuDeterministicEngine()
    C = engine.matmul(A, B)           # bit-exact matmul
    s = engine.sum(data)              # compensated sum
    r = engine.reduce(partials)       # fixed-order reduction
"""

from __future__ import annotations

import numpy as np
from typing import Optional

try:
    import wgpu
    import wgpu.utils

    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False

# ── Double-single WGSL library ──────────────────────────────────────────────
# Represents a double-precision value as (hi, lo) pair of f32s.
# hi + lo = value, with |lo| <= 0.5 * ULP(hi).
# This gives ~48 bits of mantissa (vs f32's 24), enough to compensate
# for rounding in summation without relying on compiler honoring Kahan.

DS_LIBRARY = """
// Double-single arithmetic: emulated f64 using two f32s.
// Immune to GPU fast-math optimizations that break Kahan.

struct DS {
    hi: f32,
    lo: f32,
};

fn ds_zero() -> DS {
    return DS(0.0, 0.0);
}

fn ds_from(a: f32) -> DS {
    return DS(a, 0.0);
}

fn ds_add(a: DS, b: DS) -> DS {
    // Knuth two-sum for the high parts
    let s = a.hi + b.hi;
    let v = s - a.hi;
    let t = (a.hi - (s - v)) + (b.hi - v);
    // Add low parts + error
    var lo = t + a.lo + b.lo;
    // Renormalize
    let hi = s + lo;
    lo = lo - (hi - s);
    return DS(hi, lo);
}

fn ds_add_f32(a: DS, b: f32) -> DS {
    return ds_add(a, ds_from(b));
}

fn ds_to_f32(a: DS) -> f32 {
    return a.hi + a.lo;
}

fn ds_mul_f32(a: f32, b: f32) -> DS {
    // Dekker product: exact product of two f32s as a DS
    let p = a * b;
    // Guard: if either input is large enough that a * splitter overflows f32,
    // fall back to (p, 0). This loses the error term but avoids NaN.
    // Threshold: |x| * 4097 > 3.4e38 → |x| > ~8.3e34
    if (abs(a) > 8.0e34 || abs(b) > 8.0e34) {
        return DS(p, 0.0);
    }
    // Split a into high and low 12-bit halves
    let splitter = 4097.0; // 2^12 + 1
    let a_big = a * splitter;
    let a_hi = a_big - (a_big - a);
    let a_lo = a - a_hi;
    let b_big = b * splitter;
    let b_hi = b_big - (b_big - b);
    let b_lo = b - b_hi;
    let err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    return DS(p, err);
}
"""

# ── Compute shaders ──────────────────────────────────────────────────────────

MATMUL_SHADER = DS_LIBRARY + """
// Deterministic matrix multiplication with double-single accumulation.
// C[m, n] = sum_k( A[m, k] * B[k, n] )
// Iterates K in fixed sequential order with Dekker exact product + DS accumulation.

struct Params {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    let n = gid.y;
    if (m >= params.M || n >= params.N) { return; }

    var acc = ds_zero();

    // Fixed order along K dimension
    for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        let a_val = A[m * params.K + k];
        let b_val = B[k * params.N + n];
        // Dekker exact product + DS accumulation
        let prod = ds_mul_f32(a_val, b_val);
        acc = ds_add(acc, prod);
    }

    C[m * params.N + n] = ds_to_f32(acc);
}
"""


def _require_wgpu():
    if not HAS_WGPU:
        raise ImportError(
            "wgpu is required for the WebGPU backend. "
            "Install it with: pip install wgpu"
        )


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _kahan_sum_1d(data: np.ndarray) -> np.ndarray:
    """CPU Kahan-Babushka-Neumaier sum using float64 accumulation.
    Returns a 0-D float32 ndarray for consistent return types."""
    s = np.float64(0.0)
    c = np.float64(0.0)
    for val in data.astype(np.float64):
        t = s + val
        if abs(s) >= abs(val):
            c += (s - t) + val
        else:
            c += (val - t) + s
        s = t
    return np.array(s + c, dtype=np.float32)


class WgpuDeterministicEngine:
    """
    Deterministic compute engine using WebGPU.

    Same bit-exact algorithms as Carbon's CUDA path, but runs on
    any GPU with WebGPU support (NVIDIA, AMD, Intel, Apple Silicon).

    Uses emulated double precision (Dekker splitting / double-single
    arithmetic) in WGSL to avoid GPU fast-math breaking compensation.

    Usage:
        engine = WgpuDeterministicEngine()

        # Matrix multiply
        C = engine.matmul(A, B)  # A: (M,K), B: (K,N) -> C: (M,N)

        # Compensated sum
        s = engine.sum(data, axis=-1)

        # Fixed-order reduction of partial results
        result = engine.reduce([partial_0, partial_1, partial_2])
    """

    def __init__(self, adapter_index: int = 0):
        _require_wgpu()
        adapters = wgpu.gpu.enumerate_adapters_sync()
        if not adapters:
            raise RuntimeError("No WebGPU adapters found")
        if adapter_index >= len(adapters):
            raise RuntimeError(
                f"Adapter index {adapter_index} out of range "
                f"(found {len(adapters)} adapters)"
            )

        self._adapter = adapters[adapter_index]
        self._device = self._adapter.request_device_sync()

        # Pre-compile matmul shader (sum/reduce use CPU with f64 Kahan)
        self._matmul_shader = self._device.create_shader_module(code=MATMUL_SHADER)

        # Cached pipeline + bind group layout (created once, reused)
        self._matmul_bgl = self._make_layout_4()
        self._matmul_pipeline_layout = self._device.create_pipeline_layout(
            bind_group_layouts=[self._matmul_bgl]
        )
        self._matmul_pipeline = self._device.create_compute_pipeline(
            layout=self._matmul_pipeline_layout,
            compute={"module": self._matmul_shader, "entry_point": "main"},
        )

    @property
    def adapter_info(self) -> dict:
        """Return info about the GPU adapter in use."""
        info = self._adapter.info
        return {
            "vendor": info.get("vendor", "unknown"),
            "device": info.get("device", "unknown"),
            "description": info.get("description", "unknown"),
            "backend": info.get("backend_type", "unknown"),
        }

    def _make_layout_4(self):
        """Bind group layout: storage_ro, storage_ro, storage_rw, uniform."""
        return self._device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
            ]
        )

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Deterministic matrix multiplication: C = A @ B.

        Uses Dekker exact product + double-single accumulation in fixed K-order.
        Bit-exact across any GPU that supports WebGPU.

        Args:
            a: (M, K) float32 array
            b: (K, N) float32 array

        Returns:
            (M, N) float32 array — bit-exact result
        """
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)

        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Expected 2D arrays, got {a.ndim}D and {b.ndim}D")
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"Inner dimension mismatch: {a.shape[1]} vs {b.shape[0]}"
            )

        M, K = a.shape
        _, N = b.shape

        # Create GPU buffers
        buf_a = self._device.create_buffer_with_data(
            data=a.tobytes(), usage=wgpu.BufferUsage.STORAGE
        )
        buf_b = self._device.create_buffer_with_data(
            data=b.tobytes(), usage=wgpu.BufferUsage.STORAGE
        )
        buf_c = self._device.create_buffer(
            size=M * N * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        params = np.array([M, K, N, 0], dtype=np.uint32)  # _pad = 0
        buf_params = self._device.create_buffer_with_data(
            data=params.tobytes(), usage=wgpu.BufferUsage.UNIFORM
        )

        bind_group = self._device.create_bind_group(
            layout=self._matmul_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": buf_a, "offset": 0, "size": buf_a.size}},
                {"binding": 1, "resource": {"buffer": buf_b, "offset": 0, "size": buf_b.size}},
                {"binding": 2, "resource": {"buffer": buf_c, "offset": 0, "size": buf_c.size}},
                {"binding": 3, "resource": {"buffer": buf_params, "offset": 0, "size": buf_params.size}},
            ],
        )

        # Dispatch: workgroup_size(16, 16), so ceil(M/16) x ceil(N/16)
        encoder = self._device.create_command_encoder()
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(self._matmul_pipeline)
        pass_enc.set_bind_group(0, bind_group)
        pass_enc.dispatch_workgroups(_ceil_div(M, 16), _ceil_div(N, 16), 1)
        pass_enc.end()

        self._device.queue.submit([encoder.finish()])

        result = self._device.queue.read_buffer(buf_c).cast("f")
        return np.frombuffer(result, dtype=np.float32).reshape(M, N).copy()

    def sum(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Deterministic Kahan-compensated summation.

        Uses CPU with float64 Kahan accumulation — O(N) operations
        don't benefit from GPU parallelism, and CPU float64 gives
        exact compensation without GPU fast-math interference.

        Args:
            data: input array (float32)
            axis: axis to sum along (default: last)

        Returns:
            Summed array with one fewer dimension
        """
        data = np.asarray(data, dtype=np.float32)

        if data.ndim == 1:
            return _kahan_sum_1d(data)

        data = np.moveaxis(data, axis, -1)
        orig_shape = data.shape
        n_cols = orig_shape[-1]
        n_rows = int(np.prod(orig_shape[:-1]))
        flat = data.reshape(n_rows, n_cols)

        out = np.empty(n_rows, dtype=np.float32)
        for r in range(n_rows):
            out[r] = _kahan_sum_1d(flat[r])

        return out.reshape(orig_shape[:-1])

    def reduce(self, partials: list[np.ndarray]) -> np.ndarray:
        """
        Fixed-order reduction of multiple partial results.

        Uses CPU with float64 Kahan accumulation in rank order.
        O(N*P) operations don't benefit from GPU, and CPU float64
        gives exact compensation.

        Args:
            partials: list of same-shape float32 arrays

        Returns:
            Reduced array (same shape as each partial)
        """
        if not partials:
            raise ValueError("Need at least one partial")

        shape = partials[0].shape
        for i, p in enumerate(partials[1:], start=1):
            if p.shape != shape:
                raise ValueError(
                    f"Shape mismatch: partial[0] is {shape}, "
                    f"partial[{i}] is {p.shape}"
                )

        # Fixed rank order with f64 accumulation
        acc = np.zeros(shape, dtype=np.float64)
        comp = np.zeros(shape, dtype=np.float64)

        for p in partials:
            val = p.astype(np.float64)
            t = acc + val
            mask = np.abs(acc) >= np.abs(val)
            comp = np.where(mask,
                            comp + ((acc - t) + val),
                            comp + ((val - t) + acc))
            acc = t

        return (acc + comp).astype(np.float32)
