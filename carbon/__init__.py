"""
Carbon: Deterministic Deep Learning Training

Achieves bit-exact reproducibility across different GPU counts,
hardware generations, parallelism strategies, and frameworks.
Solves the #1 blocker for alignment and interpretability research.
"""

__version__ = "0.1.0"

from carbon.deterministic import enable, disable, is_enabled
from carbon.reduction import DeterministicAllReduce, DeterministicReduceScatter
from carbon.matmul import DeterministicMatMul
from carbon.summation import CompensatedSum, KahanAccumulator
from carbon.wrapper import DeterministicTrainer

__all__ = [
    "enable",
    "disable",
    "is_enabled",
    "DeterministicAllReduce",
    "DeterministicReduceScatter",
    "DeterministicMatMul",
    "CompensatedSum",
    "KahanAccumulator",
    "DeterministicTrainer",
]
