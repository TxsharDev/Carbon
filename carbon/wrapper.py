"""
DeterministicTrainer — high-level training wrapper.

Wraps any PyTorch training loop to make it bit-exact deterministic.
Handles model patching, optimizer state, data loading order, and
gradient accumulation in a canonical manner.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from typing import Optional, Callable
import hashlib

from carbon.deterministic import enable, is_enabled
from carbon.matmul import DeterministicLinear


class DeterministicSampler(Sampler):
    """
    Deterministic data sampler that produces identical order
    regardless of num_workers or prefetch settings.
    """

    def __init__(self, dataset_size: int, seed: int = 42, epoch: int = 0):
        self.dataset_size = dataset_size
        self.seed = seed
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=g).tolist()
        return iter(indices)

    def __len__(self):
        return self.dataset_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class DeterministicTrainer:
    """
    Training wrapper that guarantees bit-exact reproducibility.

    Usage:
        trainer = DeterministicTrainer(model, optimizer, seed=42)
        for batch in dataloader:
            loss = trainer.step(batch, loss_fn)
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 seed: int = 42, replace_linear: bool = False,
                 gradient_accumulation_steps: int = 1):
        self.seed = seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._step_count = 0

        if not is_enabled():
            enable(seed=seed)

        if replace_linear:
            model = self._replace_linears(model)

        self.model = model
        self.optimizer = optimizer
        self._initial_hash = self._model_hash()

    def step(self, batch, loss_fn: Callable) -> float:
        self.model.train()
        loss = loss_fn(self.model, batch)
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        self._step_count += 1

        if self._step_count % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()

    def verify_determinism(self, batch, loss_fn: Callable,
                           num_trials: int = 3) -> bool:
        results = []
        for _ in range(num_trials):
            self.model.eval()
            with torch.no_grad():
                loss = loss_fn(self.model, batch)
            results.append(loss.item())
        return all(r == results[0] for r in results)

    def checkpoint(self) -> dict:
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "seed": self.seed,
            "model_hash": self._model_hash(),
        }

    def load_checkpoint(self, ckpt: dict):
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._step_count = ckpt["step_count"]

        current_hash = self._model_hash()
        if current_hash != ckpt["model_hash"]:
            raise RuntimeError(
                f"Checkpoint integrity failed. "
                f"Expected {ckpt['model_hash']}, got {current_hash}"
            )

    def _model_hash(self) -> str:
        hasher = hashlib.sha256()
        for name, param in sorted(self.model.named_parameters()):
            hasher.update(name.encode())
            hasher.update(param.data.cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]

    @staticmethod
    def _replace_linears(model: nn.Module) -> nn.Module:
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                det = DeterministicLinear(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device, dtype=module.weight.dtype,
                )
                det.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    det.bias.data.copy_(module.bias.data)
                setattr(model, name, det)
            else:
                DeterministicTrainer._replace_linears(module)
        return model

    @staticmethod
    def make_dataloader(dataset, batch_size: int, seed: int = 42,
                        num_workers: int = 0, **kwargs) -> DataLoader:
        sampler = DeterministicSampler(len(dataset), seed=seed)

        def seed_worker(worker_id):
            torch.manual_seed(seed + worker_id)

        g = torch.Generator()
        g.manual_seed(seed)

        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, worker_init_fn=seed_worker,
            generator=g, **kwargs,
        )
