from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from PhaseRAG.models.phase_tokenizer import PhaseTokenizer


class PhaseMemoryBank(nn.Module):
    """Tensor memory bank built only from the training split."""

    def __init__(
        self,
        key_phase: torch.Tensor,
        future_phase: torch.Tensor,
        residual_phase: torch.Tensor,
    ) -> None:
        super().__init__()
        self._validate_shapes(key_phase, future_phase, residual_phase)
        self.register_buffer("key_phase", key_phase.detach(), persistent=False)
        self.register_buffer("future_phase", future_phase.detach(), persistent=False)
        self.register_buffer("residual_phase", residual_phase.detach(), persistent=False)

    @property
    def size(self) -> int:
        return int(self.key_phase.size(0))

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        tokenizer: PhaseTokenizer,
        pred_len: int,
        batch_size: int = 256,
        stride: int = 1,
        max_items: int | None = None,
        num_workers: int = 0,
        device: str | torch.device = "cpu",
    ) -> "PhaseMemoryBank":
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if max_items is not None and max_items <= 0:
            raise ValueError("max_items must be positive when provided")

        cls._ensure_train_split(dataset)
        memory_dataset = cls._build_subset(dataset, stride, max_items)
        loader = DataLoader(
            memory_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        key_batches = []
        future_batches = []
        residual_batches = []

        tokenizer = tokenizer.to(device)
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(device).float()
                batch_y = batch[1].to(device).float()
                target = batch_y[:, -pred_len:, :]

                key_phase = tokenizer.to_phase(batch_x)
                future_phase = tokenizer.to_phase(target)
                baseline_phase = tokenizer.continuation_baseline(
                    key_phase,
                    future_phase.size(-1),
                )

                key_batches.append(key_phase.cpu())
                future_batches.append(future_phase.cpu())
                residual_batches.append((future_phase - baseline_phase).cpu())

        if not key_batches:
            raise ValueError("cannot build a memory bank from an empty dataset")

        return cls(
            key_phase=torch.cat(key_batches, dim=0),
            future_phase=torch.cat(future_batches, dim=0),
            residual_phase=torch.cat(residual_batches, dim=0),
        )

    @staticmethod
    def _validate_shapes(
        key_phase: torch.Tensor,
        future_phase: torch.Tensor,
        residual_phase: torch.Tensor,
    ) -> None:
        if key_phase.dim() != 4:
            raise ValueError("key_phase must have shape [N, C, L_phase, P_in]")
        if future_phase.dim() != 4:
            raise ValueError("future_phase must have shape [N, C, L_phase, P_out]")
        if residual_phase.shape != future_phase.shape:
            raise ValueError("residual_phase must match future_phase shape")
        if key_phase.size(0) != future_phase.size(0):
            raise ValueError("key_phase and future_phase must have the same N")
        if key_phase.size(1) != future_phase.size(1):
            raise ValueError("key_phase and future_phase must have the same C")
        if key_phase.size(2) != future_phase.size(2):
            raise ValueError("key_phase and future_phase must share L_phase")

    @staticmethod
    def _ensure_train_split(dataset: Dataset) -> None:
        split_id = getattr(dataset, "set_type", None)
        if split_id is not None and int(split_id) != 0:
            raise ValueError("PhaseMemoryBank must be built from the train split")

    @staticmethod
    def _build_subset(
        dataset: Dataset,
        stride: int,
        max_items: int | None,
    ) -> Dataset:
        indices = list(range(0, len(dataset), stride))
        if max_items is not None:
            indices = indices[:max_items]
        return Subset(dataset, indices)


def build_phase_memory_bank(
    dataset: Dataset,
    tokenizer: PhaseTokenizer,
    pred_len: int,
    batch_size: int = 256,
    stride: int = 1,
    max_items: int | None = None,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
) -> PhaseMemoryBank:
    return PhaseMemoryBank.from_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        pred_len=pred_len,
        batch_size=batch_size,
        stride=stride,
        max_items=max_items,
        num_workers=num_workers,
        device=device,
    )
