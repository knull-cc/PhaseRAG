from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from PhaseRAG.models.phase_tokenizer import PhaseTokenizer, instance_normalize


class PhaseMemoryBank(nn.Module):
    """Tensor memory bank built only from the training split.

    Both the matching key and the stored residual live in the per-window
    instance-normalized space so that neighbours selected by shape similarity
    contribute scale-comparable corrections. When a (frozen) backbone is
    provided, the residual is the backbone's *normalized prediction error*
    (RATF-style), i.e. the part of the future the backbone fails to capture.
    """

    def __init__(
        self,
        key_phase: torch.Tensor,
        residual_phase: torch.Tensor,
    ) -> None:
        super().__init__()
        self._validate_shapes(key_phase, residual_phase)
        self.register_buffer("key_phase", key_phase.detach(), persistent=False)
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
        backbone: nn.Module | None = None,
        batch_size: int = 256,
        stride: int = 1,
        max_items: int | None = None,
        num_workers: int = 0,
        device: str | torch.device = "cpu",
        norm_eps: float = 1e-5,
    ) -> "PhaseMemoryBank":
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if max_items is not None and max_items <= 0:
            raise ValueError("max_items must be positive when provided")

        memory_dataset = cls._build_subset(dataset, stride, max_items)
        loader = DataLoader(
            memory_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        tokenizer = tokenizer.to(device)
        if backbone is not None:
            backbone = backbone.to(device)
            backbone.eval()

        key_batches: list[torch.Tensor] = []
        residual_batches: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(device).float()
                batch_y = batch[1].to(device).float()
                target = batch_y[:, -pred_len:, :]

                x_norm, _, std_x = instance_normalize(batch_x, norm_eps)
                key_phase = tokenizer.to_phase(x_norm)

                residual_time = cls._residual_time(
                    backbone=backbone,
                    batch_x=batch_x,
                    target=target,
                    std_x=std_x,
                    pred_len=pred_len,
                )
                residual_phase = tokenizer.to_phase(residual_time)

                key_batches.append(key_phase.cpu())
                residual_batches.append(residual_phase.cpu())

        if not key_batches:
            raise ValueError("cannot build a memory bank from an empty dataset")

        return cls(
            key_phase=torch.cat(key_batches, dim=0),
            residual_phase=torch.cat(residual_batches, dim=0),
        )

    @staticmethod
    def _residual_time(
        backbone: nn.Module | None,
        batch_x: torch.Tensor,
        target: torch.Tensor,
        std_x: torch.Tensor,
        pred_len: int,
    ) -> torch.Tensor:
        """Normalized residual in time domain, shape [B, pred_len, C]."""
        if backbone is not None:
            output = backbone(x_enc=batch_x)
            y_base = output[0] if isinstance(output, tuple) else output
            y_base = y_base[:, -pred_len:, :]
            return (target - y_base) / std_x

        # Fallback (no backbone): residual against a persistence baseline that
        # repeats the last historical value, still in normalized space.
        last_value = batch_x[:, -1:, :]
        baseline = last_value.expand(-1, pred_len, -1)
        return (target - baseline) / std_x

    @staticmethod
    def _validate_shapes(
        key_phase: torch.Tensor,
        residual_phase: torch.Tensor,
    ) -> None:
        if key_phase.dim() != 4:
            raise ValueError("key_phase must have shape [N, C, L_phase, P_in]")
        if residual_phase.dim() != 4:
            raise ValueError("residual_phase must have shape [N, C, L_phase, P_out]")
        if key_phase.size(0) != residual_phase.size(0):
            raise ValueError("key_phase and residual_phase must have the same N")
        if key_phase.size(1) != residual_phase.size(1):
            raise ValueError("key_phase and residual_phase must have the same C")
        if key_phase.size(2) != residual_phase.size(2):
            raise ValueError("key_phase and residual_phase must share L_phase")

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
    backbone: nn.Module | None = None,
    batch_size: int = 256,
    stride: int = 1,
    max_items: int | None = None,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
    norm_eps: float = 1e-5,
) -> PhaseMemoryBank:
    return PhaseMemoryBank.from_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        pred_len=pred_len,
        backbone=backbone,
        batch_size=batch_size,
        stride=stride,
        max_items=max_items,
        num_workers=num_workers,
        device=device,
        norm_eps=norm_eps,
    )
