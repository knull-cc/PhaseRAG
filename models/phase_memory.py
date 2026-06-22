from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from PhaseRAG.models.phase_tokenizer import offset_normalize


def downsample(series: torch.Tensor, period: int) -> torch.Tensor:
    """Average-pool a [B, T, C] series along time by ``period`` (RAFT pooling)."""
    if period == 1:
        return series
    batch, length, channels = series.shape
    pooled = F.avg_pool1d(
        series.transpose(1, 2),
        kernel_size=period,
        stride=period,
    )
    return pooled.transpose(1, 2)


class RaftPhaseMemory(nn.Module):
    """Multi-period key/value retrieval bank (RAFT).

    For every historical window the *key* is the offset-normalized history and
    the *value* is the offset-normalized **real future** that follows it. Both
    are stored at several pooling periods so retrieval can match local detail
    and coarse trend. ``start_index`` records each entry's window start so the
    retriever can drop patches that overlap the current training query.
    """

    def __init__(
        self,
        periods: tuple[int, ...],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        start_index: torch.Tensor,
    ) -> None:
        super().__init__()
        if not periods:
            raise ValueError("periods must be non-empty")
        if not (len(keys) == len(values) == len(periods)):
            raise ValueError("keys/values must align with periods")

        self.periods = tuple(int(p) for p in periods)
        self.register_buffer("start_index", start_index.detach(), persistent=False)
        for idx, (key, value) in enumerate(zip(keys, values)):
            if key.size(0) != value.size(0) or key.size(0) != start_index.size(0):
                raise ValueError("all tensors must share the same N")
            self.register_buffer(f"key_{idx}", key.detach(), persistent=False)
            self.register_buffer(f"value_{idx}", value.detach(), persistent=False)

    @property
    def size(self) -> int:
        return int(self.start_index.size(0))

    def key(self, idx: int) -> torch.Tensor:
        return getattr(self, f"key_{idx}")

    def value(self, idx: int) -> torch.Tensor:
        return getattr(self, f"value_{idx}")

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        pred_len: int,
        periods: tuple[int, ...] = (1, 2, 4),
        batch_size: int = 256,
        stride: int = 1,
        max_items: int | None = None,
        num_workers: int = 0,
        device: str | torch.device = "cpu",
    ) -> "RaftPhaseMemory":
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        start_positions = list(range(0, len(dataset), stride))
        if max_items is not None:
            start_positions = start_positions[:max_items]
        subset = Subset(dataset, start_positions)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        key_batches: list[list[torch.Tensor]] = [[] for _ in periods]
        value_batches: list[list[torch.Tensor]] = [[] for _ in periods]

        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(device).float()
                batch_y = batch[1].to(device).float()
                target = batch_y[:, -pred_len:, :]

                _, offset = offset_normalize(batch_x)
                key_hat = batch_x - offset
                value_hat = target - offset

                for idx, period in enumerate(periods):
                    key_p = downsample(key_hat, period)
                    value_p = downsample(value_hat, period)
                    key_batches[idx].append(key_p.reshape(key_p.size(0), -1).cpu())
                    value_batches[idx].append(value_p.cpu())

        if not key_batches[0]:
            raise ValueError("cannot build a memory bank from an empty dataset")

        keys = [torch.cat(parts, dim=0) for parts in key_batches]
        values = [torch.cat(parts, dim=0) for parts in value_batches]
        start_index = torch.tensor(start_positions, dtype=torch.long)
        return cls(periods=tuple(periods), keys=keys, values=values, start_index=start_index)


def build_raft_memory(
    dataset: Dataset,
    pred_len: int,
    periods: tuple[int, ...] = (1, 2, 4),
    batch_size: int = 256,
    stride: int = 1,
    max_items: int | None = None,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
) -> RaftPhaseMemory:
    return RaftPhaseMemory.from_dataset(
        dataset=dataset,
        pred_len=pred_len,
        periods=periods,
        batch_size=batch_size,
        stride=stride,
        max_items=max_items,
        num_workers=num_workers,
        device=device,
    )
