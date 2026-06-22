from __future__ import annotations

import torch
from torch import nn


def offset_normalize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RAFT-style offset normalization: subtract the last timestep.

    Returns ``(x - x_last, x_last)`` for a ``[B, L, C]`` tensor so callers can
    compare shape/trend (ignoring level shifts) and add the offset back later.
    """
    if x.dim() != 3:
        raise ValueError("x must have shape [B, L, C]")
    x_last = x[:, -1:, :]
    return x - x_last, x_last


class PhaseTokenizer(nn.Module):
    """Converts time-domain sequences to phase-period matrices and back."""

    def __init__(self, phase_len: int) -> None:
        super().__init__()
        if phase_len <= 0:
            raise ValueError("phase_len must be positive")
        self.phase_len = phase_len

    def period_count(self, length: int) -> int:
        return (length + self.phase_len - 1) // self.phase_len

    def padded_length(self, length: int) -> int:
        return self.period_count(length) * self.phase_len

    def padding_length(self, length: int) -> int:
        return self.padded_length(length) - length

    def to_phase(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms [B, L, C] into [B, C, L_phase, P]."""
        if x.dim() != 3:
            raise ValueError("x must have shape [B, L, C]")

        batch_size, length, channel_count = x.shape
        series = x.permute(0, 2, 1).contiguous()
        series = self._circular_pad(series, self.padding_length(length))

        period_count = series.size(-1) // self.phase_len
        periods = series.view(batch_size, channel_count, period_count, self.phase_len)
        return periods.permute(0, 1, 3, 2).contiguous()

    def to_time(self, phase: torch.Tensor, horizon: int) -> torch.Tensor:
        """Transforms [B, C, L_phase, P] back into [B, H, C]."""
        if phase.dim() != 4:
            raise ValueError("phase must have shape [B, C, L_phase, P]")
        if phase.size(2) != self.phase_len:
            raise ValueError("phase length does not match tokenizer phase_len")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        batch_size, channel_count, _, _ = phase.shape
        series = phase.permute(0, 1, 3, 2).contiguous()
        series = series.view(batch_size, channel_count, -1)[..., :horizon]
        return series.permute(0, 2, 1).contiguous()

    def continuation_baseline(
        self,
        history_phase: torch.Tensor,
        output_periods: int,
    ) -> torch.Tensor:
        """Repeats the last historical period as a simple future baseline."""
        if history_phase.dim() != 4:
            raise ValueError("history_phase must have shape [B, C, L_phase, P]")
        if output_periods <= 0:
            raise ValueError("output_periods must be positive")

        return history_phase[..., -1:].expand(-1, -1, -1, output_periods)

    def _circular_pad(self, series: torch.Tensor, pad_length: int) -> torch.Tensor:
        if pad_length == 0:
            return series
        if series.size(-1) == 0:
            raise ValueError("cannot circular-pad an empty sequence")

        indices = torch.arange(pad_length, device=series.device) % series.size(-1)
        padding = series.index_select(-1, indices)
        return torch.cat([series, padding], dim=-1)
