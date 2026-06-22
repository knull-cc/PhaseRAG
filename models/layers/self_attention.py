from __future__ import annotations

from math import sqrt

import numpy as np
import torch
from torch import nn


class TriangularCausalMask:
    def __init__(self, batch_size: int, length: int, device: str | torch.device) -> None:
        mask_shape = [batch_size, 1, length, length]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool),
                diagonal=1,
            ).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: float | None = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask,
        tau=None,
        delta=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, query_length, _, dim = queries.shape
        scale = self.scale or 1.0 / sqrt(dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(
                    batch_size,
                    query_length,
                    device=queries.device,
                )
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attention, values)
        if self.output_attention:
            return output.contiguous(), attention
        return output.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
    ) -> None:
        super().__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads

        self.inner_attention = attention
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask,
        tau=None,
        delta=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, query_length, _ = queries.shape
        _, key_length, _ = keys.shape
        head_count = self.n_heads

        queries = self.query_projection(queries).view(
            batch_size,
            query_length,
            head_count,
            -1,
        )
        keys = self.key_projection(keys).view(batch_size, key_length, head_count, -1)
        values = self.value_projection(values).view(
            batch_size,
            key_length,
            head_count,
            -1,
        )

        output, attention = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
        )
        output = output.view(batch_size, query_length, -1)
        return self.out_projection(output), attention
