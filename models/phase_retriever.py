from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from PhaseRAG.models.phase_memory import RaftPhaseMemory, downsample


class RaftRetriever(nn.Module):
    """RAFT multi-period retrieval producing a real future evidence pattern.

    Matching uses Pearson correlation on offset-normalized, pooled windows. The
    top-m real futures are softmax-aggregated per period and projected to the
    horizon, then summed across periods.
    """

    def __init__(
        self,
        memory: RaftPhaseMemory,
        pred_len: int,
        overlap_span: int,
        top_k: int = 8,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.memory = memory
        self.periods = memory.periods
        self.pred_len = pred_len
        self.overlap_span = overlap_span
        self.top_k = top_k
        self.temperature = temperature

        self.projections = nn.ModuleList(
            [
                nn.Linear(memory.value(idx).size(1), pred_len)
                for idx in range(len(self.periods))
            ]
        )
        self._key_cache: list[torch.Tensor | None] = [None] * len(self.periods)

    def forward(
        self,
        x_hat: torch.Tensor,
        query_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        retrieval = None
        similarity_means = []
        for idx, period in enumerate(self.periods):
            query = downsample(x_hat, period).reshape(x_hat.size(0), -1)
            query = self._pearson_normalize(query)

            key = self._normalized_key(idx, query)
            similarity = query.matmul(key.transpose(0, 1))
            similarity = self._mask_overlap(similarity, query_index)

            top_k = min(self.top_k, self.memory.size)
            top_values, top_indices = torch.topk(similarity, k=top_k, dim=1)
            weights = F.softmax(top_values / self.temperature, dim=1)
            similarity_means.append(top_values.mean())

            value = self.memory.value(idx).to(device=x_hat.device, dtype=x_hat.dtype)
            candidates = value[top_indices]
            v_tilde = (candidates * weights[:, :, None, None]).sum(dim=1)

            projected = self.projections[idx](v_tilde.transpose(1, 2)).transpose(1, 2)
            retrieval = projected if retrieval is None else retrieval + projected

        topk_similarity_mean = torch.stack(similarity_means).mean()
        return retrieval, topk_similarity_mean

    def _mask_overlap(
        self,
        similarity: torch.Tensor,
        query_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if query_index is None:
            return similarity
        start_index = self.memory.start_index.to(query_index.device)
        distance = (query_index[:, None] - start_index[None, :]).abs()
        overlap = distance < self.overlap_span
        return similarity.masked_fill(overlap, float("-inf"))

    def _pearson_normalize(self, rows: torch.Tensor) -> torch.Tensor:
        rows = rows - rows.mean(dim=1, keepdim=True)
        return F.normalize(rows, dim=1, eps=1e-8)

    def _normalized_key(self, idx: int, reference: torch.Tensor) -> torch.Tensor:
        cache = self._key_cache[idx]
        if (
            cache is None
            or cache.device != reference.device
            or cache.dtype != reference.dtype
        ):
            key = self.memory.key(idx).to(device=reference.device, dtype=reference.dtype)
            cache = self._pearson_normalize(key)
            self._key_cache[idx] = cache
        return cache
