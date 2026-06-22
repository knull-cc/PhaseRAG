from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from PhaseRAG.models.phase_memory import PhaseMemoryBank
from PhaseRAG.models.phase_tokenizer import PhaseTokenizer


class PhaseRetriever(nn.Module):
    """Retrieves phase-domain residual evidence from a PhaseMemoryBank."""

    def __init__(
        self,
        tokenizer: PhaseTokenizer,
        memory_bank: PhaseMemoryBank,
        top_k: int = 8,
        temperature: float = 0.2,
        similarity: str = "cosine",
        shift_aware: bool = False,
        similarity_threshold: float | None = None,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if similarity not in {"cosine", "pearson"}:
            raise ValueError("similarity must be 'cosine' or 'pearson'")

        self.tokenizer = tokenizer
        self.memory_bank = memory_bank
        self.top_k = top_k
        self.temperature = temperature
        self.similarity = similarity
        self.shift_aware = shift_aware
        self.similarity_threshold = similarity_threshold

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        query_phase = self.tokenizer.to_phase(x)
        similarities, shifts = self._compute_similarities(query_phase)

        top_k = min(self.top_k, self.memory_bank.size)
        top_values, top_indices = torch.topk(similarities, k=top_k, dim=1)
        top_shifts = shifts.gather(dim=1, index=top_indices)

        residual_candidates = self._gather_candidates(
            self.memory_bank.residual_phase,
            top_indices,
            top_shifts,
            x,
        )
        future_candidates = self._gather_candidates(
            self.memory_bank.future_phase,
            top_indices,
            top_shifts,
            x,
        )

        weights = self._weights(top_values)
        weight_view = weights[:, :, None, None, None]
        retrieved_residual = (residual_candidates * weight_view).sum(dim=1)
        retrieved_future = (future_candidates * weight_view).sum(dim=1)

        return {
            "query_phase": query_phase,
            "indices": top_indices,
            "similarities": top_values,
            "weights": weights,
            "best_shifts": top_shifts,
            "retrieved_residual_phase": retrieved_residual,
            "retrieved_future_phase": retrieved_future,
            "confidence": self._confidence(top_values),
        }

    def _compute_similarities(
        self,
        query_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory_key = self._match_query_tensor(self.memory_bank.key_phase, query_phase)

        if not self.shift_aware:
            similarities = self._similarity(query_phase, memory_key)
            shifts = torch.zeros_like(similarities, dtype=torch.long)
            return similarities, shifts

        best_similarities = None
        best_shifts = None
        for shift in range(self.tokenizer.phase_len):
            shifted_key = torch.roll(memory_key, shifts=shift, dims=2)
            shifted_similarity = self._similarity(query_phase, shifted_key)
            shift_tensor = torch.full_like(shifted_similarity, shift, dtype=torch.long)

            if best_similarities is None:
                best_similarities = shifted_similarity
                best_shifts = shift_tensor
                continue

            is_better = shifted_similarity > best_similarities
            best_similarities = torch.where(
                is_better,
                shifted_similarity,
                best_similarities,
            )
            best_shifts = torch.where(is_better, shift_tensor, best_shifts)

        return best_similarities, best_shifts

    def _similarity(
        self,
        query_phase: torch.Tensor,
        memory_key: torch.Tensor,
    ) -> torch.Tensor:
        query = query_phase.reshape(query_phase.size(0), -1)
        key = memory_key.reshape(memory_key.size(0), -1)

        if self.similarity == "pearson":
            query = query - query.mean(dim=1, keepdim=True)
            key = key - key.mean(dim=1, keepdim=True)

        query = F.normalize(query, dim=1, eps=1e-8)
        key = F.normalize(key, dim=1, eps=1e-8)
        return query.matmul(key.transpose(0, 1))

    def _gather_candidates(
        self,
        memory: torch.Tensor,
        indices: torch.Tensor,
        shifts: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        candidates = self._match_query_tensor(memory, reference)[indices]
        return self._align_candidates(candidates, shifts)

    @staticmethod
    def _match_query_tensor(
        tensor: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return tensor.to(device=reference.device, dtype=reference.dtype)

    def _weights(self, top_values: torch.Tensor) -> torch.Tensor:
        logits = top_values / self.temperature
        if self.similarity_threshold is None:
            return F.softmax(logits, dim=1)

        is_valid = top_values >= self.similarity_threshold
        fallback_weights = F.softmax(logits, dim=1)
        masked_logits = logits.masked_fill(~is_valid, -1e9)
        masked_weights = F.softmax(masked_logits, dim=1)
        has_valid = is_valid.any(dim=1, keepdim=True)
        return torch.where(has_valid, masked_weights, fallback_weights)

    def _confidence(self, top_values: torch.Tensor) -> torch.Tensor:
        max_similarity = top_values[:, :1]
        if self.similarity_threshold is None:
            return ((max_similarity + 1.0) * 0.5).clamp(0.0, 1.0)

        denominator = max(1.0 - self.similarity_threshold, 1e-6)
        return ((max_similarity - self.similarity_threshold) / denominator).clamp(
            0.0,
            1.0,
        )

    def _align_candidates(
        self,
        candidates: torch.Tensor,
        shifts: torch.Tensor,
    ) -> torch.Tensor:
        if not self.shift_aware:
            return candidates

        aligned = torch.empty_like(candidates)
        for shift in range(self.tokenizer.phase_len):
            mask = shifts == shift
            if mask.any():
                aligned[mask] = torch.roll(candidates[mask], shifts=shift, dims=2)
        return aligned
