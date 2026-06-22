from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from PhaseRAG.models.phase_retriever import RaftRetriever
from PhaseRAG.models.phase_tokenizer import PhaseTokenizer, offset_normalize
from PhaseRAG.models.pl_bases.default_module import DefaultPLModule


class ShallowPhasePredictor(nn.Module):
    """RAFT-style shallow fusion over phase tokens.

    Embeds the query phase tokens (``P_in``) and the retrieved future phase
    tokens (``P_out``) independently along the period axis, concatenates them and
    decodes back to ``P_out`` period tokens.
    """

    def __init__(self, p_in: int, p_out: int, hidden: int) -> None:
        super().__init__()
        self.query_embed = nn.Linear(p_in, hidden)
        self.retrieval_embed = nn.Linear(p_out, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, p_out),
        )

    def forward(
        self,
        query_phase: torch.Tensor,
        retrieved_phase: torch.Tensor,
    ) -> torch.Tensor:
        query_emb = self.query_embed(query_phase)
        retrieval_emb = self.retrieval_embed(retrieved_phase)
        z = torch.cat([query_emb, retrieval_emb], dim=-1)
        return self.mlp(z)


class PhaseRAFTForecaster(DefaultPLModule):
    """RAFT-style direct future retrieval, fused in phase-token space.

    Offset-normalized history is matched against a key/value memory of real
    futures (Pearson, multi-period). The retrieved future is tokenized into
    phase tokens, concatenated with the query phase tokens and decoded into the
    forecast, with the offset added back. No backbone, no residual correction.
    """

    def __init__(self, configs, retriever: RaftRetriever) -> None:
        super().__init__(configs)
        self.retriever = retriever
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        period_len = int(configs.period_len)
        self.tokenizer = PhaseTokenizer(phase_len=period_len)
        p_in = self.tokenizer.period_count(self.seq_len)
        p_out = self.tokenizer.period_count(self.pred_len)
        hidden = int(getattr(configs, "predictor_hidden", 64))
        self.predictor = ShallowPhasePredictor(p_in=p_in, p_out=p_out, hidden=hidden)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        query_index: torch.Tensor | None = None,
        *_args,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        x_hat, x_last = offset_normalize(x_enc)
        retrieval_future, topk_similarity_mean = self.retriever(x_hat, query_index)

        query_phase = self.tokenizer.to_phase(x_hat)
        retrieved_phase = self.tokenizer.to_phase(retrieval_future)

        y_phase = self.predictor(query_phase, retrieved_phase)
        y_final = self.tokenizer.to_time(y_phase, self.pred_len) + x_last
        retrieval_direct = retrieval_future + x_last
        return {
            "y_final": y_final,
            "retrieval_direct": retrieval_direct,
            "x_last": x_last,
            "topk_similarity_mean": topk_similarity_mean,
        }

    def training_step(self, batch, _batch_idx) -> torch.Tensor:
        return self._loss_step(batch, "train")

    def validation_step(self, batch, _batch_idx) -> torch.Tensor:
        return self._loss_step(batch, "val")

    def test_step(self, batch, _batch_idx) -> dict[str, torch.Tensor]:
        batch_x, batch_y, _ = self._unpack(batch)
        out = self(x_enc=batch_x)
        y_final, target = self._align(out["y_final"], batch_y)
        retrieval_direct, _ = self._align(out["retrieval_direct"], batch_y)
        last_value = batch_x[:, -1:, :].expand(-1, self.pred_len, -1)
        last_value, _ = self._align(last_value, batch_y)

        self.log_dict(
            {
                "test_mse": F.mse_loss(y_final, target),
                "test_mae": (y_final - target).abs().mean(),
                "retrieval_only_mse": F.mse_loss(retrieval_direct, target),
                "last_value_mse": F.mse_loss(last_value, target),
                "topk_similarity_mean": out["topk_similarity_mean"],
            },
            on_epoch=True,
        )
        return {"test_mse": F.mse_loss(y_final, target)}

    def _loss_step(self, batch, stage: str) -> torch.Tensor:
        batch_x, batch_y, query_index = self._unpack(batch)
        out = self(x_enc=batch_x, query_index=query_index)
        y_final, target = self._align(out["y_final"], batch_y)
        loss = F.mse_loss(y_final, target)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def _unpack(
        self,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_x = batch[0].float()
        batch_y = batch[1].float()
        query_index = batch[4] if len(batch) > 4 else None
        return batch_x, batch_y, query_index

    def _align(
        self,
        prediction: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = prediction[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]
        if self.target_var_index != -1:
            index = self.target_var_index
            prediction = prediction[:, :, index].unsqueeze(-1)
            target = target[:, :, index].unsqueeze(-1)
        return prediction, target