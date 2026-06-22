from __future__ import annotations

import torch
from torch import nn

from PhaseRAG.models.phase_retriever import RaftRetriever
from PhaseRAG.models.phase_tokenizer import PhaseTokenizer, offset_normalize
from PhaseRAG.models.phaseformer import (
    CrossPhaseRoutingUnit,
    PhaseEmbedding,
    PhasePredictor,
)
from PhaseRAG.models.pl_bases.default_module import DefaultPLModule
from PhaseRAG.utils.metrics import metric


class PhaseRAGForecaster(DefaultPLModule):
    """RAFT-in-phase forecaster.

    Pipeline (offset-normalized space):
      1. retrieve real future patterns (multi-period RAFT retrieval),
      2. tokenize both the input and the retrieved future into phase tokens,
      3. concatenate them along the period axis,
      4. a phase predictor produces the forecast,
      5. add the offset back.

    The ablation prediction ``y_base`` reuses the same predictor with the
    retrieval evidence zeroed out, isolating what the retrieval contributes.
    """

    def __init__(self, configs, retriever: RaftRetriever) -> None:
        super().__init__(configs)
        self.retriever = retriever
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.lambda_base = float(getattr(configs, "phase_rag_lambda_base", 0.1))

        period_len = int(configs.period_len)
        self.tokenizer = PhaseTokenizer(phase_len=period_len)
        self.num_periods_input = self.tokenizer.period_count(self.seq_len)
        self.num_periods_output = self.tokenizer.period_count(self.pred_len)
        combined_periods = self.num_periods_input + self.num_periods_output

        self.latent_dim = int(getattr(configs, "latent_dim", 8))
        self.phase_layers = int(getattr(configs, "phase_layers", 1))

        self.embedding = PhaseEmbedding(
            p_in=combined_periods,
            latent_dim=self.latent_dim,
            hidden=int(getattr(configs, "phase_encoder_hidden", 32)),
            use_mlp=bool(getattr(configs, "phase_encoder_use_mlp", False)),
            dropout=float(getattr(configs, "phase_encoder_dropout", 0.0)),
        )
        self.routing_layers = nn.ModuleList(
            self._build_routing_layers(configs, combined_periods, period_len)
        )
        self.predictor = PhasePredictor(
            p_out=self.num_periods_output,
            latent_dim=self.latent_dim,
            hidden=int(getattr(configs, "predictor_hidden", 64)),
            use_mlp=bool(getattr(configs, "predictor_use_mlp", False)),
            dropout=float(getattr(configs, "predictor_dropout", 0.0)),
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        query_index: torch.Tensor | None = None,
        *_args,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_hat, x_last = offset_normalize(x_enc)
        retrieval_future = self.retriever(x_hat, query_index)

        x_phase = self.tokenizer.to_phase(x_hat)
        retrieval_phase = self.tokenizer.to_phase(retrieval_future)

        y_final = self._predict(x_phase, retrieval_phase) + x_last
        y_base = self._predict(x_phase, torch.zeros_like(retrieval_phase)) + x_last
        return y_final, y_base

    def training_step(self, batch, _batch_idx) -> torch.Tensor:
        return self._loss_step(batch, "train")

    def validation_step(self, batch, _batch_idx) -> torch.Tensor:
        return self._loss_step(batch, "val")

    def test_step(self, batch, _batch_idx) -> dict[str, torch.Tensor]:
        batch_x, batch_y, _ = self._unpack(batch)
        y_final, y_base = self(x_enc=batch_x)
        y_final, y_base, target = self._align(y_final, y_base, batch_y)

        final_metrics = metric(y_final.detach(), target.detach())
        base_metrics = metric(y_base.detach(), target.detach())
        self.log_dict(
            {
                "test_mae": final_metrics["mae"],
                "test_mse": final_metrics["mse"],
                "test_base_mae": base_metrics["mae"],
                "test_base_mse": base_metrics["mse"],
            },
            on_epoch=True,
        )
        return final_metrics

    def _loss_step(self, batch, stage: str) -> torch.Tensor:
        batch_x, batch_y, query_index = self._unpack(batch)
        y_final, y_base = self(x_enc=batch_x, query_index=query_index)
        y_final, y_base, target = self._align(y_final, y_base, batch_y)

        final_loss = nn.functional.mse_loss(y_final, target)
        base_loss = nn.functional.mse_loss(y_base, target)
        loss = final_loss + self.lambda_base * base_loss

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_final_loss", final_loss, on_epoch=True)
        self.log(f"{stage}_base_loss", base_loss, on_epoch=True)
        return loss

    def _predict(
        self,
        x_phase: torch.Tensor,
        retrieval_phase: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([x_phase, retrieval_phase], dim=-1)
        latent = self.embedding(combined)
        current = combined
        for layer_index, unit in enumerate(self.routing_layers):
            latent, phase_steps = unit(current, latent)
            if layer_index < len(self.routing_layers) - 1:
                current = phase_steps
        y_phase = self.predictor(latent)
        return self.tokenizer.to_time(y_phase, self.pred_len)

    def _build_routing_layers(self, configs, num_periods: int, period_len: int):
        layers = []
        for layer_index in range(self.phase_layers):
            is_first = layer_index == 0
            is_last = layer_index == self.phase_layers - 1
            layers.append(
                CrossPhaseRoutingUnit(
                    apply_in_proj=not is_first,
                    apply_out_proj=not is_last,
                    num_periods_input=num_periods,
                    latent_dim=self.latent_dim,
                    phase_attn_heads=int(getattr(configs, "phase_attn_heads", 4)),
                    phase_attn_dropout=float(getattr(configs, "phase_attn_dropout", 0.0)),
                    period_len=period_len,
                    phase_attention_dim=getattr(configs, "phase_attention_dim", None),
                    phase_num_routers=int(getattr(configs, "phase_num_routers", 8)),
                    phase_use_pos_embed=bool(getattr(configs, "phase_use_pos_embed", False)),
                    phase_pos_dropout=float(getattr(configs, "phase_pos_dropout", 0.0)),
                )
            )
        return layers

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
        y_final: torch.Tensor,
        y_base: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_final = y_final[:, -self.pred_len :, :]
        y_base = y_base[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]
        if self.target_var_index != -1:
            index = self.target_var_index
            y_final = y_final[:, :, index].unsqueeze(-1)
            y_base = y_base[:, :, index].unsqueeze(-1)
            target = target[:, :, index].unsqueeze(-1)
        return y_final, y_base, target
