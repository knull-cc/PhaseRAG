from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PHASEFORMER_ROOT = Path(__file__).resolve().parents[2] / "PhaseFormer"
if PHASEFORMER_ROOT.exists():
    phaseformer_path = str(PHASEFORMER_ROOT)
    if phaseformer_path not in sys.path:
        sys.path.insert(0, phaseformer_path)

from PhaseRAG.models.phase_retriever import PhaseRetriever
from PhaseRAG.models.phase_tokenizer import PhaseTokenizer
from src.models.pl_bases.default_module import DefaultPLModule
from src.utils.metrics import metric


PhaseRAGOutput = tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]


class PhaseResidualAdapter(nn.Module):
    """Gated residual fusion in phase space."""

    def __init__(
        self,
        tokenizer: PhaseTokenizer,
        pred_len: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pred_len = pred_len
        self.gate = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        y_base: torch.Tensor,
        evidence: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query_phase = evidence["query_phase"]
        y_base_phase = self.tokenizer.to_phase(y_base)
        residual_phase = evidence["retrieved_residual_phase"]

        if y_base_phase.shape != residual_phase.shape:
            raise ValueError("base prediction and retrieved residual phase shapes differ")

        gate = self._gate(x, query_phase, residual_phase, evidence)
        y_final_phase = y_base_phase + gate * residual_phase
        y_final = self.tokenizer.to_time(y_final_phase, self.pred_len)
        return y_final, {"gate": gate, "y_final_phase": y_final_phase}

    def _gate(
        self,
        x: torch.Tensor,
        query_phase: torch.Tensor,
        residual_phase: torch.Tensor,
        evidence: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        channel_count = query_phase.size(1)
        query_mean = query_phase.mean(dim=(2, 3))
        query_std = query_phase.std(dim=(2, 3), unbiased=False)
        last_value = x[:, -1, :]
        residual_scale = residual_phase.abs().mean(dim=(2, 3))

        max_similarity = evidence["similarities"].max(dim=1).values
        mean_similarity = evidence["similarities"].mean(dim=1)
        confidence = evidence["confidence"].squeeze(-1)

        max_similarity = max_similarity[:, None].expand(-1, channel_count)
        mean_similarity = mean_similarity[:, None].expand(-1, channel_count)
        confidence = confidence[:, None].expand(-1, channel_count)

        gate_input = torch.stack(
            [
                last_value,
                query_mean,
                query_std,
                residual_scale,
                max_similarity,
                mean_similarity * confidence,
            ],
            dim=-1,
        )
        return self.gate(gate_input).unsqueeze(-1)


class PhaseRAGForecaster(DefaultPLModule):
    """PhaseFormer wrapper with residual phase retrieval."""

    def __init__(
        self,
        configs,
        backbone: nn.Module,
        retriever: PhaseRetriever,
    ) -> None:
        super().__init__(configs)
        self.backbone = backbone
        self.retriever = retriever
        self.pred_len = configs.pred_len
        self.lambda_base = float(getattr(configs, "phase_rag_lambda_base", 0.1))
        self.freeze_backbone = bool(getattr(configs, "phase_rag_freeze_backbone", True))
        gate_hidden_dim = int(getattr(configs, "phase_rag_gate_hidden_dim", 32))
        self.adapter = PhaseResidualAdapter(
            tokenizer=retriever.tokenizer,
            pred_len=self.pred_len,
            hidden_dim=gate_hidden_dim,
        )

        if self.freeze_backbone:
            self._freeze_backbone()

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        *_args,
        **_kwargs,
    ) -> PhaseRAGOutput:
        y_base = self._run_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        evidence = self.retriever(x_enc)
        y_final, adapter_state = self.adapter(x_enc, y_base, evidence)
        return y_final, y_base, evidence, adapter_state

    def training_step(self, batch, _batch_idx) -> torch.Tensor:
        return self._loss_step(batch, "train")

    def validation_step(self, batch, _batch_idx) -> torch.Tensor:
        return self._loss_step(batch, "val")

    def test_step(self, batch, _batch_idx) -> dict[str, torch.Tensor]:
        batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(batch)
        dec_inp = self._build_decoder_input(batch_y)
        y_final, y_base, _, _ = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        y_final, y_base, target = self._align_outputs(y_final, y_base, batch_y)
        final_metrics = metric(y_final.detach(), target.detach())
        base_metrics = metric(y_base.detach(), target.detach())
        logs = {f"test_{name}": value for name, value in final_metrics.items()}
        logs.update({f"test_base_{name}": value for name, value in base_metrics.items()})
        self.log_dict(logs, on_epoch=True)
        return final_metrics

    def _loss_step(self, batch, stage: str) -> torch.Tensor:
        batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(batch)
        dec_inp = self._build_decoder_input(batch_y)
        y_final, y_base, evidence, adapter_state = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        y_final, y_base, target = self._align_outputs(y_final, y_base, batch_y)
        final_loss = self._forecast_loss(y_final, target)
        base_loss = self._forecast_loss(y_base, target)
        loss = final_loss + self.lambda_base * base_loss

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_final_loss", final_loss, on_epoch=True)
        self.log(f"{stage}_base_loss", base_loss, on_epoch=True)
        self.log(f"{stage}_gate_mean", adapter_state["gate"].mean(), on_epoch=True)
        self.log(
            f"{stage}_retrieval_confidence",
            evidence["confidence"].mean(),
            on_epoch=True,
        )
        return loss

    def _run_backbone(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                output = self.backbone(
                    x_enc=x_enc,
                    x_mark_enc=x_mark_enc,
                    x_dec=x_dec,
                    x_mark_dec=x_mark_dec,
                )
        else:
            output = self.backbone(
                x_enc=x_enc,
                x_mark_enc=x_mark_enc,
                x_dec=x_dec,
                x_mark_dec=x_mark_dec,
            )

        if isinstance(output, tuple):
            return output[0]
        return output

    def _prepare_batch(
        self,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        return (
            batch_x.float(),
            batch_y.float(),
            batch_x_mark.float(),
            batch_y_mark.float(),
        )

    def _align_outputs(
        self,
        y_final: torch.Tensor,
        y_base: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_final = y_final[:, -self.pred_len :, :]
        y_base = y_base[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]

        if self.target_var_index != -1:
            target_index = self.target_var_index
            y_final = y_final[:, :, target_index].unsqueeze(-1)
            y_base = y_base[:, :, target_index].unsqueeze(-1)
            target = target[:, :, target_index].unsqueeze(-1)

        return y_final, y_base, target

    def _forecast_loss(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_func = str(getattr(self.args.training_args, "loss_func", "")).lower()
        if bool(getattr(self.args, "use_huber_loss", False)) or loss_func == "huber":
            return self._huber_loss(outputs, target)

        criterion = self._get_criterion(self.args.training_args.loss_func)
        return criterion(outputs, target)

    def _huber_loss(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = torch.as_tensor(
            getattr(self.args, "huber_delta", 1.0),
            device=outputs.device,
            dtype=outputs.dtype,
        )
        diff = outputs - target
        abs_diff = diff.abs()
        quadratic = torch.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        return (0.5 * quadratic.pow(2) / delta + linear).mean()

    def _freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()
