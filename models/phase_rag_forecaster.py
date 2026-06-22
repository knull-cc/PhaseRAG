from __future__ import annotations

import torch
from torch import nn

from PhaseRAG.models.phase_retriever import PhaseRetriever
from PhaseRAG.models.phase_tokenizer import PhaseTokenizer, instance_normalize
from PhaseRAG.models.pl_bases.default_module import DefaultPLModule
from PhaseRAG.utils.metrics import metric


PhaseRAGOutput = tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]

GATE_FEATURE_DIM = 5


class PhaseResidualAdapter(nn.Module):
    """Gated fusion of a retrieved, scale-free residual correction.

    The retrieved residual lives in the per-window normalized space, so it is
    rescaled by the query's own std before being added (in the time domain) to
    the backbone prediction. The gate only sees scale-invariant features
    (similarities, confidence, normalized residual magnitude) so it is not
    confused by raw-amplitude differences between series.
    """

    def __init__(
        self,
        tokenizer: PhaseTokenizer,
        pred_len: int,
        hidden_dim: int = 32,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pred_len = pred_len
        self.norm_eps = norm_eps
        self.gate = nn.Sequential(
            nn.Linear(GATE_FEATURE_DIM, hidden_dim),
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
        residual_phase = evidence["retrieved_residual_phase"]
        residual_time = self.tokenizer.to_time(residual_phase, self.pred_len)

        _, _, std_x = instance_normalize(x, self.norm_eps)

        gate = self._gate(residual_phase, evidence)
        y_final = y_base + gate * residual_time * std_x
        return y_final, {"gate": gate, "residual_time": residual_time}

    def _gate(
        self,
        residual_phase: torch.Tensor,
        evidence: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        channel_count = residual_phase.size(1)
        residual_scale = residual_phase.abs().mean(dim=(2, 3))

        max_similarity = evidence["similarities"].max(dim=1).values
        mean_similarity = evidence["similarities"].mean(dim=1)
        confidence = evidence["confidence"].squeeze(-1)

        max_similarity = max_similarity[:, None].expand(-1, channel_count)
        mean_similarity = mean_similarity[:, None].expand(-1, channel_count)
        confidence = confidence[:, None].expand(-1, channel_count)

        gate_input = torch.stack(
            [
                residual_scale,
                max_similarity,
                mean_similarity,
                confidence,
                mean_similarity * confidence,
            ],
            dim=-1,
        )
        gate = self.gate(gate_input).squeeze(-1)
        return gate[:, None, :]


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
        norm_eps = float(getattr(configs, "revin_eps", 1e-5))
        self.adapter = PhaseResidualAdapter(
            tokenizer=retriever.tokenizer,
            pred_len=self.pred_len,
            hidden_dim=gate_hidden_dim,
            norm_eps=norm_eps,
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


class PhaseFormerForecaster(DefaultPLModule):
    """Stage-1 module that trains the PhaseFormer backbone on its own."""

    def __init__(self, configs, backbone: nn.Module) -> None:
        super().__init__(configs)
        self.backbone = backbone
        self.pred_len = configs.pred_len

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        *_args,
        **_kwargs,
    ) -> torch.Tensor:
        output = self.backbone(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
        )
        if isinstance(output, tuple):
            return output[0]
        return output

    def training_step(self, batch, _batch_idx) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, _batch_idx) -> torch.Tensor:
        return self._step(batch, "val")

    def _step(self, batch, stage: str) -> torch.Tensor:
        batch_x, batch_y = batch[0].float(), batch[1].float()
        outputs = self(x_enc=batch_x)
        outputs = outputs[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]
        if self.target_var_index != -1:
            outputs = outputs[:, :, self.target_var_index].unsqueeze(-1)
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)
        loss = criterion(outputs, target)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        return loss
