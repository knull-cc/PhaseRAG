from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn

from PhaseRAG.utils.metrics import metric


class DefaultPLModule(pl.LightningModule):
    def __init__(self, configs) -> None:
        super().__init__()
        self.args = configs
        self.target_var_index = int(configs.get("target_var_index", -1))
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.training_args.learning_rate,
        )
        if self.args.training_args.lr_schedule_config.type == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.training_args.lr_schedule_config.tmax,
                eta_min=1e-8,
            )
            return [optimizer], [scheduler]
        return optimizer

    def _build_decoder_input(self, batch_y: torch.Tensor) -> torch.Tensor:
        pred_len = self.args.dataset_args.pred_len
        label_len = self.args.dataset_args.label_len
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
        return torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float()

    def _get_criterion(self, loss_type: str) -> nn.Module:
        if loss_type == "mse":
            return nn.MSELoss()
        if loss_type == "mae":
            return nn.L1Loss()
        if loss_type == "smae":
            return nn.SmoothL1Loss()
        raise ValueError(f"loss function {loss_type} not supported yet")

    def test_step(self, batch, _batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)
        outputs = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        pred_len = self.args.dataset_args.pred_len
        outputs = outputs[:, -pred_len:, :]
        target = batch_y[:, -pred_len:, :]
        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        result = metric(outputs.detach(), target.detach())
        self.log_dict({f"test_{key}": value for key, value in result.items()})
        return result
