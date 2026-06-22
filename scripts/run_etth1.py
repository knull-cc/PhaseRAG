from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT.parent, PROJECT_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from PhaseRAG.config import base_config as config_module
from PhaseRAG.config.base_config import AttrDict
from PhaseRAG.config.data_factory import data_provider
from PhaseRAG.config.data_info import DATASET_INFO
from PhaseRAG.models import (
    PhaseFormer,
    PhaseFormerForecaster,
    PhaseRAGForecaster,
    PhaseRetriever,
    PhaseTokenizer,
    build_phase_memory_bank,
)


DEFAULT_NORM_HYPERS = {
    "revin_affine": False,
    "revin_eps": 1e-5,
}
DATASET_NAME = "ETTh1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PhaseRAG on ETTh1.")
    parser.add_argument("--seq-len", type=int, default=720)
    parser.add_argument("--pred-len", type=int, default=720)
    parser.add_argument("--period-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--memory-stride", type=int, default=1)
    parser.add_argument("--max-memory-items", type=int, default=4096)
    parser.add_argument("--lambda-base", type=float, default=0.1)
    parser.add_argument("--gate-hidden-dim", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shift-aware", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def configure_experiment(args: argparse.Namespace) -> AttrDict:
    exp_args = config_module.config
    dataset_info = DATASET_INFO[DATASET_NAME]
    model_args = exp_args.model_args
    dataset_args = exp_args.dataset_args
    training_args = exp_args.training_args

    model_args.model = "PhaseRAG"
    model_args.input_len = args.seq_len
    model_args.num_variants = int(dataset_info["num_variants"])

    dataset_args.seq_len = args.seq_len
    dataset_args.label_len = 0
    dataset_args.pred_len = args.pred_len
    dataset_args.percent = 100
    dataset_args.data = dataset_info["data"]
    dataset_args.root_path = dataset_info["root_path"]
    dataset_args.data_path = dataset_info["data_path"]
    dataset_args.batch_size = args.batch_size
    dataset_args.var_needed = model_args.num_variants
    dataset_args.noisy_ratio = 0.0
    dataset_args.num_workers = args.num_workers

    training_args.batch_size = args.batch_size
    training_args.learning_rate = args.learning_rate
    training_args.train_epochs = args.epochs
    training_args.patience = args.patience
    training_args.loss_func = "mse"
    training_args.lr_schedule_config.type = "type3"
    training_args.ema = False
    training_args.itr = 1

    return exp_args


class PhaseRAGETTh1Config:
    def __init__(self, exp_args: AttrDict, args: argparse.Namespace) -> None:
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.enc_in = exp_args.model_args.num_variants
        self.period_len = args.period_len
        self.target_var_index = -1
        self.training_args = exp_args.training_args
        self.dataset_args = exp_args.dataset_args

        self.latent_dim = 4
        self.phase_encoder_hidden = 16
        self.predictor_hidden = 32
        self.phase_layers = 3
        self.phase_attn_heads = 1
        self.phase_attn_dropout = 0.1
        self.phase_attn_use_relpos = True
        self.phase_attn_window = None
        self.phase_attention_dim = None
        self.phase_num_routers = 8
        self.phase_use_pos_embed = True
        self.phase_pos_dropout = 0.0

        self.use_revin = True
        self.revin_affine = DEFAULT_NORM_HYPERS["revin_affine"]
        self.revin_eps = DEFAULT_NORM_HYPERS["revin_eps"]
        self.use_huber_loss = False
        self.huber_delta = 1.0

        self.phase_rag_lambda_base = args.lambda_base
        self.phase_rag_freeze_backbone = args.freeze_backbone
        self.phase_rag_gate_hidden_dim = args.gate_hidden_dim

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def make_logger(args: argparse.Namespace, stage: str) -> CSVLogger:
    shift_name = "shift" if args.shift_aware else "plain"
    version = (
        f"{DATASET_NAME}-{args.seq_len}-{args.pred_len}-{stage}"
        f"-p{args.period_len}-top{args.top_k}-{shift_name}"
        f"-mem{args.max_memory_items}-tau{args.temperature}"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CSVLogger(
        save_dir="./log/training_results",
        name="PhaseRAG",
        version=f"{version}-{stage}-{timestamp}",
    )


def make_trainer(args: argparse.Namespace, stage: str, max_epochs: int) -> pl.Trainer:
    return pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        logger=make_logger(args, stage),
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience)],
        enable_checkpointing=True,
        enable_progress_bar=True,
        log_every_n_steps=1,
        deterministic=True,
    )


def pretrain_backbone(
    model_config: PhaseRAGETTh1Config,
    args: argparse.Namespace,
    train_loader,
    vali_loader,
) -> PhaseFormer:
    backbone = PhaseFormer(model_config)
    stage1 = PhaseFormerForecaster(model_config, backbone=backbone)
    trainer = make_trainer(args, "backbone", args.pretrain_epochs)
    trainer.fit(stage1, train_dataloaders=train_loader, val_dataloaders=vali_loader)
    backbone.eval()
    return backbone


def build_rag_model(
    model_config: PhaseRAGETTh1Config,
    args: argparse.Namespace,
    backbone: PhaseFormer,
    train_dataset,
) -> PhaseRAGForecaster:
    model_config.phase_rag_freeze_backbone = True
    device = next(backbone.parameters()).device

    tokenizer = PhaseTokenizer(phase_len=args.period_len)
    memory_bank = build_phase_memory_bank(
        dataset=train_dataset,
        tokenizer=tokenizer,
        pred_len=args.pred_len,
        backbone=backbone,
        batch_size=args.batch_size,
        stride=args.memory_stride,
        max_items=args.max_memory_items,
        num_workers=args.num_workers,
        device=device,
        norm_eps=model_config.revin_eps,
    )
    retriever = PhaseRetriever(
        tokenizer=tokenizer,
        memory_bank=memory_bank,
        top_k=args.top_k,
        temperature=args.temperature,
        similarity="cosine",
        shift_aware=args.shift_aware,
        norm_eps=model_config.revin_eps,
    )
    return PhaseRAGForecaster(model_config, backbone=backbone, retriever=retriever)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(2021, workers=True)

    args = parse_args()
    exp_args = configure_experiment(args)
    model_config = PhaseRAGETTh1Config(exp_args, args)

    train_dataset, train_loader = data_provider(exp_args.dataset_args, "train")
    _, vali_loader = data_provider(exp_args.dataset_args, "val")
    _, test_loader = data_provider(exp_args.dataset_args, "test")

    # Stage 1: train and freeze the PhaseFormer backbone.
    backbone = pretrain_backbone(model_config, args, train_loader, vali_loader)

    # Stage 2: build the residual memory from the frozen backbone, then train
    # only the retrieval adapter to correct the backbone's errors.
    model = build_rag_model(model_config, args, backbone, train_dataset)
    trainer = make_trainer(args, "rag", args.epochs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=vali_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
