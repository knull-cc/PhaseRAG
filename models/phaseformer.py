from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from PhaseRAG.models.layers.self_attention import AttentionLayer, FullAttention


class RevIN(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
    ) -> None:
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight + self.bias
        return x, (mean, std)

    def denormalize(
        self,
        y: torch.Tensor,
        stats: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        mean, std = stats
        return y * std + mean


class CrossPhaseRoutingLayer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_routers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
        period_len: int = 24,
        attention_dim: int | None = None,
        use_pos_embed: bool = False,
        pos_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        attention_dim = attention_dim or latent_dim
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")

        self.period_len = period_len
        self.use_pos_embed = use_pos_embed
        self.router = nn.Parameter(torch.randn(num_routers, latent_dim))
        nn.init.trunc_normal_(self.router, std=0.02)

        if use_pos_embed:
            self.pos_embedding = nn.Parameter(torch.zeros(period_len, latent_dim))
            self.pos_dropout = nn.Dropout(pos_dropout)
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.router_sender = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            latent_dim,
            num_heads,
        )
        self.router_receiver = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            latent_dim,
            num_heads,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )

    def forward(self, phase_latent: torch.Tensor) -> torch.Tensor:
        batch_size, channel_count, phase_len, latent_dim = phase_latent.shape
        x = phase_latent.view(batch_size * channel_count, phase_len, latent_dim)

        if self.use_pos_embed:
            pos_embedding = self._position_embedding(
                phase_len,
                batch_size * channel_count,
            )
            x = self.pos_dropout(x + pos_embedding)

        routers = self.router.unsqueeze(0).expand(batch_size * channel_count, -1, -1)
        router_buffer, _ = self.router_sender(routers, x, x, attn_mask=None)
        routed, _ = self.router_receiver(x, router_buffer, router_buffer, attn_mask=None)

        x = self.norm1(x + self.dropout(routed))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x.view(batch_size, channel_count, phase_len, latent_dim)

    def _position_embedding(
        self,
        phase_len: int,
        batch_size: int,
    ) -> torch.Tensor:
        if phase_len <= self.period_len:
            embedding = self.pos_embedding[:phase_len]
        else:
            repeat_count = (phase_len + self.period_len - 1) // self.period_len
            embedding = self.pos_embedding.repeat(repeat_count, 1)[:phase_len]
        return embedding.unsqueeze(0).expand(batch_size, -1, -1)


class PhaseEmbedding(nn.Module):
    def __init__(
        self,
        p_in: int,
        latent_dim: int,
        hidden: int = 32,
        use_mlp: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if use_mlp:
            self.projection = nn.Sequential(
                nn.Linear(p_in, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, latent_dim),
            )
        else:
            self.projection = nn.Linear(p_in, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, phase_series: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(phase_series))


class PhasePredictor(nn.Module):
    def __init__(
        self,
        p_out: int,
        latent_dim: int,
        hidden: int,
        use_mlp: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_mlp = use_mlp
        if use_mlp:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden, p_out),
            )
        else:
            self.decoder = nn.Linear(latent_dim, p_out)
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if self.use_mlp:
            return self.decoder(latent)
        return self.decoder(self.dropout(latent))


class CrossPhaseRoutingUnit(nn.Module):
    def __init__(
        self,
        apply_in_proj: bool,
        apply_out_proj: bool,
        num_periods_input: int,
        latent_dim: int,
        phase_attn_heads: int,
        phase_attn_dropout: float,
        period_len: int,
        phase_attention_dim: int | None,
        phase_num_routers: int,
        phase_use_pos_embed: bool,
        phase_pos_dropout: float,
    ) -> None:
        super().__init__()
        self.in_proj = None
        self.out_proj = None
        if apply_in_proj:
            self.in_proj = nn.Sequential(
                nn.Linear(num_periods_input, latent_dim),
                nn.LayerNorm(latent_dim),
            )
        if apply_out_proj:
            self.out_proj = nn.Linear(latent_dim, num_periods_input)

        self.interact = CrossPhaseRoutingLayer(
            latent_dim=latent_dim,
            num_routers=phase_num_routers,
            num_heads=phase_attn_heads,
            dropout=phase_attn_dropout,
            period_len=period_len,
            attention_dim=phase_attention_dim,
            use_pos_embed=phase_use_pos_embed,
            pos_dropout=phase_pos_dropout,
        )

    def forward(
        self,
        phase_series: torch.Tensor,
        latent: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.in_proj is not None:
            current = self.in_proj(phase_series)
            latent = current if latent is None else latent + current
        if latent is None:
            raise ValueError("latent must be provided when in projection is disabled")

        latent = self.interact(latent)
        if self.out_proj is None:
            return latent, None
        return latent, self.out_proj(latent)


class PhaseFormer(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.latent_dim = getattr(configs, "latent_dim", 8)
        self.phase_encoder_hidden = getattr(configs, "phase_encoder_hidden", 32)
        self.predictor_hidden = getattr(configs, "predictor_hidden", 64)
        self.phase_layers = getattr(configs, "phase_layers", 1)
        self.phase_attn_heads = getattr(configs, "phase_attn_heads", 4)
        self.phase_attn_dropout = getattr(configs, "phase_attn_dropout", 0.0)
        self.phase_attention_dim = getattr(configs, "phase_attention_dim", None)
        self.phase_num_routers = getattr(configs, "phase_num_routers", 8)
        self.phase_use_pos_embed = getattr(configs, "phase_use_pos_embed", False)
        self.phase_pos_dropout = getattr(configs, "phase_pos_dropout", 0.0)

        self.num_periods_input = (self.seq_len + self.period_len - 1) // self.period_len
        self.num_periods_output = (self.pred_len + self.period_len - 1) // self.period_len
        self.pad_seq_len = self.num_periods_input * self.period_len - self.seq_len

        self.use_revin = getattr(configs, "use_revin", True)
        if self.use_revin:
            self.revin = RevIN(
                num_features=self.enc_in,
                eps=getattr(configs, "revin_eps", 1e-5),
                affine=getattr(configs, "revin_affine", False),
            )

        self.embedding = PhaseEmbedding(
            p_in=self.num_periods_input,
            latent_dim=self.latent_dim,
            hidden=self.phase_encoder_hidden,
            use_mlp=getattr(configs, "phase_encoder_use_mlp", False),
            dropout=getattr(configs, "phase_encoder_dropout", 0.0),
        )
        self.routing_layers = nn.ModuleList(self._build_routing_layers())
        self.predictor = PhasePredictor(
            p_out=self.num_periods_output,
            latent_dim=self.latent_dim,
            hidden=self.predictor_hidden,
            use_mlp=getattr(configs, "predictor_use_mlp", False),
            dropout=getattr(configs, "predictor_dropout", 0.0),
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        *_args,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_revin:
            x, stats = self.revin.normalize(x_enc)
        else:
            x = x_enc.float()
            stats = None

        batch_size, _, channel_count = x.shape
        x = x.permute(0, 2, 1)
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode="circular")

        periods = x.view(
            batch_size,
            channel_count,
            self.num_periods_input,
            self.period_len,
        )
        phase_series = periods.permute(0, 1, 3, 2).contiguous()

        latent = self.embedding(phase_series)
        current_phase_series = phase_series
        for layer_index, unit in enumerate(self.routing_layers):
            latent, phase_steps = unit(current_phase_series, latent)
            if layer_index < len(self.routing_layers) - 1:
                current_phase_series = phase_steps

        y_phase_steps = self.predictor(latent)
        y_hat = self._phase_steps_to_time(y_phase_steps, batch_size, channel_count)
        if self.use_revin and stats is not None:
            y_hat = self.revin.denormalize(y_hat, stats)
        return y_hat, latent, y_phase_steps

    def _build_routing_layers(self) -> list[CrossPhaseRoutingUnit]:
        layers = []
        for layer_index in range(self.phase_layers):
            is_first = layer_index == 0
            is_last = layer_index == self.phase_layers - 1
            layers.append(
                CrossPhaseRoutingUnit(
                    apply_in_proj=not is_first,
                    apply_out_proj=not is_last,
                    num_periods_input=self.num_periods_input,
                    latent_dim=self.latent_dim,
                    phase_attn_heads=self.phase_attn_heads,
                    phase_attn_dropout=self.phase_attn_dropout,
                    period_len=self.period_len,
                    phase_attention_dim=self.phase_attention_dim,
                    phase_num_routers=self.phase_num_routers,
                    phase_use_pos_embed=self.phase_use_pos_embed,
                    phase_pos_dropout=self.phase_pos_dropout,
                )
            )
        return layers

    def _phase_steps_to_time(
        self,
        y_phase_steps: torch.Tensor,
        batch_size: int,
        channel_count: int,
    ) -> torch.Tensor:
        periods = y_phase_steps.permute(0, 1, 3, 2).contiguous()
        y_full = periods.reshape(batch_size, channel_count, -1)[..., : self.pred_len]
        return y_full.permute(0, 2, 1).contiguous()
