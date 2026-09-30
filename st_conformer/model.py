"""ST-Conformer: свёртка без сжатия каналов и осевое самовнимание.

Базовый EEG-Conformer (Song et al., 2023) после пространственной свёртки
сворачивает электроды в одну ось. Здесь сетка (электрод × время) сохраняется:
сначала Multi-Head Attention по 62 каналам, затем — по временным окнам
(Vaswani et al., 2017). Классификация — GAP и линейный слой.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .channels import xyz_matrix


class ConvModule(nn.Module):
    """Временная и точечная свёртка; ось электродов не схлопывается."""

    def __init__(
        self,
        n_bands: int = 5,
        hid_channels: int = 40,
        emb_size: int = 40,
        kernel_t: int = 3,
        dropout: float = 0.5,
        time_pool: int = 2,
    ):
        super().__init__()
        padding_t = kernel_t // 2
        layers: list[nn.Module] = [
            nn.Conv2d(n_bands, hid_channels, kernel_size=(1, kernel_t), padding=(0, padding_t)),
            nn.Conv2d(hid_channels, emb_size, kernel_size=(1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
        ]
        if time_pool and time_pool > 1:
            layers.append(nn.AvgPool2d((1, time_pool), (1, time_pool)))
        layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # [B, полосы, C, T] → [B, C, T', F]
        x = self.net(x)
        return x.permute(0, 2, 3, 1).contiguous()


class FeedForwardBlock(nn.Sequential):
    """Позиционно-независимый MLP блока трансформера."""

    def __init__(self, emb_size: int, expansion: int, drop_p: float):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


def sinusoidal_pe(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Синусоидальная временная кодировка (Vaswani et al., 2017)."""
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


class AxialAttention(nn.Module):
    """Многоголовое самовнимание вдоль одной оси сетки [B, C, T, F]."""

    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError("emb_size должно делиться на num_heads")
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(emb_size, 3 * emb_size)
        self.proj = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, axis: int, return_attn: bool = False):
        # axis=1 — электроды (токены = каналы), axis=2 — время
        if axis == 1:
            h = x.permute(0, 2, 1, 3).contiguous()
        elif axis == 2:
            h = x
        else:
            raise ValueError("axis: 1 (пространство) или 2 (время)")

        b, extra, n, f = h.shape
        qkv = self.qkv(h).view(b, extra, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=3)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        drop = self.dropout if self.training else 0.0
        if return_attn:
            scale = self.head_dim ** -0.5
            weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            weights = torch.softmax(weights, dim=-1)
            if drop > 0:
                weights = torch.nn.functional.dropout(weights, drop, training=True)
            out = torch.matmul(weights, v)
        else:
            q = q.reshape(b * extra, self.num_heads, n, self.head_dim)
            k = k.reshape(b * extra, self.num_heads, n, self.head_dim)
            v = v.reshape(b * extra, self.num_heads, n, self.head_dim)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=drop)
            out = out.view(b, extra, self.num_heads, n, self.head_dim)
            weights = None

        out = out.permute(0, 1, 3, 2, 4).reshape(b, extra, n, f)
        out = self.proj(out)
        if axis == 1:
            out = out.permute(0, 2, 1, 3).contiguous()
        return out, weights


class STBlock(nn.Module):
    """Блок: пространственное внимание → временное → FFN (pre-LN, остаток)."""

    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        dropout: float,
        ffn_expansion: int,
        ffn_dropout: float,
    ):
        super().__init__()
        self.norm_s = nn.LayerNorm(emb_size)
        self.spatial = AxialAttention(emb_size, num_heads, dropout)
        self.drop_s = nn.Dropout(dropout)

        self.norm_t = nn.LayerNorm(emb_size)
        self.temporal = AxialAttention(emb_size, num_heads, dropout)
        self.drop_t = nn.Dropout(dropout)

        self.norm_f = nn.LayerNorm(emb_size)
        self.ffn = FeedForwardBlock(emb_size, ffn_expansion, ffn_dropout)
        self.drop_f = nn.Dropout(dropout)

    def forward(self, x: Tensor, return_attn: bool = False):
        spatial_out, spatial_w = self.spatial(self.norm_s(x), axis=1, return_attn=return_attn)
        x = x + self.drop_s(spatial_out)
        temporal_out, temporal_w = self.temporal(self.norm_t(x), axis=2, return_attn=return_attn)
        x = x + self.drop_t(temporal_out)
        x = x + self.drop_f(self.ffn(self.norm_f(x)))
        attn = None
        if return_attn:
            attn = {"spatial": spatial_w, "temporal": temporal_w}
        return x, attn


class ClassificationHead(nn.Module):
    """Глобальное среднее по электродам и времени, затем линейный классификатор."""

    def __init__(self, emb_size: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=(1, 2))
        return self.mlp(self.norm(pooled))


class STConformer(nn.Module):
    """Пространственно-временной Conformer для DE-признаков SEED."""

    def __init__(
        self,
        n_channels: int = 62,
        n_bands: int = 5,
        n_classes: int = 3,
        emb_size: int = 40,
        hid_channels: int = 40,
        temporal_kernel: int = 3,
        num_heads: int = 8,
        depth: int = 2,
        ffn_expansion: int = 2,
        dropout: float = 0.5,
        ffn_dropout: float = 0.5,
        time_pool: int = 2,
    ):
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError("emb_size должно делиться на num_heads")

        self.conv = ConvModule(
            n_bands=n_bands,
            hid_channels=hid_channels,
            emb_size=emb_size,
            kernel_t=temporal_kernel,
            dropout=dropout,
            time_pool=time_pool,
        )
        xyz = torch.tensor(xyz_matrix(), dtype=torch.float32)
        self.register_buffer("electrode_xyz", xyz, persistent=False)
        self.coord_proj = nn.Linear(3, emb_size)
        self.spatial_pe = nn.Parameter(torch.zeros(1, n_channels, 1, emb_size))
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)

        self.blocks = nn.ModuleList(
            [
                STBlock(emb_size, num_heads, dropout, ffn_expansion, ffn_dropout)
                for _ in range(depth)
            ]
        )
        self.head = ClassificationHead(emb_size, n_classes, dropout=dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _add_position(self, x: Tensor) -> Tensor:
        t = x.size(2)
        spatial = self.spatial_pe + self.coord_proj(self.electrode_xyz).unsqueeze(0).unsqueeze(2)
        temporal = sinusoidal_pe(t, x.size(-1), x.device, x.dtype).view(1, 1, t, -1)
        return x + spatial + temporal

    def forward(self, x: Tensor, return_attn: bool = False):
        # x: [B, 5, 62, T]
        x = self.conv(x)
        x = self._add_position(x)
        attn_all = []
        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if return_attn:
                attn_all.append(attn)
        logits = self.head(x)
        if return_attn:
            return logits, attn_all
        return logits
