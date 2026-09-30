"""Вспомогательные функции: устройство, зерно, сборка модели."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .model import STConformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: Config) -> nn.Module:
    return STConformer(
        n_channels=cfg.n_channels,
        n_bands=cfg.n_bands,
        n_classes=cfg.n_classes,
        emb_size=cfg.emb_size,
        hid_channels=cfg.hid_channels,
        temporal_kernel=cfg.temporal_kernel,
        num_heads=cfg.num_heads,
        depth=cfg.st_depth,
        ffn_expansion=cfg.ffn_expansion,
        dropout=cfg.dropout,
        ffn_dropout=cfg.ffn_dropout,
        time_pool=cfg.time_pool,
    )
