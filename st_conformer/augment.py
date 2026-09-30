"""Mixup окон с разных роликов и маскирование каналов / секунд."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def mixup_different_trials(
    x: Tensor,
    y: Tensor,
    trials: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Смешать два окна, только если они из разных роликов.

    Возвращает `(x_mix, y_a, y_b, lam)`. При `lam == 1` смешения нет.
    """
    if alpha <= 0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    if lam > 1.0 - 1e-6:
        return x, y, y, 1.0
    if lam < 1e-6:
        lam = 0.0

    batch = x.size(0)
    device = x.device
    trials = trials.to(device)
    perm = torch.empty(batch, dtype=torch.long, device=device)
    for i in range(batch):
        others = torch.nonzero(trials != trials[i], as_tuple=False).view(-1)
        if others.numel() == 0:
            perm[i] = i
        else:
            perm[i] = others[int(torch.randint(0, others.numel(), (1,), device=device))]
    if bool((perm == torch.arange(batch, device=device)).all()):
        return x, y, y, 1.0
    mixed = lam * x + (1.0 - lam) * x[perm]
    return mixed, y, y[perm], lam


def mask_channels_time(x: Tensor, max_channels: int, max_time: int) -> Tensor:
    """Обнулить случайные электроды и непрерывный отрезок времени.

    `x`: `[B, полосы, каналы, T]`.
    """
    if max_channels <= 0 and max_time <= 0:
        return x
    _, _, n_ch, n_t = x.shape
    out = x.clone()
    for i in range(x.size(0)):
        if max_channels > 0:
            n_mask = int(torch.randint(0, max_channels + 1, (1,)))
            if n_mask:
                idx = torch.randperm(n_ch, device=x.device)[:n_mask]
                out[i, :, idx, :] = 0
        if max_time > 0 and n_t > 1:
            width = int(torch.randint(0, min(max_time, n_t) + 1, (1,)))
            if width:
                start = int(torch.randint(0, n_t - width + 1, (1,)))
                out[i, :, :, start : start + width] = 0
    return out
