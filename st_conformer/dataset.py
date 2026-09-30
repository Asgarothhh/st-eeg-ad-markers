"""Загрузка DE-признаков SEED (`de_LDS`) с разбиением train / val / test.

Используется каталог `data/seed/ExtractedFeatures_1s`: для каждого ролика
ключ `de_LDSk` имеет форму (62 канала, T секундных окон, 5 полос).
Метки в `label.mat`: +1 / 0 / −1 (позитив / нейтрал / негатив).

Протокол subject-dependent по Zheng, Lu (2015): ролики 1–9 — обучение,
10–15 — тест. Ролики 8–9 из обучающей части откладываются в валидацию.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config

# Отображение меток SEED на индексы классов классификатора
LABEL_TO_INDEX = {-1: 0, 0: 1, 1: 2}


def discover_session_files(features_dir: Path, subject: int) -> list[Path]:
    files = sorted(features_dir.glob(f"{subject}_*.mat"))
    if len(files) != 3:
        raise FileNotFoundError(
            f"Ожидались 3 сессии субъекта {subject} в {features_dir}, найдено {len(files)}"
        )
    return files


def load_seed_labels(features_dir: Path) -> np.ndarray:
    mat = scipy.io.loadmat(str(features_dir / "label.mat"))
    key = "label" if "label" in mat else next(k for k in mat if not k.startswith("__"))
    labels = np.array(mat[key]).reshape(-1).astype(np.int64)
    if labels.shape[0] != 15:
        raise ValueError(f"Ожидалось 15 меток роликов, получено {labels.shape}")
    return labels


def _trial_array(mat: dict, trial: int) -> np.ndarray:
    key = f"de_LDS{trial}"
    if key not in mat:
        raise KeyError(f"В файле сессии нет ключа {key}")
    de = np.asarray(mat[key], dtype=np.float32)
    if de.ndim != 3 or de.shape[0] != 62 or de.shape[2] != 5:
        raise ValueError(f"{key}: форма {de.shape}, ожидалось (62, T, 5)")
    return de


def load_session_trials(path: Path) -> list[np.ndarray]:
    mat = scipy.io.loadmat(str(path), verify_compressed_data_integrity=False)
    return [_trial_array(mat, trial) for trial in range(1, 16)]


def sliding_windows(de: np.ndarray, window_len: int, stride: int) -> np.ndarray:
    """Нарезать ролик на окна: (62, T, 5) → (N, 5, 62, window_len)."""
    channels, n_times, bands = de.shape
    if n_times < window_len:
        return np.empty((0, bands, channels, window_len), dtype=np.float32)
    starts = range(0, n_times - window_len + 1, stride)
    windows = np.stack(
        [np.transpose(de[:, s : s + window_len, :], (2, 0, 1)) for s in starts],
        axis=0,
    )
    return windows


def _split_of_trial(trial_idx: int, cfg: Config) -> str:
    val_end = cfg.n_train_trials + cfg.n_val_trials
    if trial_idx < cfg.n_train_trials:
        return "train"
    if trial_idx < val_end:
        return "val"
    return "test"


@dataclass
class SubjectSplit:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    sessions: list[list[np.ndarray]]
    labels: np.ndarray


def build_subject_split(
    cfg: Config,
    subject: int,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> SubjectSplit:
    labels = load_seed_labels(cfg.features_dir)
    class_ids = np.array([LABEL_TO_INDEX[int(v)] for v in labels], dtype=np.int64)
    sessions = [load_session_trials(path) for path in discover_session_files(cfg.features_dir, subject)]

    buckets: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    y_buckets: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}

    for trials in sessions:
        for trial_idx, de in enumerate(trials):
            windows = sliding_windows(de, cfg.window_len, cfg.window_stride)
            if windows.size == 0:
                continue
            y = np.full((windows.shape[0],), class_ids[trial_idx], dtype=np.int64)
            part = _split_of_trial(trial_idx, cfg)
            buckets[part].append(windows)
            y_buckets[part].append(y)

    train_x = np.concatenate(buckets["train"], axis=0)
    train_y = np.concatenate(y_buckets["train"], axis=0)
    val_x = np.concatenate(buckets["val"], axis=0)
    val_y = np.concatenate(y_buckets["val"], axis=0)
    test_x = np.concatenate(buckets["test"], axis=0)
    test_y = np.concatenate(y_buckets["test"], axis=0)

    if mean is None or std is None:
        mean = train_x.mean(axis=(0, 3), keepdims=True)
        std = np.clip(train_x.std(axis=(0, 3), keepdims=True), 1e-6, None)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    return SubjectSplit(
        train_x=train_x.astype(np.float32),
        train_y=train_y,
        val_x=val_x.astype(np.float32),
        val_y=val_y,
        test_x=test_x.astype(np.float32),
        test_y=test_y,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        sessions=sessions,
        labels=class_ids,
    )


class WindowDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


def make_loaders(split: SubjectSplit, cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    def loader(x, y, shuffle: bool) -> DataLoader:
        return DataLoader(
            WindowDataset(x, y),
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            drop_last=False,
        )

    return (
        loader(split.train_x, split.train_y, True),
        loader(split.val_x, split.val_y, False),
        loader(split.test_x, split.test_y, False),
    )


def full_trial_tensor(de: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """Один ролик целиком: [1, 5, 62, T], нормированный статистиками обучения."""
    x = np.transpose(de, (2, 0, 1))[None, ...].astype(np.float32)
    x = (x - mean) / std
    return torch.from_numpy(x)
