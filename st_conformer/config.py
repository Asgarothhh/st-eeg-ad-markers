"""Пути, протокол SEED и гиперпараметры ST-Conformer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_ROOT = PROJECT_ROOT / "data" / "seed"
FEATURES_DIR = SEED_ROOT / "ExtractedFeatures_1s"
LOCS_PATH = SEED_ROOT / "channel_62_pos.locs"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class Config:
    """Конфигурация эксперимента на SEED (Zheng, Lu, 2015)."""

    features_dir: Path = FEATURES_DIR
    locs_path: Path = LOCS_PATH
    results_dir: Path = RESULTS_DIR

    n_channels: int = 62
    n_bands: int = 5
    n_classes: int = 3
    n_subjects: int = 15
    n_trials: int = 15
    # Тест — ролики 10–15, как в subject-dependent протоколе Zheng, Lu.
    # Val — по одному фильму каждого класса из 1–9 (2=нейтрал, 3=негатив, 6=позитив).
    # Обучение — оставшиеся шесть: 1, 4, 5, 7, 8, 9 (по два на класс).
    # Индексы с нуля.
    train_trials: tuple[int, ...] = (0, 3, 4, 6, 7, 8)
    val_trials: tuple[int, ...] = (1, 2, 5)
    test_trials: tuple[int, ...] = (9, 10, 11, 12, 13, 14)

    window_len: int = 60
    window_stride: int = 30
    time_pool: int = 2

    emb_size: int = 40
    hid_channels: int = 40
    temporal_kernel: int = 3
    num_heads: int = 4
    st_depth: int = 2
    ffn_expansion: int = 2
    dropout: float = 0.5
    ffn_dropout: float = 0.5

    batch_size: int = 16
    epochs: int = 80
    lr: float = 2e-4
    weight_decay: float = 1e-3
    label_smoothing: float = 0.05
    patience: int = 12
    seed: int = 2021
    num_workers: int = 0
    grad_clip: float = 1.0
    mixup_alpha: float = 0.4
    mask_channels: int = 8
    mask_time: int = 12

    class_names: tuple[str, ...] = ("негатив", "нейтрал", "позитив")
    model_name: str = "st"
    baseline_name: str = "linear"

    def __post_init__(self) -> None:
        assigned = [*self.train_trials, *self.val_trials, *self.test_trials]
        if len(assigned) != len(set(assigned)):
            raise ValueError("Ролики train / val / test пересекаются")
        if sorted(assigned) != list(range(self.n_trials)):
            raise ValueError("Все 15 роликов должны входить ровно в одну часть выборки")

    def model_dir(self) -> Path:
        path = self.results_dir / self.model_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def baseline_dir(self) -> Path:
        path = self.results_dir / self.baseline_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def subject_dir(self, subject: int) -> Path:
        path = self.model_dir() / f"subject_{subject:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path
