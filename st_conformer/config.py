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
    # Канонический subject-dependent протокол: ролики 1–9 / 10–15.
    # Из обучающих роликов 8 и 9 откладываются в валидацию,
    # чтобы ранняя остановка не смотрела на тест.
    n_train_trials: int = 7
    n_val_trials: int = 2

    window_len: int = 60
    window_stride: int = 10
    time_pool: int = 2

    emb_size: int = 40
    hid_channels: int = 40
    temporal_kernel: int = 3
    num_heads: int = 8
    st_depth: int = 2
    ffn_expansion: int = 2
    dropout: float = 0.5
    ffn_dropout: float = 0.5

    batch_size: int = 64
    epochs: int = 80
    lr: float = 2e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    patience: int = 12
    seed: int = 2021
    num_workers: int = 0
    grad_clip: float = 1.0

    class_names: tuple[str, ...] = ("негатив", "нейтрал", "позитив")
    model_name: str = "st"

    def model_dir(self) -> Path:
        path = self.results_dir / self.model_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def subject_dir(self, subject: int) -> Path:
        path = self.model_dir() / f"subject_{subject:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path
