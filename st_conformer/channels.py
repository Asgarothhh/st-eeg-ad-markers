"""Монтаж 62 электродов SEED и пары для фронтальной асимметрии.

Порядок каналов совпадает с `channel_62_pos.locs` набора SEED
(система 10–20; Zheng, Lu, 2015). Координаты читаются из файла,
если он есть, иначе используется запасная таблица.
"""

from __future__ import annotations

import math
from pathlib import Path

# Порядок как в channel_62_pos.locs / channel-order.xlsx
CHANNEL_NAMES: list[str] = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]

assert len(CHANNEL_NAMES) == 62

# Левые и правые фронтальные пары: FAA (Davidson, 1992; Harmon-Jones, Gable, 2018)
FAA_PAIRS: list[tuple[str, str]] = [
    ("FP1", "FP2"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("AF3", "AF4"),
    ("FC3", "FC4"),
]

CHANNEL_INDEX: dict[str, int] = {name: i for i, name in enumerate(CHANNEL_NAMES)}

# Запасные координаты единичной сферы (x — вправо, y — вперёд, z — вверх)
_FALLBACK_XYZ: dict[str, tuple[float, float, float]] = {
    "FP1": (-0.31, 0.95, -0.03),
    "FPZ": (0.00, 0.99, -0.03),
    "FP2": (0.31, 0.95, -0.03),
    "AF3": (-0.34, 0.83, 0.32),
    "AF4": (0.34, 0.83, 0.32),
    "F7": (-0.81, 0.59, -0.03),
    "F5": (-0.67, 0.62, 0.32),
    "F3": (-0.50, 0.71, 0.50),
    "F1": (-0.27, 0.75, 0.61),
    "FZ": (0.00, 0.72, 0.69),
    "F2": (0.27, 0.75, 0.61),
    "F4": (0.50, 0.71, 0.50),
    "F6": (0.67, 0.62, 0.32),
    "F8": (0.81, 0.59, -0.03),
    "FT7": (-0.95, 0.31, -0.03),
    "FC5": (-0.78, 0.34, 0.45),
    "FC3": (-0.56, 0.40, 0.68),
    "FC1": (-0.29, 0.41, 0.86),
    "FCZ": (0.00, 0.39, 0.92),
    "FC2": (0.29, 0.41, 0.86),
    "FC4": (0.56, 0.40, 0.68),
    "FC6": (0.78, 0.34, 0.45),
    "FT8": (0.95, 0.31, -0.03),
    "T7": (-1.00, 0.00, -0.03),
    "C5": (-0.83, 0.00, 0.45),
    "C3": (-0.59, 0.00, 0.73),
    "C1": (-0.31, 0.00, 0.91),
    "CZ": (0.00, 0.00, 1.00),
    "C2": (0.31, 0.00, 0.91),
    "C4": (0.59, 0.00, 0.73),
    "C6": (0.83, 0.00, 0.45),
    "T8": (1.00, 0.00, -0.03),
    "TP7": (-0.95, -0.31, -0.03),
    "CP5": (-0.78, -0.34, 0.45),
    "CP3": (-0.56, -0.40, 0.68),
    "CP1": (-0.29, -0.41, 0.86),
    "CPZ": (0.00, -0.39, 0.92),
    "CP2": (0.29, -0.41, 0.86),
    "CP4": (0.56, -0.40, 0.68),
    "CP6": (0.78, -0.34, 0.45),
    "TP8": (0.95, -0.31, -0.03),
    "P7": (-0.81, -0.59, -0.03),
    "P5": (-0.67, -0.62, 0.32),
    "P3": (-0.50, -0.71, 0.50),
    "P1": (-0.27, -0.75, 0.61),
    "PZ": (0.00, -0.72, 0.69),
    "P2": (0.27, -0.75, 0.61),
    "P4": (0.50, -0.71, 0.50),
    "P6": (0.67, -0.62, 0.32),
    "P8": (0.81, -0.59, -0.03),
    "PO7": (-0.67, -0.74, -0.03),
    "PO5": (-0.50, -0.78, 0.25),
    "PO3": (-0.34, -0.83, 0.32),
    "POZ": (0.00, -0.86, 0.39),
    "PO4": (0.34, -0.83, 0.32),
    "PO6": (0.50, -0.78, 0.25),
    "PO8": (0.67, -0.74, -0.03),
    "CB1": (-0.40, -0.91, -0.15),
    "O1": (-0.31, -0.95, -0.03),
    "OZ": (0.00, -0.99, -0.03),
    "O2": (0.31, -0.95, -0.03),
    "CB2": (0.40, -0.91, -0.15),
}


def _normalize_name(name: str) -> str:
    return name.strip().upper()


def _eeglab_to_xyz(theta_deg: float, radius: float) -> tuple[float, float, float]:
    """Перевод полярных координат EEGLAB (.locs) в декартовы.

    В файле SEED угол θ = 0 соответствует носовому направлению (Fpz),
    θ < 0 — левому полушарию.
    """
    theta = math.radians(theta_deg)
    x = radius * math.sin(theta)
    y = radius * math.cos(theta)
    z_sq = max(0.0, 1.0 - x * x - y * y)
    z = math.sqrt(z_sq)
    return x, y, z


def load_xyz(locs_path: Path | None = None) -> list[list[float]]:
    """Матрица (62, 3) в порядке CHANNEL_NAMES."""
    table = dict(_FALLBACK_XYZ)
    if locs_path is not None and locs_path.is_file():
        for line in locs_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                theta = float(parts[1])
                radius = float(parts[2])
            except ValueError:
                continue
            name = _normalize_name(parts[3])
            table[name] = _eeglab_to_xyz(theta, radius)
    return [list(table[name]) for name in CHANNEL_NAMES]


def xyz_matrix() -> list[list[float]]:
    from .config import LOCS_PATH

    return load_xyz(LOCS_PATH)


def faa_indices() -> list[tuple[int, int]]:
    return [(CHANNEL_INDEX[left], CHANNEL_INDEX[right]) for left, right in FAA_PAIRS]
