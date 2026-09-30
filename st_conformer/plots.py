"""Подписи графиков на русском, кривые обучения и матрица ошибок."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .config import Config


def setup_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )


def plot_learning_curves(log_path: Path, out_path: Path, selected_epoch: int | None = None) -> None:
    """Кривые train / val (и test, если он есть в журнале, только для справки)."""
    setup_plots()
    rows = np.genfromtxt(log_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if rows.size == 0:
        return
    epoch = np.atleast_1d(rows["epoch"])
    train_acc = np.atleast_1d(rows["train_acc"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    ax = axes[0]
    ax.plot(epoch, np.atleast_1d(rows["train_loss"]), label="обучение", color="#1f4e79")
    if "val_loss" in rows.dtype.names:
        ax.plot(epoch, np.atleast_1d(rows["val_loss"]), label="валидация", color="#c45c26")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Функция потерь")
    ax.set_title("Потери")
    ax.legend()

    ax = axes[1]
    ax.plot(epoch, train_acc, label="обучение", color="#1f4e79")
    if "val_acc" in rows.dtype.names:
        ax.plot(epoch, np.atleast_1d(rows["val_acc"]), label="валидация", color="#c45c26")
    if "test_acc" in rows.dtype.names:
        ax.plot(
            epoch,
            np.atleast_1d(rows["test_acc"]),
            label="тест (справочно)",
            color="#2e7d4f",
            ls=":",
        )
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Точность")
    ax.set_title("Точность")
    ax.set_ylim(0.0, 1.05)
    if selected_epoch is not None:
        ax.axvline(selected_epoch, color="0.4", ls="--", lw=1, label=f"остановка (эп. {selected_epoch})")
    ax.legend()

    fig.suptitle("Точность и потери по эпохам", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, cfg: Config, out_path: Path) -> None:
    setup_plots()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=list(range(cfg.n_classes)),
        display_labels=list(cfg.class_names),
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    disp.ax_.set_xlabel("Прогноз")
    disp.ax_.set_ylabel("Истинный класс")
    disp.ax_.set_title("Матрица ошибок (тест)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def overfitting_gap(train_acc: float, val_acc: float) -> dict:
    """Разрыв обучение–валидация в выбранной эпохе."""
    gap = float(train_acc - val_acc)
    return {
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "gap": gap,
        "overfit": bool(gap > 0.15),
    }
