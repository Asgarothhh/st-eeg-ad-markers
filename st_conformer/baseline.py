"""Линейный SVM на среднем DE (62 × 5) — база для тех же сплитов SEED."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import Config
from .dataset import build_subject_split
from .evaluate import metrics_dict, save_results_table, write_json
from .plots import overfitting_gap
from .train import parse_subjects
from .utils import set_seed


def pooled_features(windows: np.ndarray) -> np.ndarray:
    """Окна `[N, 5, 62, T]` → среднее по времени `[N, 310]`."""
    return windows.mean(axis=3).reshape(windows.shape[0], -1)


def _score(clf, x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    pred = clf.predict(x)
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, average="macro"))
    return acc, f1, pred


def fit_subject(cfg: Config, subject: int) -> dict:
    set_seed(cfg.seed + subject)
    split = build_subject_split(cfg, subject)
    x_train = pooled_features(split.train_x)
    x_val = pooled_features(split.val_x)
    x_test = pooled_features(split.test_x)

    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, dual=False, max_iter=8000),
    )
    clf.fit(x_train, split.train_y)

    train_acc, train_f1, _ = _score(clf, x_train, split.train_y)
    val_acc, val_f1, _ = _score(clf, x_val, split.val_y)
    test_acc, test_f1, y_pred = _score(clf, x_test, split.test_y)
    gap = overfitting_gap(train_acc, val_acc)
    cm = confusion_matrix(split.test_y, y_pred, labels=list(range(cfg.n_classes)))

    out_dir = cfg.baseline_dir() / f"subject_{subject:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "subject": subject,
        "best_epoch": 0,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "gap": gap,
        "n_train": int(split.train_x.shape[0]),
        "n_val": int(split.val_x.shape[0]),
        "n_test": int(split.test_x.shape[0]),
        "confusion": cm.tolist(),
        "checkpoint": "",
    }
    write_json(out_dir / "metrics.json", result)
    print(
        f"субъект {subject:02d}  SVM  обуч. {train_acc:.3f}  "
        f"вал.F1 {val_f1:.3f}  тест {test_acc:.3f}",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Линейный SVM на среднем DE SEED")
    parser.add_argument("--subjects", default="all", help="all, 1, 1,2,5 или 1-15")
    parser.add_argument("--features-dir", default=None, help="Каталог ExtractedFeatures_1s")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.features_dir:
        cfg.features_dir = Path(args.features_dir)
    if not cfg.features_dir.is_dir():
        raise SystemExit(f"Нет каталога с признаками: {cfg.features_dir}")
    subjects = parse_subjects(args.subjects, cfg.n_subjects)
    results = [fit_subject(cfg, subject) for subject in subjects]
    summary = metrics_dict(results)
    write_json(cfg.baseline_dir() / "summary.json", summary)
    save_results_table(cfg.baseline_dir(), results)
    print(
        f"SVM среднее тест: {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}  "
        f"F1 {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
