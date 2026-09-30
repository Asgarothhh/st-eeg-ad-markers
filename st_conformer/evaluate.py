"""Оценка сохранённого чекпоинта и сводная таблица по субъектам."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import Config
from .dataset import build_subject_split, make_loaders
from .plots import overfitting_gap, plot_confusion, plot_learning_curves
from .utils import build_model, get_device


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def metrics_dict(results: list[dict]) -> dict:
    acc = np.array([r["test_acc"] for r in results], dtype=np.float64)
    f1 = np.array([r["test_f1"] for r in results], dtype=np.float64)
    val = np.array([r.get("val_acc", np.nan) for r in results], dtype=np.float64)
    val_finite = val[np.isfinite(val)]
    return {
        "n_subjects": len(results),
        "acc_mean": float(np.nanmean(acc)),
        "acc_std": float(np.nanstd(acc, ddof=1) if len(acc) > 1 else 0.0),
        "f1_mean": float(np.nanmean(f1)),
        "f1_std": float(np.nanstd(f1, ddof=1) if len(f1) > 1 else 0.0),
        "val_mean": float(val_finite.mean()) if val_finite.size else None,
        "per_subject": results,
    }


def save_results_table(results_dir: Path, results: list[dict]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    lines = ["субъект,эпоха,обуч,вал,тест,F1,разрыв"]
    for row in sorted(results, key=lambda r: r["subject"]):
        gap = row.get("gap", {})
        gap_v = gap.get("gap", "") if isinstance(gap, dict) else ""
        epoch = row.get("best_epoch", "")
        lines.append(
            f"{row['subject']},{epoch},{row.get('train_acc', float('nan')):.6f},"
            f"{row.get('val_acc', float('nan')):.6f},{row['test_acc']:.6f},"
            f"{row['test_f1']:.6f},{gap_v}"
        )
    acc = np.array([r["test_acc"] for r in results])
    f1 = np.array([r["test_f1"] for r in results])
    lines.append(f"среднее,,,,{acc.mean():.6f},{f1.mean():.6f},")
    if len(acc) > 1:
        lines.append(f"стд,,,,{acc.std(ddof=1):.6f},{f1.std(ddof=1):.6f},")
    path = results_dir / "summary.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@torch.no_grad()
def score_loader(model, loader, device):
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        ys.append(y.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro")),
        y_true,
        y_pred,
    )


def evaluate_subject(cfg: Config, subject: int, device: torch.device) -> dict:
    ckpt_path = cfg.subject_dir(subject) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Нет чекпоинта {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    split = build_subject_split(
        cfg,
        subject,
        mean=ckpt.get("mean"),
        std=ckpt.get("std"),
    )
    train_loader, val_loader, test_loader = make_loaders(split, cfg)
    train_acc, train_f1, _, _ = score_loader(model, train_loader, device)
    val_acc, val_f1, _, _ = score_loader(model, val_loader, device)
    test_acc, test_f1, y_true, y_pred = score_loader(model, test_loader, device)
    gap = overfitting_gap(train_acc, val_acc)

    out_dir = cfg.subject_dir(subject)
    plot_confusion(y_true, y_pred, cfg, out_dir / "confusion.png")
    log_path = out_dir / "log.csv"
    if log_path.exists():
        plot_learning_curves(log_path, out_dir / "learning_curves.png", selected_epoch=ckpt.get("epoch"))

    report = {
        "subject": subject,
        "best_epoch": ckpt.get("epoch"),
        "train_acc": train_acc,
        "train_f1": train_f1,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "gap": gap,
        "checkpoint": str(ckpt_path),
    }
    write_json(out_dir / "eval.json", report)
    print(
        f"субъект {subject:02d}  обуч. {train_acc:.3f}  вал. {val_acc:.3f}  "
        f"тест {test_acc:.3f}  разрыв {gap['gap']:.3f}"
        + ("  (большой разрыв)" if gap["overfit"] else "  (переобучения нет)")
    )
    return report


def collect_from_disk(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("st/subject_*/metrics.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка чекпоинта ST-Conformer на SEED")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Сводка по всем найденным metrics.json")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    device = torch.device(args.device) if args.device else get_device()

    if args.all and args.subject is None:
        rows = collect_from_disk(cfg.results_dir)
        if not rows:
            raise SystemExit(f"Нет metrics.json в {cfg.results_dir}")
        # старые прогоны писали best_acc вместо test_acc
        for row in rows:
            if "test_acc" not in row and "best_acc" in row:
                row["test_acc"] = row["best_acc"]
                row["test_f1"] = row.get("best_f1", float("nan"))
                row["val_acc"] = float("nan")
                row["train_acc"] = float("nan")
        summary = metrics_dict(rows)
        write_json(cfg.model_dir() / "summary.json", summary)
        save_results_table(cfg.results_dir, rows)
        for row in rows:
            subj = row["subject"]
            log_path = cfg.subject_dir(subj) / "log.csv"
            if log_path.exists():
                plot_learning_curves(
                    log_path,
                    cfg.subject_dir(subj) / "learning_curves.png",
                    selected_epoch=row.get("best_epoch") or row.get("epoch"),
                )
        print(
            f"тест {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}  "
            f"F1 {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}"
        )
        return

    if args.subject is None:
        raise SystemExit("Укажите --subject N или --all")
    evaluate_subject(cfg, args.subject, device)


if __name__ == "__main__":
    main()
