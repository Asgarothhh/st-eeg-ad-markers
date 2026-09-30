"""Обучение ST-Conformer на SEED: ранняя остановка по val macro-F1."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .augment import mask_channels_time, mixup_different_trials
from .config import Config
from .dataset import build_subject_split, make_loaders
from .evaluate import metrics_dict, save_results_table, write_json
from .plots import overfitting_gap, plot_confusion, plot_learning_curves
from .utils import build_model, get_device, set_seed


def _batch_xy(batch):
    return batch[0], batch[1]


@torch.no_grad()
def run_eval(model: nn.Module, loader, device: torch.device, criterion=None):
    model.eval()
    all_y = []
    all_pred = []
    total_loss = 0.0
    n_seen = 0
    for batch in loader:
        x, y = _batch_xy(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            total_loss += float(criterion(logits, y).item()) * y.size(0)
        pred = logits.argmax(dim=1)
        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        n_seen += y.size(0)
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    loss = total_loss / max(n_seen, 1) if criterion is not None else float("nan")
    return acc, f1, loss, y_true, y_pred


def train_subject(cfg: Config, subject: int, device: torch.device) -> dict:
    set_seed(cfg.seed + subject)
    split = build_subject_split(cfg, subject)
    train_loader, val_loader, test_loader = make_loaders(split, cfg)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    out_dir = cfg.subject_dir(subject)
    best_path = out_dir / "best.pt"
    log_path = out_dir / "log.csv"
    log_path.write_text(
        "epoch,train_loss,train_acc,val_loss,val_acc,val_f1,test_acc,test_f1,lr\n"
    )

    best_val_f1 = -1.0
    best_epoch = 0
    best_bundle = None
    patience_left = cfg.patience
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        n_correct = 0
        for batch in train_loader:
            x, y = _batch_xy(batch)
            trials = batch[2] if len(batch) > 2 else None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if trials is not None:
                x, y_a, y_b, lam = mixup_different_trials(x, y, trials, cfg.mixup_alpha)
            else:
                y_a, y_b, lam = y, y, 1.0
            x = mask_channels_time(x, cfg.mask_channels, cfg.mask_time)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if lam >= 1.0 - 1e-6:
                loss = criterion(logits, y_a)
                target = y_a
            else:
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
                target = y_a if lam >= 0.5 else y_b
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running_loss += float(loss.item()) * y.size(0)
            n_seen += y.size(0)
            n_correct += int((logits.argmax(1) == target).sum().item())

        scheduler.step()
        train_acc = n_correct / max(n_seen, 1)
        train_loss = running_loss / max(n_seen, 1)
        val_acc, val_f1, val_loss, _, _ = run_eval(model, val_loader, device, criterion)
        # Тест только в журнал; чекпоинт выбирается по val macro-F1.
        test_acc, test_f1, _, y_true, y_pred = run_eval(model, test_loader, device, criterion)
        lr = optimizer.param_groups[0]["lr"]
        with log_path.open("a") as fh:
            fh.write(
                f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},"
                f"{val_acc:.6f},{val_f1:.6f},{test_acc:.6f},{test_f1:.6f},{lr:.8f}\n"
            )
        print(
            f"субъект {subject:02d}  эпоха {epoch:03d}  "
            f"loss {train_loss:.4f}  обуч. {train_acc:.3f}  "
            f"вал.F1 {val_f1:.3f}  вал. {val_acc:.3f}  тест {test_acc:.3f}",
            flush=True,
        )

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            gap = overfitting_gap(train_acc, val_acc)
            best_bundle = {
                "model": model.state_dict(),
                "subject": subject,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "train_acc": train_acc,
                "gap": gap,
                "mean": split.mean,
                "std": split.std,
                "y_true": y_true,
                "y_pred": y_pred,
            }
            torch.save(
                {k: v for k, v in best_bundle.items() if k not in ("y_true", "y_pred")},
                best_path,
            )
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"субъект {subject:02d}: ранняя остановка на эпохе {epoch}", flush=True)
                break

    elapsed = time.time() - t0
    y_true = best_bundle["y_true"]
    y_pred = best_bundle["y_pred"]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.n_classes)))
    plot_learning_curves(log_path, out_dir / "learning_curves.png", selected_epoch=best_epoch)
    plot_confusion(y_true, y_pred, cfg, out_dir / "confusion.png")

    result = {
        "subject": subject,
        "best_epoch": best_epoch,
        "train_acc": best_bundle["train_acc"],
        "val_acc": best_bundle["val_acc"],
        "val_f1": best_bundle["val_f1"],
        "test_acc": best_bundle["test_acc"],
        "test_f1": best_bundle["test_f1"],
        "gap": best_bundle["gap"],
        "n_train": int(split.train_x.shape[0]),
        "n_val": int(split.val_x.shape[0]),
        "n_test": int(split.test_x.shape[0]),
        "seconds": elapsed,
        "confusion": cm.tolist(),
        "checkpoint": str(best_path),
    }
    write_json(out_dir / "metrics.json", result)
    np.save(out_dir / "y_true.npy", y_true)
    np.save(out_dir / "y_pred.npy", y_pred)
    print(
        f"субъект {subject:02d}: val F1={best_bundle['val_f1']:.4f}  "
        f"тест={best_bundle['test_acc']:.4f}  "
        f"разрыв обуч.–вал.={best_bundle['gap']['gap']:.3f}  "
        f"({elapsed / 60:.1f} мин)",
        flush=True,
    )
    return result


def parse_subjects(spec: str, n_subjects: int) -> list[int]:
    if spec == "all":
        return list(range(1, n_subjects + 1))
    subjects = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            subjects.extend(range(int(a), int(b) + 1))
        else:
            subjects.append(int(part))
    return subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение ST-Conformer на SEED (de_LDS, 1 с)")
    parser.add_argument("--subjects", default="all", help="all, 1, 1,2,5 или 1-15")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--features-dir", default=None, help="Каталог ExtractedFeatures_1s")
    parser.add_argument("--quick", action="store_true", help="Две эпохи, субъект 1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.features_dir:
        from pathlib import Path

        cfg.features_dir = Path(args.features_dir)
    if args.quick:
        cfg.epochs = 2
        cfg.patience = 2
        subjects = [1]
    else:
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        subjects = parse_subjects(args.subjects, cfg.n_subjects)

    if not cfg.features_dir.is_dir():
        raise SystemExit(f"Нет каталога с признаками: {cfg.features_dir}")

    device = torch.device(args.device) if args.device else get_device()
    print(
        f"устройство={device}  субъекты={subjects}  эпохи={cfg.epochs}  "
        f"окно={cfg.window_len}/{cfg.window_stride}c  головы={cfg.num_heads}  "
        f"данные={cfg.features_dir}",
        flush=True,
    )

    results = []
    for subject in subjects:
        results.append(train_subject(cfg, subject, device))
    summary = metrics_dict(results)
    write_json(cfg.model_dir() / "summary.json", summary)
    save_results_table(cfg.results_dir, results)
    print(
        f"среднее тест: {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}  "
        f"F1 {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
