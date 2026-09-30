"""Карты пространственного и временного внимания для интерпретации ЭЭГ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .channels import CHANNEL_NAMES, faa_indices
from .config import Config
from .dataset import build_subject_split, full_trial_tensor
from .plots import setup_plots
from .utils import build_model, get_device


def _load_checkpoint(path: Path, cfg: Config, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _stack_attention(attn_blocks: list[dict], key: str) -> np.ndarray:
    stacked = torch.stack([block[key] for block in attn_blocks], dim=0)
    return stacked.detach().cpu().numpy()


def summarize_spatial(spatial: np.ndarray) -> dict[str, np.ndarray]:
    # [слои, B, T, головы, C, C]
    heatmap = spatial.mean(axis=(0, 1, 2, 3))
    importance = heatmap.mean(axis=0)
    return {"heatmap": heatmap, "importance": importance}


def summarize_temporal(temporal: np.ndarray) -> np.ndarray:
    # входящее внимание по времени ролика
    return temporal.mean(axis=(0, 1, 2, 3, 4))


def faa_series(spatial: np.ndarray) -> dict[str, np.ndarray]:
    s = spatial.mean(axis=(0, 3))[0]
    out = {}
    for left, right in faa_indices():
        name = f"{CHANNEL_NAMES[left]}–{CHANNEL_NAMES[right]}"
        out[name] = 0.5 * (s[:, left, right] + s[:, right, left])
    return out


def plot_spatial_heatmap(heatmap: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    setup_plots()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(heatmap, cmap="viridis", interpolation="nearest")
    ax.set_title("Пространственное внимание (электрод × электрод)")
    ax.set_xlabel("Ключ")
    ax.set_ylabel("Запрос")
    ticks = np.arange(0, len(CHANNEL_NAMES), 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([CHANNEL_NAMES[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([CHANNEL_NAMES[i] for i in ticks], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Вес внимания")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_topomap(importance: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt
    import mne

    from .channels import xyz_matrix

    setup_plots()
    names = CHANNEL_NAMES
    ch_pos = {name: np.asarray(xyz, dtype=float) for name, xyz in zip(names, xyz_matrix())}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info = mne.create_info(list(names), sfreq=1.0, ch_types="eeg")
    info.set_montage(montage)
    fig, ax = plt.subplots(figsize=(5, 4))
    mne.viz.plot_topomap(importance, info, axes=ax, show=False, contours=4, cmap="RdBu_r")
    ax.set_title("Значимость электродов\n(входящее пространственное внимание)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_temporal(curve: np.ndarray, path: Path, title: str, sec_per_token: float) -> None:
    import matplotlib.pyplot as plt

    setup_plots()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    seconds = np.arange(curve.shape[0]) * sec_per_token
    ax.plot(seconds, curve, color="#1f4e79", lw=1.8)
    ax.set_xlabel("Время ролика, с")
    ax.set_ylabel("Масса внимания")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_faa(series: dict[str, np.ndarray], path: Path, sec_per_token: float) -> None:
    import matplotlib.pyplot as plt

    setup_plots()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for name, values in series.items():
        ax.plot(np.arange(values.shape[0]) * sec_per_token, values, label=name, lw=1.4)
    ax.set_xlabel("Время ролика, с")
    ax.set_ylabel("Внимание влево–вправо")
    ax.set_title("Фронтальные пары (FAA)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def run_trial(
    cfg: Config,
    subject: int,
    session: int,
    trial: int,
    ckpt_path: Path,
    out_dir: Path,
    device: torch.device,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = _load_checkpoint(ckpt_path, cfg, device)
    split = build_subject_split(cfg, subject)
    de = split.sessions[session][trial]
    x = full_trial_tensor(de, ckpt["mean"], ckpt["std"]).to(device)
    logits, attn_blocks = model(x, return_attn=True)
    pred = int(logits.argmax(dim=1).item())
    true = int(split.labels[trial])

    spatial = _stack_attention(attn_blocks, "spatial")
    temporal = _stack_attention(attn_blocks, "temporal")
    spatial_sum = summarize_spatial(spatial)
    temporal_curve = summarize_temporal(temporal)
    faa = faa_series(spatial)
    sec_per_token = float(cfg.time_pool)

    np.save(out_dir / "spatial_attn.npy", spatial)
    np.save(out_dir / "temporal_attn.npy", temporal)
    np.save(out_dir / "spatial_heatmap.npy", spatial_sum["heatmap"])
    np.save(out_dir / "electrode_importance.npy", spatial_sum["importance"])
    np.save(out_dir / "temporal_curve.npy", temporal_curve)

    plot_spatial_heatmap(spatial_sum["heatmap"], out_dir / "spatial_heatmap.png")
    try:
        plot_topomap(spatial_sum["importance"], out_dir / "spatial_topomap.png")
    except Exception as exc:
        print(f"топография пропущена: {exc}")
    plot_temporal(
        temporal_curve,
        out_dir / "temporal_curve.png",
        title=(
            f"Временное внимание, субъект {subject}, сессия {session + 1}, "
            f"ролик {trial + 1}: истина — {cfg.class_names[true]}, "
            f"прогноз — {cfg.class_names[pred]}"
        ),
        sec_per_token=sec_per_token,
    )
    plot_faa(faa, out_dir / "faa_attention.png", sec_per_token)
    print(
        f"карты сохранены в {out_dir}  "
        f"прогноз={cfg.class_names[pred]}  истина={cfg.class_names[true]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Карты внимания ST-Conformer на ролике SEED")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--session", type=int, default=0, help="0, 1 или 2")
    parser.add_argument("--trial", type=int, default=0, help="индекс ролика в сессии, с нуля")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    device = torch.device(args.device) if args.device else get_device()
    ckpt = Path(args.ckpt) if args.ckpt else cfg.subject_dir(args.subject) / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"Чекпоинт не найден: {ckpt}")
    out_dir = (
        cfg.results_dir
        / "attn"
        / f"subject_{args.subject:02d}_sess{args.session}_trial{args.trial + 1}"
    )
    run_trial(cfg, args.subject, args.session, args.trial, ckpt, out_dir, device)


if __name__ == "__main__":
    main()
