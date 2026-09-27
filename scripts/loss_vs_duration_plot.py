"""Joint plot of model loss vs utterance duration for a given checkpoint."""

import logging
import os
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data.datasets.librispeech import collate_fn
from src.pipeline.stage_01_data_preparation import run_load_datasets
from src.pipeline.stage_03_training import run_create_model

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
for _lib in ("torch", "hydra", "omegaconf", "mlflow", "sklearn", "espnet", "espnet2",
             "matplotlib", "transformers", "datasets", "PIL", "urllib3", "asyncio"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

torch.multiprocessing.set_sharing_strategy("file_system")


def _strip_lightning_prefix(state_dict: dict) -> dict:
    return {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }


def _load_checkpoint(model, ckpt_path: str, device) -> None:
    print(f"[loss_vs_dur] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = _strip_lightning_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[loss_vs_dur] WARNING: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[loss_vs_dur] WARNING: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")


def _collect_loss_and_duration(model, dataset, device, num_workers: int, max_samples: int,
                                sample_rate: int, desc: str) -> pd.DataFrame:
    """Returns a DataFrame with columns [speaker_id, duration_sec, loss]."""
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,
    )

    rows = []
    model.eval()
    model.to(device)

    for batch in tqdm(loader, desc=desc):
        if max_samples > 0 and len(rows) >= max_samples:
            break

        wav_lens = batch["wav_lens"]  # tensor of shape (1,) — actual sample count
        duration_sec = float(wav_lens[0].item()) / sample_rate
        speaker_id = batch["speaker_id"][0] if "speaker_id" in batch else "unknown"

        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            loss = model(batch_on_device)

        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu()
        if loss.ndim == 0:
            loss = loss.unsqueeze(0)

        rows.append({
            "speaker_id": str(speaker_id),
            "duration_sec": duration_sec,
            "loss": float(loss[0].item()),
        })

    return pd.DataFrame(rows)


def _aggregate_by_speaker(df: pd.DataFrame) -> pd.DataFrame:
    """Mean loss and total duration per speaker."""
    return (
        df.groupby("speaker_id")
        .agg(
            duration_sec=("duration_sec", "mean"),
            loss=("loss", "mean"),
            num_utterances=("loss", "count"),
        )
        .reset_index()
    )


_PALETTE = {"train": "#2196F3", "test": "#F44336", "forget": "#FF9800", "retain": "#4CAF50"}
_DEFAULT_COLOR = "#9C27B0"


def _single_jointplot(df: pd.DataFrame, color: str,
                      xlabel: str, ylabel: str, title: str) -> sns.JointGrid:
    g = sns.JointGrid(data=df, x="duration_sec", y="loss", height=6)
    g.plot_joint(sns.scatterplot, alpha=0.35, s=12, color=color, linewidth=0)
    g.plot_marginals(sns.kdeplot, fill=True, color=color, alpha=0.4)
    sns.regplot(
        data=df, x="duration_sec", y="loss",
        ax=g.ax_joint, scatter=False, color=color,
        line_kws={"linewidth": 1.5, "linestyle": "--"},
    )
    r = df["duration_sec"].corr(df["loss"])
    g.ax_joint.annotate(
        f"r = {r:.3f}",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    g.set_axis_labels(xlabel, ylabel)
    g.figure.suptitle(title, y=1.01, fontsize=12)
    return g


def _make_jointplot(dfs: dict[str, pd.DataFrame], output_path: str,
                    xlabel: str = "Utterance duration (s)",
                    ylabel: str = "Model loss",
                    tag: str = "") -> None:
    """dfs: split_name -> DataFrame with [duration_sec, loss]."""
    base, ext = os.path.splitext(output_path)
    ext = ext or ".pdf"

    for split_name, df in dfs.items():
        if df.empty:
            print(f"[loss_vs_dur] Skipping {split_name}{tag} — no data.")
            continue

        color = _PALETTE.get(split_name, _DEFAULT_COLOR)
        g = _single_jointplot(
            df, color, xlabel, ylabel,
            title=f"{ylabel} vs {xlabel} — {split_name}{tag}",
        )
        out = f"{base}{tag}_{split_name}{ext}"
        g.figure.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(g.figure)
        print(f"[loss_vs_dur] Saved: {out}")

    if len(dfs) > 1:
        combined = pd.concat(
            [df.assign(split=name) for name, df in dfs.items() if not df.empty],
            ignore_index=True,
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        for split_name, grp in combined.groupby("split"):
            color = _PALETTE.get(split_name, _DEFAULT_COLOR)
            ax.scatter(grp["duration_sec"], grp["loss"],
                       alpha=0.3, s=10, label=split_name, color=color, linewidths=0)
            sns.regplot(data=grp, x="duration_sec", y="loss",
                        ax=ax, scatter=False, color=color,
                        line_kws={"linewidth": 1.5, "linestyle": "--"})
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs {xlabel} — all splits{tag}")
        ax.legend()
        fig.tight_layout()
        out_combined = f"{base}{tag}_combined{ext}"
        fig.savefig(out_combined, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[loss_vs_dur] Saved: {out_combined}")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    lvd = cfg.get("loss_vs_dur", None)
    if lvd is None:
        raise ValueError(
            "Provide at least +loss_vs_dur.checkpoint_path=... "
            "(or +loss_vs_dur.pretrained=true for the base model)."
        )

    checkpoint_path = lvd.get("checkpoint_path", None)
    use_pretrained   = bool(lvd.get("pretrained", False))
    output_path      = str(lvd.get("output_path", "loss_vs_duration.pdf"))
    sets_str         = str(lvd.get("sets", "train,test"))
    sets_to_eval     = [s.strip() for s in sets_str.split(",") if s.strip()]
    sample_rate      = int(lvd.get("sample_rate", 16000))
    num_workers      = int(lvd.get("num_workers", cfg.evaluation.num_workers))
    max_samples      = int(lvd.get("max_samples", 0))

    if not use_pretrained and not checkpoint_path:
        raise ValueError("Provide +loss_vs_dur.checkpoint_path=... or +loss_vs_dur.pretrained=true")
    if checkpoint_path and not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[loss_vs_dur] Using device: {device}")

    # ── Load datasets (no MLflow/caching side-effects needed for a plot script)
    from unittest.mock import MagicMock

    class _DummyArtifacts:
        tracker = MagicMock()

        def log_metrics(self, *a, **kw):
            pass

        def log_artifact(self, *a, **kw):
            pass

        def get_cached(self, artifact_type, config, loader, suffix=None):
            # Always recompute — no persistent cache for the plot script
            return loader(), False

        def start_run(self, *a, **kw):
            return None

        def end_run(self, *a, **kw):
            pass

    artifacts = _DummyArtifacts()
    data_handler = run_load_datasets(cfg, artifacts)

    print(f"[loss_vs_dur] Datasets loaded. Available sets: {list(data_handler.datasets.keys())}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = run_create_model(cfg=cfg)

    if use_pretrained:
        print("[loss_vs_dur] Using pretrained model weights (no checkpoint loaded).")
    else:
        _load_checkpoint(model, checkpoint_path, device)

    model.to(device)
    model.eval()

    # ── Collect loss + duration per utterance for each requested split ───────
    utt_dfs: dict[str, pd.DataFrame] = {}
    for split in sets_to_eval:
        if split not in data_handler.datasets:
            print(f"[loss_vs_dur] WARNING: split '{split}' not found — skipping.")
            continue

        df = _collect_loss_and_duration(
            model=model,
            dataset=data_handler.datasets[split],
            device=device,
            num_workers=num_workers,
            max_samples=max_samples,
            sample_rate=sample_rate,
            desc=f"[{split}] loss + duration",
        )
        print(f"[loss_vs_dur] {split}: {len(df)} utterances, "
              f"duration {df['duration_sec'].min():.1f}–{df['duration_sec'].max():.1f}s, "
              f"loss {df['loss'].mean():.4f} ± {df['loss'].std():.4f}")
        utt_dfs[split] = df

    if not utt_dfs:
        raise RuntimeError("No data collected — check that the requested splits exist.")

    # ── Per-utterance joint plots ─────────────────────────────────────────────
    _make_jointplot(
        utt_dfs, output_path,
        xlabel="Utterance duration (s)",
        ylabel="Model loss",
        tag="",
    )

    # ── Per-speaker joint plots (mean loss and mean duration per speaker) ────
    spk_dfs = {split: _aggregate_by_speaker(df) for split, df in utt_dfs.items()}
    for split, df in spk_dfs.items():
        print(f"[loss_vs_dur] {split} (speaker-level): {len(df)} speakers, "
              f"mean duration {df['duration_sec'].min():.1f}–{df['duration_sec'].max():.1f}s, "
              f"mean loss {df['loss'].mean():.4f} ± {df['loss'].std():.4f}")

    _make_jointplot(
        spk_dfs, output_path,
        xlabel="Mean utterance duration (s)",
        ylabel="Mean model loss",
        tag="_speaker",
    )

    print("[loss_vs_dur] Done.")


if __name__ == "__main__":
    main()
