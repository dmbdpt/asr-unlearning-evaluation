#!/usr/bin/env python3
"""Unified plotting CLI for the LeaF machine-unlearning framework."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Subcommand: kde
# ---------------------------------------------------------------------------

def cmd_kde(args: argparse.Namespace) -> None:
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.lines import Line2D

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.json_path) as f:
        data = json.load(f)

    forget_subjects = set(args.forget_subjects)

    def _set_losses(split: str, exclude_forget: bool) -> np.ndarray:
        losses = []
        for sub, utt_dict in data.get("per_utt_los", {}).get(split, {}).items():
            if exclude_forget and sub in forget_subjects:
                continue
            if not exclude_forget and sub not in forget_subjects:
                continue
            losses.extend(utt_dict.values())
        return np.array(losses, dtype=float)

    def _legend_line(label: str, color: str, lw: float, vals: np.ndarray) -> Line2D:
        txt = f"{label}  [{vals.min():.1f}, {vals.max():.1f}]  μ={vals.mean():.1f}±{vals.std():.1f}"
        return Line2D([0], [0], color=color, lw=lw, label=txt)

    train_losses = _set_losses("train", exclude_forget=True)
    test_losses = _set_losses("test", exclude_forget=True)

    legend_elements = [
        _legend_line("Train Set", "#1f77b4", 2, train_losses),
        _legend_line("Test Set", "#ff7f0e", 2, test_losses),
    ]
    for sub in sorted(forget_subjects):
        sub_losses = np.concatenate(
            [
                np.array(list(data["per_utt_los"][sp][sub].values()), dtype=float)
                for sp in ["train", "test"]
                if sub in data.get("per_utt_los", {}).get(sp, {})
            ]
        )
        if len(sub_losses):
            legend_elements.append(_legend_line(f"  {sub}", "#2ca02c", 1.5, sub_losses))

    fig, ax = plt.subplots(figsize=(10, 6))
    for split, color in [("train", "#1f77b4"), ("test", "#ff7f0e")]:
        losses = _set_losses(split, exclude_forget=True)
        if len(losses):
            sns.kdeplot(losses, ax=ax, color=color, lw=2, label=split)
    for sub in sorted(forget_subjects):
        sub_losses = np.concatenate(
            [
                np.array(list(data["per_utt_los"][sp][sub].values()), dtype=float)
                for sp in ["train", "test"]
                if sub in data.get("per_utt_los", {}).get(sp, {})
            ]
        )
        if len(sub_losses):
            sns.kdeplot(sub_losses, ax=ax, lw=1.5, label=sub)

    ax.legend(handles=legend_elements, fontsize=8)
    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.set_title("Per-utterance loss distributions")
    plt.tight_layout()
    fig.savefig(out / "loss_distributions_kde.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"KDE plot saved to {out / 'loss_distributions_kde.pdf'}")


# ---------------------------------------------------------------------------
# Subcommand: loss-vs-duration (Hydra-based)
# ---------------------------------------------------------------------------

def cmd_loss_vs_duration(extra_args: list) -> None:
    """Delegate to the loss_vs_duration_plot script via Hydra."""
    import importlib.util, runpy
    script = Path(__file__).parent / "loss_vs_duration_plot.py"
    sys.argv = [str(script)] + extra_args
    runpy.run_path(str(script), run_name="__main__")


# ---------------------------------------------------------------------------
# Subcommand: distances (Hydra-based)
# ---------------------------------------------------------------------------

def cmd_distances(extra_args: list) -> None:
    import runpy
    script = Path(__file__).parent / "all_distances_updated.py"
    sys.argv = [str(script)] + extra_args
    runpy.run_path(str(script), run_name="__main__")


# ---------------------------------------------------------------------------
# Subcommand: utt-loss
# ---------------------------------------------------------------------------

def cmd_utt_loss(args: argparse.Namespace) -> None:
    import runpy
    script = Path(__file__).parent / "utt_loss_calc.py"
    argv = [str(script)]
    if args.run_id:
        argv += ["--run-id", args.run_id]
    if args.config:
        argv += ["--config", args.config]
    if args.splits:
        argv += ["--splits"] + args.splits
    sys.argv = argv
    runpy.run_path(str(script), run_name="__main__")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeaF unified plotting CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- kde --
    p_kde = sub.add_parser("kde", help="KDE loss distribution plots")
    p_kde.add_argument("--json-path", required=True, help="Path to evaluation JSON")
    p_kde.add_argument("--output-dir", default="plots", help="Output directory")
    p_kde.add_argument(
        "--forget-subjects",
        nargs="+",
        default=["103", "1034", "1040", "1069", "1081", "1088", "1098", "1116", "118", "1183"],
        help="Forget speaker IDs",
    )

    # -- loss-vs-duration --
    p_lvd = sub.add_parser(
        "loss-vs-duration",
        help="Loss vs utterance duration scatter plot (passes remaining args to Hydra)",
    )
    p_lvd.add_argument("hydra_overrides", nargs=argparse.REMAINDER)

    # -- distances --
    p_dist = sub.add_parser(
        "distances",
        help="Pairwise speaker distance computation (passes remaining args to Hydra)",
    )
    p_dist.add_argument("hydra_overrides", nargs=argparse.REMAINDER)

    # -- utt-loss --
    p_utt = sub.add_parser("utt-loss", help="Utterance-level loss from MLflow run")
    p_utt.add_argument("--run-id", required=True)
    p_utt.add_argument("--config", default="config/config.yaml")
    p_utt.add_argument("--splits", nargs="+", default=["test"])

    args, remaining = parser.parse_known_args()

    if args.command == "kde":
        cmd_kde(args)
    elif args.command == "loss-vs-duration":
        cmd_loss_vs_duration(getattr(args, "hydra_overrides", []) + remaining)
    elif args.command == "distances":
        cmd_distances(getattr(args, "hydra_overrides", []) + remaining)
    elif args.command == "utt-loss":
        cmd_utt_loss(args)


if __name__ == "__main__":
    main()
