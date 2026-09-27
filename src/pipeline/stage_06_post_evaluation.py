import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import torch.distributed as dist

from src.artifacts.simplified_artifacts import SimplifiedArtifacts
from src.data.data_handler import DataHandler
from src.evaluation.visualize_metrics import log_summary_to_mlflow, plot_metrics_kde
from src.pipeline.stage_04_evaluation import (
    make_serializable,
    process_post_unlearn_metrics,
    run_evaluation,
)
from src.utils.utils import (
    ClusterMembershipResult,
    compute_deltas,
    extract_speaker_metrics,
    rank0_print,
)

logger = logging.getLogger(__name__)


def _generate_kde_plots(
    artifacts: SimplifiedArtifacts,
    post_unlearn_results: Dict[str, Any],
    post_sets: List[str],
    cluster_result: ClusterMembershipResult,
    run_label: str,
) -> None:
    try:
        rank0_print("[Experiment] Generating KDE plots for metric distributions...")

        per_utt_loss = post_unlearn_results.get("per_utt_loss", {})
        kde_per_utt: Dict[str, Any] = {}
        if per_utt_loss:
            for set_name in post_sets:
                if set_name in per_utt_loss:
                    set_data = per_utt_loss[set_name]
                    if isinstance(set_data, dict):
                        kde_per_utt[set_name] = {}
                        for speaker_id, utt_losses in set_data.items():
                            if isinstance(utt_losses, dict):
                                loss_values = list(utt_losses.values())
                            elif isinstance(utt_losses, list):
                                loss_values = utt_losses
                            else:
                                loss_values = []
                            if loss_values:
                                kde_per_utt[set_name][speaker_id] = {"losses": loss_values}

        if kde_per_utt:
            figures = plot_metrics_kde(
                metrics_dict=kde_per_utt,
                forget_speakers=cluster_result.forget_set,
                log_to_mlflow=True,
                artifact_path=f"kde_plots/per_utt_loss_{run_label}",
            )
            rank0_print(f"[Experiment] Generated {len(figures)} KDE plots for per-utt loss")

        kde_speaker = {
            set_name: post_unlearn_results.get(set_name, {})
            for set_name in post_sets
            if isinstance(post_unlearn_results.get(set_name, {}), dict)
        }
        if kde_speaker:
            figures = plot_metrics_kde(
                metrics_dict=kde_speaker,
                forget_speakers=cluster_result.forget_set,
                log_to_mlflow=True,
                artifact_path=f"kde_plots/speaker_loss_{run_label}",
            )
            rank0_print(f"[Experiment] Generated {len(figures)} KDE plots for speaker loss")
            log_summary_to_mlflow(metrics_by_category={}, artifact_path=f"kde_plots/per_utt_loss_{run_label}")
            log_summary_to_mlflow(metrics_by_category={}, artifact_path=f"kde_plots/speaker_loss_{run_label}")

    except Exception as e:
        rank0_print(f"[Experiment] Warning: Could not generate KDE plots: {e}")


def _log_mia_metrics(artifacts: SimplifiedArtifacts, post_unlearn_results: Dict[str, Any]) -> None:
    mia_results = post_unlearn_results.get("mia", {})
    if not mia_results:
        return

    rank0_print("[Experiment] Logging MIA metrics to MLflow...")
    mia_to_log: Dict[str, float] = {}
    for metric_name, metric_val in mia_results.items():
        if isinstance(metric_val, (int, float)):
            mia_to_log[f"mia.{metric_name}"] = float(metric_val)
        elif isinstance(metric_val, dict):
            for sub_key, sub_val in metric_val.items():
                if isinstance(sub_val, (int, float)):
                    mia_to_log[f"mia.{metric_name}.{sub_key}"] = float(sub_val)
    if mia_to_log:
        artifacts.log_metrics(mia_to_log, stage="evaluation")


def _compute_and_log_deltas(
    artifacts: SimplifiedArtifacts,
    pre_unlearn_results: Dict[str, Any],
    post_unlearn_results: Dict[str, Any],
) -> None:
    avg_metrics_to_log: Dict[str, float] = {}
    for set_name in post_unlearn_results:
        pre_set_data = pre_unlearn_results.get(set_name, {})
        post_set_data = post_unlearn_results.get(set_name, {})
        if not (isinstance(pre_set_data, dict) and isinstance(post_set_data, dict)):
            continue
        pre_by_speaker = extract_speaker_metrics(pre_set_data)
        post_by_speaker = extract_speaker_metrics(post_set_data)
        speaker_deltas, avg_deltas = compute_deltas(pre_by_speaker, post_by_speaker)
        if speaker_deltas:
            delta_df = pd.DataFrame(speaker_deltas)
            preferred_cols = ["speaker_id", "metric_name", "pre_value", "post_value", "delta"]
            ordered_cols = [c for c in preferred_cols if c in delta_df.columns]
            ordered_cols += [c for c in delta_df.columns if c not in preferred_cols]
            delta_df = delta_df[ordered_cols]
            artifacts.tracker.log_table(data=delta_df, artifact_file=f"comparison/delta_{set_name}.json")
        for metric_name, delta in avg_deltas.items():
            if isinstance(delta, (int, float)):
                avg_metrics_to_log[f"delta.{set_name}.avg_{metric_name}"] = float(delta)
    if avg_metrics_to_log:
        artifacts.log_metrics(avg_metrics_to_log, stage="comparison")


def _export_json_artifacts(
    artifacts: SimplifiedArtifacts,
    pre_unlearn_results: Dict[str, Any],
    post_unlearn_results: Dict[str, Any],
    pre_mia_results: Optional[Dict[str, Any]],
    post_mia_results: Optional[Dict[str, Any]],
    run_label: str,
) -> None:
    try:
        artifacts.tracker.log_dict(
            make_serializable(pre_unlearn_results),
            f"evaluation/{run_label}_pre_unlearn_results.json",
        )
        artifacts.tracker.log_dict(
            make_serializable(post_unlearn_results),
            f"evaluation/{run_label}_post_unlearn_results.json",
        )
        if pre_mia_results:
            artifacts.tracker.log_dict(
                make_serializable(pre_mia_results),
                f"evaluation/{run_label}_mia_pre.json",
            )
        if post_mia_results:
            artifacts.tracker.log_dict(
                make_serializable(post_mia_results),
                f"evaluation/{run_label}_mia_post.json",
            )
    except Exception as e:
        rank0_print(f"[Experiment] Warning: Could not log evaluation artifacts: {e}")


def run_post_unlearning_pass(
    cfg,
    data_handler: DataHandler,
    artifacts: SimplifiedArtifacts,
    pre_unlearn_results: Dict[str, Any],
    cluster_result: ClusterMembershipResult,
    unlearned_model,
    pre_mia_results: Optional[Dict[str, Any]],
    run_label: str,
) -> Dict[str, Any]:
    post_sets = ["test", "train"]

    post_unlearn_results = run_evaluation(
        cfg=cfg,
        data_handler=data_handler,
        artifacts=artifacts,
        sets_to_evaluate=post_sets,
        model=unlearned_model,
        artifact_type=f"evaluation_post_{run_label}",
        forget_speakers=cluster_result.forget_set,
        use_cache=False,
    )

    post_mia_results = post_unlearn_results.get("mia", None)

    process_post_unlearn_metrics(artifacts=artifacts, post_unlearn_results=post_unlearn_results)
    _generate_kde_plots(artifacts, post_unlearn_results, post_sets, cluster_result, run_label)
    _log_mia_metrics(artifacts, post_unlearn_results)
    _compute_and_log_deltas(artifacts, pre_unlearn_results, post_unlearn_results)
    _export_json_artifacts(
        artifacts, pre_unlearn_results, post_unlearn_results,
        pre_mia_results, post_mia_results, run_label,
    )

    artifacts.tracker.set_tags({"pipeline_stage": "unlearning_complete"})
    artifacts.end_run(status="FINISHED")

    return {
        "cluster_members": cluster_result.cluster_members,
        "forget_set": cluster_result.forget_set,
        "retain_set": cluster_result.retain_set,
        "pre": make_serializable(pre_unlearn_results),
        "post": make_serializable(post_unlearn_results),
        "distances": cluster_result.distances,
    }
