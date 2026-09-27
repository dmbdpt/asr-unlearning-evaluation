from collections import defaultdict
import logging
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch.distributed as dist
from torch.utils.data import Subset


from src.artifacts.simplified_artifacts import SimplifiedArtifacts
from src.data.data_handler import DataHandler
from src.evaluation.evaluate_metrics import run_evaluate_metrics
from src.evaluation.evaluate_loss import run_evaluate_loss, run_evaluate_loss_per_utt, compute_emd_forget_test
from src.evaluation.evaluate_mia import (
    evaluate_mia_forget_utt_level,
    evaluate_mia_forget_speaker_level,
)
from src.evaluation.utils import aggregate_results, get_avg_duration, downsample_dataset_by_duration
from src.utils.utils import rank0_print

logger = logging.getLogger(__name__)


def make_serializable(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable values to strings."""

    if isinstance(obj, DictConfig):
        obj = OmegaConf.to_container(obj, resolve=True)

    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj

    return str(obj)


def run_evaluation_main(
    cfg,
    model,
    data_handler,
    artifacts: SimplifiedArtifacts,
    sets_to_evaluate: Tuple = ("test",),
    set_to_match_downsample: str = "test",
    metrics: bool = True,
    losses: bool = True,
    mia: bool = True,
    emd: Optional[bool] = None,
    artifact_dir: Optional[str] = None,
    forget_speakers: Optional[List] = None,
    stage: str = "post",
    model_tag: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    rank0_print("[Evaluation] Starting evaluation stage...")

    rank = dist.get_rank() if dist.is_initialized() else 0
    device = 'cuda:' + str(rank) if dist.is_initialized() else 'cpu'
    model.to(device)

    datasets = data_handler.datasets

    metric_results = {}
    loss_results = {}
    mia_results = None
    loss_per_utt = None
    emd_results = None

    if metrics:
        datasets_downsampled = deepcopy(datasets)

        if cfg.downsample_to_test:
            if rank == 0:
                avg_duration = get_avg_duration(datasets_downsampled[set_to_match_downsample])
                downsample_indices = {}

                for set_name in sets_to_evaluate:
                    if set_name == "test":
                        continue

                    downsample_indices[set_name] = downsample_dataset_by_duration(
                        datasets_downsampled[set_name], avg_duration
                    )
            else:
                downsample_indices = None

            if dist.is_initialized():
                dist.barrier()
                obj = [downsample_indices]
                dist.broadcast_object_list(obj, src=0)
                downsample_indices = obj[0]
                dist.barrier()

            # Only override the sets that were downsampled
            for set_name, indices in downsample_indices.items():
                datasets_downsampled[set_name] = Subset(
                    datasets_downsampled[set_name].dataset,
                    indices
                )

        model.to(device)
        metric_results = run_evaluate_metrics(model, datasets_downsampled, sets_to_evaluate, cfg)
    if losses:
        model.to(device)
        loss_results = run_evaluate_loss(model, datasets, sets_to_evaluate, cfg)
        loss_per_utt = run_evaluate_loss_per_utt(model, datasets, sets_to_evaluate, cfg)

    # EMD needs only the forget and test losses, so it is separable from the full per-set...
    if emd is None:
        emd = losses
    if emd:
        model.to(device)
        if 'forget' in datasets and 'test' in datasets:
            if rank == 0:
                rank0_print("[Evaluation] Computing Earth Mover's Distance between forget and test set losses...")
            emd_results = compute_emd_forget_test(model, datasets, cfg)
        else:
            emd_results = {'emd_forget_test': None}
    if mia:
        if rank == 0:
            rank0_print("[Evaluation] Running Membership Inference Attack evaluation...")

            mia_cfg = cfg.mia_config
            speaker_level = mia_cfg.speaker_level

            mia_results = {}

            rank0_print("[Evaluation] Running speaker-level MIA...")
            mia_results_spk = evaluate_mia_forget_speaker_level(
                model,
                datasets,
                forget_speakers=forget_speakers,
                cfg=mia_cfg,
                artifact_dir=artifact_dir,
                stage=stage,
            )
            mia_results["speaker_level"] = mia_results_spk

            if not speaker_level:
                rank0_print("[Evaluation] Running utterance-level MIA...")
                mia_results_utt = evaluate_mia_forget_utt_level(
                    model,
                    datasets,
                    forget_speakers=forget_speakers,
                    cfg=mia_cfg,
                    artifact_dir=artifact_dir,
                    stage=stage,
                )
                mia_results["utterance_level"] = mia_results_utt

        if dist.is_initialized():
            dist.barrier()

    aggregated_results = aggregate_results(metric_results, loss_results, sets_to_evaluate)

    if emd_results:
        aggregated_results["emd"] = emd_results

    # Add MIA results at the top level (not per-set, as MIA is a global metric)
    if mia and mia_results:
        aggregated_results["mia"] = mia_results

    if losses and loss_per_utt:
        aggregated_results["per_utt_loss"] = loss_per_utt

    return aggregated_results


def run_evaluation(cfg: DictConfig, data_handler: DataHandler, artifacts: SimplifiedArtifacts, model=None, artifact_type: str = "evaluation_pre", use_cache=True, **kwargs):
    rank0_print("[Experiment] Running evaluation...")

    eval_cfg = cfg.evaluation

    sets_to_evaluate = kwargs.get("sets_to_evaluate", eval_cfg.sets_to_evaluate)
    set_to_match_downsample = kwargs.get("set_to_match_downsample",
                                         eval_cfg.set_to_match_downsample)
    metrics = kwargs.get("metrics", eval_cfg.metrics)
    losses = kwargs.get("losses", eval_cfg.losses)
    mia = kwargs.get("mia", eval_cfg.mia)
    emd = kwargs.get("emd", eval_cfg.get("emd", None))

    forget_speakers = kwargs.get("forget_speakers", [])
    stage = "pre" if "pre" in artifact_type else "post"
    model_tag = cfg.training.model_tag

    if use_cache:
        aggregated_results, was_cached = artifacts.get_cached(
            artifact_type=artifact_type,
            config=eval_cfg,
            loader=lambda: run_evaluation_main(
                cfg=eval_cfg,
                model=model,
                data_handler=data_handler,
                artifacts=artifacts,
                sets_to_evaluate=sets_to_evaluate,
                forget_speakers=forget_speakers,
                set_to_match_downsample=set_to_match_downsample,
                metrics=metrics,
                losses=losses,
                mia=mia,
                emd=emd,
                artifact_dir=cfg.artifacts.cache_dir,
                stage=stage,
                model_tag=model_tag,
            )
        )
    else:
        aggregated_results = run_evaluation_main(
            cfg=eval_cfg,
            model=model,
            data_handler=data_handler,
            artifacts=artifacts,
            sets_to_evaluate=sets_to_evaluate,
            forget_speakers=forget_speakers,
            set_to_match_downsample=set_to_match_downsample,
            metrics=metrics,
            losses=losses,
            mia=mia,
            emd=emd,
            artifact_dir=cfg.artifacts.cache_dir,
            stage=stage,
            model_tag=model_tag,
        )
        was_cached = False

    if dist.is_initialized():
        dist.barrier()

    if was_cached:
        rank0_print("[Experiment] Evaluation results loaded from cache")
    else:
        rank0_print("[Experiment] Evaluation completed and cached")

    def _sanitize_metric_key(key: str) -> str:
        return key.replace("@", "_at_").replace("=", "_eq_").replace("%", "_pct_")

    def log_eval_results_fast(aggregated_results: dict) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        metrics_to_log = {}

        for set_name, set_data in aggregated_results.items():
            if not isinstance(set_data, dict):
                continue

            if set_name == "mia":
                for mia_key, mia_val in set_data.items():
                    if isinstance(mia_val, (int, float)):
                        sanitized_key = _sanitize_metric_key(mia_key)
                        metrics_to_log[f"eval.mia.{sanitized_key}"] = float(mia_val)
                continue
            
            if set_name == "emd":
                for emd_key, emd_val in set_data.items():
                    if isinstance(emd_val, (int, float)):
                        sanitized_key = _sanitize_metric_key(emd_key)
                        metrics_to_log[f"eval.emd.{sanitized_key}"] = float(emd_val)
                continue

            for speaker_id, speaker_data in set_data.items():
                if not isinstance(speaker_data, dict):
                    continue

                speaker_metrics = speaker_data.get("metrics")
                if not isinstance(speaker_metrics, dict):
                    continue

                prefix = f"eval.{set_name}.{speaker_id}."
                for metric_name, metric_val in speaker_metrics.items():
                    if isinstance(metric_val, (int, float)):
                        metrics_to_log[prefix + metric_name] = float(metric_val)

        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log)

        results_serializable = make_serializable(aggregated_results)
        utt_los_ser = results_serializable.get("per_utt_loss")
        mia_ser = results_serializable.get("mia")
        if utt_los_ser is not None:
            artifacts.log_artifact(utt_los_ser, name="per_utt_loss.json", artifact_type="evaluation")
            del results_serializable["per_utt_loss"]
        if mia_ser is not None:
            artifacts.log_artifact(mia_ser, name="mia_results.json", artifact_type="evaluation")
            del results_serializable["mia"]

        artifacts.log_artifact(results_serializable, name="evaluation_results", artifact_type="evaluation")
        artifacts.tracker.set_tags({"pipeline_stage": "evaluation_complete"})

        rank0_print("[Evaluation] Logged evaluation results to MLflow")

    log_eval_results_fast(aggregated_results)

    return aggregated_results


def process_pre_unlearn_metrics(
    artifacts: SimplifiedArtifacts,
    pre_unlearn_results: Dict[str, Any],
) -> None:

    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

    if is_rank0:
        metrics_to_log: dict[str, float] = {}
        rows: list[dict[str, Any]] = []

        for set_name, set_data in pre_unlearn_results.items():
            if not isinstance(set_data, dict):
                continue

            for speaker_id, speaker_data in set_data.items():
                if not isinstance(speaker_data, dict):
                    continue

                prefix = f"pre_unlearn.{set_name}.{speaker_id}"
                row: dict[str, Any] = {
                    "set_name": set_name,
                    "speaker_id": speaker_id,
                }
                has_payload = False

                metrics = speaker_data.get("metrics")
                if isinstance(metrics, dict):
                    for metric_name, metric_val in metrics.items():
                        if isinstance(metric_val, Number):
                            value = float(metric_val)
                            metrics_to_log[f"{prefix}.{metric_name}"] = value
                            row[f"{metric_name}_pre"] = value
                            has_payload = True

                loss_val = speaker_data.get("losses")
                if isinstance(loss_val, Number):
                    value = float(loss_val)
                    metrics_to_log[f"{prefix}.loss"] = value
                    row["loss_pre"] = value
                    has_payload = True

                if has_payload:
                    rows.append(row)

        if metrics_to_log:
            artifacts.log_metrics(metrics_to_log, stage="evaluation")

        if rows:
            df = pd.DataFrame(rows)
            mlflow.log_table(
                data=df,
                artifact_file="evaluation/pre_unlearn_metrics_table.json",
            )

    if dist.is_initialized():
        dist.barrier()


def process_post_unlearn_metrics(
    artifacts: SimplifiedArtifacts,
    post_unlearn_results: Dict[str, Any],
):
    if not dist.is_initialized() or dist.get_rank() == 0:
        table_rows = []
        aggregate_metrics_to_log = {}

        for set_name, set_data in post_unlearn_results.items():
            if not isinstance(set_data, dict):
                continue

            all_metrics = defaultdict(list)

            for speaker_id, speaker_data in set_data.items():
                if not (isinstance(speaker_data, dict) and "metrics" in speaker_data):
                    continue

                metrics = speaker_data["metrics"]
                if not isinstance(metrics, dict):
                    continue

                for metric_name, metric_val in metrics.items():
                    if isinstance(metric_val, (int, float)):
                        metric_val = float(metric_val)
                        all_metrics[metric_name].append(metric_val)

                        table_rows.append({
                            "set_name": set_name,
                            "speaker_id": speaker_id,
                            "metric_name": metric_name,
                            "metric_value": metric_val,
                        })

            for metric_name, values in all_metrics.items():
                if not values:
                    continue

                aggregate_metrics_to_log[
                    f"post_unlearn.{set_name}.avg_{metric_name}"
                ] = float(sum(values) / len(values))
                aggregate_metrics_to_log[
                    f"post_unlearn.{set_name}.num_{metric_name}_samples"
                ] = int(len(values))

        if table_rows:
            df = pd.DataFrame(table_rows)
            mlflow.log_table(
                data=df,
                artifact_file="evaluation/post_unlearn_metrics_table.json",
            )

        if aggregate_metrics_to_log:
            artifacts.log_metrics(aggregate_metrics_to_log, stage="evaluation")

    if dist.is_initialized():
        dist.barrier()

def log_mia_results(
    artifacts: SimplifiedArtifacts,
    pre_mia_results: Optional[Dict[str, Any]],
    post_mia_results: Optional[Dict[str, Any]],
    stage: str = "unlearning"
):
    if pre_mia_results:
        for metric_name, metric_val in pre_mia_results.items():
            if isinstance(metric_val, (int, float)):
                artifacts.log_metrics({f"mia_pre.{metric_name}": metric_val}, stage=stage)

    if post_mia_results:
        for metric_name, metric_val in post_mia_results.items():
            if isinstance(metric_val, (int, float)):
                artifacts.log_metrics({f"mia_post.{metric_name}": metric_val}, stage=stage)

    if pre_mia_results and post_mia_results:
        for metric_name in pre_mia_results:
            if metric_name in post_mia_results:
                delta = post_mia_results[metric_name] - pre_mia_results[metric_name]
                artifacts.log_metrics({f"mia_delta.{metric_name}": delta}, stage=stage)
