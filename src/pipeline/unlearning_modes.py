import gc
import logging
import os
import tempfile
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from src.artifacts.simplified_artifacts import SimplifiedArtifacts
from src.data.data_handler import DataHandler
from src.pipeline.stage_03_training import run_create_model
from src.pipeline.stage_04_evaluation import run_evaluation
from src.pipeline.stage_05_unlearning import run_unlearning
from src.pipeline.stage_06_post_evaluation import run_post_unlearning_pass
from src.utils.utils import ClusterMembershipResult, compute_cluster_membership, rank0_print

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _reset_model(cfg: DictConfig, finetuned_model_state: Dict[str, Any]):
    model = run_create_model(cfg=cfg)
    model.load_state_dict(finetuned_model_state)
    return model


def _run_pre_mia(
    cfg: DictConfig,
    data_handler: DataHandler,
    artifacts: SimplifiedArtifacts,
    model,
    cluster_result: ClusterMembershipResult,
    artifact_type: str,
) -> Dict[str, Any]:
    return run_evaluation(
        cfg=cfg, data_handler=data_handler, artifacts=artifacts, model=model,
        use_cache=True, metrics=False, losses=False, mia=True,
        artifact_type=artifact_type,
        forget_speakers=cluster_result.forget_set,
    ).get("mia", None)


def _log_cluster_metrics(
    artifacts: SimplifiedArtifacts,
    cluster_result: ClusterMembershipResult,
    subject_idx: int | None = None,
    target_subject: str | None = None,
) -> None:
    artifacts.log_metrics({
        "unlearning.cluster.size": cluster_result.cluster_size,
        "unlearning.cluster.has_cluster": cluster_result.has_cluster,
        "unlearning.cluster.is_single_subject": cluster_result.is_single,
        "unlearning.forget_set.size": len(cluster_result.forget_set),
        "unlearning.retain_set.size": len(cluster_result.retain_set),
    }, stage="unlearning")

    member_metrics = {
        f"unlearning.cluster.member_{i}": member
        for i, member in enumerate(cluster_result.cluster_members)
    }
    if member_metrics:
        artifacts.log_metrics(member_metrics, stage="unlearning")

    forget_set_s = set(cluster_result.forget_set)
    rows = [
        {"speaker": spk, "distance": float(d), "in_forget_set": spk in forget_set_s}
        for spk, d in cluster_result.distances.items()
        if isinstance(d, (int, float))
    ]
    if rows:
        artifacts.tracker.log_table(
            data=pd.DataFrame(rows), artifact_file="unlearning/distances.json"
        )

    if subject_idx is not None:
        try:
            artifacts.tracker.log_dict(
                {
                    "target_subject": str(target_subject),
                    "subject_idx": subject_idx,
                    "cluster_members": cluster_result.cluster_members,
                    "forget_set": cluster_result.forget_set,
                    "retain_set": cluster_result.retain_set,
                    "distances": {k: float(v) for k, v in cluster_result.distances.items()},
                    "cluster_size": cluster_result.cluster_size,
                    "is_single": cluster_result.is_single,
                    "has_cluster": cluster_result.has_cluster,
                    "stage": "pre_unlearning",
                },
                f"subject_{subject_idx}_cluster_info_pre.json",
            )
        except Exception as e:
            rank0_print(f"[Experiment] Warning: Could not log cluster info: {e}")


def _save_sequential_checkpoint(
    artifacts: SimplifiedArtifacts,
    model,
    step_idx: int,
    total_steps: int,
    forget_speaker: str,
    forget_set: List[str],
) -> None:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        torch.save(
            {
                "step": step_idx + 1,
                "total_steps": total_steps,
                "speaker": forget_speaker,
                "forget_set": forget_set,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            },
            tmp_path,
        )
        artifacts.tracker.log_artifact(tmp_path, artifact_path="checkpoints/sequential")
        rank0_print(f"[Experiment] Saved sequential checkpoint: step {step_idx + 1}/{total_steps}")
    except Exception as e:
        rank0_print(f"[Experiment] Warning: Could not save sequential checkpoint: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _run_sequential(
    cfg: DictConfig,
    data_handler: DataHandler,
    artifacts: SimplifiedArtifacts,
    model,
    cluster_result: ClusterMembershipResult,
    subject_run_id: str | None,
) -> Any:
    n = len(cluster_result.forget_set)
    rank0_print(f"[Experiment] Sequential unlearning of {n} subjects, one at a time")

    current_model = model
    for step_idx, forget_speaker in enumerate(cluster_result.forget_set):
        rank0_print(f"[Experiment] Sequential step {step_idx + 1}/{n}: unlearning {forget_speaker}")
        other_forget = [s for s in cluster_result.forget_set if s != forget_speaker]
        step_cluster = ClusterMembershipResult(
            target_subject=forget_speaker,
            cluster_members=[forget_speaker],
            forget_set_original=[forget_speaker],
            retain_set=[
                s for s in data_handler.speaker_map_per_set["train"].keys()
                if s not in cluster_result.forget_set
            ],
            distances=cluster_result.distances,
            cluster_size=1,
            is_single=True,
            has_cluster=True,
        )
        data_handler.set_unlearning_datasets(
            forget_speakers=[forget_speaker],
            neutral_speakers=other_forget,
        )
        current_model = run_unlearning(
            cfg=cfg, model=current_model, data_handler=data_handler,
            artifacts=artifacts, cluster_results=step_cluster,
            subject_run_id=subject_run_id, target_subject=forget_speaker,
        )
        _save_sequential_checkpoint(
            artifacts, current_model, step_idx, n, forget_speaker, cluster_result.forget_set
        )

    return current_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_per_subject_unlearning(
    cfg: DictConfig,
    data_handler: DataHandler,
    artifacts: SimplifiedArtifacts,
    model,
    finetuned_model_state: Dict[str, Any],
    pre_unlearn_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    all_speakers = list(data_handler.data["speaker_to_indices"].keys())
    mia_enabled = cfg.evaluation.mia
    clustering_cfg = cfg.clustering
    all_results: List[Dict[str, Any]] = []

    rank0_print(f"[Experiment] Running per-subject unlearning for {len(all_speakers)} subjects")

    for subject_idx, target_subject in enumerate(data_handler.speaker_map_per_set["train"].keys()):
        limit_subjects = cfg.limit if cfg.limit is not None else len(all_speakers)
        if subject_idx >= limit_subjects:
            break

        skip_subjects = cfg.skip
        if subject_idx < skip_subjects:
            rank0_print(
                f"[Experiment] Skipping subject {target_subject} "
                f"(index {subject_idx}) due to skip limit ({skip_subjects})"
            )
            continue

        rank0_print(f"\n{'=' * 60}")
        rank0_print(f"[Experiment] Processing target subject {subject_idx + 1}/{limit_subjects}: {target_subject}")
        rank0_print(f"{'=' * 60}")

        target_speaker_id = str(target_subject) if target_subject else f"subject_{subject_idx}"

        subject_run_id = artifacts.start_run(
            config=cfg,
            run_name=f"unlearn_{target_subject}",
            tags={"subject": str(target_subject), "stage": "unlearning"},
            nested=True,
        )

        cluster_result: ClusterMembershipResult | None = None
        if data_handler is not None and hasattr(data_handler, "datasets_features") and data_handler.datasets_features:
            rank0_print(f"[Cluster] Computing cluster membership for target: {target_speaker_id}")
            cluster_result = compute_cluster_membership(
                data_handler=data_handler,
                target_subject=target_speaker_id,
                clustering_config=clustering_cfg,
                dataset_name="train",
            )
            rank0_print(f"[Cluster] Cluster size: {cluster_result.cluster_size}")
            rank0_print(f"[Cluster] Forget set ({len(cluster_result.forget_set)}): {cluster_result.forget_set}")
            rank0_print(f"[Cluster] Retain set ({len(cluster_result.retain_set)}): {cluster_result.retain_set}")

        try:
            if cluster_result is None:
                # No features extracted — forget only the target speaker
                all_train = [str(s) for s in data_handler.speaker_map_per_set["train"].keys()]
                cluster_result = ClusterMembershipResult(
                    target_subject=target_speaker_id,
                    cluster_members=[target_speaker_id],
                    forget_set_original=[target_speaker_id],
                    retain_set=[s for s in all_train if s != target_speaker_id],
                    distances={},
                    cluster_size=1,
                    is_single=True,
                    has_cluster=True,
                )
            if not cluster_result.has_cluster:
                logging.warning(
                    "[Cluster] No cluster found for target %s - skipping unlearning",
                    target_speaker_id,
                )
                continue
            if cluster_result.is_single:
                logging.warning(
                    "[Cluster] Single subject cluster for target %s - proceeding with unlearning",
                    target_speaker_id,
                )
            _log_cluster_metrics(
                artifacts, cluster_result, subject_idx=subject_idx, target_subject=target_subject
            )
        except Exception as e:
            rank0_print(f"[Experiment] ERROR: Could not compute cluster membership: {e}")
            logging.error("[Cluster] Failed to compute cluster membership: %s", e)
            artifacts.log_metrics({
                "unlearning.cluster.error": 1,
                "unlearning.cluster.error_msg": str(e),
            }, stage="unlearning")
            continue

        try:
            artifacts.tracker.log_metrics({"nested_run_started": 1})
        except Exception as e:
            rank0_print(f"[Experiment] Warning: Could not log nested run start: {e}")

        model = _reset_model(cfg, finetuned_model_state)
        rank0_print(f"[Experiment] Reset model to finetuned state for subject {target_subject}")

        data_handler.set_unlearning_datasets(forget_speakers=cluster_result.forget_set)
        rank0_print(
            f"[Experiment] Updated forget/retain sets: "
            f"forget={len(cluster_result.forget_set)}, retain={len(cluster_result.retain_set)}"
        )
        logging.info(
            "[Unlearning] Updated datasets: forget_set=%s, retain_set size=%s",
            cluster_result.forget_set, len(cluster_result.retain_set),
        )

        pre_mia_results = (
            _run_pre_mia(cfg, data_handler, artifacts, model, cluster_result, f"mia_pre_{target_speaker_id}")
            if mia_enabled else None
        )

        if dist.is_initialized():
            dist.barrier()


        unlearned_model = run_unlearning(
            cfg=cfg, model=model, data_handler=data_handler,
            artifacts=artifacts, cluster_results=cluster_result,
            subject_run_id=subject_run_id, target_subject=target_subject,
        )

        result = run_post_unlearning_pass(
            cfg=cfg, data_handler=data_handler, artifacts=artifacts,
            pre_unlearn_results=pre_unlearn_results, cluster_result=cluster_result,
            unlearned_model=unlearned_model, pre_mia_results=pre_mia_results,
            run_label=f"subject_{subject_idx}",
        )
        all_results.append({"target_subject": target_subject, **result})

        del model, unlearned_model
        plt.close("all")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def run_multi_subject_unlearning(
    cfg: DictConfig,
    data_handler: DataHandler,
    artifacts: SimplifiedArtifacts,
    model,
    finetuned_model_state: Dict[str, Any],
    pre_unlearn_results: Dict[str, Any],
    unlearning_mode: str,
) -> List[Dict[str, Any]]:
    mia_enabled = cfg.evaluation.mia
    clustering_cfg = cfg.clustering

    first_subject = str(list(data_handler.speaker_map_per_set["train"].keys())[0])
    rank0_print(f"[Experiment] Computing forget set (target anchor: {first_subject})")

    cluster_result = compute_cluster_membership(
        data_handler=data_handler,
        target_subject=first_subject,
        clustering_config=clustering_cfg,
        dataset_name="train",
    )

    rank0_print(f"[Experiment] Forget set ({len(cluster_result.forget_set)}): {cluster_result.forget_set}")
    rank0_print(f"[Experiment] Retain set size: {len(cluster_result.retain_set)}")

    if not cluster_result.has_cluster:
        raise RuntimeError(
            f"[Experiment] No cluster found for anchor {first_subject} — "
            f"cannot proceed with {unlearning_mode} unlearning"
        )

    subject_run_id = artifacts.start_run(
        config=cfg,
        run_name=f"unlearn_{unlearning_mode}_{len(cluster_result.forget_set)}subj",
        tags={"mode": unlearning_mode, "stage": "unlearning",
              "n_subjects": str(len(cluster_result.forget_set))},
        nested=True,
    )

    artifacts.log_metrics({
        "unlearning.cluster.size": cluster_result.cluster_size,
        "unlearning.forget_set.size": len(cluster_result.forget_set),
        "unlearning.retain_set.size": len(cluster_result.retain_set),
    }, stage="unlearning")

    try:
        artifacts.tracker.log_metrics({"nested_run_started": 1})
    except Exception as e:
        rank0_print(f"[Experiment] Warning: Could not log nested run start: {e}")

    model = _reset_model(cfg, finetuned_model_state)
    rank0_print("[Experiment] Reset model to finetuned state")

    data_handler.set_unlearning_datasets(forget_speakers=cluster_result.forget_set)

    pre_mia_results = (
        _run_pre_mia(cfg, data_handler, artifacts, model, cluster_result, "mia_pre_multi")
        if mia_enabled else None
    )

    if dist.is_initialized():
        dist.barrier()


    if unlearning_mode == "simultaneous":
        rank0_print(f"[Experiment] Simultaneous unlearning of {len(cluster_result.forget_set)} subjects")
        unlearned_model = run_unlearning(
            cfg=cfg, model=model, data_handler=data_handler,
            artifacts=artifacts, cluster_results=cluster_result,
            subject_run_id=subject_run_id, target_subject=first_subject,
        )
    else:
        unlearned_model = _run_sequential(
            cfg, data_handler, artifacts, model, cluster_result, subject_run_id
        )
        data_handler.set_unlearning_datasets(forget_speakers=cluster_result.forget_set)

    result = run_post_unlearning_pass(
        cfg=cfg, data_handler=data_handler, artifacts=artifacts,
        pre_unlearn_results=pre_unlearn_results, cluster_result=cluster_result,
        unlearned_model=unlearned_model, pre_mia_results=pre_mia_results,
        run_label="multi",
    )

    del model, unlearned_model
    plt.close("all")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [{"mode": unlearning_mode, "forget_set": cluster_result.forget_set, **result}]
