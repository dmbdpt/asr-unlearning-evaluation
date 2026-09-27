"""Eval-only entrypoint for re-evaluating a single (checkpoint, forgotten subject) pair from a..."""

import json
import logging
import os
import random
import sys
from datetime import timedelta
from pathlib import Path

# Make `src` importable when launched from outside the repo root (e.g. by SLURM).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from src.pipeline.stage_00_initialization import run_initialization
from src.pipeline.stage_01_data_preparation import run_extract_features, run_load_datasets
from src.pipeline.stage_03_training import run_create_model
from src.pipeline.stage_04_evaluation import (
    make_serializable,
    process_post_unlearn_metrics,
    run_evaluation,
)
from src.utils.utils import compute_cluster_membership, rank0_print


logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
for _lib in (
    "torch", "hydra", "omegaconf", "mlflow", "sklearn", "espnet", "espnet2",
    "matplotlib", "transformers", "datasets", "PIL", "urllib3", "asyncio",
):
    logging.getLogger(_lib).setLevel(logging.WARNING)

torch.multiprocessing.set_sharing_strategy('file_system')

def _setup_dist(timeout_hours: int = 2):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=timeout_hours),
            device_id=device,
        )
    return True, rank, local_rank, world_size, device


def _strip_lightning_prefix(state_dict: dict) -> dict:
    """Lightning saves keys with one extra `model.` prefix (since the LightningModule stores the..."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            out[k[len("model."):]] = v
        else:
            # Unexpected key; keep it so load_state_dict raises informatively.
            out[k] = v
    return out


def _load_checkpoint_into_model(model, ckpt_path: str, device) -> None:
    rank0_print(f"[EvalFromRun] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = _strip_lightning_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        rank0_print(f"[EvalFromRun] WARNING: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        rank0_print(f"[EvalFromRun] WARNING: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")


def _build_native_model(spec_path: str, cfg: DictConfig, device):
    """Build the wrapper from an ESPnet train config + weights + its OWN bpemodel."""
    from espnet2.bin.asr_inference import Speech2Text

    from src.models.espnet import ESPnetASRWrapper

    meta = json.loads(Path(spec_path).read_text())
    for field in ("asr_train_config", "asr_model_file", "bpemodel"):
        if field not in meta:
            raise ValueError(f"{spec_path}: missing '{field}'")
        if not os.path.isfile(meta[field]):
            raise FileNotFoundError(f"{spec_path}: {field} -> {meta[field]} not found")

    decode_cfg = OmegaConf.to_container(cfg.training.model_config, resolve=True)
    rank0_print(f"[EvalFromRun] Building native ESPnet model from {spec_path}")
    rank0_print(f"[EvalFromRun]   weights: {meta['asr_model_file']}")
    rank0_print(f"[EvalFromRun]   bpemodel: {meta['bpemodel']}")
    rank0_print(f"[EvalFromRun]   decode: {decode_cfg}")

    speech2text = Speech2Text(
        asr_train_config=meta["asr_train_config"],
        asr_model_file=meta["asr_model_file"],
        bpemodel=meta["bpemodel"],
        device=str(device),
        **decode_cfg,
    )
    model = ESPnetASRWrapper(speech2text)
    model.model.eval()
    return model


# Sections that depend only on the model, not on which subject is being forgotten.
_MODEL_LEVEL_SECTIONS = ("test", "train")
# Per-speaker payload keys that only exist to be dumped; reeval_to_csv.py drops them when...
_BULK_KEYS = ("transcripts", "ground_truths", "speaker_ids")


def _strip_bulk(section: dict) -> dict:
    out = {}
    for speaker, payload in section.items():
        if not isinstance(payload, dict):
            out[speaker] = payload
            continue
        out[speaker] = {
            group: ({k: v for k, v in values.items() if k not in _BULK_KEYS}
                    if isinstance(values, dict) else values)
            for group, values in payload.items()
        }
    return out


def _merge_shared_metrics(results: dict, shared_path: str, subject: str) -> dict:
    """Fill the model-level sections of `results` from another run of the same model."""
    shared = json.loads(Path(shared_path).read_text())
    inherited = []
    for section in _MODEL_LEVEL_SECTIONS:
        if results.get(section):
            rank0_print(f"[EvalFromRun] {section!r} was computed by this run -- not inheriting it")
            continue
        if not shared.get(section):
            rank0_print(f"[EvalFromRun] WARNING: shared metrics have no {section!r} section")
            continue
        results[section] = _strip_bulk(shared[section])
        inherited.append(section)

    if not inherited:
        raise ValueError(
            f"{shared_path} provided none of {_MODEL_LEVEL_SECTIONS} -- refusing to "
            f"emit a subject-{subject} row with no model-level metrics")

    results["metrics_source"] = {
        "path": str(shared_path),
        "sections": inherited,
        "note": ("model-level metrics measured once on this same model; only the MIA "
                 "and EMD in this payload are specific to this subject"),
    }
    rank0_print(f"[EvalFromRun] Inherited {inherited} from {shared_path}")
    return results


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ---- Required overrides ----------------------------------------------
    efr = cfg.get("eval_from_run", None)
    if efr is None:
        raise ValueError("Missing required group: +eval_from_run.{checkpoint_path,target_subject}")

    checkpoint_path = efr.get("checkpoint_path", None)
    espnet_model_json = efr.get("espnet_model_json", None)
    target_subject = efr.get("target_subject", None)
    if bool(checkpoint_path) == bool(espnet_model_json):
        raise ValueError(
            "Exactly one of +eval_from_run.checkpoint_path (a Lightning checkpoint) or "
            "+eval_from_run.espnet_model_json (an ESPnet-native model) is required."
        )
    if target_subject in (None, ""):
        raise ValueError("+eval_from_run.target_subject is required.")
    target_subject = str(target_subject)
    subject_idx = int(efr.get("subject_idx", -1))

    is_native = bool(espnet_model_json)
    source_artifact = espnet_model_json if is_native else checkpoint_path
    if not os.path.isfile(source_artifact):
        raise FileNotFoundError(f"Model artifact not found: {source_artifact}")

    # ---- Reproducibility --------------------------------------------------
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- Distributed (single-GPU here; we parallelize per-task in SLURM) -
    is_distributed, rank, local_rank, world_size, device = _setup_dist()

    # ---- Init MLflow + data + features + model ---------------------------
    artifacts = run_initialization(cfg, run_name=f"reeval_{target_subject}")
    artifacts.tracker.set_tags({
        "stage": "reeval",
        "subject": target_subject,
        "subject_idx": str(subject_idx),
        "source_checkpoint": source_artifact,
        "model_kind": "espnet_native" if is_native else "lightning_ckpt",
    })

    data_handler = run_load_datasets(cfg, artifacts)
    _features, _fe = run_extract_features(cfg, data_handler, artifacts)

    # ---- Build the model ------------------------------------------------
    # Lightning path loads the base wrapper so the pre eval sees it unmodified.
    model = (_build_native_model(source_artifact, cfg, device) if is_native
             else run_create_model(cfg=cfg))
    model.to(device)

    # ---- Reconstruct forget/retain splits from the saved clustering cfg --
    cluster_result = compute_cluster_membership(
        data_handler=data_handler,
        target_subject=target_subject,
        clustering_config=cfg.clustering,
        dataset_name="train",
    )
    if not cluster_result.has_cluster:
        raise RuntimeError(f"No cluster found for target subject {target_subject}")

    rank0_print(
        f"[EvalFromRun] Cluster size={cluster_result.cluster_size} "
        f"forget={cluster_result.forget_set} retain={len(cluster_result.retain_set)}"
    )
    data_handler.set_unlearning_datasets(forget_speakers=cluster_result.forget_set)

    # Optional: speakers that were unlearned by this model but are NOT the target of this...
    neutral = efr.get("neutral_speakers", None)
    if neutral:
        from src.evaluation.evaluate_mia import _resolve_forget_and_retain_indices
        excl = {str(s) for s in neutral} | {str(s) for s in cluster_result.forget_set}
        f_idx, r_all = _resolve_forget_and_retain_indices(data_handler.datasets, cluster_result.forget_set)
        _, r_excl = _resolve_forget_and_retain_indices(data_handler.datasets, sorted(excl))
        data_handler.datasets["forget_idx"], data_handler.datasets["retain_idx"] = f_idx, r_excl
        rank0_print(f"[EvalFromRun] MIA pools: forget={len(f_idx)} utt; retain {len(r_all)} -> {len(r_excl)} "
                    f"after excluding {len(excl) - len(cluster_result.forget_set)} neutral speaker(s)")

    artifacts.log_metrics({
        "reeval.cluster.size": cluster_result.cluster_size,
        "reeval.forget_set.size": len(cluster_result.forget_set),
        "reeval.retain_set.size": len(cluster_result.retain_set),
    }, stage="evaluation")

    sets_to_evaluate = ["test", "train"]
    skip_pre  = bool(efr.get("skip_pre",  True))
    skip_post = bool(efr.get("skip_post", False))

    # For a native model there is no "before": `model` is already the model under test, so a...
    if is_native and not skip_pre:
        rank0_print("[EvalFromRun] Native model has no pre-unlearning state -- forcing skip_pre")
        skip_pre = True

    # ---- Pre-unlearn eval (pretrained model) ----------------------------
    if skip_pre:
        rank0_print("[EvalFromRun] Skipping PRE eval (skip_pre=true)")
        pre_results = {}
    else:
        rank0_print("[EvalFromRun] Running PRE eval on pretrained model...")
        pre_results = run_evaluation(
            cfg=cfg,
            data_handler=data_handler,
            artifacts=artifacts,
            sets_to_evaluate=sets_to_evaluate,
            model=model,
            artifact_type=f"reeval_pre_{target_subject}",
            forget_speakers=cluster_result.forget_set,
            use_cache=False,
        )
        try:
            artifacts.tracker.log_dict(
                make_serializable(pre_results),
                f"reeval/subject_{target_subject}_pre_results.json",
            )
        except Exception as err:
            rank0_print(f"[EvalFromRun] Warning: could not log pre results artifact: {err}")

    # ---- Load unlearned checkpoint and run post-unlearn eval ------------
    if skip_post:
        rank0_print("[EvalFromRun] Skipping POST eval (skip_post=true)")
    else:
        if is_native:
            rank0_print("[EvalFromRun] Running eval on the natively-built model...")
        else:
            rank0_print("[EvalFromRun] Loading unlearned checkpoint and running POST eval...")
            _load_checkpoint_into_model(model, checkpoint_path, device)
        model.to(device)

        results = run_evaluation(
            cfg=cfg,
            data_handler=data_handler,
            artifacts=artifacts,
            sets_to_evaluate=sets_to_evaluate,
            model=model,
            artifact_type=f"reeval_{target_subject}",
            forget_speakers=cluster_result.forget_set,
            use_cache=False,
        )

        shared_metrics_json = efr.get("shared_metrics_json", None)
        if shared_metrics_json:
            if not os.path.isfile(shared_metrics_json):
                raise FileNotFoundError(f"shared_metrics_json not found: {shared_metrics_json}")
            results = _merge_shared_metrics(results, shared_metrics_json, target_subject)

        process_post_unlearn_metrics(artifacts=artifacts, post_unlearn_results=results)

        try:
            artifacts.tracker.log_dict(
                make_serializable(results),
                f"reeval/subject_{target_subject}_results.json",
            )
        except Exception as err:
            rank0_print(f"[EvalFromRun] Warning: could not log results artifact: {err}")

    artifacts.tracker.set_tags({"pipeline_stage": "reeval_complete"})
    artifacts.end_run(status="FINISHED")
    rank0_print(f"[EvalFromRun] Done: subject={target_subject}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        rank0_print(f"[EvalFromRun] ERROR: {e}")
        logging.error("eval_from_run failed", exc_info=True)
        sys.exit(1)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
