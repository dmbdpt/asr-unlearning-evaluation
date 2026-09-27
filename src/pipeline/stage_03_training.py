import os
import logging

import mlflow
import torch
from typing import Dict, Any
from omegaconf import DictConfig
from torch.utils.data import Subset
import torch.distributed as dist

from src.training.trainer import train_model, load_model
from src.models.base import ModelWrapper
from src.models.espnet import ESPnetASRWrapper
from src.utils.utils import rank0_print
from src.artifacts.simplified_artifacts import SimplifiedArtifacts
from src.data.data_handler import DataHandler

logger = logging.getLogger(__name__)


def run_create_model(cfg: DictConfig, artifacts: SimplifiedArtifacts|None = None) -> ModelWrapper:
    training_cfg = cfg.training
    model_tag = training_cfg.model_tag
    model_config = training_cfg.model_config
    rank0_print(f"[Training] Creating model with tag: {model_tag}")
    rank0_print(f"[Training] Model config: {model_config}")

    model = ESPnetASRWrapper.from_pretrained(model_tag, **model_config)

    if artifacts is not None:
        artifacts.tracker.log_params(training_cfg.model_config)

    return model


def run_train_model_main(config: DictConfig, model, data_handler, clusters=None,
                         target_cluster=None, cfg=None, artifacts=None):
    if cfg is None:
        cfg = {}

    rank0_print("[Training] Starting model training stage...")

    if artifacts is not None and hasattr(artifacts, "cache_dir"):
        artifact_dir = str(artifacts.cache_dir)
    else:
        artifact_dir = config.artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)

    model_tag_clean = config.model_tag.replace("/", "_")
    cluster_suffix = f"_cluster_{target_cluster}" if target_cluster is not None else ""
    model_filename = f"{model_tag_clean}{cluster_suffix}.pt"
    model_path = os.path.join(artifact_dir, model_filename)

    if os.path.exists(model_path):
        rank0_print(f"[Training] Found cached model at {model_path}. Loading...")
        try:
            model = load_model(model, model_path)
            return model
        except Exception as e:
            rank0_print(f"[Training] Failed to load cached model: {e}. Proceeding to train.")

    datasets = data_handler.datasets

    if "retain" not in datasets:
        rank0_print("[Training] 'retain' set not found. Creating default retain=train, forget=empty.")
        datasets["retain"] = datasets["train"]
        datasets["forget"] = Subset(datasets["train"].dataset, [])

    cfg["checkpoint_dir"] = os.path.join(artifact_dir, "checkpoints")
    cfg["log_dir"] = os.path.join(artifact_dir, "logs")
    cfg["num_workers"] = config.train_config.num_workers
    cfg["tracking_uri"] = artifacts.tracking_uri if artifacts is not None else None
    cfg["mlflow_experiment_name"] = artifacts.experiment_name if artifacts is not None else None

    train_model(model, datasets, clusters=clusters, config=config, target_cluster_id=target_cluster, cfg=cfg)

    if dist.is_initialized():
        pass

    rank0_print(f"[Training] Saving trained model to {model_path}...")
    torch.save(model.state_dict(), model_path)

    if artifacts is not None:
        artifacts.tracker.log_artifact(model_path, name="finetuned")

    return model


def run_train_model(cfg: DictConfig, data_handler: DataHandler, artifacts: SimplifiedArtifacts,
                    model: ModelWrapper, clusters=None, target_cluster=None) -> tuple[str, Dict[str, Any]]:

    training_config = cfg.training

    model, was_cached = artifacts.get_cached(
        artifact_type="model",
        config=training_config,
        loader=lambda: run_train_model_main(
            model=model,
            data_handler=data_handler,
            config=training_config,
            clusters=clusters,
            target_cluster=target_cluster,
            artifacts=artifacts,
        ) if training_config.finetune_model else model
    )

    if was_cached:
        rank0_print("[Experiment] Model loaded from cache")
    else:
        rank0_print("[Experiment] Model trained and cached")

    def save_finetuned_model(
        model: torch.nn.Module,
        output_dir: str,
        tracker,
        rank: int = 0
    ) -> tuple[str, Dict[str, Any]]:

        finetuned_ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(finetuned_ckpt_dir, exist_ok=True)
        checkpoint_path = os.path.join(finetuned_ckpt_dir, "finetuned_model.pt")
        model_state = model.state_dict()

        if rank == 0:
            try:
                torch.save(model_state, checkpoint_path)
                tracker.log_artifact(checkpoint_path, artifact_path="checkpoints/finetuned")
                rank0_print(f"[TrainingUtils] Logged finetuned model checkpoint: {checkpoint_path}")
            except Exception as e:
                rank0_print(f"[TrainingUtils] Warning: Could not log finetuned checkpoint: {e}")

        return checkpoint_path, model_state

    checkpoint_path, model_state = save_finetuned_model(
        model=model,
        output_dir=cfg.artifacts.cache_dir,
        tracker=artifacts.tracker,
        rank=dist.get_rank() if dist.is_initialized() else 0
    )
    rank0_print("[Experiment] Saved finetuned model state for resetting between unlearning iterations")

    if dist.is_initialized():
        dist.barrier()

    return checkpoint_path, model_state
