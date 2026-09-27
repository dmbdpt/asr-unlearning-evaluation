import logging

import pytorch_lightning as pl
import torch.distributed as dist
from omegaconf import DictConfig

from src.artifacts.simplified_artifacts import SimplifiedArtifacts


def run_initialization(cfg: DictConfig, run_name: str = "main_experiment") -> SimplifiedArtifacts:
    """Set global seed, start the MLflow run, and return the artifact tracker."""
    seed = int(cfg.seed)
    pl.seed_everything(seed, workers=True)

    if not dist.is_initialized() or dist.get_rank() == 0:
        logging.info("Seed set to %d. Running initialization stage.", seed)

    artifacts = SimplifiedArtifacts(cfg.artifacts)
    artifacts.start_run(config=cfg, run_name=run_name)
    artifacts.tracker.log_cfg(cfg)

    logging.info("Initialization complete.")
    return artifacts
