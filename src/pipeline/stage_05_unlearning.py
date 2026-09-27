from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist

from src.artifacts.simplified_artifacts import RankAwareTracker, SimplifiedArtifacts
from src.data.data_handler import DataHandler
from src.models.base import ModelWrapper
from src.unlearning.unlearning import Unlearning

from src.utils.utils import ClusterMembershipResult, rank0_print


def run_unlearning(cfg: DictConfig, model: ModelWrapper, data_handler: DataHandler,
                   artifacts: SimplifiedArtifacts, cluster_results: ClusterMembershipResult,
                   subject_run_id: str | None, target_subject: str):
    unlearning_config = {
        **cfg.unlearning.unlearner_config,
        "unlearner": cfg.unlearning.unlearner,
        "mlflow_experiment_name": cfg.artifacts.experiment_name,
        "mlflow_run_name": f"unlearn_{target_subject}",
        "mlflow_run_id": subject_run_id,
        "target_subject": str(target_subject),
    }

    unlearning_config['tracking_uri'] = cfg.artifacts.tracking_uri
    if unlearning_config['tracking_uri'] is None:
        raise ValueError("Tracking URI for MLflow must be specified in cfg.artifacts.tracking_uri or cfg.tracking_uri")

    unlearning_config = OmegaConf.create(unlearning_config)

    unlearning_module = Unlearning(unlearning_config)

    log_unlearner_metadata(artifacts.tracker, unlearning_config)

    rank0_print(f"[Experiment] Executing unlearning for target subject: {target_subject}")
    rank0_print(f"[Experiment] Forget set: {cluster_results.forget_set}")
    unlearned_model = unlearning_module.run(model, data_handler.datasets)

    artifacts.log_metrics({
        "unlearning.completed": 1,
    }, stage="unlearning")

    if dist.is_initialized():
        dist.barrier()

    return unlearned_model

def log_unlearner_metadata(
    tracker: RankAwareTracker,
    unlearning: Dict[str, Any]
):
    # unlearning is flat (run_unlearning spreads unlearner_config to the top level), and...
    params = {
        "unlearner.num_epochs": OmegaConf.select(unlearning, "num_epochs"),
        "unlearner.batch_size_forget": OmegaConf.select(unlearning, "batch_size_forget"),
        "unlearner.batch_size_retain": OmegaConf.select(unlearning, "batch_size_retain"),
        "unlearner.lr": OmegaConf.select(unlearning, "optimizer.lr"),
        "unlearner.weight_decay": OmegaConf.select(unlearning, "optimizer.weight_decay"),
    }
    tracker.log_params({k: v for k, v in params.items() if v is not None})
