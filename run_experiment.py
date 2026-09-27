import logging
import random

import hydra
import numpy as np
from omegaconf import DictConfig

import torch
import torch.distributed as dist

from src.utils.distributed import setup_dist
from src.pipeline.stage_00_initialization import run_initialization
from src.pipeline.stage_01_data_preparation import run_extract_features, run_load_datasets
from src.pipeline.stage_03_training import run_create_model, run_train_model
from src.pipeline.stage_04_evaluation import process_pre_unlearn_metrics, run_evaluation
from src.pipeline.unlearning_modes import run_multi_subject_unlearning, run_per_subject_unlearning
from src.utils.utils import rank0_print

torch.multiprocessing.set_sharing_strategy('file_system')

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logging.getLogger('src').setLevel(logging.DEBUG)
logging.getLogger(__name__).setLevel(logging.DEBUG)

for _lib in [
    'torch', 'torch.cuda', 'torch.distributed', 'torch.nn',
    'hydra', 'hydra.core', 'omegaconf',
    'sklearn', 'numpy', 'matplotlib', 'PIL',
    'transformers', 'datasets', 'torchvision', 'torchaudio',
    'mlflow', 'tensorflow', 'tensorboard', 'asyncio', 'espnet', 'espnet2',
]:
    logging.getLogger(_lib).setLevel(logging.WARNING)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ── Distributed setup ────────────────────────────────────────────────
    setup_dist()

    # ── Reproducibility ──────────────────────────────────────────────────
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Stage 0: Initialization ──────────────────────────────────────────
    artifacts = run_initialization(cfg)

    # ── Stage 1: Data preparation ────────────────────────────────────────
    data_handler = run_load_datasets(cfg, artifacts)

    # ── Stage 2: Feature extraction ──────────────────────────────────────
    if cfg.clustering.enabled:
        run_extract_features(cfg, data_handler, artifacts)

    # ── Stage 3/4: Model training ────────────────────────────────────────
    model = run_create_model(cfg=cfg)
    _, finetuned_model_state = run_train_model(
        cfg=cfg, data_handler=data_handler, artifacts=artifacts, model=model
    )
    model.load_state_dict(finetuned_model_state)

    # ── Stage 5: Pre-unlearning evaluation ──────────────────────────────
    if cfg.skip_pre:
        rank0_print("[Experiment] Skipping pre-unlearning evaluation (skip_pre=true)")
        pre_unlearn_results = {}
    else:
        pre_unlearn_results = run_evaluation(
            cfg=cfg, data_handler=data_handler, artifacts=artifacts, model=model,
            artifact_type="evaluation_pre", sets_to_evaluate=["test", "train"], mia=False,
        )
        process_pre_unlearn_metrics(artifacts=artifacts, pre_unlearn_results=pre_unlearn_results)

    # ── Stage 6: Unlearning ──────────────────────────────────────────────
    unlearning_mode = cfg.unlearning.mode
    rank0_print(f"[Experiment] Unlearning mode: {unlearning_mode}")
    rank0_print(f"[Experiment] Using clustering config: {cfg.clustering.distance_config}")

    if unlearning_mode == "per_subject":
        all_results = run_per_subject_unlearning(
            cfg=cfg, data_handler=data_handler, artifacts=artifacts,
            model=model, finetuned_model_state=finetuned_model_state,
            pre_unlearn_results=pre_unlearn_results,
        )
    else:
        all_results = run_multi_subject_unlearning(
            cfg=cfg, data_handler=data_handler, artifacts=artifacts,
            model=model, finetuned_model_state=finetuned_model_state,
            pre_unlearn_results=pre_unlearn_results,
            unlearning_mode=unlearning_mode,
        )

    # ── Wrap-up ──────────────────────────────────────────────────────────
    rank0_print(f"[Experiment] Completed {len(all_results)} unlearning runs")
    artifacts.log_artifact(
        artifact=all_results, name="all_unlearning_results.json", artifact_type="summary"
    )
    artifacts.end_run(status="FINISHED")


if __name__ == "__main__":
    try:
        main()
        rank0_print("[Experiment] Experiment complete!")
    except Exception as e:
        rank0_print(f"[Experiment] ERROR: Experiment failed with exception: {e}")
        logging.error("Experiment failed with exception", exc_info=True)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
