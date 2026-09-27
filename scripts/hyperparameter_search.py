#!/usr/bin/env python3
"""Optuna search over one unlearner's hyperparameters, scored by forget-vs-test EMD."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import optuna
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from optuna.visualization import plot_optimization_history, plot_param_importances

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.artifacts.simplified_artifacts import SimplifiedArtifacts
from src.evaluation.evaluate_loss import (
    compute_emd_forget_post_vs_test_pre_computed,
    compute_test_losses_pre_unlearning,
)
from src.pipeline.stage_00_initialization import run_initialization
from src.pipeline.stage_01_data_preparation import run_extract_features, run_load_datasets
from src.pipeline.stage_03_training import run_create_model, run_train_model
from src.unlearning.unlearning import Unlearning
from src.utils.utils import rank0_print

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Per-method search spaces
# ---------------------------------------------------------------------------

SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "scrub": {
        "forget_steps_per_epoch": {"type": "int", "low": 10, "high": 200, "default": 50},
        "retain_steps_per_epoch": {"type": "int", "low": 10, "high": 200, "default": 10},
        "alpha_kl_retain": {"type": "float", "low": 0.01, "high": 1.0, "default": 0.2},
        "retain_loss_weight": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.2},
        "lr": {"type": "float", "low": 1e-7, "high": 1e-3, "default": 1e-4, "log": True},
        "grad_clip_norm": {"type": "float", "low": 0.10, "high": 2.0, "default": 0.25},
    },
    "bad_teacher_smooth": {
        "lambda_forget": {"type": "float", "low": 0.01, "high": 1.0, "default": 0.5},
        "temperature": {"type": "float", "low": 1.0, "high": 10.0, "default": 2.5},
        "output_temperature": {"type": "float", "low": 1.0, "high": 4.0, "default": 1.0},
        "kl_ctc_weight": {"type": "float", "low": 0.0, "high": 1.0, "default": 1.0},
        "kl_att_weight": {"type": "float", "low": 0.0, "high": 1.0, "default": 1.0},
        "lr": {"type": "float", "low": 1e-7, "high": 1e-3, "default": 1e-4, "log": True},
        "grad_clip_norm": {"type": "float", "low": 0.10, "high": 2.0, "default": 0.25},
    },
    "neggrad": {
        "threshold": {"type": "float", "low": 0.9, "high": 1.0, "default": 0.95},
        "lr": {"type": "float", "low": 1e-7, "high": 1e-3, "default": 1e-4, "log": True},
        "grad_clip_norm": {"type": "float", "low": 0.10, "high": 2.0, "default": 0.25},
    },
    "neggradplus": {
        "lambda_forget": {"type": "float", "low": 0.001, "high": 1.0, "default": 1.0},
        "lambda_retain": {"type": "float", "low": 0.001, "high": 1.0, "default": 1.0},
        "threshold": {"type": "float", "low": 0.9, "high": 1.0, "default": 0.95},
        "lr": {"type": "float", "low": 1e-7, "high": 1e-3, "default": 1e-4, "log": True},
        "grad_clip_norm": {"type": "float", "low": 0.10, "high": 2.0, "default": 0.25},
    },
    "finetune": {
        "threshold": {"type": "float", "low": 0.9, "high": 1.0, "default": 0.95},
        "lr": {"type": "float", "low": 1e-7, "high": 1e-3, "default": 1e-4, "log": True},
        "grad_clip_norm": {"type": "float", "low": 0.10, "high": 2.0, "default": 0.25},
    },
    "cfk": {
        "k": {"type": "int", "low": 1, "high": 12, "default": 4},
        "lr": {"type": "float", "low": 1e-7, "high": 1e-3, "default": 1e-4, "log": True},
        "grad_clip_norm": {"type": "float", "low": 0.10, "high": 2.0, "default": 0.25},
    },
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str, method: str, extra_overrides: Optional[List[str]] = None) -> DictConfig:
    resolved = Path(config_path).resolve()
    config_dir, config_name = str(resolved.parent), resolved.stem

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    overrides = [f"unlearning={method}", *(extra_overrides or [])]
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=config_name, overrides=overrides)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def sample_params(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, cfg in search_space.items():
        if cfg["type"] == "int":
            params[name] = trial.suggest_int(name, low=cfg["low"], high=cfg["high"])
        elif cfg["type"] == "float":
            params[name] = trial.suggest_float(
                name, low=cfg["low"], high=cfg["high"], log=cfg.get("log", False)
            )
        elif cfg["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, choices=cfg["choices"])
    return params


def update_config_with_params(
    base_config: DictConfig, method: str, params: Dict[str, Any]
) -> DictConfig:
    config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    uc = config.unlearning.unlearner_config

    # Shared gradient-descent params.
    if "grad_clip_norm" in params:
        uc["gradient_clip_val"] = params["grad_clip_norm"]
    if "lr" in params:
        if "optimizer" not in uc:
            uc["optimizer"] = {}
        uc["optimizer"]["lr"] = params["lr"]
    if "weight_decay" in params:
        uc["optimizer"]["weight_decay"] = params["weight_decay"]

    # Method-specific params (all flat — no nested sub-blocks)
    for key in (
        "lambda_forget", "lambda_retain", "temperature", "output_temperature",
        "kl_ctc_weight", "kl_att_weight", "threshold",
        "cycle_length", "num_epochs", "k",
        "forget_steps_per_epoch", "retain_steps_per_epoch",
        "alpha_kl_retain", "retain_loss_weight",
        "alpha",
    ):
        if key in params:
            uc[key] = params[key]

    return config


def save_study_results(
    study: optuna.Study, output_dir: str, study_name: str
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "study_name": study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_completed_trials": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ),
    }
    (out / f"{study_name}_results.json").write_text(json.dumps(results, indent=2))

    trials_data = [
        {"number": t.number, "state": str(t.state), "value": t.value,
         "params": t.params, "metrics": t.user_attrs}
        for t in study.trials
    ]
    (out / f"{study_name}_all_trials.json").write_text(json.dumps(trials_data, indent=2))

    rank0_print(f"\nBest EMD: {study.best_value:.4f}")
    rank0_print(f"Best params: {study.best_params}")
    rank0_print(f"Results saved to: {out}")

    try:
        plot_optimization_history(study).write_image(
            str(out / f"{study_name}_optimization_history.png")
        )
        plot_param_importances(study).write_image(
            str(out / f"{study_name}_param_importance.png")
        )
    except Exception as exc:
        logger.warning("Could not generate visualizations: %s", exc)


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def create_objective(
    method: str,
    base_config: DictConfig,
    model_state: Dict,
    data_handler,
    artifacts: SimplifiedArtifacts,
    forget_speakers: List[str],
    retain_speakers: List[str],
    test_speakers: List[str],
    study: optuna.Study,
    test_losses_precomputed=None,
):
    search_space = SEARCH_SPACES[method]


    def objective(trial: optuna.Trial) -> float:
        if dist.is_initialized():
            trial = optuna.integration.TorchDistributedTrial(trial)

        params = sample_params(trial, search_space)
        is_rank0 = dist.get_rank() == 0 if dist.is_initialized() else True
        trial_num = trial.number if is_rank0 else -1
        rank0_print(f"\n[Trial {trial_num}] params={params}")

        cfg = update_config_with_params(base_config, method, params)

        model = run_create_model(cfg=cfg)
        model.load_state_dict(model_state)

        unlearning_cfg = OmegaConf.create({
            **cfg.unlearning.unlearner_config,
            "unlearner": cfg.unlearning.unlearner,
            "target_subject": forget_speakers[0] if forget_speakers else "unknown",
            "tracking_uri": cfg.artifacts.tracking_uri,
            "mlflow_experiment_name": cfg.artifacts.experiment_name,
            "mlflow_run_name": f"trial_{trial_num}",
            "mlflow_run_id": None,
        })
        unlearning_module = Unlearning(unlearning_cfg, optuna_trial=trial)
        unlearned_model = unlearning_module.run(model, data_handler.datasets)

        emd_result = compute_emd_forget_post_vs_test_pre_computed(
            unlearned_model,
            test_losses_precomputed,
            data_handler.datasets,
            cfg=cfg.evaluation,
        )
        emd = emd_result["emd_forget_post_test_pre"]
        if emd is None:
            rank0_print(f"[Trial {trial_num}] EMD unavailable (too few samples) — pruning trial")
            raise optuna.TrialPruned()
        rank0_print(f"[Trial {trial_num}] EMD={emd:.4f}")
        return float(emd)

    return objective


# ---------------------------------------------------------------------------
# Search runner
# ---------------------------------------------------------------------------

def run_hyperparameter_search(
    method: str,
    base_config: DictConfig,
    model_state: Dict,
    data_handler,
    artifacts: SimplifiedArtifacts,
    forget_speakers: List[str],
    retain_speakers: List[str],
    test_speakers: List[str],
    n_trials: int = 30,
    study_name: str = "hp_search",
    output_dir: str = "hyperparam_results",
    timeout: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    rank0_print(f"\n{'='*60}\nBayesian HP Search — method={method}\n{'='*60}")
    rank0_print(f"Trials: {n_trials}  |  Search space: {list(SEARCH_SPACES[method].keys())}")

    pre_model = run_create_model(cfg=base_config)
    pre_model.load_state_dict(model_state)
    pre_model.eval()
    test_losses = compute_test_losses_pre_unlearning(
        pre_model, data_handler.datasets, cfg=base_config.evaluation
    )
    rank0_print(f"Pre-computed test losses: {len(test_losses)} samples")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out / f'{study_name}.db'}?timeout=600"
    rank0_print(f"Optuna storage: {storage}")
    study = None
    remaining_trials = n_trials
    if not dist.is_initialized() or dist.get_rank() == 0:
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
            storage=storage,
            load_if_exists=True,
        )

        zombies = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        for t in zombies:
            study._storage.set_trial_state_values(t._trial_id, state=optuna.trial.TrialState.FAIL)
        if zombies:
            rank0_print(f"Marked {len(zombies)} stale RUNNING trial(s) from a prior run as FAILED.")

        if not study.trials:
            defaults = {k: v["default"] for k, v in SEARCH_SPACES[method].items()}
            study.enqueue_trial(defaults)

        n_completed = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        remaining_trials = max(0, n_trials - n_completed)
        if n_completed:
            rank0_print(
                f"Resuming study '{study_name}': {n_completed}/{n_trials} trials "
                f"already completed. Running {remaining_trials} more."
            )

    if dist.is_initialized():
        obj_list = [remaining_trials]
        dist.broadcast_object_list(obj_list, src=0)
        remaining_trials = obj_list[0]
        dist.barrier()

    objective = create_objective(
        method=method,
        base_config=base_config,
        model_state=model_state,
        data_handler=data_handler,
        artifacts=artifacts,
        forget_speakers=forget_speakers,
        retain_speakers=retain_speakers,
        test_speakers=test_speakers,
        study=study,
        test_losses_precomputed=test_losses,
    )

    if not dist.is_initialized() or dist.get_rank() == 0:
        if remaining_trials > 0:
            study.optimize(objective, n_trials=remaining_trials, timeout=timeout, show_progress_bar=True)
        save_study_results(study, output_dir, study_name)
        return study.best_params
    else:
        for _ in range(remaining_trials):
            try:
                objective(None)
            except optuna.TrialPruned:
                continue
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_dist(timeout_hours: int = 2):
    is_dist = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_dist:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend="nccl",
            timeout=__import__("datetime").timedelta(hours=timeout_hours),
        )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return is_dist, rank, local_rank, world_size, device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Bayesian HP search for machine unlearning methods"
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=list(SEARCH_SPACES.keys()),
        help="Unlearning method to optimise",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to Hydra config")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides, e.g. data_preparation=spot",
    )
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="hyperparam_results")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--model_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument(
        "--forget_speakers",
        nargs="+",
        #default=["103", "1034", "1040", "1069", "1081", "1088", "1098", "1116", "118", "1183"],
        default=["163"],
        help="Forget speaker IDs",
    )
    args = parser.parse_args()

    study_name = args.study_name or f"{args.method}_emd_optimization"
    cfg = load_config(args.config, args.method, extra_overrides=args.overrides)
    cfg.artifacts.experiment_name = "hyperparam_search"

    setup_dist()
    artifacts = run_initialization(cfg, run_name=study_name)
    data_handler = run_load_datasets(cfg, artifacts)
    run_extract_features(cfg, data_handler, artifacts)

    if args.model_path:
        model = run_create_model(cfg=cfg)
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        model_state = model.state_dict()
    else:
        model = run_create_model(cfg=cfg)
        _, model_state = run_train_model(
            cfg=cfg, data_handler=data_handler, artifacts=artifacts, model=model
        )

    all_speakers = list(data_handler.data["speaker_to_indices"].keys())
    forget_speakers = args.forget_speakers
    retain_speakers = [s for s in all_speakers if s not in forget_speakers]

    # The forget/retain split is fixed across trials (same forget speakers throughout).
    data_handler.set_unlearning_datasets(forget_speakers=forget_speakers)

    best = run_hyperparameter_search(
        method=args.method,
        base_config=cfg,
        model_state=model_state,
        data_handler=data_handler,
        artifacts=artifacts,
        forget_speakers=forget_speakers,
        retain_speakers=retain_speakers,
        test_speakers=[],
        n_trials=args.n_trials,
        study_name=study_name,
        output_dir=args.output_dir,
        timeout=args.timeout,
    )

    rank0_print(f"\nBest params: {best}")


if __name__ == "__main__":
    main()
