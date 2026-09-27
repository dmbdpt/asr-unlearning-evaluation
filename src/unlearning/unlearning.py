import logging
import os

import optuna
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from src.models.base import ModelWrapper
from src.unlearning.unlearners.neggrad import NegGradTrainer
from src.unlearning.unlearners.neggradplus import NegGradPlusTrainer
from src.unlearning.unlearners.scrub import SCRUBTrainer, ScrubSchedule
from src.unlearning.unlearners.finetune import FineTuneTrainer
from src.unlearning.unlearners.bad_teacher_smooth import BadTeacherSmoothTrainer
from src.unlearning.unlearners.catastrophic_forgetting import CatastrophicForgettingTrainer

logger = logging.getLogger(__name__)


class DistributedPruningCallback(Callback):
    """Optuna pruning callback that synchronises the prune decision across DDP ranks."""

    def __init__(self, trial, monitor: str):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        if self.monitor not in trainer.callback_metrics:
            return
        current_score = trainer.callback_metrics[self.monitor]
        should_prune = torch.tensor([0], dtype=torch.int32, device=pl_module.device)
        if (not dist.is_initialized() or dist.get_rank() == 0) and self.trial is not None:
            self.trial.report(current_score.item(), step=trainer.global_step)
            if self.trial.should_prune():
                should_prune[0] = 1
        if dist.is_initialized():
            dist.broadcast(should_prune, src=0)
        if should_prune.item() == 1:
            raise optuna.TrialPruned()


_UNLEARNER_REGISTRY = {
    "neggrad": NegGradTrainer,
    "neggradplus": NegGradPlusTrainer,
    "scrub": SCRUBTrainer,
    "finetune": FineTuneTrainer,
    "bad_teacher_smooth": BadTeacherSmoothTrainer,
    "cfk": CatastrophicForgettingTrainer,
}


class Unlearning:

    def __init__(self, cfg: DictConfig, optuna_trial=None):
        self.cfg = cfg
        method = cfg.unlearner.lower()

        if method not in _UNLEARNER_REGISTRY:
            raise ValueError(
                f"Unsupported unlearner: {method!r}. Valid: {list(_UNLEARNER_REGISTRY)}"
            )
        self.unlearner_class = _UNLEARNER_REGISTRY[method]

        unlearner_config = cfg
        if isinstance(unlearner_config, DictConfig):
            self.unlearner_config = OmegaConf.to_container(unlearner_config, resolve=True)
        else:
            self.unlearner_config = dict(unlearner_config)

        # SCRUB: build schedule to derive total epoch count for the Trainer
        if method == "scrub":
            self.scrub_schedule = ScrubSchedule(
                cycles=int(self.unlearner_config["cycles"]),
                extra_retain_epochs=int(self.unlearner_config["extra_retain_epochs"]),
                forget_steps_per_epoch=self.unlearner_config["forget_steps_per_epoch"],
                retain_steps_per_epoch=self.unlearner_config["retain_steps_per_epoch"],
            )
            self.unlearner_config["num_epochs"] = self.scrub_schedule.total_epochs
        else:
            self.scrub_schedule = None

        callbacks = []

        early_stopping_config = self.unlearner_config["early_stopping"]
        if early_stopping_config["active"]:
            callbacks.append(EarlyStopping(
                monitor=early_stopping_config["monitor"],
                stopping_threshold=early_stopping_config["stopping_threshold"],
                patience=early_stopping_config["patience"],
                min_delta=early_stopping_config["min_delta"],
                mode=early_stopping_config["mode"],
                verbose=True,
            ))

        if optuna_trial is not None:
            callbacks.append(DistributedPruningCallback(trial=optuna_trial, monitor="val/proxy_emd"))

        callbacks.append(ModelCheckpoint(
            filename="unlearned-{epoch:02d}",
            save_top_k=0,
            save_last=True,
            verbose=False,
        ))

        # MLflow logger — rank 0 only
        mlflow_tracking_uri = cfg.tracking_uri
        mlflow_experiment_name = cfg.mlflow_experiment_name
        mlflow_run_name = cfg.mlflow_run_name
        mlflow_run_id = cfg.mlflow_run_id

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            mlflow_logger = MLFlowLogger(
                experiment_name=mlflow_experiment_name,
                run_name=mlflow_run_name,
                run_id=mlflow_run_id,
                tracking_uri=mlflow_tracking_uri,
                log_model=True,
                synchronous=True,
            )
        else:
            mlflow_logger = None

        val_check_interval = self.unlearner_config["val_check_interval"]

        self.trainer = pl.Trainer(
            accelerator="auto",
            strategy="ddp_find_unused_parameters_true",
            max_epochs=self.unlearner_config["num_epochs"],
            val_check_interval=val_check_interval,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            logger=mlflow_logger,
        )

    def run(self, model: ModelWrapper, datasets):
        instance = self.unlearner_class(
            model=model,
            datasets=datasets,
            train_config=self.unlearner_config,
            schedule=self.scrub_schedule,
        )
        self.trainer.fit(instance)
        return instance.model
