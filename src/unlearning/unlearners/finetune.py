"""FineTune unlearner: fine-tune on the retain set to forget by overwriting."""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from src.unlearning.unlearners.base import BaseUnlearner

logger = logging.getLogger(__name__)


class FineTuneTrainer(BaseUnlearner):
    """Fine-tunes exclusively on the retain set."""

    def __init__(self, model, datasets, train_config: Dict[str, Any], schedule=None):
        super().__init__(model, datasets, train_config, schedule)
        self.threshold: float = float(train_config["threshold"])
        self.cycle_length: int = int(train_config["cycle_length"])

    def _mlflow_params(self) -> Dict[str, Any]:
        return {
            **super()._mlflow_params(),
            "threshold": self.threshold,
            "cycle_length": self.cycle_length,
        }

    # ------------------------------------------------------------------
    # DataLoaders — train on retain only; validate on forget to probe acc
    # ------------------------------------------------------------------

    def train_dataloader(self):
        return self._make_loader(self.retain_set, self.batch_size_retain, True)

    def validation_step(self, batch, batch_idx):
        forget_loss = self.model(batch["forget"])
        self.log("val/forget_task_loss", forget_loss.detach(),
                 on_epoch=True, prog_bar=True, sync_dist=True)

        if self.cycle_length > 0 and self.global_step > 0:
            acc = self._probe_forget_accuracy()
            self.log("train/forget_acc_probe", acc, on_epoch=True, prog_bar=True)
            self.log("train/forget_below_threshold", float(acc <= self.threshold),
                     on_epoch=True)
            if self._sync_stop(acc <= self.threshold):
                self.trainer.should_stop = True

    # ------------------------------------------------------------------
    # Training — minimise retain loss
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        retain_loss = self.model(batch)
        self.log("train/retain_task_loss", retain_loss.detach(),
                 on_epoch=True, prog_bar=True)
        return retain_loss

