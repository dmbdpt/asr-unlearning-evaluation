"""NegGrad unlearner: maximise forget-set loss to drive forgetting."""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from src.unlearning.unlearners.base import BaseUnlearner

logger = logging.getLogger(__name__)


class NegGradTrainer(BaseUnlearner):
    """Plain NegGrad: negate the forget-set loss; stop once forget accuracy drops below ``threshold``."""

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
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        forget_loss = self.model(batch["forget"])

        with torch.no_grad():
            retain_loss = self.model(batch["retain"])

        loss = -forget_loss

        self.log("train/forget_task_loss", forget_loss.detach(),
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/retain_task_loss", retain_loss.detach(),
                 on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/forget_objective_loss", loss.detach(),
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Periodic forget-accuracy probe for threshold-based early stopping
        if (
            self.cycle_length > 0
            and self.global_step > 0
            and self.global_step % self.cycle_length == 0
        ):
            acc = self._probe_forget_accuracy()
            self.log("train/forget_acc_probe", acc,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/forget_below_threshold", float(acc <= self.threshold),
                     on_step=True, on_epoch=True, sync_dist=True)

            if self._sync_stop(acc <= self.threshold):
                self.trainer.should_stop = True

        self.log("train/did_probe",
                 float(self.cycle_length > 0 and self.global_step > 0
                       and self.global_step % self.cycle_length == 0),
                 on_step=True, on_epoch=True, sync_dist=True)

        return loss
