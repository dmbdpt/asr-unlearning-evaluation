"""NegGradPlus unlearner: joint retain-maximise / forget-minimise objective."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import mlflow
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.datasets.librispeech import collate_fn as collate_fn_espnet
from src.unlearning.unlearners.base import BaseUnlearner

logger = logging.getLogger(__name__)


class NegGradPlusTrainer(BaseUnlearner):
    """NegGrad+ joint objective."""

    def __init__(self, model, datasets, train_config: Dict[str, Any], **kwargs):
        super().__init__(model, datasets, train_config)

        self.lambda_forget: float = float(train_config["lambda_forget"])
        self.lambda_retain: float = float(train_config["lambda_retain"])
        self.threshold: float = float(train_config["threshold"])
        self.cycle_length: int = int(train_config["cycle_length"])
        self.accumulate_grad_batches: int = max(
            1, int(train_config["accumulate_grad_batches"])
        )
        self.persistent_workers: bool = train_config["persistent_workers"]

        self.automatic_optimization = False
        self._last_probe_step: int = -1
        self._retain_train_sampler: Optional[DistributedSampler] = None
        self._forget_train_sampler: Optional[DistributedSampler] = None

    # ------------------------------------------------------------------
    # DataLoaders — use explicit DistributedSamplers for epoch syncing
    # ------------------------------------------------------------------

    def _make_train_sampler(self, dataset) -> Optional[DistributedSampler]:
        return DistributedSampler(dataset, shuffle=True, drop_last=False) \
            if self._is_distributed() else None

    def _make_eval_sampler(self, dataset) -> Optional[DistributedSampler]:
        return DistributedSampler(dataset, shuffle=False, drop_last=False) \
            if self._is_distributed() else None

    def _make_loader(self, dataset, batch_size: int, shuffle: bool = False,
                     sampler=None):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle if sampler is None else False),
            sampler=sampler,
            collate_fn=collate_fn_espnet,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self):
        from pytorch_lightning.utilities.combined_loader import CombinedLoader
        self._retain_train_sampler = self._make_train_sampler(self.retain_set)
        self._forget_train_sampler = self._make_train_sampler(self.forget_set)
        return CombinedLoader(
            {
                "forget": self._make_loader(
                    self.forget_set, self.batch_size_forget, True,
                    sampler=self._forget_train_sampler,
                ),
                "retain": self._make_loader(
                    self.retain_set, self.batch_size_retain, True,
                    sampler=self._retain_train_sampler,
                ),
            },
            mode="max_size_cycle",
        )

    def val_dataloader(self):
        from pytorch_lightning.utilities.combined_loader import CombinedLoader
        return CombinedLoader(
            {
                "forget": self._make_loader(
                    self.forget_set, self.batch_size_forget, False,
                    sampler=self._make_eval_sampler(self.forget_set),
                ),
                "retain": self._make_loader(
                    self.retain_set, self.batch_size_retain, False,
                    sampler=self._make_eval_sampler(self.retain_set),
                ),
            },
            mode="max_size_cycle",
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self):
        if self._retain_train_sampler is not None:
            self._retain_train_sampler.set_epoch(self.current_epoch)
        if self._forget_train_sampler is not None:
            self._forget_train_sampler.set_epoch(self.current_epoch)
        self.log("train/lambda_forget", float(self.lambda_forget), on_epoch=True, sync_dist=True)
        self.log("train/lambda_retain", float(self.lambda_retain), on_epoch=True, sync_dist=True)

    def on_train_start(self):
        super().on_train_start()
        self.optimizers().zero_grad()
        if self.global_rank != 0:
            return
        try:
            if mlflow.active_run():
                mlflow.log_params({
                    "neggradplus.lambda_forget": self.lambda_forget,
                    "neggradplus.lambda_retain": self.lambda_retain,
                    "neggradplus.cycle_length": self.cycle_length,
                    "neggradplus.threshold": self.threshold,
                    "neggradplus.batches_to_check": self.batches_to_check,
                    "neggradplus.accumulate_grad_batches": self.accumulate_grad_batches,
                    "neggradplus.gradient_clip_val": self.gradient_clip_val,
                    "neggradplus.lr": self.optimizer_config["lr"],
                    "neggradplus.optimizer": self.optimizer_config["type"],
                })
                mlflow.set_tag("unlearner.class", "NegGradPlusTrainer")
        except Exception as exc:
            logger.debug("MLflow logging failed: %s", exc)

    def on_train_end(self):
        super().on_train_end()
        try:
            final_forget_acc = self._probe_forget_accuracy()
            if self.global_rank == 0 and mlflow.active_run():
                mlflow.log_metric("unlearning.final_forget_acc", final_forget_acc)
                mlflow.set_tag("unlearning_status", "completed")
        except Exception as exc:
            logger.debug("Final metric logging failed: %s", exc)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if self.trainer.sanity_checking:
            return None

        optimizer = self.optimizers()
        retain_loss = self.model(batch["retain"])
        forget_loss = self.model(batch["forget"])
        total_loss = self.lambda_retain * retain_loss - self.lambda_forget * forget_loss

        self.manual_backward(total_loss / self.accumulate_grad_batches)

        should_step = ((batch_idx + 1) % self.accumulate_grad_batches == 0)
        try:
            is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches
        except Exception:
            is_last_batch = False

        if should_step or is_last_batch:
            max_norm = self.gradient_clip_val if self.gradient_clip_val > 0 else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
            optimizer.step()
            optimizer.zero_grad()
            self.log("train/grad_norm", grad_norm.detach(), on_step=True, sync_dist=True)
            self.log("train/will_optimizer_step", 1.0, on_step=True, sync_dist=True)
        else:
            self.log("train/will_optimizer_step", 0.0, on_step=True, sync_dist=True)

        # Periodic forget-accuracy probe
        did_probe = 0.0
        if (
            self.cycle_length > 0
            and self.global_step > 0
            and self.global_step % self.cycle_length == 0
            and self.global_step != self._last_probe_step
        ):
            did_probe = 1.0
            self._last_probe_step = self.global_step
            acc = self._probe_forget_accuracy()
            self.log("train/forget_acc_probe", acc,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/forget_below_threshold", float(acc <= self.threshold),
                     on_step=True, on_epoch=True, sync_dist=True)
            if self._sync_stop(acc <= self.threshold):
                self.trainer.should_stop = True

        self.log("train/did_probe", did_probe, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/retain_task_loss", retain_loss.detach(),
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/forget_task_loss", forget_loss.detach(),
                 on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/retain_loss_weighted", (self.lambda_retain * retain_loss).detach(),
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/forget_loss_weighted", (self.lambda_forget * forget_loss).detach(),
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/total_loss", total_loss.detach(),
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        denom = retain_loss.detach().abs().clamp_min(1e-8)
        self.log("train/forget_to_retain_loss_ratio",
                 forget_loss.detach().abs() / denom,
                 on_step=True, on_epoch=True, sync_dist=True)

        return total_loss.detach()
