"""Abstract base class for all machine-unlearning Lightning modules."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

import mlflow
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.utilities.combined_loader import CombinedLoader

from src.data.datasets.librispeech import collate_fn as collate_fn_espnet

logger = logging.getLogger(__name__)


def make_unlearning_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Build a forget/retain DataLoader with the conventions shared by all unlearners (ESPnet..."""
    distributed = dist.is_available() and dist.is_initialized()
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        collate_fn=collate_fn_espnet,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


class BaseUnlearner(pl.LightningModule):
    """Shared scaffold for unlearning Lightning modules."""

    def __init__(
        self,
        model,
        datasets: Dict[str, Any],
        train_config: Dict[str, Any],
        schedule=None,
    ) -> None:
        super().__init__()
        self.model = model
        self.datasets = datasets
        self.cfg = train_config

        self.optimizer_config: Dict[str, Any] = train_config["optimizer"]

        self.batch_size_forget: int = int(train_config["batch_size_forget"])
        self.batch_size_retain: int = int(train_config["batch_size_retain"])
        self.num_workers: int = int(train_config["num_workers"])
        self.pin_memory: bool = bool(train_config["pin_memory"])

        self.gradient_clip_val: float = float(train_config["gradient_clip_val"])
        self.batches_to_check: int = int(train_config["batches_to_check"])

        # Used by on_validation_epoch_end to compute a pruning proxy metric for Optuna
        self.test_loss_mean: Optional[float] = train_config["test_loss_mean"]

        self.forget_set = datasets.get("forget")
        self.retain_set = datasets.get("retain")
        self.test_set = datasets.get("test")

        # Built lazily on first probe and reused for every subsequent one — see...
        self._probe_loader: Optional[DataLoader] = None

    # ------------------------------------------------------------------
    # Distributed utilities
    # ------------------------------------------------------------------

    def _is_distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _make_loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return make_unlearning_loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _sync_stop(self, should_stop: bool) -> bool:
        """Broadcast an early-stop decision across all DDP ranks."""
        if self._is_distributed():
            flag = torch.tensor([1.0 if should_stop else 0.0], device=self.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            return bool(flag.item() > 0.5)
        return should_stop

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self):
        return CombinedLoader(
            {
                "forget": self._make_loader(self.forget_set, self.batch_size_forget, True),
                "retain": self._make_loader(self.retain_set, self.batch_size_retain, True),
            },
            mode="max_size_cycle",
        )

    def val_dataloader(self):
        return CombinedLoader(
            {
                "forget": self._make_loader(self.forget_set, self.batch_size_forget, False),
                "retain": self._make_loader(self.retain_set, self.batch_size_retain, False),
            },
            mode="max_size_cycle",
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            forget_loss = self.model(batch["forget"])
            retain_loss = self.model(batch["retain"])
        self.log("val/forget_task_loss", forget_loss.detach(),
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/retain_task_loss", retain_loss.detach(),
                 on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        if self.test_loss_mean is None:
            return
        mean_forget = self.trainer.callback_metrics.get("val/forget_task_loss")
        if mean_forget is None:
            return
        proxy = torch.abs(mean_forget - self.test_loss_mean)
        self.log("val/proxy_emd", proxy, prog_bar=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        lr = self.optimizer_config["lr"]
        wd = self.optimizer_config["weight_decay"]
        opt_type = self.optimizer_config["type"].lower()

        if opt_type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.optimizer_config["momentum"],
                weight_decay=wd,
            )
        return optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=tuple(self.optimizer_config["betas"]),
            weight_decay=wd,
        )

    # ------------------------------------------------------------------
    # Gradient clipping
    # ------------------------------------------------------------------

    def on_before_optimizer_step(self, optimizer):
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_clip_val
            )

    # ------------------------------------------------------------------
    # Forget-accuracy probe (shared by threshold-based early-stop methods)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _probe_forget_accuracy(self) -> float:
        """Compute mean forget-set accuracy over a few batches."""
        was_training = self.model.training
        self.model.eval()

        # Reuse one persistent-worker loader across every probe call instead of building a fresh...
        if self._probe_loader is None:
            self._probe_loader = self._make_loader(self.forget_set, self.batch_size_forget, False)
        it = iter(self._probe_loader)
        total, count = 0.0, 0.0
        for _ in range(self.batches_to_check):
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            _, stats, _ = self.model.original_forward(batch)
            total += float(stats.get("acc", 0.0))
            count += 1

        tensor = torch.tensor([total, count], device=self.device)
        if self._is_distributed():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total, count = tensor.tolist()

        self.model.train(was_training)
        return float(total / count) if count > 0 else 0.0

    # ------------------------------------------------------------------
    # MLflow hooks
    # ------------------------------------------------------------------

    def _mlflow_params(self) -> Dict[str, Any]:
        """Params logged on train start. Subclasses extend by calling super()."""
        return {
            "batches_to_check": self.batches_to_check,
        }

    def on_train_start(self):
        if self.global_rank != 0:
            return
        try:
            if mlflow.active_run():
                params = {
                    "gradient_clip_val": self.gradient_clip_val,
                    "lr": self.optimizer_config["lr"],
                    "optimizer": self.optimizer_config["type"],
                    **self._mlflow_params(),
                }
                prefix = self.__class__.__name__.lower()
                mlflow.log_params({f"{prefix}.{k}": v for k, v in params.items()})
                mlflow.set_tag("unlearner.class", self.__class__.__name__)
        except Exception as exc:
            logger.debug("MLflow logging failed: %s", exc)

    def on_train_end(self):
        if self.forget_set is None:
            return
        try:
            # _probe_forget_accuracy uses dist.all_reduce — ALL ranks must participate.
            acc = self._probe_forget_accuracy()
            if self.global_rank == 0 and mlflow.active_run():
                mlflow.log_metric("unlearning.final_forget_acc", acc)
                mlflow.set_tag("unlearning_status", "completed")
        except Exception as exc:
            logger.debug("MLflow logging failed: %s", exc)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Compute and return the training loss for one batch."""
        ...
