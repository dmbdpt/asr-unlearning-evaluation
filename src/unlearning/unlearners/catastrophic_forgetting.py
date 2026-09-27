"""CF-k unlearner: fine-tune only the last k transformer blocks on the retain set."""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
from torch import optim

from src.unlearning.unlearners.base import BaseUnlearner

logger = logging.getLogger(__name__)

# A block ModuleList entry: "encoder.encoders.11", with or without a wrapper prefix.
_BLOCK_RE = re.compile(r"^(?P<container>.*(?:encoders|decoders))\.(?P<index>\d+)$")


def _param_group(param_name: str) -> str:
    """Map a parameter name to the layer group that owns it."""
    parts = param_name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part.isdigit():
            return ".".join(parts[: i + 1])
    return ".".join(parts[:-1]) if len(parts) > 1 else param_name


def param_groups(model: torch.nn.Module) -> "OrderedDict[str, List[str]]":
    """Group ``model``'s parameters by owning module, in registration order."""
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    for name, _ in model.named_parameters():
        groups.setdefault(_param_group(name), []).append(name)
    return groups


def ordered_blocks(groups: "OrderedDict[str, List[str]]") -> List[str]:
    """Transformer/E-Branchformer blocks from ``groups``, ordered by depth."""
    container_rank: Dict[str, int] = {}
    blocks = []
    for name in groups:
        match = _BLOCK_RE.match(name)
        if match is None:
            continue
        container = match.group("container")
        container_rank.setdefault(container, len(container_rank))
        blocks.append((container_rank[container], int(match.group("index")), name))
    return [name for _, _, name in sorted(blocks)]


def freeze_except_last_k(model: torch.nn.Module, k: int) -> List[str]:
    """Freeze all parameters except those in the last k blocks."""
    groups = param_groups(model)
    selectable = ordered_blocks(groups)
    if selectable:
        kind = "transformer blocks"
    else:
        # Non-block architecture (or a test stub): fall back to module groups in registration order.
        selectable, kind = list(groups), "module groups (no blocks found)"
        logger.warning(
            "CF-k: no transformer blocks found in %s; falling back to the last "
            "k module groups in registration order.", type(model).__name__,
        )

    logger.info(
        "CF-k: %d layer groups, %d selectable %s in depth order: %s",
        len(groups), len(selectable), kind, selectable,
    )

    if k > len(selectable):
        logger.warning(
            "CF-k: k=%d exceeds the %d available %s; unfreezing all of them.",
            k, len(selectable), kind,
        )
    unfrozen_names = selectable[-k:] if k > 0 else []
    unfrozen = set(unfrozen_names)

    frozen_n, unfrozen_n = 0, 0
    for name, param in model.named_parameters():
        if _param_group(name) in unfrozen:
            param.requires_grad_(True)
            unfrozen_n += 1
        else:
            param.requires_grad_(False)
            frozen_n += 1

    logger.info(
        "CF-k: froze %d params across %d groups; unfroze %d params in the last "
        "%d %s: %s",
        frozen_n, len(groups) - len(unfrozen), unfrozen_n, len(unfrozen), kind,
        unfrozen_names,
    )
    return unfrozen_names


class CatastrophicForgettingTrainer(BaseUnlearner):
    """CF-k: fine-tune only the last k transformer-block layer groups."""

    def __init__(self, model, datasets, train_config: Dict[str, Any], schedule=None):
        super().__init__(model, datasets, train_config, schedule)
        self.k: int = int(train_config["k"])
        self._frozen: bool = False
        self._unfrozen_layers: List[str] = []

    # ------------------------------------------------------------------
    # DataLoaders — train on retain only; validate on forget to monitor
    # ------------------------------------------------------------------

    def train_dataloader(self):
        return self._make_loader(self.retain_set, self.batch_size_retain, True)

    def validation_step(self, batch, batch_idx):
        forget_loss = self.model(batch["forget"])
        self.log("val/forget_task_loss", forget_loss.detach(),
                 on_epoch=True, prog_bar=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Layer freezing — must happen before configure_optimizers
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        # Freezing in on_train_start would be too late: the optimizer is already built, which...
        if stage not in (None, "fit"):
            return
        if not self._frozen:
            self._unfrozen_layers = freeze_except_last_k(self.model, self.k)
            self._frozen = True

    # ------------------------------------------------------------------
    # Optimizer — only passes parameters with requires_grad=True
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        all_params = list(self.model.parameters())
        trainable = [p for p in all_params if p.requires_grad]

        assert self._frozen, "CF-k: setup() must run before configure_optimizers()"
        assert len(trainable) < len(all_params), (
            f"CF-k: freezing had no effect — all {len(all_params)} params are "
            f"trainable (k={self.k})"
        )
        groups = param_groups(self.model)
        expected = sum(len(groups[name]) for name in self._unfrozen_layers)
        assert len(trainable) == expected, (
            f"CF-k: {len(trainable)} trainable params but the {len(self._unfrozen_layers)} "
            f"selected groups hold {expected}"
        )

        lr = self.optimizer_config["lr"]
        wd = self.optimizer_config["weight_decay"]
        if self.optimizer_config["type"].lower() == "sgd":
            return optim.SGD(
                trainable, lr=lr,
                momentum=self.optimizer_config["momentum"],
                weight_decay=wd,
            )
        return optim.Adam(
            trainable, lr=lr,
            betas=tuple(self.optimizer_config["betas"]),
            weight_decay=wd,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        retain_loss = self.model(batch)
        self.log("train/retain_task_loss", retain_loss.detach(),
                 on_epoch=True, prog_bar=True)
        return retain_loss

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------

    def _mlflow_params(self) -> Dict[str, Any]:
        return {
            **super()._mlflow_params(),
            "k": self.k,
            "unfrozen_layers": str(self._unfrozen_layers),
        }
