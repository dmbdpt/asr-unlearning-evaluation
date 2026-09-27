from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

from src.data.datasets.librispeech import collate_fn as collate_fn_espnet
from src.unlearning.unlearners.base import BaseUnlearner


def _kl_from_logprobs(
    logp_s: torch.Tensor,  # [B, T, V]
    logp_t: torch.Tensor,  # [B, T, V]
    lengths: Optional[torch.Tensor] = None,  # [B]
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean KL(student || teacher) over valid time steps."""
    p = logp_s.exp()
    kl_bt = torch.sum(p * (logp_s - logp_t), dim=-1)  # [B, T]

    if lengths is not None:
        B, T = kl_bt.shape
        t = torch.arange(T, device=kl_bt.device).unsqueeze(0).expand(B, T)
        mask = t < lengths.unsqueeze(1)
        kl_bt = kl_bt * mask.float()
        denom = mask.float().sum().clamp_min(eps)
        return kl_bt.sum() / denom

    return kl_bt.mean()

def _kl_from_logprobs_tokenwise(
    logp_s: torch.Tensor,   # [B, U, V]
    logp_t: torch.Tensor,   # [B, U, V]
    target_pad: torch.Tensor,  # [B, U]
    ignore_id: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    p = logp_s.exp()
    kl_bu = torch.sum(p * (logp_s - logp_t), dim=-1)  # [B, U]

    mask = (target_pad != ignore_id)
    kl_bu = kl_bu * mask.float()
    denom = mask.float().sum().clamp_min(eps)
    return kl_bu.sum() / denom


@dataclass
class ScrubSchedule:
    cycles: int
    extra_retain_epochs: int
    forget_steps_per_epoch: Optional[int]
    retain_steps_per_epoch: Optional[int]

    @property
    def total_epochs(self) -> int:
        return self.cycles * 2 + self.extra_retain_epochs

    def phase_for_epoch(self, epoch_idx: int) -> str:
        main = self.cycles * 2
        if epoch_idx < main:
            return "forget" if (epoch_idx % 2 == 0) else "retain"
        return "retain"

    def phase_for_batch(self, epoch_idx: int, batch_idx: int) -> Optional[str]:
        phase = self.phase_for_epoch(epoch_idx)

        if phase == "forget":
            if self.forget_steps_per_epoch is not None and batch_idx >= self.forget_steps_per_epoch:
                return None
        else:
            if self.retain_steps_per_epoch is not None and batch_idx >= self.retain_steps_per_epoch:
                return None

        return phase


class SCRUBTrainer(BaseUnlearner):

    def __init__(self, model, datasets, train_config: Dict[str, Any], schedule: Optional[ScrubSchedule] = None):
        super().__init__(model, datasets, train_config)

        # SCRUB-specific hyperparams — read flat (no nested scrub: sub-block)
        self.alpha_kl_retain = float(train_config["alpha_kl_retain"])
        self.beta_kl_forget = float(train_config["beta_kl_forget"])
        self.retain_loss_weight = float(train_config["retain_loss_weight"])

        if schedule is not None:
            self.schedule = schedule
        else:
            self.schedule = ScrubSchedule(
                cycles=int(train_config["cycles"]),
                extra_retain_epochs=int(train_config["extra_retain_epochs"]),
                forget_steps_per_epoch=train_config["forget_steps_per_epoch"],
                retain_steps_per_epoch=train_config["retain_steps_per_epoch"],
            )

        # Teacher snapshot (frozen copy of the student at t=0)
        self.teacher = self.model.clone(freeze=True)

        # Manual optimisation — alternating forget/retain phases
        self.automatic_optimization = False

        # Model-architecture flags for CTC / attention heads
        self.model_lm_weight = getattr(self.model.model, "lm_weight", 0.0)
        self.model_ctc_weight = getattr(self.model.model, "ctc_weight", 0.0)
        self.has_att = hasattr(self.model.model, "decoder") and self.model_ctc_weight < 1.0
        self.has_ctc = getattr(self.model.model, "ctc", None) is not None

    # -------------------------
    # Dataloaders (Lightning)
    # -------------------------
    def train_dataloader(self):
        pw = self.num_workers > 0
        retain_loader = DataLoader(self.datasets["retain"], batch_size=self.batch_size_retain,
                                   collate_fn=collate_fn_espnet, num_workers=self.num_workers,
                                   shuffle=True, pin_memory=True, persistent_workers=pw)
        forget_loader = DataLoader(self.datasets["forget"], batch_size=self.batch_size_forget,
                                   collate_fn=collate_fn_espnet, num_workers=self.num_workers,
                                   shuffle=True, pin_memory=True, persistent_workers=pw)
        return CombinedLoader({"retain": retain_loader, "forget": forget_loader}, mode="max_size_cycle")

    def val_dataloader(self):
        pw = self.num_workers > 0
        retain_loader = DataLoader(self.datasets["retain"], batch_size=self.batch_size_retain,
                                   collate_fn=collate_fn_espnet, num_workers=self.num_workers,
                                   shuffle=False, pin_memory=True, persistent_workers=pw)
        forget_loader = DataLoader(self.datasets["forget"], batch_size=self.batch_size_forget,
                                   collate_fn=collate_fn_espnet, num_workers=self.num_workers,
                                   shuffle=False, pin_memory=True, persistent_workers=pw)
        return CombinedLoader({"retain": retain_loader, "forget": forget_loader}, mode="max_size_cycle")

    # -------------------------
    # Helpers
    # -------------------------
    def _phase(self, batch_idx: Optional[int] = None) -> Optional[str]:
        if batch_idx is None:
            return self.schedule.phase_for_epoch(self.current_epoch)

        return self.schedule.phase_for_batch(self.current_epoch, batch_idx)

    def _ctc_logprobs(self, wrapper, batch: Dict[str, Any]):
        ctc_logits, encoder_out_lens = wrapper.forward_ctc_logits(batch)  # [B, T, V]
        return F.log_softmax(ctc_logits, dim=-1), encoder_out_lens

    def _att_logprobs(self, wrapper, batch):
        att_logits, ys_out_pad, ys_in_lens = wrapper.forward_att_logits(batch)
        return F.log_softmax(att_logits, dim=-1), ys_out_pad, ys_in_lens

    # -------------------------
    # Training
    # -------------------------
    def on_train_epoch_start(self):
        self.teacher.eval()
        self.log("train/phase_is_forget", float(self._phase() == "forget"), prog_bar=True)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        phase = self._phase(batch_idx)
        
        # Early exit if we are pausing for this batch
        if phase is None:
            return None

        b_forget = batch["forget"]
        b_retain = batch["retain"]

        if phase == "forget":
            kl_forget = torch.tensor(0.0, device=self.device)
            if self.has_ctc:
                logp_s_ctc_f, lens_f = self._ctc_logprobs(self.model, b_forget)
                with torch.no_grad():
                    logp_t_ctc_f, _ = self._ctc_logprobs(self.teacher, b_forget)
                kl_forget += _kl_from_logprobs(logp_s_ctc_f, logp_t_ctc_f, lengths=lens_f)

            if self.has_att:
                logp_s_att_f, ys_out_pad_f, _ = self._att_logprobs(self.model, b_forget)
                with torch.no_grad():
                    logp_t_att_f, _, _ = self._att_logprobs(self.teacher, b_forget)
                kl_forget += _kl_from_logprobs_tokenwise(
                    logp_s_att_f, logp_t_att_f, ys_out_pad_f, self.model.model.ignore_id
                )
            
            loss = -self.beta_kl_forget * kl_forget
            self.log("train/kl_forget", kl_forget.detach(), sync_dist=True)

        elif phase == "retain":
            retain_task_loss = self.model(b_retain)

            kl_retain = torch.tensor(0.0, device=self.device)
            if self.has_ctc:
                logp_s_ctc_r, lens_r = self._ctc_logprobs(self.model, b_retain)
                with torch.no_grad():
                    logp_t_ctc_r, _ = self._ctc_logprobs(self.teacher, b_retain)
                kl_retain += _kl_from_logprobs(logp_s_ctc_r, logp_t_ctc_r, lengths=lens_r)

            if self.has_att:
                logp_s_att_r, ys_out_pad_r, _ = self._att_logprobs(self.model, b_retain)
                with torch.no_grad():
                    logp_t_att_r, _, _ = self._att_logprobs(self.teacher, b_retain)
                kl_retain += _kl_from_logprobs_tokenwise(
                    logp_s_att_r, logp_t_att_r, ys_out_pad_r, self.model.model.ignore_id
                )
            
            loss = (self.retain_loss_weight * retain_task_loss) + (self.alpha_kl_retain * kl_retain)
            self.log("train/kl_retain", kl_retain.detach(), sync_dist=True)
            self.log("train/retain_task_loss", retain_task_loss.detach(), sync_dist=True)

        self.log("train/loss", loss.detach(), prog_bar=True, sync_dist=True)

        if loss.requires_grad:
            self.manual_backward(loss)

            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

            opt.step()
        else:
            # still step optimizer to avoid hanging schedulers/DDP sync issues
            opt.step()

        return loss
    # -------------------------
    # Validation
    # -------------------------
    def validation_step(self, batch, batch_idx):
        forget_loss = self.model(batch["forget"])
        self.log("val/forget_task_loss", forget_loss.detach(), on_epoch=True, prog_bar=True, sync_dist=True)

    # -------------------------
    # Optimizer — AdamW (SCRUB-specific; base uses Adam/SGD)
    # -------------------------

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer_config["lr"],
            weight_decay=self.optimizer_config["weight_decay"],
        )
