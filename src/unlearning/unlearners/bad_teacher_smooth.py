import os
import math
import logging
from typing import Dict, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader
from pytorch_lightning.utilities.combined_loader import CombinedLoader

from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
    LegacyRelPositionMultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from src.data.datasets.librispeech import collate_fn as collate_fn_espnet
from src.models.base import ModelWrapper
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from src.unlearning.unlearners.base import BaseUnlearner

logger = logging.getLogger(__name__)


# =========================================================
# KL helpers
# =========================================================
def _kl_from_logprobs(logp_s, logp_t, lengths=None, eps=1e-8):
    # The teacher cache is built with a plain DataLoader while training is DDP-sharded, so...
    if logp_s.size(1) != logp_t.size(1):
        T = min(logp_s.size(1), logp_t.size(1))
        logp_s = logp_s[:, :T, :]
        logp_t = logp_t[:, :T, :]

    p = logp_s.exp()
    kl_bt = torch.sum(p * (logp_s - logp_t), dim=-1)

    if lengths is not None:
        B, T = kl_bt.shape
        lengths = lengths.clamp_max(T)
        t = torch.arange(T, device=kl_bt.device).unsqueeze(0).expand(B, T)
        mask = t < lengths.unsqueeze(1)
        kl_bt = kl_bt * mask.float()
        denom = mask.float().sum().clamp_min(eps)
        return kl_bt.sum() / denom

    return kl_bt.mean()


def _kl_from_logprobs_tokenwise(logp_s, logp_t, target_pad, ignore_id, eps=1e-8):
    # Same DDP/cache mismatch as in CTC; trim target_pad too so the mask stays aligned.
    if logp_s.size(1) != logp_t.size(1):
        T = min(logp_s.size(1), logp_t.size(1))
        logp_s = logp_s[:, :T, :]
        logp_t = logp_t[:, :T, :]
        target_pad = target_pad[:, :T]

    p = logp_s.exp()
    kl_bu = torch.sum(p * (logp_s - logp_t), dim=-1)

    mask = (target_pad != ignore_id)
    kl_bu = kl_bu * mask.float()
    denom = mask.float().sum().clamp_min(eps)
    return kl_bu.sum() / denom


# Temperature Attention
# =========================================================
class TemperatureMultiHeadedAttention(MultiHeadedAttention):
    def __init__(self, n_head, n_feat, dropout_rate, temperature=1.0, **kwargs):
        super().__init__(n_head, n_feat, dropout_rate, **kwargs)
        self.temperature = temperature

    def forward(self, query, key, value, mask, expand_kv=False):
        """Compute scaled dot product attention with temperature."""
        
        if getattr(self, "use_sdpa", False) or self.use_flash_attn:
            query = query / self.temperature
            return super().forward(query, key, value, mask, expand_kv)

        q, k, v = self.forward_qkv(query, key, value, expand_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(self.d_k) * self.temperature)        
        return self.forward_attention(v, scores, mask)

_REL_POS_ATTN_TYPES = (RelPositionMultiHeadedAttention, LegacyRelPositionMultiHeadedAttention)


def _make_temperature_rel_pos_class(base_cls):
    """Builds a temperature-scaled subclass of a relative-position attention class."""

    class _TemperatureRelPositionMultiHeadedAttention(base_cls):
        def __init__(self, *args, temperature=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.temperature = temperature

        def forward(self, query, key, value, pos_emb, mask):
            q, k, v = self.forward_qkv(query, key, value)
            q = q.transpose(1, 2)

            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)

            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            matrix_bd = self.rel_shift(matrix_bd)

            scores = (matrix_ac + matrix_bd) / (math.sqrt(self.d_k) * self.temperature)
            return self.forward_attention(v, scores, mask)

    return _TemperatureRelPositionMultiHeadedAttention


# =========================================================
# Decoder Smooth Attention Teacher — patches the decoder only
# =========================================================
class SmoothDecoderAttentionTeacher(torch.nn.Module):
    """Smoothed bad teacher that applies temperature scaling to the decoder attention (affects ATT..."""

    def __init__(self, teacher_model, temperature=2.0, output_temperature=1.0):
        super().__init__()
        self.model = teacher_model
        self.temperature = temperature
        self.output_temperature = output_temperature

        self._patch(self.model)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Attention patching
    # ------------------------------------------------------------------
    def _patch(self, model):
        self._patch_decoder(model)

    def _replace_plain(self, attn):
        old_state = attn.state_dict()
        new_attn = TemperatureMultiHeadedAttention(
            n_head=attn.h,
            n_feat=attn.d_k * attn.h,
            dropout_rate=attn.dropout_rate,
            qk_norm=True if not isinstance(attn.q_norm, nn.Identity) else False,
            use_flash_attn=getattr(attn, "use_flash_attn", False),
            causal=attn.causal if hasattr(attn, "causal") else False,
            cross_attn=attn.cross_attn if hasattr(attn, "cross_attn") else False,
            use_sdpa=getattr(attn, "use_sdpa", False),
            temperature=self.temperature,
        )
        new_attn.load_state_dict(old_state)
        return new_attn

    def _replace_rel_pos(self, attn):
        old_state = attn.state_dict()
        temp_cls = _make_temperature_rel_pos_class(type(attn))
        new_attn = temp_cls(
            attn.h,
            attn.d_k * attn.h,
            attn.dropout_rate,
            zero_triu=getattr(attn, "zero_triu", False),
            temperature=self.temperature,
        )
        new_attn.load_state_dict(old_state)
        return new_attn

    def _replace(self, attn):
        if isinstance(attn, _REL_POS_ATTN_TYPES):
            return self._replace_rel_pos(attn)
        if isinstance(attn, MultiHeadedAttention):
            return self._replace_plain(attn)
        logger.warning(f"Skipping unsupported attention type for smoothing: {type(attn).__name__}")
        return None

    def _patch_decoder(self, model):
        for layer in model.model.decoder.decoders:
            new_self_attn = self._replace(layer.self_attn)
            if new_self_attn is not None:
                layer.self_attn = new_self_attn
            if hasattr(layer, "src_attn") and layer.src_attn is not None:
                new_src_attn = self._replace(layer.src_attn)
                if new_src_attn is not None:
                    layer.src_attn = new_src_attn

    @torch.no_grad()
    def forward_all(self, batch):
        wav = batch["wav"]
        wav_lens = batch["wav_lens"]
        text = batch["wrd"]

        encoder_out, encoder_out_lens = self.model.model.encode(wav, wav_lens)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # CTC — smoothed via output temperature (and, in the encoder variant, via the patched...
        ctc_logits = self.model.model.ctc.ctc_lo(encoder_out)
        ctc_logp = F.log_softmax(ctc_logits / self.output_temperature, dim=-1)

        # ATT — smoothed via patched decoder attention
        text, text_lengths = self.model.pad_and_tokenize_text(text)
        text = text[:, : text_lengths.max()]

        ys_in_pad, ys_out_pad = add_sos_eos(
            text, self.model.model.sos, self.model.model.eos, self.model.model.ignore_id
        )
        ys_in_lens = text_lengths + 1

        decoder_out, _ = self.model.model.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )
        att_logp = F.log_softmax(decoder_out / self.output_temperature, dim=-1)

        return ctc_logp, att_logp, encoder_out_lens, ys_out_pad, ys_in_lens


# =========================================================
# Trainer
# =========================================================
class BadTeacherSmoothTrainer(BaseUnlearner):
    """Bad-Teacher unlearning with temperature-smoothed decoder attention (ATT branch)."""

    teacher_cls = SmoothDecoderAttentionTeacher
    cache_tag = "bad_teacher_smooth"

    def __init__(self, model: ModelWrapper, datasets, train_config, schedule=None):
        super().__init__(model, datasets, train_config, schedule)

        self.lambda_forget = float(train_config["lambda_forget"])
        self.kl_ctc_weight = float(train_config.get("kl_ctc_weight", 1.0))
        self.kl_att_weight = float(train_config.get("kl_att_weight", 1.0))

        temperature = train_config["temperature"]
        output_temperature = train_config.get("output_temperature", 1.0)
        self.bad_teacher = self.teacher_cls(
            teacher_model=self.model.clone(freeze=True),
            temperature=temperature,
            output_temperature=output_temperature,
        )

        forget_spks = "_".join(sorted(set(str(d["speaker_id"]) for d in self.datasets["forget"])))
        self.cache_file = train_config.get(
            "cache_file",
            f".cache/bad_teacher/{self.cache_tag}_cache_{temperature}_ot{output_temperature}_{forget_spks}.pt",
        )

        self.use_teacher_cache = True
        self.teacher_cache = {}
        self.automatic_optimization = False

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self):
        return CombinedLoader(
            {
                "retain": DataLoader(
                    self.datasets["retain"],
                    batch_size=self.batch_size_retain,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    persistent_workers=False,
                    collate_fn=collate_fn_espnet,
                ),
                "forget": DataLoader(
                    self.datasets["forget"],
                    batch_size=self.batch_size_forget,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    persistent_workers=False,
                    collate_fn=collate_fn_espnet,
                ),
            },
            mode="max_size_cycle",
        )

    def val_dataloader(self):
        return CombinedLoader(
            {
                "retain": DataLoader(
                    self.datasets["retain"],
                    batch_size=self.batch_size_retain,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    persistent_workers=False,
                    collate_fn=collate_fn_espnet,
                ),
                "forget": DataLoader(
                    self.datasets["forget"],
                    batch_size=self.batch_size_forget,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    persistent_workers=False,
                    collate_fn=collate_fn_espnet,
                ),
            },
            mode="max_size_cycle",
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        b_retain = batch["retain"]
        b_forget = batch["forget"]

        # 1. Retain loss (standard task loss)
        retain_loss = self.model(b_retain)
        if isinstance(retain_loss, tuple):
            retain_loss = retain_loss[0]

        # 2. Student forward on forget set
        if hasattr(self.model, "forward_all"):
            s_ctc_logits, s_ctc_lens, s_att_logits, s_ys_pad, s_ys_lens = \
                self.model.forward_all(b_forget)
        else:
            s_ctc_logits, s_ctc_lens = self.model.forward_ctc_logits(b_forget)
            s_att_logits, s_ys_pad, s_ys_lens = self.model.forward_att_logits(b_forget)

        compute_ctc = self.kl_ctc_weight != 0.0
        compute_att = self.kl_att_weight != 0.0

        s_ctc_logp = F.log_softmax(s_ctc_logits, dim=-1) if compute_ctc else None
        s_att_logp = F.log_softmax(s_att_logits, dim=-1) if compute_att else None

        # 3. Bad teacher outputs (from cache or computed)
        cache_keys = [
            f"{spk}_{uid}"
            for spk, uid in zip(b_forget["speaker_id"], b_forget["utt_id"])
        ]

        if self.use_teacher_cache and all(k in self.teacher_cache for k in cache_keys):
            # Cache tensors already live on this rank's device (see on_fit_start), so pad_sequence...
            bt_ctc_logp = pad_sequence(
                [self.teacher_cache[k][0] for k in cache_keys],
                batch_first=True, padding_value=0.0,
            ) if compute_ctc else None
            bt_att_logp = pad_sequence(
                [self.teacher_cache[k][1] for k in cache_keys],
                batch_first=True, padding_value=0.0,
            ) if compute_att else None
        else:
            with torch.no_grad():
                bt_ctc_logp, bt_att_logp, _, _, _ = self.bad_teacher.forward_all(b_forget)
            bt_ctc_logp = bt_ctc_logp.detach()
            bt_att_logp = bt_att_logp.detach()

        # 4. KL divergence losses
        ignore_id = getattr(self.model, "ignore_id", -1)

        if compute_ctc:
            kl_ctc = _kl_from_logprobs(s_ctc_logp, bt_ctc_logp, lengths=s_ctc_lens)
        else:
            kl_ctc = torch.zeros((), device=s_ctc_logits.device)
        del s_ctc_logp, bt_ctc_logp

        if compute_att:
            kl_att = _kl_from_logprobs_tokenwise(
                s_att_logp, bt_att_logp,
                target_pad=s_ys_pad,
                ignore_id=ignore_id,
            )
        else:
            kl_att = torch.zeros((), device=s_ctc_logits.device)
        del s_att_logp, bt_att_logp, s_ys_pad

        forget_loss = self.kl_ctc_weight * kl_ctc + self.kl_att_weight * kl_att

        # 5. Combined loss
        loss = (1 - self.lambda_forget) * retain_loss + self.lambda_forget * forget_loss

        self.manual_backward(loss)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        opt.step()

        self.log("train/loss", loss.detach(), prog_bar=True)
        self.log("train/retain_loss", retain_loss.detach())
        self.log("train/forget_loss", forget_loss.detach())
        self.log("train/kl_ctc", kl_ctc.detach())
        self.log("train/kl_att", kl_att.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        b_retain = batch["retain"]
        b_forget = batch["forget"]

        retain_loss = self.model(b_retain)
        if isinstance(retain_loss, tuple):
            retain_loss = retain_loss[0]

        forget_loss = self.model(b_forget)
        if isinstance(forget_loss, tuple):
            forget_loss = forget_loss[0]

        self.log("val/retain_task_loss", retain_loss.detach(), prog_bar=True, sync_dist=True)
        self.log("val/forget_task_loss", forget_loss.detach(), prog_bar=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer_config["lr"],
            weight_decay=self.optimizer_config["weight_decay"],
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def on_fit_start(self):
        if not self.use_teacher_cache:
            return

        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0

        if is_rank0:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

            if not os.path.exists(self.cache_file):
                logger.info("Teacher cache not found. Computing smoothed bad teacher outputs for 'forget' set...")

                device = self.device
                self.bad_teacher.to(device)

                forget_dl = DataLoader(
                    self.datasets["forget"],
                    batch_size=self.batch_size_forget,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=collate_fn_espnet,
                )

                temp_cache = {}
                with torch.no_grad():
                    for batch in forget_dl:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        bt_ctc_logp, bt_att_logp, ctc_lens, _, ys_in_lens = \
                            self.bad_teacher.forward_all(batch)

                        for i, (uid, spk) in enumerate(zip(batch["utt_id"], batch["speaker_id"])):
                            ctc_len = ctc_lens[i].item()
                            att_len = ys_in_lens[i].item()
                            temp_cache[f"{spk}_{uid}"] = (
                                bt_ctc_logp[i, :ctc_len].cpu(),
                                bt_att_logp[i, :att_len].cpu(),
                            )

                torch.save(temp_cache, self.cache_file)
                logger.info(f"Saved bad teacher cache to {self.cache_file}")

        # All ranks wait for rank 0 to finish writing, then load together.
        if dist.is_initialized():
            dist.barrier()

        logger.info(f"Loading bad teacher cache from {self.cache_file}")
        raw_cache = torch.load(self.cache_file, map_location="cpu")
        # Move once, here, instead of paying a blocking host->device copy every training step...
        device = self.device
        self.teacher_cache = {
            k: (ctc_logp.to(device), att_logp.to(device))
            for k, (ctc_logp, att_logp) in raw_cache.items()
        }

    def on_fit_end(self):
        if self.use_teacher_cache:
            self.teacher_cache.clear()

    def _mlflow_params(self) -> Dict[str, Any]:
        return {
            "lambda_forget": self.lambda_forget,
            "temperature": self.bad_teacher.temperature,
            "output_temperature": self.bad_teacher.output_temperature,
            "kl_ctc_weight": self.kl_ctc_weight,
            "kl_att_weight": self.kl_att_weight,
        }
