import copy
import io

import torch
from torch.nn.utils.rnn import pad_sequence

from espnet2.bin.asr_inference import Speech2Text as Speech2TextASR
from espnet2.bin.s2t_inference import Speech2Text as Speech2TextS2T
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos

from .base import ModelWrapper


class ESPnetASRWrapper(ModelWrapper):
    def __init__(self, speech2text: Speech2TextASR):
        super().__init__()

        self.model = speech2text.asr_model

        self._speech2text = [speech2text]
        self._tokenizer = [speech2text.tokenizer]
        self._converter = [speech2text.converter]
        self.text_pad_token_id = -1

        if hasattr(self.model, "config"):
            self.config = self.model.config
        # Some versions use asr_model.config
        elif hasattr(self.model, "asr_model") and hasattr(self.model.asr_model, "config"):
            self.config = self.model.asr_model.config
        else:
            self.config = {
                "encoder": getattr(self.model, "encoder", None),
                "decoder": getattr(self.model, "decoder", None),
                "ctc": getattr(self.model, "ctc", None),
            }

        if hasattr(self.model, "frontend"):
            pass

        # Fix for LogMel melmat device issue
        self.logmel_layers = [m for m in self.model.modules() if m.__class__.__name__ == "LogMel"]

    def pad_and_tokenize_text(self, texts):
        tokens_ids = [torch.tensor(self.tokenize(
            t), dtype=torch.long) for t in texts]
        tokens_lens = torch.tensor([len(t)
                                    for t in tokens_ids], dtype=torch.long)

        padded_tokens = pad_sequence(
            tokens_ids,
            batch_first=True,
            padding_value=self.text_pad_token_id
        )
        device = next(self.model.parameters()).device
        return padded_tokens.to(device), tokens_lens.to(device)

    def tokenize(self, text):
        return self._converter[0].tokens2ids(
            self._tokenizer[0].text2tokens(text)
        )

    def original_forward(self, *args):
        if len(args) == 3:
            wav, wav_lens, wrd = args
        else:
            batch = args[0]
            wav = batch['wav']
            wav_lens = batch['wav_lens']
            wrd = batch['wrd']

        wrd_padded, wrd_lens = self.pad_and_tokenize_text(wrd)

        return self.model(
            wav,
            wav_lens,
            wrd_padded,
            wrd_lens
        )

    def forward(self, *args):
        if len(args) == 3:
            wav, wav_lens, wrd = args
        else:
            batch = args[0]
            wav = batch['wav']
            wav_lens = batch['wav_lens']
            wrd = batch['wrd']

        for m in self.logmel_layers:
            if hasattr(m, "melmat") and m.melmat.device != wav.device:
                m.melmat = m.melmat.to(wav.device)

        wrd_padded, wrd_lens = self.pad_and_tokenize_text(wrd)

        loss, _, _ = self.model(
            wav,
            wav_lens,
            wrd_padded,
            wrd_lens
        )

        return loss

    def forward_att_logits(self, *args):
        if len(args) == 2:
            wav, wav_lens = args
            text, text_lens = None, None
        elif len(args) == 4:
            wav, wav_lens, text, text_lens = args
        else:
            batch = args[0]
            wav = batch['wav']
            wav_lens = batch['wav_lens']
            text = batch.get('wrd', None)

        encoder_out, encoder_out_lens = self.model.encode(wav, wav_lens)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        text, text_lengths = self.pad_and_tokenize_text(text)

        # Important: trim first
        text = text[:, : text_lengths.max()]

        ys_in_pad, ys_out_pad = add_sos_eos(
            text, self.model.sos, self.model.eos, self.model.ignore_id
        )
        ys_in_lens = text_lengths + 1

        decoder_out, _ = self.model.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [B, U, V]

        return decoder_out, ys_out_pad, ys_in_lens

    def forward_ctc_logits(self, *args):
        if len(args) == 2:
            wav, wav_lens = args
        elif len(args) == 4:
            wav, wav_lens, _, _ = args
        else:
            batch = args[0]
            wav = batch["wav"]
            wav_lens = batch["wav_lens"]

        encoder_out, encoder_out_lens = self.model.encode(wav, wav_lens)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        if self.model.ctc is None:
            raise RuntimeError("Model has no CTC head")

        ctc_logits = self.model.ctc.ctc_lo(encoder_out)  # [B, T, V]
        return ctc_logits, encoder_out_lens

    def forward_all(self, batch):
        wav = batch["wav"]
        wav_lens = batch["wav_lens"]
        text = batch["wrd"]

        # =========================
        # Fix LogMel device ONCE
        # =========================
        for m in self.logmel_layers:
            if hasattr(m, "melmat") and m.melmat.device != wav.device:
                m.melmat = m.melmat.to(wav.device)

        # =========================
        # Encode ONCE
        # =========================
        encoder_out, encoder_out_lens = self.model.encode(wav, wav_lens)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # =========================
        # CTC branch
        # =========================
        if self.model.ctc is None:
            raise RuntimeError("Model has no CTC head")

        ctc_logits = self.model.ctc.ctc_lo(encoder_out)

        # =========================
        # ATT branch
        # =========================
        text, text_lengths = self.pad_and_tokenize_text(text)
        text = text[:, : text_lengths.max()]

        ys_in_pad, ys_out_pad = add_sos_eos(
            text, self.model.sos, self.model.eos, self.model.ignore_id
        )
        ys_in_lens = text_lengths + 1

        decoder_out, _ = self.model.decoder(
            encoder_out, encoder_out_lens,
            ys_in_pad, ys_in_lens
        )

        return (
            ctc_logits,
            encoder_out_lens,
            decoder_out,
            ys_out_pad,
            ys_in_lens
        )

    def compute_loss(self, loss_fn, logits, text, text_lengths):

        loss = loss_fn(
            logits,
            text,
            text_lengths
        )

        return loss

    def inference(self, speech):
        if speech.dim() == 2:
            return [self._speech2text[0](s) for s in speech]
        return self._speech2text[0](speech)

    def to_device(self, device):
        self.model.to(device)
        self._speech2text[0].device = device

        def _propagate_device(module, device):
            for attr in dir(module):
                if isinstance(getattr(module, attr), torch.nn.Module):
                    getattr(module, attr).to(device)
                    _propagate_device(getattr(module, attr), device)

        _propagate_device(self._speech2text[0], device)

    def to(self, device):
        self.to_device(device)

    @staticmethod
    def from_pretrained(model_tag, **kwargs):
        speech2text = Speech2TextASR.from_pretrained(
            model_tag=model_tag, **kwargs)
        return ESPnetASRWrapper(speech2text)
    
    def clone(self, freeze: bool = True):
        model_state = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
        }

        # copy.deepcopy fails on weight_norm's non-leaf tensors; pickle handles them.
        buf = io.BytesIO()
        torch.save(self._speech2text[0], buf)
        buf.seek(0)
        speech2text = torch.load(buf, weights_only=False)

        new_wrapper = ESPnetASRWrapper(speech2text)

        new_wrapper.model.load_state_dict(model_state)

        device = next(self.model.parameters()).device
        new_wrapper.model.to(device)

        if freeze:
            new_wrapper.model.eval()
            for p in new_wrapper.model.parameters():
                p.requires_grad_(False)

        return new_wrapper

class ESPnetS2TWrapper(ModelWrapper):
    def __init__(self, speech2text):
        super().__init__()

        self.model = speech2text.s2t_model

        self._speech2text = [speech2text]
        self._tokenizer = [speech2text.tokenizer]
        self._converter = [speech2text.converter]
        self.text_pad_token_id = -1

        if hasattr(self.model, "config"):
            self.config = self.model.config
        elif hasattr(self.model, "s2t_model") and hasattr(self.model.asr_model, "config"):
            self.config = self.model.s2t_model.config
        else:
            self.config = {
                "encoder": getattr(self.model, "encoder", None),
                "decoder": getattr(self.model, "decoder", None),
                "ctc": getattr(self.model, "ctc", None),
            }

    def pad_and_tokenize_text(self, texts):
        tokens_ids = [torch.tensor(self.tokenize(
            t), dtype=torch.long) for t in texts]
        tokens_lens = torch.tensor([len(t)
                                    for t in tokens_ids], dtype=torch.long)

        padded_tokens = pad_sequence(
            tokens_ids,
            batch_first=True,
            padding_value=self.text_pad_token_id
        )
        device = next(self.model.parameters()).device
        return padded_tokens.to(device), tokens_lens.to(device)

    def original_forward(self, *args):
        if len(args) == 3:
            wav, wav_lens, wrd = args
        else:
            batch = args[0]
            wav = batch['wav']
            wav_lens = batch['wav_lens']
            wrd = batch['wrd']

        wrd_padded, wrd_lens = self.pad_and_tokenize_text(wrd)

        return self.model(
            wav,
            wav_lens,
            wrd_padded,
            wrd_lens
        )

    def forward(self, *args):
        if len(args) == 3:
            wav, wav_lens, wrd = args
        else:
            batch = args[0]
            wav = batch['wav']
            wav_lens = batch['wav_lens']
            wrd = batch['wrd']

        wrd_padded, wrd_lens = self.pad_and_tokenize_text(wrd)

        loss, _, _ = self.model(
            wav,
            wav_lens,
            wrd_padded,
            wrd_lens
        )
        return loss

    def forward_logits(self, *args):
        if len(args) == 2:
            wav, wav_lens = args
            text, text_lens = None, None
        elif len(args) == 4:
            wav, wav_lens, text, text_lens = args
        else:
            batch = args[0]
            wav = batch['wav']
            wav_lens = batch['wav_lens']
            text = batch.get('wrd', None)
            text_lens = batch.get('wrd_lens', None)

        if text is not None and text_lens is not None:
            text, text_lens = self.pad_and_tokenize_text(text)

        for m in self.logmel_layers:
            if hasattr(m, "melmat") and m.melmat.device != wav.device:
                m.melmat = m.melmat.to(wav.device)

        encoder_out, encoder_out_lens = self.model.encode(wav, wav_lens)

        decoder_out, _ = self.model.decoder(
            encoder_out, encoder_out_lens, text, text_lens
        )  # [batch, seqlen, dim]

        return decoder_out, encoder_out_lens

    def compute_loss(self, loss_fn, logits, text, text_lengths):

        loss = loss_fn(
            logits,
            text,
            text_lengths
        )

        return loss

    def inference(self, speech):
        for m in self.model.modules():
            if m.__class__.__name__ == "LogMel":
                if hasattr(m, 'melmat') and m.melmat.device != speech.device:
                    m.melmat = m.melmat.to(speech.device)

        if speech.dim() == 2:
            return [self._speech2text[0](s) for s in speech]
        return self._speech2text[0](speech)

    def to_device(self, device):
        self.model.to(device)
        self._speech2text[0].device = device

        def _propagate_device(module, device):
            for attr in dir(module):
                if isinstance(getattr(module, attr), torch.nn.Module):
                    getattr(module, attr).to(device)
                    _propagate_device(getattr(module, attr), device)

        _propagate_device(self._speech2text[0], device)

    @staticmethod
    def from_pretrained(model_tag):
        speech2text = Speech2TextS2T.from_pretrained(model_tag=model_tag)
        return ESPnetS2TWrapper(speech2text)
