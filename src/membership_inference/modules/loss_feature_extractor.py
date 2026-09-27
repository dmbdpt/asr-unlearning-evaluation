import torch
from .abs_feature_extractor import FeatureExtractor


class LossFeatureExtractorEspnetASR(FeatureExtractor):
    def __init__(self, model_to_attack, cfg):
        super().__init__(model_to_attack, cfg)
        self.name = cfg["feature_extractor_name"]
        self.loss_att = cfg["loss_att"]
        self.loss_ctc = cfg["loss_ctc"]
        self.loss_cer = cfg["loss_cer"]  # Not a loss but for consistency let's call it one

        if self.loss_att:
            self.name += "_att"

        if self.loss_ctc:
            self.name += "_ctc"

        if self.loss_cer:
            self.name += "_cer"

    # Dependent on type of dataloader/dataset used
    def get_sample_ids(self, batch):
        return batch["ID"]

    def get_labels(self, batch, label):
        if isinstance(label, str):
            return batch[label]
        elif isinstance(label, list):
            return [{key: batch[key][i] for key in label} for i in range(len(batch[label[0]]))]
        else:
            raise ValueError("Label must be either a string or a list of strings.")

    def extract_from_batch(self, batch):
        wav, wav_lens, text = batch["wav"], batch["wav_lens"], batch["wrd"]

        device = next(self.model_to_attack.parameters()).device

        wav = wav.to(device)
        wav_lens = wav_lens.to(device)

        features = []
        for i in range(wav.shape[0]):
            with torch.no_grad():
                # first output is full loss with grad, we only need stats
                _, stats, _ = self.model_to_attack.original_forward(wav[i][:wav_lens[i]].unsqueeze(0),
                                                                    wav_lens[i].unsqueeze(0),
                                                                    [text[i]])
            feats = []
            if self.loss_att:
                feats.append(stats['loss_att'].item())
            if self.loss_ctc:
                feats.append(stats['loss_ctc'].item())
            if self.loss_cer:
                feats.append(stats['cer'].item())
            features.append(feats)
        return features

    def extract_features(self, batch):
        return self.extract_from_batch(batch)
