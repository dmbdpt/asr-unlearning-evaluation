import os
import torch
import torchaudio
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence


class LibriSpeechDataset(Dataset):
    def __init__(self, root, url="train-clean-100", download=False, subset=None):
        self.root = root
        self.url = url
        self.subset_name = subset if subset else url

        if download:
            # torchaudio writes the archive into `root` before extracting, so it must exist.
            os.makedirs(self.root, exist_ok=True)

        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=self.root, url=self.url, download=download
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
        item = self.dataset[idx]
        return {
            "wav": item[0].squeeze(0),
            "wav_lens": item[0].shape[-1],
            "wrd": item[2],
            "speaker_id": item[3],
            "id": str(item[5])
        }

    def get_metadata(self, idx, dicted=False):
        ret = None
        if hasattr(self.dataset, "dataset") and isinstance(self.dataset, Subset):
            real_idx = self.dataset.indices[idx]
            if hasattr(self.dataset.dataset, "get_metadata"):
                return self.dataset.dataset.get_metadata(real_idx)

        if hasattr(self.dataset, "get_metadata"):
            # ret: (audio_path, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
            ret = self.dataset.get_metadata(idx)

        if dicted and ret is not None:
            ret = {
                "wav": ret[0],
                "sample_rate": ret[1],
                "wrd": ret[2],
                "spk_id":  str(ret[3]),
                "speaker_id":  str(ret[3]),
                "chapter_id": str(ret[4]),
                "utt_id": str(ret[5])
            }

        return ret

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


def collate_fn(batch, padding=True):
    speech = [item["wav"].clone().detach() for item in batch]
    speech = pad_sequence([x.clone().detach()
                           for x in speech], batch_first=True)

    speech_lengths = torch.tensor([item["wav_lens"]
                                  for item in batch], dtype=torch.long)
    text = [item["wrd"] for item in batch]

    batch_dict = {
        "wav": speech,
        "wav_lens": speech_lengths,
        "wrd": text,
        "wrd_raw": text,
    }

    if "speaker_id" in batch[0]:
        # Keep speaker_ids as strings, not tensors, to avoid JSON serialization issues
        speaker_ids = [str(item["speaker_id"]) for item in batch]
        batch_dict["speaker_id"] = speaker_ids
    elif "spk_id" in batch[0]:
        speaker_ids = [str(item["spk_id"]) for item in batch]
        batch_dict["speaker_id"] = speaker_ids

    if "id" in batch[0] or "utt_id" in batch[0]:
        ids = [str(item.get("id", item.get("utt_id", idx))) for idx, item in enumerate(batch)]
        batch_dict["utt_id"] = ids

    return batch_dict
