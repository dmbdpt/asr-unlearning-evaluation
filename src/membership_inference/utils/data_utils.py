import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Dict, Any, List


class MembershipInferenceDataset(Dataset):
    """PyTorch Dataset for membership inference experiments."""
    
    def __init__(self, csv_path: str, sample_rate: int = 16000):
        self.sample_rate = sample_rate

        self.df = pd.read_csv(csv_path)

        required_columns = [
            "wrd", "ID", "dataset", "language", 
            "utt_in_set", "spk_in_set", "speaker_id", 
            "duration", "wav"
        ]
        missing_cols = set(required_columns) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load audio and metadata for a single sample."""
        row = self.df.iloc[idx]

        wav_path = row["wav"]
        waveform, orig_sr = torchaudio.load(wav_path)

        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze(0)
        
        return {
            "wav": waveform,
            "wrd": row["wrd"],
            "ID": row["ID"],
            "dataset": row["dataset"],
            "language": row["language"],
            "utt_in_set": int(row["utt_in_set"]),
            "spk_in_set": int(row["spk_in_set"]),
            "speaker_id": row["speaker_id"],
            "duration": float(row["duration"]),
        }


def create_mi_dataset(csv: str, sample_rate: int = 16000) -> MembershipInferenceDataset:
    """Create a PyTorch Dataset for membership inference experiments."""
    return MembershipInferenceDataset(csv, sample_rate)


def collate_fn_mi(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad audio arrays and collect metadata for membership inference batches."""
    wav = [sample["wav"] for sample in batch]
    wav_lens = torch.tensor([w.numel() for w in wav], dtype=torch.long)
    wav_padded = torch.nn.utils.rnn.pad_sequence(wav, batch_first=True)

    wrd = [sample["wrd"] for sample in batch]
    ID = [str(sample["ID"]) for sample in batch]
    spk_id = [sample["speaker_id"] for sample in batch]

    dataset = [sample["dataset"] for sample in batch]
    language = [sample["language"] for sample in batch]

    utt_in_set = torch.tensor([sample["utt_in_set"] for sample in batch], dtype=torch.long)
    spk_in_set = torch.tensor([sample["spk_in_set"] for sample in batch], dtype=torch.long)

    duration = torch.tensor([sample["duration"] for sample in batch], dtype=torch.float32)


    return {
        "wav": wav_padded,
        "wav_lens": wav_lens,
        "wrd": wrd,
        "ID": ID,
        "spk_id": spk_id,
        "speaker_id": spk_id,
        "dataset": dataset,
        "language": language,
        "utt_in_set": utt_in_set,
        "spk_in_set": spk_in_set,
        "duration": duration,
    }