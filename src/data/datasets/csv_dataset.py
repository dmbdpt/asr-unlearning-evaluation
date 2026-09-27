import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd



class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # split,wav,sampl_rate,wrd,speaker_id,utt_id,chapt_id,dur_sec
        self.transform = transform
        COLS = ['wav', 'sampl_rate', 'wrd', 'speaker_id', 'utt_id']

        missing = set(COLS) - set(self.data.columns)
        if missing:
            raise ValueError(f"CSV file must contain the following columns: {COLS}")

        if "duration" not in self.data.columns and "dur_sec" in self.data.columns:

            self.data['duration'] = self.data['dur_sec']
            self.data['wav_lens'] = self.data['duration'] #* self.data['sampl_rate']

            COLS.append('duration')
            COLS.append('wav_lens')

        self.data = self.data[COLS].copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        # Work with a plain dict, not a pandas view/Series
        sample = self.data.iloc[idx].to_dict()

        if self.transform:
            sample = self.transform(sample)

        audio_path = sample['wav']
        waveform, sample_rate = torchaudio.load(audio_path)

        sample['wav'] = waveform.squeeze(0)   # Remove channel dimension if mono
        sample['wav_lens'] = waveform.shape[-1]

        exp_sample_rate = sample['sampl_rate']
        if exp_sample_rate != sample_rate:
            raise ValueError(
                f"sample rate should be {exp_sample_rate}, but got {sample_rate}"
            )

        return sample

    def get_metadata(self, idx, dicted=False):
        row = self.data.iloc[idx]
        return row.to_dict() if dicted else row.to_list()

    def get_subj_indices(self, speaker_id):
        return self.data.index[self.data['speaker_id'] == speaker_id].tolist()
