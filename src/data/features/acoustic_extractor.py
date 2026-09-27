import os
import pickle
import torch
import torchaudio
import numpy as np
import warnings
from torch.utils.data import Dataset, Subset, DataLoader
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import hashlib
from speechbrain.inference.speaker import EncoderClassifier
from espnet2.bin.spk_inference import Speech2Embedding
from transformers import AutoTokenizer, AutoModel
import torch.multiprocessing as mp
import torch.distributed as dist

from src.data.datasets.librispeech import LibriSpeechDataset as LibriSpeechWrapper, collate_fn as collate_fn_espnet

class AcousticExtractor:
    def __init__(self, hparams=None, device=None):
        self.hparams = hparams or {}
        
        self.is_distributed = dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = f"cuda:{self.rank}"
        else:
            self.rank = 0
            self.world_size = 1
            if device:
                self.device = device
            else:
                self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.pretrained_model = self.hparams["pretrained_model"]
        self.batch_size = self.hparams["batch_size"]

        self.embedder = Speech2Embedding.from_pretrained(
            self.pretrained_model, device=self.device, batch_size=self.batch_size)

        # Might be wrong but should not be used either way
        self.fallback_embedding_size = self.embedder.spk_model.projector.fc.out_features

    def extract(self, data):
        total_len = len(data)
        if self.is_distributed:
            chunk_size = int(np.ceil(total_len / self.world_size))
            start_idx = self.rank * chunk_size
            end_idx = min((self.rank + 1) * chunk_size, total_len)
            subset_indices = list(range(start_idx, end_idx))
            
            dist.barrier()
            print(f"[Acoustic] Rank {self.rank}/{self.world_size} processing {len(subset_indices)} samples")
        else:
            subset_indices = list(range(total_len))
            print(f"[Acoustic] Processing {len(subset_indices)} samples")

        local_features = []
        
        if subset_indices:
            dataset = data
            if len(subset_indices) < total_len:
                dataset = Subset(dataset, subset_indices)

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn_espnet,
                num_workers=8
            )

            desc = f"Acoustic Rank {self.rank}" if self.is_distributed else "Acoustic Extraction"
            with torch.no_grad():
                for speech in tqdm(dataloader, desc=desc, position=self.rank):
                    waveforms = speech['wav'].to(self.device)
                    wav_lens = speech['wav_lens'].to(self.device)

                    batch = {
                        "speech": waveforms,
                        "speech_lengths": wav_lens,
                        "extract_embd": True
                    }
                    
                    embeddings = self.embedder.spk_model(**batch)

                    if isinstance(embeddings, tuple):
                        embeddings = embeddings[0]

                    local_features.append(embeddings.cpu().numpy())

                    del waveforms, wav_lens, batch, embeddings
                    torch.cuda.empty_cache()

        if local_features:
            local_features_np = np.vstack(local_features)
            embedding_dim = local_features_np.shape[1]
        else:
            warnings.warn(f"No features found for rank {self.rank}. Proceeding with empty embedding.")
            embedding_dim = self.fallback_embedding_size
            local_features_np = np.empty((0, embedding_dim))
            
        print(f"[Acoustic] Rank {self.rank} finished. Extracted {len(local_features_np)} samples.")

        if self.is_distributed:
            all_features = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_features, local_features_np)
            return np.vstack(all_features)
        else:
            return local_features_np