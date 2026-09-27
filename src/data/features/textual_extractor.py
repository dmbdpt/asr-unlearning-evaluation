import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
import numpy as np
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import warnings
from src.data.datasets.librispeech import collate_fn as collate_fn_espnet

class TextualExtractor:
    def __init__(self, hparams=None, device=None):
        self.hparams = hparams or {}

        self.is_distributed = dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.rank)
            self.device = torch.device(f"cuda:{self.rank}")
        else:
            self.rank = 0
            self.world_size = 1
            if device:
                self.device = device
            else:
                self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model_name = self.hparams["textual_model"]
        self.batch_size = self.hparams["batch_size"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

		# Might be wrong but should not be used either way
        self.fallback_embedding_size = self.hparams["fallback_embedding_size"]

    def extract(self, data):
        total_len = len(data)
        if self.is_distributed:
            chunk_size = int(np.ceil(total_len / self.world_size))
            start_idx = self.rank * chunk_size
            end_idx = min((self.rank + 1) * chunk_size, total_len)
            subset_indices = list(range(start_idx, end_idx))
            
            dist.barrier()
            print(f"[Textual] Rank {self.rank}/{self.world_size} processing {len(subset_indices)} samples")
        else:
            subset_indices = list(range(total_len))
            print(f"[Textual] Processing {len(subset_indices)} samples")

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
                num_workers=0
            )

            desc = f"Textual Rank {self.rank}" if self.is_distributed else "Textual Extraction"
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=desc, position=self.rank):
                    texts = batch['wrd_raw']
                    inputs = self.tokenizer(
                        texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    
                    local_features.append(embeddings.cpu().numpy())
                    
                    del inputs, outputs, embeddings
                    torch.cuda.empty_cache()

        if local_features:
            local_features_np = np.vstack(local_features)
            embedding_dim = local_features_np.shape[1]
        else:
            warnings.warn(f"No features found for rank {self.rank}. Proceeding with empty embedding.")
            embedding_dim = self.fallback_embedding_size
            local_features_np = np.empty((0, embedding_dim))
            
        print(f"[Textual] Rank {self.rank} finished. Extracted {len(local_features_np)} samples.")

        if self.is_distributed:
            all_features = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_features, local_features_np)
            return np.vstack(all_features)
        else:
            return local_features_np
