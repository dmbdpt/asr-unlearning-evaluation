import os
import pickle
import json
import hashlib
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from src.utils.utils import rank0_print
from src.data.features.acoustic_extractor import AcousticExtractor
from src.data.features.textual_extractor import TextualExtractor

class FeatureExtractor:
    def __init__(self, hparams=None, device=None):
        self.hparams = hparams or {}
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.embedding_type = self.hparams["embedding_type"]
        self.acoustic_model = self.hparams["acoustic_model"]
        self.textual_model = self.hparams["textual_model"]
        self.batch_size = self.hparams["batch_size"]
        self.cache_dir = self.hparams["cache_dir"]
        self.seed = self.hparams["seed"]

        self.acoustic_dim = None
        self.textual_dim = None

    def _get_config_hash(self):
        relevant_keys = [
            "embedding_type",
            "acoustic_model",
            "textual_model",
            "batch_size",
            "seed"
        ]
        config = {k: self.hparams.get(k) for k in relevant_keys}
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self, dataset_name):
        config_hash = self._get_config_hash()
        filename = f"features_{dataset_name}_{config_hash}.pkl"
        return os.path.join(self.cache_dir, filename)

    def extract(self, data: Dataset, dataset_name: str = "default"):
        cache_path = self._get_cache_path(dataset_name)
        if os.path.exists(cache_path):
            rank0_print(f"[Feature Extraction] Loading cached features from {cache_path}")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            
            self.acoustic_dim = cached_data.get("acoustic_dim")
            self.textual_dim = cached_data.get("textual_dim")
            return cached_data["features"]

        rank0_print(
            f"[Feature Extraction] Extracting {self.embedding_type} embeddings for {dataset_name}...")
            
        extractor_hparams = OmegaConf.to_container(self.hparams, resolve=True)

        features = None
        if self.embedding_type == "acoustic":
            extractor_hparams["pretrained_model"] = self.acoustic_model
            extractor_hparams["batch_size"] = self.batch_size
            extractor = AcousticExtractor(hparams=extractor_hparams, device=self.device)
            features = extractor.extract(data)
            
        elif self.embedding_type == "textual":
            extractor_hparams["textual_model"] = self.textual_model
            extractor_hparams["batch_size"] = self.batch_size
            extractor = TextualExtractor(hparams=extractor_hparams, device=self.device)
            features = extractor.extract(data)
            
        elif self.embedding_type == "concat":
            extractor_hparams["pretrained_model"] = self.acoustic_model
            extractor_hparams["batch_size"] = self.batch_size
            acoustic_extractor = AcousticExtractor(hparams=extractor_hparams, device=self.device)
            acoustic = acoustic_extractor.extract(data)
            
            extractor_hparams["textual_model"] = self.textual_model
            textual_extractor = TextualExtractor(hparams=extractor_hparams, device=self.device)
            textual = textual_extractor.extract(data)
            
            features = np.concatenate([acoustic, textual], axis=1)
            self.acoustic_dim = acoustic.shape[1]
            self.textual_dim = textual.shape[1]
        else:
            raise ValueError(f"Unknown embedding_type: {self.embedding_type}")

        return features

    def get_feature_dims(self):
        return {"acoustic": self.acoustic_dim, "textual": self.textual_dim}

    def separate_features(self, features):
        if self.embedding_type == "concat":
            acoustic = features[:, :self.acoustic_dim]
            textual = features[:, self.acoustic_dim:]
            return acoustic, textual
        else:
            return features