"""Membership inference attack implementation."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from torch.utils.data import DataLoader

from src.membership_inference.utils import create_mi_dataset, collate_fn_mi, compute_binary_mia_metrics


class Attacker(object):
    """Attacker class for membership inference attacks."""

    def __init__(
        self,
        cfg: dict[str, Any],
    ) -> None:
        """Initialize the Attacker."""
        super().__init__()
        self.cfg = cfg
        self.datasets = cfg.get('datasets')
        self.hf_datasets = self.create_datasets(self.datasets)
        self.classifier = cfg.get('classifier')
        self.feature_extractors = cfg.get('feature_extractors')
        self.label = cfg.get('label', 'utt_in_set')

    def create_datasets(self, datasets: dict[str, Any]) -> dict[str, Any]:
        """Create Hugging Face datasets from the provided files."""
        assert set(datasets.keys()).issubset({'train', 'dev', 'test'})
        hf_datasets = {}
        for split, csv in datasets.items():
            hf_datasets[split] = create_mi_dataset(csv)
        return hf_datasets

    def train_dataloader(self) -> DataLoader:
        g = torch.Generator().manual_seed(self.cfg.get('seed', 42))
        return DataLoader(
            self.hf_datasets['train'],
            batch_size=self.cfg.get('batch_size', 32),
            shuffle=True,
            num_workers=self.cfg.get('num_workers', 4),
            collate_fn=collate_fn_mi,
            generator=g,
        )

    def dev_dataloader(self) -> DataLoader:
        return DataLoader(
            self.hf_datasets['dev'],
            batch_size=self.cfg.get('batch_size', 32),
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 4),
            collate_fn=collate_fn_mi,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.hf_datasets['test'],
            batch_size=self.cfg.get('batch_size', 32),
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 4),
            collate_fn=collate_fn_mi,
        )

    def extract_features(
        self,
        split: str,
        override: bool = False,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Extract features using the feature extractor with caching."""

        if split == 'train':
            dataloader = self.train_dataloader()
        elif split == 'dev':
            dataloader = self.dev_dataloader()
        elif split == 'test':
            dataloader = self.test_dataloader()
        else:
            raise ValueError(f"Unknown split: {split}")

        X_list = []
        for feature_extractor in self.feature_extractors:
            X, y = feature_extractor.extract_features_with_cache(
                dataloader, label=self.label, override=override
            )
            X_list.append(X)

        X = np.concatenate(X_list, axis=1)
        return X, y

    def train_classifier(self,
                         override: bool = False) -> None:
        """Train the membership inference classifier."""
        X, y = self.extract_features('train', override)
        self.classifier = self.classifier.fit(X, y)

    def evaluate(self, 
                 split: str = 'dev', 
                 override: bool = False) -> dict[str, float]:
        """Evaluate the membership inference attack."""

        X, y = self.extract_features(split, override=override)
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)[:, 1]
        elif hasattr(self.classifier, "decision_function"):
            probs = self.classifier.decision_function(X)
        else:
            raise ValueError("Classifier must have either predict_proba or decision_function method.")
        preds = self.classifier.predict(X)

        return compute_binary_mia_metrics(y, probs, preds)

