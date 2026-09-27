"""Membership inference attack implementation."""

from typing import Any

import numpy as np
import numpy.typing as npt

import pandas as pd

from torch.utils.data import DataLoader

from .abs_attacker import Attacker
from src.membership_inference.utils import create_mi_dataset, collate_fn_mi, compute_binary_mia_metrics


class SpeakerLevelAttacker(Attacker):
    """Attacker class for membership inference attacks."""

    def __init__(
        self,
        cfg: dict[str, Any],
    ) -> None:
        """Initialize the Attacker."""
        super().__init__(cfg)
        self.class_label = cfg['class_label']

    def extract_features(
        self,
        split: str,
        override: bool = False,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]] | tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[Any]]]:
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
        y = y[self.class_label]
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

        y, spk_ids = y[self.class_label], y['speaker_id']

        df = pd.DataFrame({
            'y': y,
            'preds': preds,
            'probs': probs,
            'speaker_id': spk_ids
        })

        df = df.groupby("speaker_id").mean()
        y, preds, probs = df['y'].values, df['preds'].values, df['probs'].values

        preds = np.where(preds >= 0.5, 1, 0)
        y = np.where(y >= 0.5, 1, 0)

        return compute_binary_mia_metrics(y, probs, preds)

