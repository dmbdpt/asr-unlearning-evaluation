
import os
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

import torch
import tqdm


class FeatureExtractor(object):
    """Feature extraction and caching for membership inference attacks."""

    def __init__(
        self,
        model_to_attack: Any,
        cfg: dict[str, Any],
    ) -> None:
        """Initialize the FeatureExtractor."""
        super().__init__()
        self.name = cfg.get('feature_extractor_name', 'abstract_extractor')
        self.model_to_attack = model_to_attack
        self.cfg = cfg

    def extract_features(self, batch: Any) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Extract features from a batch."""
        raise NotImplementedError("extract_features must be implemented by the user.")

    def get_sample_ids(self, batch: Any) -> list[str]:
        """Get sample IDs from a batch."""
        raise NotImplementedError("get_sample_ids must be implemented by the user.")

    def get_labels(self, batch: Any, label: str | list) -> list[int] | list[dict[str, Any]]:
        """Get labels from a batch."""
        raise NotImplementedError("get_labels must be implemented by the user.")

    def extract_features_with_cache(
        self,
        dataloader: torch.utils.data.DataLoader,
        override: bool = False,
        save_frequency: int | None = None,
        label: str | list = 'spk_in_set',
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]] | tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[Any]]]:
        """Extract features with subset saving."""

        save_folder = self.cfg.get('save_features_folder', './features')
        os.makedirs(save_folder, exist_ok=True)
        feature_file = os.path.join(save_folder, f"{self.name}_features.h5")

        if save_frequency is None:
            save_frequency = self.cfg.get('cache_save_frequency', 10)

        # Ensure eval mode so dropout / BN are frozen during extraction
        if hasattr(self.model_to_attack, "eval"):
            self.model_to_attack.eval()

        cached_data = {}
        if os.path.exists(feature_file) and not override:
            with h5py.File(feature_file, 'r') as f:
                stored_ids = f['sample_ids']
                features_data = f['features']

                for idx, sid in enumerate(stored_ids):
                    cached_data[sid.decode("utf-8")] = features_data[idx]

        all_features = []
        all_labels = []
        
        batch_count = 0
        for batch in tqdm.tqdm(dataloader, desc="Extracting features"):
            sample_ids = self.get_sample_ids(batch)
            all_labels.extend(self.get_labels(batch, label))

            if all(sid in cached_data for sid in sample_ids):
                for sid in sample_ids:
                    feat = cached_data[sid]
                    all_features.append(feat)

            else:
                feats = self.extract_features(batch)
                all_features.extend(feats)

                for i, sid in enumerate(sample_ids):
                    cached_data[sid] = feats[i]

                if batch_count % save_frequency == 0:
                    self._save_cache_to_file(feature_file, cached_data)
                batch_count += 1

        features = np.array(all_features)
        if isinstance(all_labels[0], dict):
            labels = {key: np.array([l[key] for l in all_labels]) for key in label}
        else:
            labels = np.array(all_labels)
        
        if batch_count != 0 and batch_count % save_frequency > 0:
            self._save_cache_to_file(feature_file, cached_data)

        if batch_count > 0:
            print(f"Computed {batch_count} new batches for {self.name} feature extractor.")

        return features, labels

    def _save_cache_to_file(
        self,
        feature_file: str,
        cached_data: dict[str, tuple[npt.NDArray, npt.NDArray]],
    ) -> None:
        """Save cache to HDF5 file."""
        sample_id_list = list(cached_data.keys())
        sample_id_list.sort()
        sample_id_list = [str(sid) for sid in sample_id_list]
        features_list = [cached_data[sid] for sid in sample_id_list]

        with h5py.File(feature_file, 'w') as f:
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('sample_ids', data=np.array(sample_id_list, dtype=object), dtype=dt)
            f.create_dataset('features', data=np.array(features_list))

