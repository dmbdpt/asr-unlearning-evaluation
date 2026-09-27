import os
import pickle
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm

from src.data.datasets.csv_dataset import CSVDataset
from src.data.datasets.librispeech import LibriSpeechDataset
from src.utils.utils import rank0_print


class DataHandler:
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else {}

        self.data = {}
        self.datasets = {}
        self.datasets_features = {}

        # split_name -> {speaker_id: [positions inside that split dataset]}
        self.speaker_map_per_set = {}

        # split_name -> raw indices into the relevant raw dataset
        self.indices = {}

        self.dataset_hash = None

    @property
    def speaker_map(self):
        if not self.data or "all_speakers" not in self.data:
            return None
        return {spk: i for i, spk in enumerate(self.data["all_speakers"])}

    def _cfg_get(self, key: str, default=None):
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def _dataset_cfg_to_kwargs(self):
        dataset_cfg = self.cfg.dataset
        if isinstance(dataset_cfg, dict):
            return dataset_cfg
        if hasattr(dataset_cfg, "items"):
            return dict(dataset_cfg.items())
        return vars(dataset_cfg)

    def load_data(self):
        dataset_name = self.cfg.dataset.name

        if dataset_name == "librispeech":
            return self.load_librispeech(**self._dataset_cfg_to_kwargs())
        
        elif dataset_name == "custom_csv":
            return self.load_custom_csv(**self._dataset_cfg_to_kwargs())

        raise ValueError(f"Unknown dataset name: {dataset_name}")

    @staticmethod
    def _ensure_list(item):
        if isinstance(item, str):
            return [item]
        return list(item)

    @staticmethod
    def _shuffle_and_split_indices(
        num_items: int,
        valid_fraction: float,
        seed: int = 42,
    ) -> tuple[list[int], list[int]]:
        indices = np.arange(num_items)

        if valid_fraction <= 0.0:
            return indices.tolist(), []

        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        split = int(np.floor(valid_fraction * num_items))
        valid_idx = indices[:split].tolist()
        train_idx = indices[split:].tolist()
        return train_idx, valid_idx

    @staticmethod
    def _get_speaker_id_from_dataset(dataset, idx: int) -> Optional[str]:
        """Return speaker_id for a raw index in a dataset."""
        if hasattr(dataset, "get_metadata"):
            metadata = dataset.get_metadata(idx)
            if metadata:
                return str(metadata[3])
            return None

        if hasattr(dataset, "dataset") and hasattr(dataset, "indices") and hasattr(dataset.dataset, "get_metadata"):
            real_idx = dataset.indices[idx]
            metadata = dataset.dataset.get_metadata(real_idx)
            if metadata:
                return str(metadata[3])
            return None

        raise ValueError("Dataset does not have a recognized method for retrieving speaker ID.")

    def _build_speaker_to_raw_indices_for_concat_train(self, train_datasets):
        """
        Build global speaker -> raw indices map for the concatenated train dataset.
        """
        speaker_to_indices = defaultdict(list)
        current_offset = 0

        for ds in train_datasets:
            ds_name = getattr(ds, "url", ds.__class__.__name__)
            rank0_print(f"[Dataset] Indexing {ds_name}...")

            for i in tqdm(range(len(ds)), desc=f"Indexing {ds_name}", leave=False):
                speaker_id = self._get_speaker_id_from_dataset(ds, i)
                if speaker_id is not None:
                    speaker_to_indices[speaker_id].append(i + current_offset)

            current_offset += len(ds)

        return {spk: sorted(idxs) for spk, idxs in speaker_to_indices.items()}

    def _build_speaker_position_map_from_raw_indices(
        self,
        raw_dataset,
        raw_indices: Sequence[int],
        desc: str = "Building speaker index map",
    ) -> dict[str, list[int]]:
        """For a split defined by raw indices, return: speaker_id -> [positions within the split dataset]."""
        speaker_to_positions = defaultdict(list)

        for pos, raw_idx in enumerate(tqdm(raw_indices, desc=desc, leave=False)):
            speaker_id = self._get_speaker_id_from_dataset(raw_dataset, raw_idx)
            if speaker_id is not None:
                speaker_to_positions[speaker_id].append(pos)

        return {spk: positions for spk, positions in speaker_to_positions.items()}

    def _register_split(
        self,
        name: str,
        raw_dataset,
        raw_indices: Sequence[int],
        build_speaker_map: bool = True,
    ):
        raw_indices = list(raw_indices)
        self.indices[name] = raw_indices
        self.datasets[name] = Subset(raw_dataset, raw_indices)

        if build_speaker_map:
            self.speaker_map_per_set[name] = self._build_speaker_position_map_from_raw_indices(
                raw_dataset=raw_dataset,
                raw_indices=raw_indices,
                desc=f"Building speaker index map [{name}]",
            )

    def load_librispeech(
        self,
        root,
        train_subset,
        test_subset,
        valid_fraction,
        seed,
        download,
        **kwargs,
    ) -> dict:
        train_subsets = self._ensure_list(train_subset)
        test_subsets = self._ensure_list(test_subset)

        train_datasets = []
        train_datasets_map = {}
        for subset in train_subsets:
            rank0_print(f"[Dataset] Loading train subset: {subset}...")
            ds = LibriSpeechDataset(root=root, url=subset, download=download)
            train_datasets.append(ds)
            train_datasets_map[subset] = ds

        train_data_raw = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

        test_datasets = []
        test_datasets_map = {}
        for subset in test_subsets:
            rank0_print(f"[Dataset] Loading test subset: {subset}...")
            ds = LibriSpeechDataset(root=root, url=subset, download=download)
            test_datasets.append(ds)
            test_datasets_map[subset] = ds

        test_data_raw = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]

        speaker_to_train_raw_indices = self._build_speaker_to_raw_indices_for_concat_train(train_datasets)
        all_speakers = sorted(speaker_to_train_raw_indices.keys())

        self.data = {
            "speaker_to_indices": speaker_to_train_raw_indices,   # raw train index space
            "all_speakers": all_speakers,
        }

        self.datasets = {
            "train_raw": train_data_raw,
            "test_raw": test_data_raw,
        }
        self.indices = {}
        self.speaker_map_per_set = {}

        num_train = len(train_data_raw)
        train_idx, valid_idx = self._shuffle_and_split_indices(
            num_items=num_train,
            valid_fraction=valid_fraction,
            seed=seed,
        )

        self._register_split("train", train_data_raw, train_idx, build_speaker_map=True)

        if valid_idx:
            self._register_split("valid", train_data_raw, valid_idx, build_speaker_map=True)

        test_idx = list(range(len(test_data_raw)))
        self._register_split("test", test_data_raw, test_idx, build_speaker_map=True)

        # These are raw datasets, not Subsets, unlike self.datasets["train"/"test"]
        for name, ds in train_datasets_map.items():
            self.datasets[name] = ds
            self.speaker_map_per_set[name] = self._build_speaker_position_map_from_raw_indices(
                raw_dataset=ds,
                raw_indices=list(range(len(ds))),
                desc=f"Building speaker index map [{name}]",
            )

        for name, ds in test_datasets_map.items():
            self.datasets[name] = ds
            self.speaker_map_per_set[name] = self._build_speaker_position_map_from_raw_indices(
                raw_dataset=ds,
                raw_indices=list(range(len(ds))),
                desc=f"Building speaker index map [{name}]",
            )

        return self.datasets

    def load_custom_csv(self, train_csv, test_csv, valid_fraction, seed, **kwargs):
        train_csvs = self._ensure_list(train_csv)
        test_csvs = self._ensure_list(test_csv)

        train_datasets = []
        train_datasets_map = {}
        for csv in train_csvs:
            rank0_print(f"[Dataset] Loading train CSV: {csv}...")
            ds = CSVDataset(csv)
            train_datasets.append(ds)
            train_datasets_map[csv] = ds

        train_data_raw = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

        test_datasets = []
        test_datasets_map = {}
        for csv in test_csvs:
            rank0_print(f"[Dataset] Loading test CSV: {csv}...")
            ds = CSVDataset(csv)
            test_datasets.append(ds)
            test_datasets_map[csv] = ds

        test_data_raw = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]

        speaker_to_train_raw_indices = self._build_speaker_to_raw_indices_for_concat_train(train_datasets)
        all_speakers = sorted(speaker_to_train_raw_indices.keys())

        self.data = {
            "speaker_to_indices": speaker_to_train_raw_indices,   # raw train index space
            "all_speakers": all_speakers,
        }
        self.datasets = {
            "train_raw": train_data_raw,
            "test_raw": test_data_raw,
        }
        self.indices = {}
        self.speaker_map_per_set = {}

        num_train = len(train_data_raw)
        train_idx, valid_idx = self._shuffle_and_split_indices(
            num_items=num_train,
            valid_fraction=valid_fraction,
            seed=seed,
        )

        self._register_split("train", train_data_raw, train_idx, build_speaker_map=True)
        if valid_idx:
            self._register_split("valid", train_data_raw, valid_idx, build_speaker_map=True)
        test_idx = list(range(len(test_data_raw)))
        self._register_split("test", test_data_raw, test_idx, build_speaker_map=True)

        # These are raw datasets, not Subsets, unlike self.datasets["train"/"test"]
        for name, ds in train_datasets_map.items():
            self.datasets[name] = ds
            self.speaker_map_per_set[name] = self._build_speaker_position_map_from_raw_indices(
                raw_dataset=ds,
                raw_indices=list(range(len(ds))),
                desc=f"Building speaker index map [{name}]",
            )

        for name, ds in test_datasets_map.items():
            self.datasets[name] = ds
            self.speaker_map_per_set[name] = self._build_speaker_position_map_from_raw_indices(
                raw_dataset=ds,
                raw_indices=list(range(len(ds))),
                desc=f"Building speaker index map [{name}]",
            )
        return self.datasets


    def set_unlearning_datasets(self, forget_speakers: Sequence = (), neutral_speakers: Sequence = ()):
        """Build forget and retain splits."""
        if "train_raw" not in self.datasets or "train" not in self.indices:
            raise ValueError("Datasets not loaded. Call load_data() first.")

        train_raw_dataset = self.datasets["train_raw"]
        train_raw_set = set(self.indices["train"])

        forget_raw_indices = set()
        for spk in forget_speakers:
            if spk not in self.data["speaker_to_indices"]:
                raise ValueError(f"Speaker {spk} not found in training data.")
            forget_raw_indices.update(self.data["speaker_to_indices"][spk])
        forget_raw_indices = sorted(train_raw_set & forget_raw_indices)

        neutral_raw_indices: set = set()
        for spk in neutral_speakers:
            if spk not in self.data["speaker_to_indices"]:
                raise ValueError(f"Speaker {spk} not found in training data.")
            neutral_raw_indices.update(self.data["speaker_to_indices"][spk])
        neutral_raw_indices &= train_raw_set

        retain_raw_indices = sorted(train_raw_set - set(forget_raw_indices) - neutral_raw_indices)

        self._register_split("forget", train_raw_dataset, forget_raw_indices, build_speaker_map=True)
        self._register_split("retain", train_raw_dataset, retain_raw_indices, build_speaker_map=True)

        forget_set = set(self.indices["forget"])
        retain_set = set(self.indices["retain"])
        train_set = set(self.indices["train"])

        assert forget_set.isdisjoint(retain_set), "Forget and retain sets are not disjoint!"
        if not neutral_speakers:
            assert forget_set | retain_set == train_set, "Forget + retain do not reconstruct train!"

        # Verify no forgotten or neutral speaker bleeds into retain
        retain_speakers = set(self.speaker_map_per_set["retain"].keys())
        overlap = retain_speakers.intersection(set(map(str, forget_speakers)))
        assert not overlap, f"Forgotten speakers still present in retain: {sorted(overlap)}"
        neutral_overlap = retain_speakers.intersection(set(map(str, neutral_speakers)))
        assert not neutral_overlap, f"Neutral speakers still present in retain: {sorted(neutral_overlap)}"

        return self.datasets

    def random_unlearning(self, num_forget: int = 1, seed: Optional[int] = None):
        if "train_raw" not in self.datasets or "train" not in self.indices:
            raise ValueError("Datasets not loaded. Call load_data() first.")

        train_raw_dataset = self.datasets["train_raw"]
        train_raw_indices = np.array(self.indices["train"], dtype=np.int64)

        if num_forget < 0:
            raise ValueError("num_forget must be >= 0")
        if num_forget > len(train_raw_indices):
            raise ValueError(
                f"num_forget={num_forget} is larger than train split size={len(train_raw_indices)}"
            )

        rng = np.random.default_rng(seed)
        forget_raw_indices = sorted(rng.choice(train_raw_indices, size=num_forget, replace=False).tolist())
        forget_raw_set = set(forget_raw_indices)
        retain_raw_indices = sorted([i for i in train_raw_indices.tolist() if i not in forget_raw_set])

        self._register_split("forget", train_raw_dataset, forget_raw_indices, build_speaker_map=True)
        self._register_split("retain", train_raw_dataset, retain_raw_indices, build_speaker_map=True)

        return self.datasets

    def get_subject_index_on_dataset(self, subject: str = None, dataset: str = "train"):
        """Return indices for a subject."""
        if subject is None:
            raise ValueError("subject must not be None")

        if dataset == "train_raw":
            if "speaker_to_indices" not in self.data:
                raise ValueError("Dataset not loaded. Call load_data() first.")
            if subject not in self.data["speaker_to_indices"]:
                raise ValueError(f"Subject {subject} not found in train_raw.")
            return self.data["speaker_to_indices"][subject]

        if dataset not in self.speaker_map_per_set:
            raise ValueError(f"Dataset {dataset} not loaded or has no speaker map.")

        if subject not in self.speaker_map_per_set[dataset]:
            raise ValueError(f"Subject {subject} not found in dataset {dataset}.")

        return self.speaker_map_per_set[dataset][subject]

    def add_features(self, features: dict = None, to_extract=None):
        if features is None:
            raise ValueError("features must not be None")
        if to_extract is None:
            to_extract = ["train", "test"]

        for dataset_name in to_extract:
            if dataset_name not in features:
                raise ValueError(f"Missing features for dataset {dataset_name}")
            self.datasets_features[dataset_name] = features[dataset_name]

    def get_subject_features(self, subject: str = None, dataset: str = None):
        if subject is None:
            raise ValueError("subject must not be None")

        if dataset is not None:
            if dataset not in self.datasets_features:
                raise ValueError(f"Features for dataset {dataset} not loaded. Call add_features() first.")
            if dataset not in self.speaker_map_per_set:
                raise ValueError(f"Speaker map for dataset {dataset} not available.")
            if subject not in self.speaker_map_per_set[dataset]:
                raise ValueError(f"Subject {subject} not found in dataset {dataset}.")
            return self.datasets_features[dataset][self.speaker_map_per_set[dataset][subject]]

        # Global lookup over loaded feature-bearing datasets
        for ds_name, speaker_map in self.speaker_map_per_set.items():
            if ds_name in self.datasets_features and subject in speaker_map:
                return self.datasets_features[ds_name][speaker_map[subject]]

        raise ValueError(f"Subject {subject} not found in any loaded dataset features.")

    def save_distances(
        self,
        distances: dict,
        target: str,
        dataset_name: str = "default",
        prefix: str = "distances",
    ):
        cache_dir = self._cfg_get("cache_dir")
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(cache_dir, f"{prefix}_{target}_{dataset_name}.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump(distances, f)