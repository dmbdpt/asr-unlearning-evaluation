from typing import List, Optional, Tuple, Union

from omegaconf import DictConfig
from tqdm import tqdm

import torch.distributed as dist

from src.utils import rank0_print
from src.artifacts.simplified_artifacts import SimplifiedArtifacts
from src.data.data_handler import DataHandler
from src.data.features.feature_extractor import FeatureExtractor


def run_load_datasets_main(config: DictConfig) -> DataHandler:
    data_handler = DataHandler(config)
    data_handler.load_data()
    return data_handler


def run_load_datasets(cfg: DictConfig, artifacts: SimplifiedArtifacts) -> DataHandler:
    data_handler = None

    print("[Experiment] Loading dataset...")

    dataset_cfg = cfg.data_preparation.dataset

    was_cached = False

    if not dist.is_initialized() or dist.get_rank() == 0:
        data_handler, was_cached = artifacts.get_cached(
            artifact_type="dataset",
            config=dataset_cfg,
            loader=lambda: run_load_datasets_main(cfg.data_preparation)
        )
    if dist.is_initialized():
        dist.barrier()
        if dist.get_rank() != 0:
            data_handler, was_cached = artifacts.get_cached(
                artifact_type="dataset",
                config=dataset_cfg,
                loader=lambda: run_load_datasets_main(cfg.data_preparation)
            )

    num_speakers = len(data_handler.data.get("speaker_to_indices", {}))
    artifacts.log_metrics({
        "dataset.num_speakers": num_speakers,
        "dataset.was_cached": int(was_cached)
    }, stage="data_preparation")

    dataset_meta = {
        "name": dataset_cfg.name,
        "root": str(dataset_cfg.root),
        "train_subset": dataset_cfg.train_subset,
        "test_subset": dataset_cfg.test_subset,
        "valid_fraction": dataset_cfg.valid_fraction,
        "seed": dataset_cfg.seed,
        "num_speakers": num_speakers,
        "was_cached": was_cached,
    }
    artifacts.tracker.log_dict(dataset_meta, "data/dataset_metadata.json")

    artifacts.tracker.log_params({
        "dataset.root": str(dataset_cfg.root),
        "dataset.valid_fraction": dataset_cfg.valid_fraction,
        "dataset.seed": dataset_cfg.seed,
        "dataset.num_speakers": num_speakers,
    })


    if was_cached:
        rank0_print("[Experiment] Dataset loaded from cache")
    else:
        rank0_print("[Experiment] Dataset loaded and cached")

    rank0_print(f"[Experiment] Logged dataset metadata: {num_speakers} speakers")

    return data_handler


def run_extract_features_main(cfg: DictConfig, data_handler: DataHandler, to_extract: Union[List[str], Tuple[str, ...]] = ("train", "test")) -> Tuple[dict, Optional[FeatureExtractor]]:
    feature_extractor = FeatureExtractor(cfg)
    features = {}

    for dataset_name, dataset in tqdm(data_handler.datasets.items(), desc="Extracting features for datasets"):
        if dataset_name not in to_extract:
            continue
        features[dataset_name] = feature_extractor.extract(dataset, dataset_name=dataset_name)

    return features, feature_extractor

def run_extract_features(cfg: DictConfig, data_handler: DataHandler, artifacts: SimplifiedArtifacts,
                         to_extract: Union[List[str], Tuple[str, ...]] = ("train", "test")) -> Tuple[dict, Optional[FeatureExtractor]]:
 
    print("[Experiment] Extracting features...")

    result, was_cached = artifacts.get_cached(
        artifact_type="features",
        config=cfg.data_preparation.features,
        loader=lambda: run_extract_features_main(cfg=cfg.data_preparation.features,
                        data_handler=data_handler, to_extract=to_extract)
    )

    if isinstance(result, tuple):
        features, feature_extractor = result
    else:
        features = result
        feature_extractor = None

    if was_cached:
        print("[Experiment] Features loaded from cache")
    else:
        print("[Experiment] Features extracted and cached")

    data_handler.add_features(features)

    return features, feature_extractor



def run_init_unlearning_datasets(data_handler: DataHandler, cfg: dict, indices_forget: Optional[List[int]] = None) -> DataHandler:
    if cfg.get("unlearning_set") is None:
        raise ValueError("Unlearning not specified in cfg")

    cfg = cfg["unlearning_set"]

    if cfg["type"] == "speaker":
        if cfg["num_forget"]:
            data_handler.random_unlearning(num_forget=cfg["num_forget"])
        elif cfg["indice_forget_speakers"]:
            data_handler.set_unlearning_datasets(forget_speakers=cfg["indice_forget_speakers"])
        else:
            raise ValueError("Unlearning not specified in cfg")
    elif cfg["type"] == "cluster":
        if indices_forget:
            data_handler.set_unlearning_datasets(forget_speakers=indices_forget)
        else:
            raise ValueError("Unlearning not specified in cfg")
    else:
        raise ValueError("Unlearning type not specified in cfg")

    return data_handler
