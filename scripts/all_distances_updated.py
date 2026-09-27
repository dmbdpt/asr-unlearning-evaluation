#!/usr/bin/env python3
"""Compute pairwise distances between all subjects in the dataset."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data.data_handler import DataHandler
from src.data.features.feature_extractor import FeatureExtractor
from src.utils.utils import rank0_print

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("all_distances")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def load_config(config_path: str, overrides: Optional[list[str]] = None) -> DictConfig:
    """Load a YAML config and apply OmegaConf-style overrides."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    cfg = OmegaConf.create(config_dict)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


# -----------------------------------------------------------------------------
# Distributed setup
# -----------------------------------------------------------------------------
def setup_dist() -> tuple[int, torch.device, bool]:
    """Initialize distributed process group if launched with torchrun."""
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if not is_distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, device, False

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed launch detected but CUDA is not available.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=2),
            device_id=device,
        )

    rank = dist.get_rank()
    return rank, device, True


def cleanup_dist(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# Feature loading
# -----------------------------------------------------------------------------
def load_dataset_and_features(cfg: DictConfig, recompute: bool = False) -> tuple[DataHandler, FeatureExtractor]:
    """Load dataset and extract/load features for split-local train and test datasets, keeping..."""
    logger.info("Loading dataset and features...")
    data_prep_cfg = cfg.data_preparation

    logger.info("Loading dataset...")
    data_handler = DataHandler(data_prep_cfg)
    data_handler.load_data()

    num_speakers = len(data_handler.data.get("all_speakers", []))
    logger.info("Loaded dataset with %d train speakers", num_speakers)

    features_cfg = data_prep_cfg.features

    if recompute:
        cache_dir = OmegaConf.select(cfg, "data_preparation.features.cache_dir")
        if os.path.exists(cache_dir):
            removed = 0
            for filename in os.listdir(cache_dir):
                if filename.startswith("features_") and filename.endswith(".pkl"):
                    cache_file = os.path.join(cache_dir, filename)
                    logger.info("Removing cached feature file: %s", cache_file)
                    os.remove(cache_file)
                    removed += 1
            logger.info("Removed %d cached feature files", removed)

    logger.info("Initializing feature extractor...")
    feature_extractor = FeatureExtractor(features_cfg)

    # Extract split-local features only, to stay aligned with speaker_map_per_set
    wanted_datasets = ["train", "test"]
    features: dict[str, dict[str, Any]] = {}

    for dataset_name in wanted_datasets:
        if dataset_name not in data_handler.datasets:
            logger.warning("Dataset %s not available; skipping feature extraction", dataset_name)
            continue

        dataset = data_handler.datasets[dataset_name]
        logger.info("Extracting features for dataset: %s", dataset_name)

        dataset_features = feature_extractor.extract(dataset, dataset_name=dataset_name)

        if not hasattr(dataset_features, "shape"):
            raise TypeError(
                f"Feature extractor for {dataset_name} returned an object without .shape: "
                f"{type(dataset_features).__name__}"
            )

        if len(dataset_features) != len(dataset):
            raise ValueError(
                f"Feature count mismatch for {dataset_name}: "
                f"{len(dataset_features)} features vs {len(dataset)} samples"
            )

        features[dataset_name] = {
            "features": dataset_features,
            "acoustic_dim": getattr(feature_extractor, "acoustic_dim", 0),
            "textual_dim": getattr(feature_extractor, "textual_dim", 0),
        }

        logger.info(
            "Extracted features for %s: shape=%s, acoustic_dim=%s, textual_dim=%s",
            dataset_name,
            getattr(dataset_features, "shape", None),
            features[dataset_name]["acoustic_dim"],
            features[dataset_name]["textual_dim"],
        )

    if not features:
        raise RuntimeError("No features were extracted for any dataset.")

    data_handler.add_features(features, to_extract=list(features.keys()))
    logger.info("Loaded features for datasets: %s", sorted(features.keys()))

    return data_handler, feature_extractor


# -----------------------------------------------------------------------------
# Distance helpers
# -----------------------------------------------------------------------------
def pairwise_distance_matrix(
    f1: np.ndarray,
    f2: np.ndarray,
    distance_metric: str = "cosine",
) -> np.ndarray:
    """
    Compute full pairwise distance matrix between two 2D feature arrays.
    """
    if f1.ndim == 1:
        f1 = f1.reshape(1, -1)
    if f2.ndim == 1:
        f2 = f2.reshape(1, -1)

    if distance_metric == "cosine":
        return cosine_distances(f1, f2)
    if distance_metric == "euclidean":
        return euclidean_distances(f1, f2)

    raise ValueError(f"Unknown distance metric: {distance_metric}")


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Row-wise L2 normalization.
    """
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (denom + eps)


def compute_distance_all_pairs(
    features1: np.ndarray | tuple[np.ndarray, np.ndarray],
    features2: np.ndarray | tuple[np.ndarray, np.ndarray],
    distance_metric: str = "cosine",
    normalize_modalities_before_distance: bool = True,
) -> tuple[float, float]:
    """Minimum all-pairs distance between two speakers -> (acoustic, textual)."""
    if isinstance(features1, tuple) and isinstance(features2, tuple):
        f1_acoustic, f1_textual = features1
        f2_acoustic, f2_textual = features2

        if f1_acoustic.ndim == 1:
            f1_acoustic = f1_acoustic.reshape(1, -1)
        if f2_acoustic.ndim == 1:
            f2_acoustic = f2_acoustic.reshape(1, -1)
        if f1_textual.ndim == 1:
            f1_textual = f1_textual.reshape(1, -1)
        if f2_textual.ndim == 1:
            f2_textual = f2_textual.reshape(1, -1)

        if normalize_modalities_before_distance:
            f1_acoustic = l2_normalize_rows(f1_acoustic)
            f2_acoustic = l2_normalize_rows(f2_acoustic)
            f1_textual = l2_normalize_rows(f1_textual)
            f2_textual = l2_normalize_rows(f2_textual)

        d_acoustic = pairwise_distance_matrix(
            f1_acoustic, f2_acoustic, distance_metric=distance_metric
        )
        d_textual = pairwise_distance_matrix(
            f1_textual, f2_textual, distance_metric=distance_metric
        )

        acoustic_dist = float(np.min(d_acoustic))
        textual_dist = float(np.min(d_textual))
        return acoustic_dist, textual_dist

    # Single-view case
    f1 = features1
    f2 = features2

    if not isinstance(f1, np.ndarray) or not isinstance(f2, np.ndarray):
        raise TypeError(
            "Single-view distance expects numpy arrays; got "
            f"{type(f1).__name__} and {type(f2).__name__}"
        )

    dists = pairwise_distance_matrix(f1, f2, distance_metric=distance_metric)
    min_dist = float(np.min(dists))
    return min_dist, 0.0


# -----------------------------------------------------------------------------
# Speaker aggregation
# -----------------------------------------------------------------------------
def build_speaker_samples(data_handler: DataHandler) -> dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]]:
    """Build speaker -> sample-feature matrix (or modality tuple) from loaded split-local features."""
    datasets_features = data_handler.datasets_features
    speaker_map_per_set = data_handler.speaker_map_per_set

    speaker_samples: dict[str, list[np.ndarray | tuple[np.ndarray, np.ndarray]]] = {}

    for dataset_name, features_data in datasets_features.items():
        if dataset_name not in speaker_map_per_set:
            logger.warning(
                "Features exist for dataset %s but no matching speaker map exists; skipping",
                dataset_name,
            )
            continue

        logger.info("Processing features for dataset: %s", dataset_name)

        if isinstance(features_data, dict):
            features = features_data.get("features")
            acoustic_dim = int(features_data.get("acoustic_dim", 0) or 0)
        else:
            features = features_data
            acoustic_dim = 0

        if features is None:
            logger.warning("No features found for dataset %s", dataset_name)
            continue

        speaker_map = speaker_map_per_set[dataset_name]

        for speaker_id, positions in speaker_map.items():
            if speaker_id not in speaker_samples:
                speaker_samples[speaker_id] = []

            speaker_feat_array = features[positions]

            if acoustic_dim > 0:
                acoustic_features = speaker_feat_array[:, :acoustic_dim]
                textual_features = speaker_feat_array[:, acoustic_dim:]
                speaker_samples[speaker_id].append((acoustic_features, textual_features))
            else:
                speaker_samples[speaker_id].append(speaker_feat_array)

    combined_speaker_samples: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]] = {}

    for speaker_id, chunks in speaker_samples.items():
        if not chunks:
            continue

        first_chunk = chunks[0]

        if isinstance(first_chunk, tuple):
            acoustic_list = [chunk[0] for chunk in chunks]  # type: ignore[index]
            textual_list = [chunk[1] for chunk in chunks]   # type: ignore[index]
            combined_speaker_samples[speaker_id] = (
                np.vstack(acoustic_list),
                np.vstack(textual_list),
            )
        else:
            combined_speaker_samples[speaker_id] = np.vstack(chunks)  # type: ignore[arg-type]

    return combined_speaker_samples


# -----------------------------------------------------------------------------
# Main computation
# -----------------------------------------------------------------------------
def compute_pairwise_distances(
    data_handler: DataHandler,
    distance_metric: str = "cosine",
) -> list[tuple[str, str, float, float]]:
    """
    Compute pairwise distances between all speakers visible in loaded split-local datasets.
    """
    train_speakers = set(data_handler.data.get("all_speakers", []))
    test_speakers = set(data_handler.speaker_map_per_set.get("test", {}).keys())
    all_speakers = sorted(train_speakers | test_speakers)

    if not all_speakers:
        raise ValueError("No speakers found in dataset")

    logger.info("Preparing speaker sample matrices...")
    speaker_samples = build_speaker_samples(data_handler)

    valid_speakers = [
        speaker_id
        for speaker_id in all_speakers
        if speaker_id in speaker_samples
    ]

    logger.info(
        "Computing pairwise distances for %d speakers (%d valid with features)",
        len(all_speakers),
        len(valid_speakers),
    )

    if not valid_speakers:
        raise ValueError("No speakers with valid features found")

    total_pairs = len(valid_speakers) * (len(valid_speakers) - 1) // 2
    distances: list[tuple[str, str, float, float]] = []
    processed = 0

    for i, source in enumerate(valid_speakers):
        source_features = speaker_samples[source]

        for target in valid_speakers[i + 1:]:
            target_features = speaker_samples[target]

            acoustic_dist, textual_dist = compute_distance_all_pairs(
                source_features,
                target_features,
                distance_metric=distance_metric,
                normalize_modalities_before_distance=True,
            )

            distances.append((source, target, acoustic_dist, textual_dist))
            processed += 1

            if processed % 5000 == 0 or processed == total_pairs:
                logger.info(
                    "Progress: %d/%d pairs (%.1f%%)",
                    processed,
                    total_pairs,
                    100.0 * processed / max(total_pairs, 1),
                )

    logger.info("Completed computing %d pairwise distances", len(distances))
    return distances


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------
def save_distances_to_csv(
    distances: list[tuple[str, str, float, float]],
    output_path: str = "all_distances.csv",
) -> pd.DataFrame:
    df = pd.DataFrame(
        distances,
        columns=["source", "target", "acoustic_distance", "textual_distance"],
    )
    df.to_csv(output_path, index=False)
    logger.info("Saved %d distances to %s", len(distances), output_path)
    return df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise distances between all subjects"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Configuration overrides in OmegaConf/Hydra key=value format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="all_distances.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default=None,
        choices=["cosine", "euclidean"],
        help="Distance metric to use; if omitted, uses cfg.clustering.distance_calc when available",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of features by clearing cached feature files",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> pd.DataFrame | None:
    rank, device, is_distributed = setup_dist()

    try:
        args = parse_args()

        if rank == 0:
            logger.info("=" * 60)
            logger.info("Starting pairwise distance computation")
            logger.info("=" * 60)
            logger.info("Device: %s", device)
            logger.info("Config: %s", args.config)
            if args.overrides:
                logger.info("Overrides: %s", args.overrides)

        cfg = load_config(args.config, args.overrides)

        required_keys = ["data_preparation"]
        for key in required_keys:
            if key not in cfg:
                raise ValueError(f"Missing required configuration section: {key}")

        if rank == 0:
            logger.info("Loading dataset and features...")
        data_handler, _feature_extractor = load_dataset_and_features(
            cfg=cfg,
            recompute=args.recompute,
        )

        distance_metric = args.distance_metric
        if distance_metric is None:
            distance_metric = OmegaConf.select(cfg, "clustering.distance_calc")

        if rank == 0:
            logger.info("Using distance metric: %s", distance_metric)

        if is_distributed and rank != 0:
            logger.info("Rank %d skipping final computation; only rank 0 computes and writes output", rank)
            return None

        logger.info("Computing pairwise distances...")
        distances = compute_pairwise_distances(
            data_handler=data_handler,
            distance_metric=distance_metric,
        )

        logger.info("Saving results to: %s", args.output)
        df = save_distances_to_csv(distances, args.output)

        logger.info("=" * 60)
        logger.info("Summary:")
        logger.info("  Total unique pairs: %d", len(distances))
        logger.info(
            "  Acoustic distance range: [%.4f, %.4f]",
            df["acoustic_distance"].min(),
            df["acoustic_distance"].max(),
        )
        logger.info(
            "  Acoustic mean distance: %.4f",
            df["acoustic_distance"].mean(),
        )
        logger.info(
            "  Textual distance range: [%.4f, %.4f]",
            df["textual_distance"].min(),
            df["textual_distance"].max(),
        )
        logger.info(
            "  Textual mean distance: %.4f",
            df["textual_distance"].mean(),
        )
        logger.info("  Output file: %s", args.output)
        logger.info("=" * 60)
        logger.info("Complete!")

        return df

    except Exception as e:
        logger.error("An error occurred during distance computation", exc_info=True)
        if rank == 0:
            logger.error(f"Experiment failed with exception: {e}")
        return None

    finally:
        cleanup_dist(is_distributed)


if __name__ == "__main__":
    main()